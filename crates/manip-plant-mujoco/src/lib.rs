//! Exposes articara's MuJoCo as a fixed-base arm [`Plant`].
//!
//! # The motors' MIT law runs here
//!
//! Real motors run `τ = kp*(q* − q) + kd*(v* − v) + τff` internally at kHz rates.
//! Here the same expression is evaluated **every physics step** and applied as
//! torque (the command is held for the whole control period). So from the control
//! law's point of view, the only differences between hardware and MuJoCo are
//! "latency, noise and friction".
//!
//! Being an explicit PD, it is only stable for `kd < 2*I/dt` (`I` = joint inertia +
//! armature). Lower [`SimOptions::timestep_s`] if you use a stiff `kd` on light
//! axes such as the wrist or gripper.
//!
//! # Going limp really goes limp
//!
//! Unlike misa-runner's quadruped Plant, `Idle` axes get zero torque (+ joint
//! damping). An arm that goes limp **falls**. That matches the hardware, and
//! catching mistimed disarms in sim matters more.
//!
//! # Mimic-dependent joints
//!
//! articara's MJCF export does not turn mimics into equality constraints, so
//! dependent joints (one finger of the B601) track the leader joint's angle with
//! a stiff PD. This approximates the rigid rack-and-pinion coupling.
//!
//! # What this cannot verify
//!
//! CAN round-trip latency, bus jitter, friction, backlash, dropped receptions.
//! Passing here does not mean the hardware will work (same caveat as misa-runner).

use std::time::Duration;

use articara::mjcf::MjcfExportOptions;
use articara::mujoco_sim::MujocoSim;
use articara::rbd::model::ActuatorMode;
use articara::robot::RobotModel;
use misa_core::{
    Axis, AxisHealth, AxisId, AxisRole, AxisState, AxisTable, Command, ControlMode, Observation,
    Plant, PlantCaps, Time,
};

/// Configuration of a single axis.
#[derive(Debug, Clone)]
pub struct SimAxis {
    /// Joint name in the model (independent DOF).
    pub joint: String,
    /// Reflected rotor inertia [kg·m²] ([kg] for prismatic). MuJoCo's `armature`.
    pub armature: f64,
    /// Joint damping [N·m·s/rad]. Rough stand-in for motor back-EMF and gearbox losses.
    pub damping: f64,
    /// Torque limit [N·m]. No limit if 0 or less.
    pub effort: f64,
}

/// A mimic-dependent joint.
#[derive(Debug, Clone)]
pub struct SimMimic {
    pub joint: String,
    pub source: String,
    pub multiplier: f64,
    pub offset: f64,
    pub armature: f64,
}

#[derive(Debug, Clone)]
pub struct SimOptions {
    pub misa_path: String,
    pub axes: Vec<SimAxis>,
    pub mimics: Vec<SimMimic>,
    /// Control period [s]. Each `exchange` advances physics by this much.
    pub control_period_s: f64,
    /// Physics timestep [s].
    pub timestep_s: f64,
    /// Initial pose (in `axes` order).
    pub initial_q: Vec<f64>,
    /// Base position [m].
    pub base_pos: [f64; 3],
    /// Whether to add a floor (at base height 0). True for a tabletop arm.
    pub ground: bool,
    /// Whether MuJoCo enforces joint limits (in place of mechanical stops).
    pub joint_limits: bool,
    /// Whether to compute link-to-link contacts.
    ///
    /// **Off by default.** The vendor collision meshes interpenetrate in the folded
    /// pose (the B601's zero point) and with the gripper closed, so contact forces push
    /// the joints back (the closed fingers opened by 0.03 m and the holding PD fought
    /// it with thousands of N). To check self-collision, fix the collision meshes first.
    pub self_collision: bool,
}

pub struct MujocoArmPlant {
    robot: RobotModel,
    sim: MujocoSim,
    table: AxisTable,
    caps: PlantCaps,
    axes: Vec<SimAxis>,
    /// Joint index of `axes[i]` in the RobotModel.
    axis_joint: Vec<usize>,
    mimics: Vec<(usize, usize, SimMimic)>,
    substeps: u32,
    armed: bool,
    time: Duration,
    /// Torque applied most recently (reported in the observation).
    last_tau: Vec<f64>,
    tau_buf: Vec<f64>,
}

impl MujocoArmPlant {
    pub fn new(opts: SimOptions) -> Result<Self, String> {
        let mut robot = RobotModel::from_misa(std::path::Path::new(&opts.misa_path))?;
        if opts.initial_q.len() != opts.axes.len() {
            return Err(format!(
                "initial_q length {} does not match axis count {}",
                opts.initial_q.len(),
                opts.axes.len()
            ));
        }
        let mut axis_joint = Vec::with_capacity(opts.axes.len());
        for (a, &q0) in opts.axes.iter().zip(&opts.initial_q) {
            let ji = *robot
                .joint_map
                .get(&a.joint)
                .ok_or_else(|| format!("joint {} is not in the model", a.joint))?;
            axis_joint.push(ji);
            robot.joint_positions[ji] = q0;
        }
        let mut mimics = Vec::new();
        for m in &opts.mimics {
            let ji = *robot
                .joint_map
                .get(&m.joint)
                .ok_or_else(|| format!("mimic joint {} is not in the model", m.joint))?;
            let src = opts
                .axes
                .iter()
                .position(|a| a.joint == m.source)
                .ok_or_else(|| format!("mimic source {} is not an axis", m.source))?;
            robot.joint_positions[ji] = m.multiplier * opts.initial_q[src] + m.offset;
            mimics.push((ji, src, m.clone()));
        }
        // Make every joint torque-driven. The PD runs in `exchange` below.
        for j in robot.joints.iter_mut() {
            j.actuator_mode = ActuatorMode::Torque;
            j.actuator_kp = 0.0;
            j.actuator_kv = 0.0;
        }
        for (a, &ji) in opts.axes.iter().zip(&axis_joint) {
            robot.joints[ji].armature = a.armature;
            robot.joints[ji].joint_damping = a.damping;
        }
        for (ji, _, m) in &mimics {
            robot.joints[*ji].armature = m.armature;
        }
        if !opts.self_collision {
            let names: Vec<String> = robot.links.iter().map(|l| l.name.clone()).collect();
            for (i, a) in names.iter().enumerate() {
                for b in &names[i + 1..] {
                    robot
                        .collision_pairs
                        .push(articara::rbd::model::CollisionPair::new(a.clone(), b.clone(), false));
                }
            }
        }
        robot.rebuild_misarta_model();

        let mj = MjcfExportOptions {
            base_pos: Some(opts.base_pos),
            base_locked_axes: [true; 6],
            add_actuators: true,
            // Torque limits are clamped here instead (to use the per-axis settings).
            bake_actuator_limits: false,
            bake_joint_position_limits: opts.joint_limits,
            timestep: Some(opts.timestep_s),
            ground_plane: if opts.ground {
                Some(articara::mjcf::GroundPlaneCfg { z: 0.0, half_size: 2.0, roll: 0.0, pitch: 0.0 })
            } else {
                None
            },
            ..MjcfExportOptions::default()
        };
        let mut sim = MujocoSim::new(&robot, mj)?;
        // No rewind history needed (it eats memory on long runs).
        sim.set_trace_max(0);

        let substeps = (opts.control_period_s / sim.timestep()).round().max(1.0) as u32;
        let table = AxisTable::new(
            opts.axes
                .iter()
                .map(|a| Axis {
                    name: a.joint.clone(),
                    role: AxisRole::Aux,
                })
                .collect(),
        )?;
        let n = opts.axes.len();
        let caps = PlantCaps {
            modes: vec![
                ControlMode::Position,
                ControlMode::Velocity,
                ControlMode::Torque,
                ControlMode::Impedance,
            ],
            has_imu: false,
            has_contacts: false,
            driven: vec![true; n],
        };
        let nj = robot.joints.len();
        Ok(Self {
            robot,
            sim,
            table,
            caps,
            axes: opts.axes,
            axis_joint,
            mimics,
            substeps,
            armed: false,
            time: Duration::ZERO,
            last_tau: vec![0.0; n],
            tau_buf: vec![0.0; nj],
        })
    }

    /// Physics timestep [s].
    pub fn timestep(&self) -> f64 {
        self.sim.timestep()
    }

    fn joint_state(&self, ji: usize) -> (f64, f64) {
        self.sim
            .joint_q_qd(&self.robot.joints[ji].name)
            .unwrap_or((f64::NAN, f64::NAN))
    }

    fn fill(&self, obs: &mut Observation) {
        obs.time = Time::from_nanos(self.time.as_nanos() as u64);
        for (i, &ji) in self.axis_joint.iter().enumerate() {
            let (q, v) = self.joint_state(ji);
            if let Some(s) = obs.get_mut(AxisId::new(i as u16)) {
                *s = AxisState {
                    position_rad: q,
                    velocity_rad_s: v,
                    torque_nm: Some(self.last_tau[i]),
                    health: AxisHealth {
                        valid: q.is_finite(),
                        age: Duration::ZERO,
                        fault_raw: 0,
                        temperature_c: None,
                        voltage_v: None,
                    },
                };
            }
        }
    }
}

impl Plant for MujocoArmPlant {
    fn axes(&self) -> &AxisTable {
        &self.table
    }

    fn capabilities(&self) -> &PlantCaps {
        &self.caps
    }

    fn arm(&mut self) -> Result<(), String> {
        self.armed = true;
        Ok(())
    }

    fn disarm(&mut self) -> Result<(), String> {
        self.armed = false;
        Ok(())
    }

    fn status_line(&self) -> String {
        format!(
            "mujoco dt={:.1}ms ×{} t={:.2}s",
            self.sim.timestep() * 1e3,
            self.substeps,
            self.time.as_secs_f64()
        )
    }

    fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
        if cmd.len() != self.axes.len() {
            return Err(format!("command axis count {} != {}", cmd.len(), self.axes.len()));
        }
        // Don't advance time before arming. The real arm rests on its stops while the
        // pose is read before arming, but the sim starts from a pose floating in mid-air,
        // so free-falling even one period here would start control with nonzero velocity.
        if !self.armed {
            self.fill(obs);
            return Ok(());
        }
        let dt = self.sim.timestep();
        for _ in 0..self.substeps {
            self.tau_buf.iter_mut().for_each(|t| *t = 0.0);
            for (i, &ji) in self.axis_joint.iter().enumerate() {
                let (q, v) = self.joint_state(ji);
                let a = &cmd.axes()[i];
                let tau = if !self.armed {
                    0.0
                } else {
                    match a.mode {
                        ControlMode::Idle => 0.0,
                        ControlMode::Impedance => {
                            a.kp_nm_per_rad * (a.position_rad - q)
                                + a.kd_nm_s_per_rad * (a.velocity_rad_s - v)
                                + a.torque_ff_nm
                        }
                        // Position control stands in for the real motor's position mode.
                        // Uses the gains carried in the command (no effect if absent).
                        ControlMode::Position => {
                            a.kp_nm_per_rad * (a.position_rad - q) - a.kd_nm_s_per_rad * v
                                + a.torque_ff_nm
                        }
                        ControlMode::Velocity => {
                            a.kd_nm_s_per_rad * (a.velocity_rad_s - v) + a.torque_ff_nm
                        }
                        ControlMode::Torque => a.torque_ff_nm,
                    }
                };
                let e = self.axes[i].effort;
                let tau = if e > 0.0 { tau.clamp(-e, e) } else { tau };
                self.last_tau[i] = tau;
                self.tau_buf[ji] = tau;
            }
            // Dependent joints: stiff PD toward the leader (rack-and-pinion approximation).
            // Stays active while limp (it is a mechanical coupling, independent of power).
            for (ji, src, m) in &self.mimics {
                let (qs, vs) = self.joint_state(self.axis_joint[*src]);
                let (q, v) = self.joint_state(*ji);
                let target = m.multiplier * qs + m.offset;
                let k = m.armature.max(1e-3) * (2.0 * std::f64::consts::PI * 30.0).powi(2);
                let d = 2.0 * m.armature.max(1e-3) * (2.0 * std::f64::consts::PI * 30.0);
                self.tau_buf[*ji] = k * (target - q) + d * (m.multiplier * vs - v);
            }
            self.sim.set_wbc_torques(&self.tau_buf);
            self.sim.step_n_frames(&mut self.robot, 1, false);
            self.time += Duration::from_secs_f64(dt);
        }
        self.fill(obs);
        Ok(())
    }
}
