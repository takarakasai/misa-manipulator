//! A virtual CAN arm: motors that speak the `Actuator` trait in **motor frame**
//! units, all backed by one shared rigid-body simulation of the arm.
//!
//! Plugged into [`manip_plant_can::CanArmPlant::with_actuators`], it exercises
//! the real hardware path end to end without hardware: the per-bus threads,
//! the model <-> motor conversion (`sign`, `zero`, `ratio`, and the `ratio²`
//! on the MIT gains), arm/disarm, and the stale-setpoint fallback. A mistake in
//! any of those shows up as an arm that sags, flies off or oscillates, not as
//! a number in a unit test.
//!
//! Like a real MIT motor, each virtual motor runs its PD on every physics
//! substep from the last command it received; the bus thread only updates the
//! command. Physics advances with wall-clock time, so this runs in real time
//! (no `--fast`). A disabled motor produces no torque: the arm falls, as the
//! real one does.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use manip_model::ArmModel;
use manip_plant_can::{BusSpec, MotorSpec};
use misa_actuator::{Actuator, Error, ErrorFlags, MotorFeedback, MotorStatus, Result, RunMode};
use nalgebra::DVector;

/// Longest wall-clock gap simulated in one catch-up, so a stalled caller does
/// not make the next call integrate for seconds.
const MAX_CATCH_UP_S: f64 = 0.05;

/// Time one request/reply takes on the wire, emulated per call. An MIT
/// exchange is two 8-byte frames (~110 µs each at 1 Mbps with stuffing and
/// inter-frame space) plus the motor's turnaround: ~250 µs, so seven motors
/// make a bus cycle of ~1.75 ms (~570 Hz), close to what the real bus will
/// allow. Without it the bus thread spins at over a million cycles per second
/// and hides any timing problem.
pub const DEFAULT_TRANSACTION: std::time::Duration = std::time::Duration::from_micros(250);

#[derive(Debug, Clone, Copy, Default)]
struct MotorCmd {
    enabled: bool,
    /// Motor frame: q [rad], v [rad/s], kp, kd, tau [N·m].
    q: f64,
    v: f64,
    kp: f64,
    kd: f64,
    tau: f64,
}

struct Physics {
    arm: ArmModel,
    q: DVector<f64>,
    v: DVector<f64>,
    /// Per DOF: (sign, zero, ratio) of the motor driving it.
    frames: Vec<(f64, f64, f64)>,
    cmds: Vec<MotorCmd>,
    /// Per DOF: (Coulomb, viscous) joint friction.
    friction: Vec<(f64, f64)>,
    friction_v_eps: f64,
    /// Last applied motor torque per DOF, model frame.
    tau: DVector<f64>,
    h: f64,
    t_sim: f64,
    t0: Instant,
}

impl Physics {
    fn advance(&mut self) {
        let now = self.t0.elapsed().as_secs_f64();
        if now - self.t_sim > MAX_CATCH_UP_S {
            self.t_sim = now - MAX_CATCH_UP_S;
        }
        while self.t_sim + self.h <= now {
            self.step();
            self.t_sim += self.h;
        }
    }

    fn step(&mut self) {
        let n = self.q.len();
        let s = self.arm.evaluate(self.q.as_slice(), self.v.as_slice());
        let mut total = DVector::zeros(n);
        for i in 0..n {
            let (sign, zero, ratio) = self.frames[i];
            let c = self.cmds[i];
            let tau_model = if c.enabled {
                let qm = sign * (self.q[i] - zero) / ratio;
                let vm = sign * self.v[i] / ratio;
                let tau_m = c.kp * (c.q - qm) + c.kd * (c.v - vm) + c.tau;
                sign * tau_m / ratio
            } else {
                0.0
            };
            let e = self.arm.dofs()[i].effort;
            let tau_model = if e.is_finite() { tau_model.clamp(-e, e) } else { tau_model };
            self.tau[i] = tau_model;
            let (fc, fv) = self.friction[i];
            total[i] = tau_model - fc * (self.v[i] / self.friction_v_eps).tanh() - fv * self.v[i];
        }
        let Some(chol) = s.mass.clone().cholesky() else { return };
        let qdd = chol.solve(&(total - &s.nle));
        self.v += qdd * self.h;
        self.q += &self.v * self.h;
        for (i, d) in self.arm.dofs().iter().enumerate() {
            if !d.within(self.q[i]) {
                self.q[i] = d.clamp(self.q[i]);
                self.v[i] = 0.0;
            }
        }
    }

    fn feedback(&self, i: usize) -> MotorFeedback {
        let (sign, zero, ratio) = self.frames[i];
        MotorFeedback {
            position_rad: (sign * (self.q[i] - zero) / ratio) as f32,
            velocity_rad_per_s: (sign * self.v[i] / ratio) as f32,
            torque_nm: (sign * self.tau[i] * ratio) as f32,
            current_a: f32::NAN,
            temperature_c: 30.0,
        }
    }
}

/// Build the virtual motors for `buses` (same shape as `[hardware] bus`), with
/// the arm starting at `q0` (model frame).
pub fn virtual_motors(
    arm: &ArmModel,
    buses: &[BusSpec],
    q0: DVector<f64>,
    friction: Vec<(f64, f64)>,
    friction_v_eps: f64,
    timestep_s: f64,
    transaction: std::time::Duration,
) -> std::result::Result<Vec<Vec<manip_plant_can::Motor>>, String> {
    let n = arm.n();
    let mut frames = vec![(1.0, 0.0, 1.0); n];
    for b in buses {
        for m in &b.motor {
            let i = arm.dof(&m.joint).map_err(|e| e.to_string())?;
            frames[i] = (m.sign, m.zero, m.ratio);
        }
    }
    let phys = Arc::new(Mutex::new(Physics {
        arm: arm.clone(),
        q: q0,
        v: DVector::zeros(n),
        frames,
        cmds: vec![MotorCmd::default(); n],
        friction,
        friction_v_eps: friction_v_eps.max(1e-4),
        tau: DVector::zeros(n),
        h: timestep_s,
        t_sim: 0.0,
        t0: Instant::now(),
    }));
    buses
        .iter()
        .map(|b| {
            b.motor
                .iter()
                .map(|m: &MotorSpec| -> std::result::Result<manip_plant_can::Motor, String> {
                    Ok(Box::new(VirtualMotor {
                        id: m.id,
                        dof: arm.dof(&m.joint).map_err(|e| e.to_string())?,
                        phys: phys.clone(),
                        transaction,
                    }))
                })
                .collect()
        })
        .collect()
}

struct VirtualMotor {
    id: u8,
    dof: usize,
    phys: Arc<Mutex<Physics>>,
    transaction: std::time::Duration,
}

impl VirtualMotor {
    /// One wire transaction: the command lands and the reply is sampled after
    /// the transaction time, like a real request/reply.
    fn with<T>(&self, f: impl FnOnce(&mut Physics) -> T) -> T {
        if !self.transaction.is_zero() {
            std::thread::sleep(self.transaction);
        }
        let mut p = self.phys.lock().unwrap();
        p.advance();
        f(&mut p)
    }
}

impl Actuator for VirtualMotor {
    fn motor_id(&self) -> u8 {
        self.id
    }

    fn enable(&mut self) -> Result<MotorFeedback> {
        let i = self.dof;
        Ok(self.with(|p| {
            // A real MIT motor enables with zero gains until the first command.
            p.cmds[i] = MotorCmd { enabled: true, ..MotorCmd::default() };
            p.feedback(i)
        }))
    }

    fn disable(&mut self) -> Result<()> {
        let i = self.dof;
        self.with(|p| p.cmds[i].enabled = false);
        Ok(())
    }

    fn set_zero(&mut self) -> Result<()> {
        Err(Error::Unsupported("virtual motor: zero is fixed by the profile"))
    }

    fn set_run_mode(&mut self, mode: RunMode) -> Result<()> {
        match mode {
            RunMode::Mit => Ok(()),
            _ => Err(Error::Unsupported("virtual motor: MIT only")),
        }
    }

    fn set_position(&mut self, _pos_rad: f32, _max_speed_rad_s: f32) -> Result<MotorFeedback> {
        Err(Error::Unsupported("virtual motor: MIT only"))
    }

    fn set_velocity(&mut self, _vel_rad_s: f32) -> Result<MotorFeedback> {
        Err(Error::Unsupported("virtual motor: MIT only"))
    }

    fn set_torque(&mut self, torque_nm: f32) -> Result<MotorFeedback> {
        self.mit_control(0.0, 0.0, 0.0, 0.0, torque_nm)
    }

    fn mit_control(&mut self, pos: f32, vel: f32, kp: f32, kd: f32, tau: f32) -> Result<MotorFeedback> {
        let (i, id) = (self.dof, self.id);
        self.with(|p| {
            if !p.cmds[i].enabled {
                return Err(Error::NotEnabled { motor_id: id });
            }
            p.cmds[i] = MotorCmd {
                enabled: true,
                q: pos as f64,
                v: vel as f64,
                kp: kp as f64,
                kd: kd as f64,
                tau: tau as f64,
            };
            Ok(p.feedback(i))
        })
    }

    fn measure(&mut self) -> Result<MotorFeedback> {
        let i = self.dof;
        Ok(self.with(|p| p.feedback(i)))
    }

    fn read_status(&mut self) -> Result<MotorStatus> {
        Ok(MotorStatus {
            voltage_v: 24.0,
            temperature_c: 30.0,
            error: ErrorFlags::default(),
        })
    }
}
