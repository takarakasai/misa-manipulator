//! One control cycle as a function: mode requests + target + observation in,
//! gated command out.
//!
//! The live loop (`app.rs`) and the replayer (`replay.rs`) both go through
//! [`Policy::step`] and nothing else, so "same inputs, same command" can be
//! checked bit for bit. Everything that is not a pure function of those
//! inputs — wall-clock time, Ctrl-C, the leader thread, the startup sequence —
//! stays in the live loop and reaches the policy only as recorded inputs
//! (requests and target).

use std::time::Duration;

use manip_control::JointCommand;
use manip_mpc::JointPlan;
use manip_wbc::{JointTracking, Osc};
use manip_model::{ArmModel, ArmState};
use misa_core::{AxisCommand, AxisId, Command, ControlMode, Observation, SafetyGate, SafetyVerdict};
use nalgebra::DVector;

use crate::assemble;
use crate::config::RobotProfile;
use crate::supervisor::{Mode, Supervisor, Target, TickInfo};

pub struct Policy {
    sup: Supervisor,
    gate: SafetyGate,
    cmd: Command,
    dt: f64,
    /// Per axis: cap on the commanded impedance force (see `cap_force`).
    force_limit: Vec<f64>,
}

/// What one step produced (beyond the command itself).
pub struct StepOut {
    pub state: ArmState,
    /// The control law's output before the SafetyGate.
    pub joint_command: JointCommand,
    pub info: TickInfo,
    pub verdict: SafetyVerdict,
    /// Axes whose command `cap_force` pulled in this step.
    pub force_capped: Vec<usize>,
}

impl Policy {
    /// `q0` is the measured pose before energizing; the policy starts holding there.
    pub fn new(profile: &RobotProfile, arm: &ArmModel, q0: &DVector<f64>) -> Result<Self, String> {
        Ok(Self {
            sup: Supervisor::new(
                assemble::supervisor_config(profile, arm)?,
                Osc::new(assemble::osc_config(profile, arm)?),
                JointTracking::new(assemble::tracking_config(profile, arm)?),
                arm,
                q0,
            ),
            gate: SafetyGate::new(assemble::safety_config(profile, arm)),
            cmd: Command::idle(arm.n()),
            dt: 1.0 / profile.control.rate_hz,
            force_limit: assemble::force_limits(profile, arm),
        })
    }

    pub fn mode(&self) -> Mode {
        self.sup.mode()
    }

    /// Clock of the next step [s] (the time a plan requested now starts at).
    pub fn time(&self) -> f64 {
        self.sup.time()
    }

    pub fn osc_failures(&self) -> u64 {
        self.sup.osc_failures
    }

    /// The gated command from the last step.
    pub fn command(&self) -> &Command {
        &self.cmd
    }

    /// Arm state for an observation (positions / velocities in model frame).
    pub fn state(arm: &ArmModel, obs: &Observation) -> ArmState {
        let q: Vec<f64> = obs.axes().iter().map(|a| a.position_rad).collect();
        let v: Vec<f64> = obs.axes().iter().map(|a| a.velocity_rad_s).collect();
        arm.evaluate(&q, &v)
    }

    /// One cycle: apply `requests` in order, take a newly arrived `plan` (an
    /// input like the target: the planner runs outside, and the log records
    /// which cycle each plan reached), run the active control law toward
    /// `target`, then the SafetyGate.
    pub fn step(&mut self, arm: &ArmModel, obs: &Observation, requests: &[Mode], target: &Target, plan: Option<&JointPlan>) -> StepOut {
        let s = Self::state(arm, obs);
        for &m in requests {
            self.sup.request(m, arm, &s);
        }
        if let Some(p) = plan {
            self.sup.set_plan(p.clone());
        }
        let (jc, info) = self.sup.tick(arm, &s, target, self.dt);
        to_command(&jc, &mut self.cmd);
        let verdict = self.gate.apply(&mut self.cmd, obs, Duration::from_secs_f64(self.dt));
        // Last, after the gate (which may move the position command).
        let force_capped = cap_force(&mut self.cmd, obs, &self.force_limit);
        StepOut {
            state: s,
            joint_command: jc,
            info,
            verdict,
            force_capped,
        }
    }
}

fn to_command(jc: &JointCommand, cmd: &mut Command) {
    for (i, a) in jc.axes.iter().enumerate() {
        if let Some(c) = cmd.get_mut(AxisId::new(i as u16)) {
            *c = AxisCommand {
                mode: ControlMode::Impedance,
                position_rad: a.q,
                velocity_rad_s: a.v,
                torque_ff_nm: a.tau,
                kp_nm_per_rad: a.kp,
                kd_nm_s_per_rad: a.kd,
            };
        }
    }
}

/// Keep what each motor is told to exert within `limit`: the MIT law
/// `kp·(q_cmd − q) + kd·(v_cmd − v) + τ_ff`, evaluated at the measurement, is
/// scaled back toward the measured state (position and velocity errors
/// together; `τ_ff` itself clamped first). The SafetyGate only bounds `τ_ff`;
/// the motor's own PD had no bound, so a blocked joint whose reference kept
/// moving was pushed with the motor's full torque. Returns the capped axes.
pub(crate) fn cap_force(cmd: &mut Command, obs: &Observation, limit: &[f64]) -> Vec<usize> {
    let mut capped = Vec::new();
    for (i, lim) in limit.iter().enumerate() {
        let id = AxisId::new(i as u16);
        let (Some(o), Some(c)) = (obs.get(id).copied(), cmd.get_mut(id)) else { continue };
        if !lim.is_finite() {
            continue;
        }
        let lim = lim.max(0.0);
        if c.mode == ControlMode::Torque {
            let t = c.torque_ff_nm.clamp(-lim, lim);
            if t != c.torque_ff_nm {
                c.torque_ff_nm = t;
                capped.push(i);
            }
            continue;
        }
        if c.mode != ControlMode::Impedance {
            continue;
        }
        let ff = c.torque_ff_nm.clamp(-lim, lim);
        let pd = c.kp_nm_per_rad * (c.position_rad - o.position_rad) + c.kd_nm_s_per_rad * (c.velocity_rad_s - o.velocity_rad_s);
        let total = pd + ff;
        if ff != c.torque_ff_nm || total.abs() > lim {
            capped.push(i);
            c.torque_ff_nm = ff;
            if total.abs() > lim && pd != 0.0 {
                // Scale both errors so that pd + ff = ±lim.
                let s = ((lim * total.signum() - ff) / pd).clamp(0.0, 1.0);
                c.position_rad = o.position_rad + (c.position_rad - o.position_rad) * s;
                c.velocity_rad_s = o.velocity_rad_s + (c.velocity_rad_s - o.velocity_rad_s) * s;
            }
        }
    }
    capped
}
