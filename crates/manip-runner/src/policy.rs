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
use manip_wbc::Osc;
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
}

/// What one step produced (beyond the command itself).
pub struct StepOut {
    pub state: ArmState,
    /// The control law's output before the SafetyGate.
    pub joint_command: JointCommand,
    pub info: TickInfo,
    pub verdict: SafetyVerdict,
}

impl Policy {
    /// `q0` is the measured pose before energizing; the policy starts holding there.
    pub fn new(profile: &RobotProfile, arm: &ArmModel, q0: &DVector<f64>) -> Result<Self, String> {
        Ok(Self {
            sup: Supervisor::new(
                assemble::supervisor_config(profile, arm)?,
                Osc::new(assemble::osc_config(profile, arm)?),
                arm,
                q0,
            ),
            gate: SafetyGate::new(assemble::safety_config(profile, arm)),
            cmd: Command::idle(arm.n()),
            dt: 1.0 / profile.control.rate_hz,
        })
    }

    pub fn mode(&self) -> Mode {
        self.sup.mode()
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

    /// One cycle: apply `requests` in order, run the active control law toward
    /// `target`, then the SafetyGate.
    pub fn step(&mut self, arm: &ArmModel, obs: &Observation, requests: &[Mode], target: &Target) -> StepOut {
        let s = Self::state(arm, obs);
        for &m in requests {
            self.sup.request(m, arm, &s);
        }
        let (jc, info) = self.sup.tick(arm, &s, target, self.dt);
        to_command(&jc, &mut self.cmd);
        let verdict = self.gate.apply(&mut self.cmd, obs, Duration::from_secs_f64(self.dt));
        StepOut {
            state: s,
            joint_command: jc,
            info,
            verdict,
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
