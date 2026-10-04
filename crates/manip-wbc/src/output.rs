//! From a QP solution (`q̈`, `τ`) to the per-axis MIT command.
//!
//! `τ` goes out as `τff`. If `kd > 0`, motor-side damping is added with
//! `v = v + q̈·dt` as the target velocity (it doesn't act on motion that
//! follows the QP solution, only suppresses high-frequency jitter). If
//! `kp > 0`, the target position is the QP solution integrated since the
//! controller was (re)started, kept within `lead_max` of the measured joint,
//! so the motor's own PD holds the arm on the solved motion between PC
//! cycles. On the real B601-DM, OSC with feedback only in the 500 Hz PC loop
//! chattered at ~30 Hz; motor-side damping removed most of it.

use manip_control::{AxisCmd, JointCommand};
use nalgebra::DVector;

use crate::chain::ChainState;

/// Motor-side PD around the solved motion (per chain DOF).
#[derive(Debug, Clone)]
pub struct MotorGains {
    pub kp: DVector<f64>,
    pub kd: DVector<f64>,
    /// Bound on |integrated reference − measured| [rad].
    pub lead_max: f64,
}

impl MotorGains {
    pub fn zeros(n: usize) -> Self {
        Self {
            kp: DVector::zeros(n),
            kd: DVector::zeros(n),
            lead_max: 0.03,
        }
    }
}

/// Holds the integrated position reference between cycles.
#[derive(Debug, Clone, Default)]
pub struct MotorOutput {
    q_ref: Option<DVector<f64>>,
}

impl MotorOutput {
    /// Forget the integrated reference (on mode switches).
    pub fn reset(&mut self) {
        self.q_ref = None;
    }

    /// Overwrite the chain DOFs of `base` with the MIT command for the solution.
    pub fn command(
        &mut self,
        c: &ChainState,
        qddot: &DVector<f64>,
        tau: &DVector<f64>,
        dt: f64,
        gains: &MotorGains,
        mut base: JointCommand,
    ) -> JointCommand {
        let n = c.n();
        let v_next = &c.v + qddot * dt;
        let q_target = if gains.kp.iter().any(|&k| k > 0.0) {
            let prev = self.q_ref.take().filter(|q| q.len() == n).unwrap_or_else(|| c.q.clone());
            let q_next = DVector::from_iterator(
                n,
                (0..n).map(|k| (prev[k] + v_next[k] * dt).clamp(c.q[k] - gains.lead_max, c.q[k] + gains.lead_max)),
            );
            self.q_ref = Some(q_next.clone());
            q_next
        } else {
            c.q.clone()
        };
        assert!(c.idx.iter().all(|&i| i < base.len()), "base is shorter than the independent DOFs");
        for (k, &i) in c.idx.iter().enumerate() {
            base.axes[i] = AxisCmd {
                q: q_target[k],
                v: v_next[k],
                kp: gains.kp[k],
                kd: gains.kd[k],
                tau: tau[k],
            };
        }
        base
    }
}
