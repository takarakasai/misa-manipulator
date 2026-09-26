//! Joint-space control laws: gravity compensation and joint impedance.

use manip_model::{ArmModel, ArmState};
use nalgebra::DVector;

use crate::command::{AxisCmd, JointCommand};
use crate::gains::JointGains;
use crate::shaper::JointRef;

/// What to put into `τff`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feedforward {
    /// Nothing (motor PD only). For comparison when the model is suspect.
    None,
    /// Gravity `g(q)` only. **Default.** Independent of velocity, so robust to measurement noise.
    #[default]
    Gravity,
    /// Inverse dynamics `M(q)·a_ref + h(q, v_ref)`. Anticipates the inertial
    /// part for fast tracking. A rough reference acceleration makes the torque
    /// equally rough, so use it with a shaped reference.
    InverseDynamics,
}

/// Joint impedance: `τ = kp·(q_ref − q) + kd·(v_ref − v) + τff`.
///
/// The PD runs on the motor side; only `τff` is computed here (see the
/// division of labor in [`crate`]). Setting `gains` to 0 with
/// [`Feedforward::Gravity`] gives gravity compensation (movable by hand).
#[derive(Debug, Clone)]
pub struct JointImpedance {
    pub gains: JointGains,
    pub feedforward: Feedforward,
    /// Per-axis scale on the gravity term. An escape hatch to absorb mass
    /// differences between model and hardware. Record the reason whenever you
    /// move it away from 1.0.
    pub gravity_scale: DVector<f64>,
}

impl JointImpedance {
    pub fn new(gains: JointGains, feedforward: Feedforward) -> Self {
        let n = gains.len();
        Self {
            gains,
            feedforward,
            gravity_scale: DVector::from_element(n, 1.0),
        }
    }

    /// Gravity compensation only (kp = 0). `kd` is the damping felt when moving
    /// by hand; with 0 the arm won't stop after being swung.
    pub fn gravity_comp(n: usize, kd: f64) -> Self {
        Self::new(JointGains::uniform(n, 0.0, kd), Feedforward::Gravity)
    }

    pub fn command(&self, arm: &ArmModel, s: &ArmState, r: &JointRef) -> JointCommand {
        let n = arm.n();
        assert_eq!(r.q.len(), n, "reference length differs from the number of independent DOFs");
        let tau = match self.feedforward {
            Feedforward::None => DVector::zeros(n),
            Feedforward::Gravity => s.gravity.component_mul(&self.gravity_scale),
            Feedforward::InverseDynamics => {
                // Apply the scale to the gravity part only: ID − g + scale·g.
                let id = arm.inverse_dynamics(s.q.as_slice(), r.v.as_slice(), r.a.as_slice());
                id - &s.gravity + s.gravity.component_mul(&self.gravity_scale)
            }
        };
        JointCommand {
            axes: (0..n)
                .map(|i| AxisCmd {
                    q: r.q[i],
                    v: r.v[i],
                    kp: self.gains.kp[i],
                    kd: self.gains.kd[i],
                    tau: tau[i],
                })
                .collect(),
        }
    }
}
