//! Joint friction model for feedforward compensation.
//!
//! `τ_f(v) = Fc·tanh(v / ε) + Fv·v`: the torque the motor has to add to cancel
//! Coulomb and viscous friction at joint velocity `v`.
//!
//! # Which velocity
//!
//! Compensating with the **measured** velocity is the classic way to make a
//! limit cycle: near standstill the measured sign flips with noise (DAMIAO
//! reports velocity in 12 bits, 0.015 rad/s per LSB on a DM4310), and a
//! Coulomb term that follows it pushes the joint back and forth. So the joint
//! impedance law uses the **reference** velocity from the shaper, which is
//! clean and is what the joint is supposed to do. The OSC has no joint
//! reference, so it uses the measured velocity with a wider `ε`.
//!
//! The model does nothing at standstill (`tanh(0) = 0`): it cannot remove the
//! static error friction leaves when holding. That needs integral action.

use nalgebra::DVector;

#[derive(Debug, Clone, PartialEq)]
pub struct FrictionModel {
    /// Coulomb friction per joint [N·m] ([N] for prismatic).
    pub coulomb: DVector<f64>,
    /// Viscous friction per joint [N·m·s/rad].
    pub viscous: DVector<f64>,
    /// Velocity over which the Coulomb term ramps in [rad/s].
    pub v_eps: f64,
}

impl FrictionModel {
    pub fn zeros(n: usize) -> Self {
        Self {
            coulomb: DVector::zeros(n),
            viscous: DVector::zeros(n),
            v_eps: 0.1,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coulomb.iter().all(|&c| c == 0.0) && self.viscous.iter().all(|&c| c == 0.0)
    }

    /// Torque that cancels friction at velocity `v`.
    pub fn compensation(&self, v: &DVector<f64>) -> DVector<f64> {
        assert_eq!(v.len(), self.coulomb.len(), "velocity length differs from friction model");
        let eps = self.v_eps.max(1e-6);
        DVector::from_iterator(
            v.len(),
            (0..v.len()).map(|i| self.coulomb[i] * (v[i] / eps).tanh() + self.viscous[i] * v[i]),
        )
    }

    /// The model restricted to the given joints (OSC works on the TCP chain only).
    pub fn select(&self, idx: &[usize]) -> Self {
        Self {
            coulomb: DVector::from_iterator(idx.len(), idx.iter().map(|&i| self.coulomb[i])),
            viscous: DVector::from_iterator(idx.len(), idx.iter().map(|&i| self.viscous[i])),
            v_eps: self.v_eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposes_nothing_at_rest_and_saturates_when_moving() {
        let f = FrictionModel {
            coulomb: DVector::from_vec(vec![0.3, 0.1]),
            viscous: DVector::from_vec(vec![0.0, 0.5]),
            v_eps: 0.05,
        };
        let z = f.compensation(&DVector::zeros(2));
        assert_eq!(z, DVector::zeros(2));
        let t = f.compensation(&DVector::from_vec(vec![1.0, -2.0]));
        assert!((t[0] - 0.3).abs() < 1e-6);
        assert!((t[1] - (-0.1 - 1.0)).abs() < 1e-6);
    }
}
