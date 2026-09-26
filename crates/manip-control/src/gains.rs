//! Per-joint gains.

use nalgebra::DVector;

/// Per-joint PD gains [N·m/rad, N·m·s/rad] (N/m, N·s/m for prismatic).
#[derive(Debug, Clone, PartialEq)]
pub struct JointGains {
    pub kp: DVector<f64>,
    pub kd: DVector<f64>,
}

impl JointGains {
    pub fn new(kp: DVector<f64>, kd: DVector<f64>) -> Self {
        assert_eq!(kp.len(), kd.len(), "kp and kd have different lengths");
        Self { kp, kd }
    }

    pub fn uniform(n: usize, kp: f64, kd: f64) -> Self {
        Self::new(DVector::from_element(n, kp), DVector::from_element(n, kd))
    }

    pub fn zeros(n: usize) -> Self {
        Self::uniform(n, 0.0, 0.0)
    }

    pub fn len(&self) -> usize {
        self.kp.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kp.is_empty()
    }

    /// Linearly blends from `self` to `other` with `s ∈ [0, 1]`.
    ///
    /// When moving from "stiff hold" to "soft tracking" at startup, switching
    /// gains in one step turns the target-vs-actual error into torque instantly.
    pub fn lerp(&self, other: &JointGains, s: f64) -> JointGains {
        let s = s.clamp(0.0, 1.0);
        JointGains {
            kp: &self.kp * (1.0 - s) + &other.kp * s,
            kd: &self.kd * (1.0 - s) + &other.kd * s,
        }
    }
}
