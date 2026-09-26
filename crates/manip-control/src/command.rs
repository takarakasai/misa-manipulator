//! Control law output: per-axis MIT commands.

use nalgebra::DVector;

/// MIT command to one axis: `τ = kp·(q − q_meas) + kd·(v − v_meas) + tau`.
///
/// Units are SI in the model frame. With `kp = kd = 0` it is a pure torque command.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct AxisCmd {
    pub q: f64,
    pub v: f64,
    pub kp: f64,
    pub kd: f64,
    pub tau: f64,
}

impl AxisCmd {
    /// Torque this command produces at the current state (including the
    /// motor-side PD). For logging and saturation checks.
    pub fn torque_at(&self, q: f64, v: f64) -> f64 {
        self.kp * (self.q - q) + self.kd * (self.v - v) + self.tau
    }
}

/// Commands for all axes, in independent-DOF order.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct JointCommand {
    pub axes: Vec<AxisCmd>,
}

impl JointCommand {
    pub fn len(&self) -> usize {
        self.axes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// Torque produced at the current state (all axes).
    pub fn torque_at(&self, q: &DVector<f64>, v: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(
            self.axes.len(),
            self.axes.iter().enumerate().map(|(i, a)| a.torque_at(q[i], v[i])),
        )
    }
}
