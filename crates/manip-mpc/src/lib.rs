//! Model predictive control for fixed-base arms.
//!
//! ```text
//!   goal (TCP pose over time, posture)
//!        │                          every MPC period (tens of ms)
//!        ▼
//!   Planner::plan(arm, q, v, t) ──► JointPlan (q, v, q̈ knots over the horizon)
//!                                        │ sample(now)   every control cycle (500 Hz)
//!                                        ▼
//!                     manip_wbc::JointTracking (exact torque / joint limits, barriers)
//!                                        │
//!                                        ▼ per-axis MIT command
//! ```
//!
//! The planner works on the TCP-chain DOFs (`ArmModel::tcp_chain`); other
//! DOFs (the gripper) keep the caller's reference (`JointPlan::sample_full`).
//!
//! Implementations:
//! - [`LtvMpc`]: linear time-varying MPC at the acceleration level, one dense
//!   QP per period (misa-wbc), real-time iteration around the previous plan.
//! - [`IlqrMpc`]: torque-level nonlinear MPC (iLQR) on the full rigid-body
//!   dynamics with misarta's analytical derivatives; soft state constraints.
//!
//! No I/O; time is the caller's clock in seconds.

pub mod condense;
pub mod ilqr;
pub mod ltv;
pub mod plan;

pub use ilqr::{IlqrConfig, IlqrMpc};
pub use ltv::{LtvConfig, LtvMpc, WorkspaceBox};
pub use plan::JointPlan;

use manip_model::ArmModel;
use nalgebra::{DVector, Isometry3};

/// What the planner should achieve.
pub struct MpcGoal<'a> {
    /// TCP target as a function of time (the caller's clock), evaluated at
    /// every knot of the horizon.
    pub tcp: &'a dyn Fn(f64) -> Isometry3<f64>,
    /// Posture for the redundancy (all independent DOFs); `None` = keep the
    /// current configuration.
    pub posture: Option<&'a DVector<f64>>,
}

/// Per-plan breakdown.
#[derive(Debug, Clone, Default)]
pub struct MpcReport {
    pub cost: f64,
    /// Wall time of the whole plan, the linearization and the QP [µs].
    pub total_us: f64,
    pub linearize_us: f64,
    pub qp_us: f64,
    pub qp_iterations: usize,
    pub status: String,
    /// The state constraints were dropped (the current state violated one).
    pub relaxed: bool,
    /// TCP position error now, and predicted at the end of the horizon [m];
    /// orientation error at the end [rad].
    pub tcp_pos_err_now: f64,
    pub tcp_pos_err_end: f64,
    pub tcp_rot_err_end: f64,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MpcError {
    #[error("could not set up the MPC: {0}")]
    Setup(String),
    #[error("QP not solved: {0}")]
    Solve(String),
}

/// A receding-horizon planner.
pub trait Planner {
    /// Plan from the state `(q, v)` (all independent DOFs) at time `t`.
    fn plan(&mut self, arm: &ArmModel, q: &[f64], v: &[f64], t: f64, goal: &MpcGoal) -> Result<(JointPlan, MpcReport), MpcError>;
    /// Forget the warm start (on mode switches).
    fn reset(&mut self);
    /// Distance the plan keeps from the joint limits [rad].
    fn q_margin(&self) -> f64;
}

/// A joint-space target moved inside the range a planner aims for (the
/// TCP-chain DOFs clamped to the limits less `margin`; other DOFs unchanged).
///
/// A TCP goal computed from a joint target on or past a limit is out of reach
/// of a planner that keeps `margin` away from it. On the real B601-DM, the
/// leader folded at joint2 = joint3 = 0 (their upper limits) left a 5 mm
/// residual, and the planner chased it by reshaping the arm: a 2 Hz, ±1°
/// swing on joint2 / joint4 at rest.
pub fn reachable_joint_target(arm: &ArmModel, q: &DVector<f64>, margin: f64) -> DVector<f64> {
    let mut out = q.clone();
    for i in arm.tcp_chain() {
        let d = &arm.dofs()[i];
        let (lo, hi) = (d.q_min + margin, d.q_max - margin);
        if lo <= hi {
            out[i] = out[i].clamp(lo, hi);
        }
    }
    out
}
