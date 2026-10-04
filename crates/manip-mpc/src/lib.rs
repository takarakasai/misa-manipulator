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
//!
//! No I/O; time is the caller's clock in seconds.

pub mod condense;
pub mod ltv;
pub mod plan;

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
}
