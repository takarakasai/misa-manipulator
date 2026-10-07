//! Whole-body control for fixed-base arms on misa-wbc's hierarchical QP.
//!
//! ```text
//!  ArmState ──► ChainState (TCP-chain DOFs) ──► chain_dynamics (M, h + friction)
//!                                                   │
//!   task builders (tasks.rs) ──► levels [0: physics & safety, 1: objective, 2: redundancy]
//!                                                   │ solve_levels
//!                                                   ▼
//!                      MotorOutput: q̈, τ ──► per-axis MIT command (q*, v*, kp, kd, τff)
//! ```
//!
//! Controllers built from these blocks:
//! - [`Osc`]: TCP pose target (teleop from a TCP, Cartesian moves).
//! - [`JointTracking`]: a joint trajectory (the WBC half of MPC + WBC: the
//!   planner hands over `(q, v, a)(t)`, this keeps the exact limits).
//!
//! Everything is in manip-model's independent-DOF order and SI units of the
//! model frame. No I/O.

pub mod chain;
pub mod osc;
pub mod output;
pub mod solve;
pub mod tasks;
pub mod tracking;

pub use chain::ChainState;
pub use osc::{Osc, OscConfig, OscError, OscReport, TcpExtras};
pub use output::{MotorGains, MotorOutput};
pub use solve::{solve_levels, Solved, WbcError};
pub use tasks::{pose_error, Cbf, Compliance, ComplianceLimits, JointLimitParams, SingularityParams, TcpGains, TcpRef, TcpTask};
pub use tracking::{JointTracking, TrackingConfig, TrackingReport};
