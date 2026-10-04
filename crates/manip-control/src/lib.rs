//! Arm control laws.
//!
//! ```text
//!  target (leader / planner)       ArmState (built once per cycle by manip-model)
//!        │                               │
//!   shaper (velocity/accel limits)       │
//!        │ q_ref, v_ref, a_ref           │
//!        ▼                               ▼
//!   ┌─────────── Controller ─────────────────┐
//!   │ GravityComp / JointImpedance           │
//!   └────────────────────────────────────────┘
//!        │ per-axis MIT command (q, v, kp, kd, τff)
//!        ▼
//!   SafetyGate → Plant (hardware / MuJoCo)
//! ```
//!
//! # Division of labor: stiffness in the motor, model as feedforward
//!
//! Joint position stiffness (kp, kd) is run by the PD inside the motor at kHz.
//! With our cycle (a few hundred Hz) plus CAN round-trip latency, closing the
//! same stiffness in torque adds phase lag from the delay and tends to
//! oscillate. The model (gravity, inertia, Coriolis) goes into `τff` as
//! **feedforward**, which is harmless even when delayed. Torque-level
//! controllers (OSC and the other whole-body controllers) live in manip-wbc.
//!
//! # Ordering and units
//!
//! Everything is in manip-model's independent-DOF order, in SI units of the
//! model frame (rad / m, rad/s / m/s, N·m / N). Motor sign, zero offset and
//! gear ratio are unknown here.

pub mod command;
pub mod friction;
pub mod gains;
pub mod joint;
pub mod shaper;

pub use command::{AxisCmd, JointCommand};
pub use friction::FrictionModel;
pub use gains::JointGains;
pub use joint::{Feedforward, JointImpedance};
pub use shaper::{JointRef, JointShaper, ShaperLimits, TcpShaper, TcpShaperLimits};
