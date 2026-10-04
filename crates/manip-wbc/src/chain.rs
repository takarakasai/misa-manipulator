//! The arm state restricted to the DOFs that move the TCP.
//!
//! DOFs that don't move the TCP, like gripper fingers, are kept out of the QP
//! (the caller's `base` command is used as-is for them). Including them put the
//! fingers' reflected inertia (tens of kg equivalent via rack & pinion) and the
//! wrist (0.003 kg·m²) in the same `M`, pushing the condition number above
//! 10⁴, and Clarabel returned NumericalFailure at level 0. The reaction force
//! of finger acceleration on the arm is ignored (the fingers weigh tens of g).

use manip_control::JointRef;
use manip_model::{ArmModel, ArmState, Dof};
use nalgebra::{DMatrix, DVector, Isometry3, Vector6};

/// [`ArmState`] restricted to [`ArmModel::tcp_chain`], in chain order.
#[derive(Debug, Clone)]
pub struct ChainState {
    /// Index of each chain DOF among all independent DOFs.
    pub idx: Vec<usize>,
    pub dofs: Vec<Dof>,
    pub q: DVector<f64>,
    pub v: DVector<f64>,
    pub mass: DMatrix<f64>,
    /// Nonlinear effects `h(q, v)` (Coriolis + gravity), without friction.
    pub nle: DVector<f64>,
    pub tcp_pose: Isometry3<f64>,
    pub tcp_twist: Vector6<f64>,
    /// `6 × n`, `[ω; v]` rows, world coordinates.
    pub tcp_jacobian: DMatrix<f64>,
    pub tcp_jdot_v: Vector6<f64>,
}

impl ChainState {
    pub fn new(arm: &ArmModel, s: &ArmState) -> Self {
        let idx = arm.tcp_chain();
        let all = arm.dofs();
        let dofs: Vec<_> = idx.iter().map(|&i| all[i].clone()).collect();
        let n = idx.len();
        let sub = |v: &DVector<f64>| DVector::from_iterator(n, idx.iter().map(|&i| v[i]));
        Self {
            q: sub(&s.q),
            v: sub(&s.v),
            mass: s.mass.select_rows(&idx).select_columns(&idx),
            nle: sub(&s.nle),
            tcp_pose: s.tcp_pose,
            tcp_twist: s.tcp_twist,
            tcp_jacobian: s.tcp_jacobian.select_columns(&idx),
            tcp_jdot_v: s.tcp_jdot_v,
            dofs,
            idx,
        }
    }

    /// Number of chain DOFs.
    pub fn n(&self) -> usize {
        self.idx.len()
    }

    /// Restrict a per-DOF vector (all independent DOFs) to the chain.
    pub fn pick(&self, v: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(self.n(), self.idx.iter().map(|&i| v[i]))
    }

    /// Restrict a joint reference (all independent DOFs) to the chain.
    pub fn pick_ref(&self, r: &JointRef) -> JointRef {
        JointRef {
            q: self.pick(&r.q),
            v: self.pick(&r.v),
            a: self.pick(&r.a),
        }
    }

    /// Torque limit per chain DOF: model effort × `scale` (1e3 if unbounded).
    pub fn torque_limits(&self, scale: f64) -> DVector<f64> {
        DVector::from_iterator(
            self.n(),
            self.dofs.iter().map(|d| {
                let e = d.effort * scale;
                if e.is_finite() { e } else { 1e3 }
            }),
        )
    }
}

pub(crate) fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}
