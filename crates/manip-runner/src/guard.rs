//! Workspace and self-collision safety.
//!
//! Two enforcement paths, one model:
//!
//! - **OSC** gets control barrier functions in its top priority level
//!   ([`SafetyModel::cbfs`]): each wall of the workspace box, for each
//!   monitored point, and each link pair closer than the activation distance.
//!   The arm decelerates and stops at the boundary instead of hitting it.
//! - **Joint tracking** has no QP, so the shaped reference is guarded
//!   ([`SafetyModel::violation`]): a reference step that would make the
//!   violation worse is refused and the reference holds. Steps that reduce
//!   it are allowed, so an arm that starts outside can still come back.
//!
//! Park is not guarded: it must be able to fold the arm from anywhere.

use manip_control::Cbf;
use manip_model::collision::SelfCollision;
use manip_model::{ArmModel, ArmState};
use nalgebra::{DVector, Vector3};

use crate::config::SafetySection;

/// Only walls / pairs closer than this become QP constraints [m].
const ACTIVATION: f64 = 0.15;

pub struct SafetyModel {
    box_min: Vector3<f64>,
    box_max: Vector3<f64>,
    /// (link, point in link frame).
    points: Vec<(String, Vector3<f64>)>,
    collision: Option<SelfCollision>,
    margin: f64,
    joint_margin: f64,
}

impl SafetyModel {
    pub fn build(cfg: &SafetySection, arm: &ArmModel, exclude_pose: Option<&[f64]>) -> Result<Self, String> {
        let tcp = arm.tcp_spec();
        let mut points = vec![(tcp.link.clone(), Vector3::from(tcp.xyz))];
        for p in &cfg.points {
            arm.link_joint(&p.link).map_err(|e| e.to_string())?;
            points.push((p.link.clone(), Vector3::from(p.xyz)));
        }
        let collision = if cfg.self_collision {
            let mut sc = SelfCollision::build(arm).map_err(|e| e.to_string())?;
            if let Some(q) = exclude_pose {
                let dropped = sc.exclude_close_at(arm, q, cfg.collision_margin);
                if !dropped.is_empty() {
                    log::info!(
                        "self-collision: not checking {} pairs already in contact at the exclusion pose: {:?}",
                        dropped.len(),
                        dropped
                    );
                }
            }
            Some(sc)
        } else {
            None
        };
        Ok(Self {
            box_min: Vector3::from(cfg.box_min),
            box_max: Vector3::from(cfg.box_max),
            points,
            collision,
            margin: cfg.collision_margin,
            joint_margin: cfg.joint_margin,
        })
    }

    /// Violation for the joint-tracking guard: like [`Self::violation`] but
    /// against the box shrunk by the joint margin (room for tracking error).
    pub fn joint_violation(&self, arm: &ArmModel, q: &[f64]) -> f64 {
        self.violation_inset(arm, q, self.joint_margin)
    }

    /// How far `q` is from being safe: the sum of every monitored point's
    /// distance outside the box plus the self-collision margin intrusion.
    /// 0 when safe.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn violation(&self, arm: &ArmModel, q: &[f64]) -> f64 {
        self.violation_inset(arm, q, 0.0)
    }

    fn violation_inset(&self, arm: &ArmModel, q: &[f64], inset: f64) -> f64 {
        let zero = vec![0.0; q.len()];
        let mut v = 0.0;
        for (link, local) in &self.points {
            if let Ok(ps) = arm.point_state(q, &zero, link, *local) {
                for k in 0..3 {
                    v += (self.box_min[k] + inset - ps.p[k]).max(0.0) + (ps.p[k] - self.box_max[k] + inset).max(0.0);
                }
            }
        }
        if let Some(sc) = &self.collision {
            v += (self.margin - sc.min_distance(arm, q)).max(0.0);
        }
        v
    }

    /// Barrier constraints for the OSC at the current state.
    pub fn cbfs(&self, arm: &ArmModel, s: &ArmState) -> Vec<Cbf> {
        let (q, v) = (s.q.as_slice(), s.v.as_slice());
        let n = q.len();
        let mut out = Vec::new();
        for (link, local) in &self.points {
            let Ok(ps) = arm.point_state(q, v, link, *local) else { continue };
            for k in 0..3 {
                let row = ps.jacobian.row(k).transpose();
                // Upper wall: h = max − p.
                let h = self.box_max[k] - ps.p[k];
                if h < ACTIVATION {
                    out.push(Cbf { grad: -&row, drift: -ps.jdot_v[k], h, h_dot: -ps.v[k] });
                }
                // Lower wall: h = p − min.
                let h = ps.p[k] - self.box_min[k];
                if h < ACTIVATION {
                    out.push(Cbf { grad: row.clone(), drift: ps.jdot_v[k], h, h_dot: ps.v[k] });
                }
            }
        }
        if let Some(sc) = &self.collision {
            let vv = DVector::from_column_slice(v);
            for p in sc.close_pairs(arm, q, ACTIVATION) {
                // h = d − margin; ḣ = n·(v_b − v_a); ḧ ≈ n·(J_b − J_a)·q̈ (the
                // J̇v and normal-rotation terms are dropped: small at arm speeds,
                // and α keeps the approach slow near the margin anyway).
                let ja = arm.point_jacobian_at(q, p.joint_a, &p.point_a);
                let jb = arm.point_jacobian_at(q, p.joint_b, &p.point_b);
                let nt = p.normal.transpose();
                let grad = (nt * (&jb - &ja)).transpose();
                let h_dot = grad.dot(&vv);
                out.push(Cbf { grad: DVector::from_iterator(n, grad.iter().cloned()), drift: 0.0, h: p.distance - self.margin, h_dot });
            }
        }
        out
    }
}
