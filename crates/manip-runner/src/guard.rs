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

use std::collections::HashMap;

use manip_wbc::Cbf;
use manip_model::collision::{PairDistance, SelfCollision};
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
    /// Model joint index of the last proximal (shoulder / elbow) joint: the
    /// first half of the TCP chain. See [`Self::allowance`].
    proximal: usize,
    /// Per link pair, the clearance above the margin at the exclusion pose,
    /// which caps the allowance: that pose is known to be safe.
    allowance_cap: HashMap<(String, String), f64>,
}

impl SafetyModel {
    pub fn build(cfg: &SafetySection, arm: &ArmModel, exclude_pose: Option<&[f64]>) -> Result<Self, String> {
        let tcp = arm.tcp_spec();
        let mut points = vec![(tcp.link.clone(), Vector3::from(tcp.xyz))];
        for p in &cfg.points {
            arm.link_joint(&p.link).map_err(|e| e.to_string())?;
            points.push((p.link.clone(), Vector3::from(p.xyz)));
        }
        let mut allowance_cap = HashMap::new();
        let collision = if cfg.self_collision {
            let mut sc = SelfCollision::build(arm).map_err(|e| e.to_string())?;
            let named: Vec<(String, String)> = cfg
                .exclude_pairs
                .iter()
                .map(|[a, b]| (a.clone(), b.clone()))
                .collect();
            let unmatched = sc.exclude_pairs(&named);
            if !unmatched.is_empty() {
                return Err(format!(
                    "[safety] exclude_pairs: no such checked link pair: {unmatched:?}"
                ));
            }
            if let Some(q) = exclude_pose {
                let dropped = sc.exclude_close_at(arm, q, cfg.collision_margin);
                if !dropped.is_empty() {
                    log::info!(
                        "self-collision: not checking {} pairs already in contact at the exclusion pose: {:?}",
                        dropped.len(),
                        dropped
                    );
                }
                for p in sc.close_pairs(arm, q, cfg.collision_margin + cfg.joint_margin) {
                    let cap = allowance_cap.entry(pair_key(&p)).or_insert(f64::INFINITY);
                    // 1 µm below, so that rounding keeps the pose itself clear.
                    *cap = f64::min(*cap, p.distance - cfg.collision_margin - 1e-6);
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
            allowance_cap,
            proximal: {
                let chain = arm.tcp_chain();
                chain
                    .get(chain.len().div_ceil(2).saturating_sub(1))
                    .map_or(0, |&i| arm.dofs()[i].joint_idx)
            },
        })
    }

    /// Violation for the joint-tracking guard: like [`Self::violation`] but
    /// with room for tracking error: the box shrunk by the joint margin, and
    /// the collision margin grown by it for pairs that a proximal joint moves
    /// apart (see [`Self::allowance`]).
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
            v += sc
                .close_pairs(arm, q, self.margin + inset)
                .iter()
                .map(|p| self.margin + self.allowance(p, inset) - p.distance)
                .fold(0.0, f64::max);
        }
        v
    }

    /// What is violated at `q` against the joint-guard box (with the joint
    /// margin) and the collision margin, for the operator log. Empty if nothing.
    pub fn explain(&self, arm: &ArmModel, q: &[f64]) -> String {
        let zero = vec![0.0; q.len()];
        let inset = self.joint_margin;
        let mut out = Vec::new();
        for (link, local) in &self.points {
            if let Ok(ps) = arm.point_state(q, &zero, link, *local) {
                for (k, axis) in ["x", "y", "z"].iter().enumerate() {
                    let lo = self.box_min[k] + inset - ps.p[k];
                    let hi = ps.p[k] - (self.box_max[k] - inset);
                    if lo > 0.0 {
                        out.push(format!(
                            "{link} {axis}={:.3} below {:.3} (box min + joint_margin)",
                            ps.p[k],
                            self.box_min[k] + inset
                        ));
                    }
                    if hi > 0.0 {
                        out.push(format!(
                            "{link} {axis}={:.3} above {:.3} (box max − joint_margin)",
                            ps.p[k],
                            self.box_max[k] - inset
                        ));
                    }
                }
            }
        }
        if let Some(sc) = &self.collision {
            for p in sc.close_pairs(arm, q, self.margin + inset) {
                let within = self.margin + self.allowance(&p, inset);
                if p.distance >= within {
                    continue;
                }
                out.push(format!(
                    "{} – {} {:.1} mm apart (< {:.1} mm)",
                    p.link_a,
                    p.link_b,
                    p.distance * 1e3,
                    within * 1e3
                ));
            }
        }
        out.join("; ")
    }

    /// Extra collision margin for tracking error. The arm lags its reference
    /// mostly through the shoulder and elbow, whose long levers turn small
    /// angle errors into centimetres (the gripper overshot a held reference
    /// by 10 mm toward the base). Pairs closed by the wrist alone see a
    /// fraction of that, and their hulls sit permanently close (DM link3 and
    /// link6 never come closer than 10 mm), so the full margin there would
    /// stop the wrist for nothing. Folded poses bring proximal pairs close too
    /// (RS link1–link3 is 12 mm apart at rest), hence the cap.
    fn allowance(&self, p: &PairDistance, inset: f64) -> f64 {
        if p.joint_a.min(p.joint_b) >= self.proximal {
            return 0.0;
        }
        self.allowance_cap
            .get(&pair_key(p))
            .map_or(inset, |&c| inset.min(c))
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

fn pair_key(p: &PairDistance) -> (String, String) {
    if p.link_a <= p.link_b {
        (p.link_a.clone(), p.link_b.clone())
    } else {
        (p.link_b.clone(), p.link_a.clone())
    }
}
