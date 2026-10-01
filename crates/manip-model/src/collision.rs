//! Self-collision model: one convex hull per collision geometry, built once.
//!
//! misarta's `collision` module rebuilds parry shapes from the meshes on every
//! call, and parry has no mesh-vs-mesh distance (misarta maps the `Unsupported`
//! error to "infinitely far"), so it cannot be queried every control cycle.
//! Convex hulls built at load time support GJK distance and contact queries
//! in microseconds, and match what MuJoCo does with collision meshes anyway.
//!
//! Hulls are conservative (a hull contains the part), so distances are
//! slightly pessimistic, which is the safe side.
//!
//! # Which pairs are checked
//!
//! All pairs of geometries on different links, except
//! - links joined directly by a joint (they touch at the joint by design);
//! - pairs already within the margin at a reference pose, typically the folded
//!   rest pose, where vendor meshes interpenetrate
//!   ([`SelfCollision::exclude_close_at`], the SRDF "default" rule).

use misarta::native::schema::{Geom, Origin};
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
use parry3d::query;
use parry3d::shape::ConvexPolyhedron;

use crate::{ArmModel, ModelError};

struct Hull {
    link: String,
    /// misarta joint whose child link carries the geometry.
    joint: usize,
    placement: Isometry3<f64>,
    shape: ConvexPolyhedron,
}

/// Distance between one checked pair, with the witness points (world frame).
#[derive(Debug, Clone)]
pub struct PairDistance {
    pub link_a: String,
    pub link_b: String,
    pub joint_a: usize,
    pub joint_b: usize,
    /// Signed distance [m] (negative = penetrating).
    pub distance: f64,
    /// Closest point on A / on B.
    pub point_a: Vector3<f64>,
    pub point_b: Vector3<f64>,
    /// Unit normal from A toward B (the direction in which the distance grows).
    pub normal: Vector3<f64>,
}

pub struct SelfCollision {
    hulls: Vec<Hull>,
    pairs: Vec<(usize, usize)>,
}

impl SelfCollision {
    /// Build hulls from the `.misa` collision geometries of `arm`.
    pub fn build(arm: &ArmModel) -> Result<Self, ModelError> {
        let dir = arm.source_dir().map(|d| d.to_path_buf()).unwrap_or_default();
        let mut hulls = Vec::new();
        for link in &arm.file().link {
            let Ok(joint) = arm.link_joint(&link.name) else { continue };
            for c in &link.collision {
                let Some(points) = geom_points(&c.geom, &dir)? else { continue };
                let Some(shape) = ConvexPolyhedron::from_convex_hull(&points) else { continue };
                hulls.push(Hull {
                    link: link.name.clone(),
                    joint,
                    placement: origin_iso(&c.origin),
                    shape,
                });
            }
        }
        // Links joined by fixed joints are one rigid body (link6 -> end_link ->
        // fingers on the B601 have no moving joint between link6 and the
        // finger bases); adjacency is judged between those bodies.
        let joints = &arm.raw().joints;
        let body = |mut j: usize| {
            while j != 0 && matches!(joints[j].joint_type, misarta::joint::JointType::Fixed) {
                j = joints[j].parent;
            }
            j
        };
        let parent_body = |j: usize| if j == 0 { 0 } else { body(joints[j].parent) };
        let adjacent = |a: usize, b: usize| {
            let (ra, rb) = (body(a), body(b));
            ra == rb || parent_body(ra) == rb || parent_body(rb) == ra
        };
        let mut pairs = Vec::new();
        for i in 0..hulls.len() {
            for j in i + 1..hulls.len() {
                if !adjacent(hulls[i].joint, hulls[j].joint) {
                    pairs.push((i, j));
                }
            }
        }
        Ok(Self { hulls, pairs })
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Drop pairs closer than `margin` at `q` (e.g. the folded rest pose).
    /// Returns the dropped link pairs.
    pub fn exclude_close_at(&mut self, arm: &ArmModel, q: &[f64], margin: f64) -> Vec<(String, String)> {
        let poses = arm.joint_poses(q);
        let mut dropped = Vec::new();
        self.pairs.retain(|&(i, j)| {
            let (a, b) = (&self.hulls[i], &self.hulls[j]);
            let d = query::distance(&(poses[a.joint] * a.placement), &a.shape, &(poses[b.joint] * b.placement), &b.shape)
                .unwrap_or(f64::INFINITY);
            if d < margin {
                dropped.push((a.link.clone(), b.link.clone()));
                false
            } else {
                true
            }
        });
        dropped.sort();
        dropped.dedup();
        dropped
    }

    /// Stop checking the given link pairs (either order). Returns the names
    /// that matched no checked pair (typos, or pairs already dropped).
    pub fn exclude_pairs(&mut self, names: &[(String, String)]) -> Vec<(String, String)> {
        let hulls = &self.hulls;
        let same = |i: usize, j: usize, (a, b): &(String, String)| {
            (hulls[i].link == *a && hulls[j].link == *b)
                || (hulls[i].link == *b && hulls[j].link == *a)
        };
        let unmatched = names
            .iter()
            .filter(|n| !self.pairs.iter().any(|&(i, j)| same(i, j, n)))
            .cloned()
            .collect();
        self.pairs
            .retain(|&(i, j)| !names.iter().any(|n| same(i, j, n)));
        unmatched
    }

    /// Checked pairs closer than `within` at `q`, closest first.
    pub fn close_pairs(&self, arm: &ArmModel, q: &[f64], within: f64) -> Vec<PairDistance> {
        let poses = arm.joint_poses(q);
        let mut out = Vec::new();
        for &(i, j) in &self.pairs {
            let (a, b) = (&self.hulls[i], &self.hulls[j]);
            let (pa, pb) = (poses[a.joint] * a.placement, poses[b.joint] * b.placement);
            if let Ok(Some(c)) = query::contact(&pa, &a.shape, &pb, &b.shape, within) {
                out.push(PairDistance {
                    link_a: a.link.clone(),
                    link_b: b.link.clone(),
                    joint_a: a.joint,
                    joint_b: b.joint,
                    distance: c.dist,
                    point_a: c.point1.coords,
                    point_b: c.point2.coords,
                    normal: c.normal1.into_inner(),
                });
            }
        }
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Smallest distance over the checked pairs (∞ if none within 1 m).
    pub fn min_distance(&self, arm: &ArmModel, q: &[f64]) -> f64 {
        self.close_pairs(arm, q, 1.0).first().map(|p| p.distance).unwrap_or(f64::INFINITY)
    }
}

fn origin_iso(o: &Origin) -> Isometry3<f64> {
    let t = Translation3::new(o.xyz[0], o.xyz[1], o.xyz[2]);
    let r = if let Some(q) = o.quat {
        // [x, y, z, w]
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))
    } else if let Some(rpy) = o.rpy {
        UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2])
    } else {
        UnitQuaternion::identity()
    };
    Isometry3::from_parts(t, r)
}

/// Points whose hull is the geometry (mesh vertices, or sampled primitives).
fn geom_points(g: &Geom, dir: &std::path::Path) -> Result<Option<Vec<Point3<f64>>>, ModelError> {
    let ring = |r: f64, z: f64| -> Vec<Point3<f64>> {
        (0..16)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 16.0;
                Point3::new(r * a.cos(), r * a.sin(), z)
            })
            .collect()
    };
    Ok(Some(match g {
        Geom::Box { size } => {
            let [x, y, z] = size.map(|s| s / 2.0);
            let mut p = Vec::new();
            for sx in [-x, x] {
                for sy in [-y, y] {
                    for sz in [-z, z] {
                        p.push(Point3::new(sx, sy, sz));
                    }
                }
            }
            p
        }
        Geom::Sphere { radius } => {
            let mut p = Vec::new();
            for k in 0..9 {
                let el = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 8.0;
                p.extend(ring(radius * el.cos(), radius * el.sin()));
            }
            p
        }
        Geom::Cylinder { radius, length } => [ring(*radius, -length / 2.0), ring(*radius, length / 2.0)].concat(),
        Geom::Capsule { radius, length } => [
            ring(*radius, -length / 2.0),
            ring(*radius, length / 2.0),
            vec![Point3::new(0.0, 0.0, -length / 2.0 - radius), Point3::new(0.0, 0.0, length / 2.0 + radius)],
        ]
        .concat(),
        Geom::Mesh { file, scale } => {
            let path = dir.join(file);
            let m = misarta::mesh::MeshData::from_stl(&path).map_err(|e| ModelError::Load {
                path: path.display().to_string(),
                msg: e,
            })?;
            m.vertices
                .iter()
                .map(|v| Point3::new(v.x * scale[0], v.y * scale[1], v.z * scale[2]))
                .collect()
        }
    }))
}
