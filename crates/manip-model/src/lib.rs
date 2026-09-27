//! Reshapes a misarta model into the form an arm control law needs.
//!
//! # Speak in independent DOFs
//!
//! Control laws, profiles and recordings all speak in terms of a sequence of
//! **independent DOFs** ([`Dof`]), not URDF joints. In a model like the
//! reBot B601 gripper, where "one of the two prismatic fingers is a mimic
//! follower", the model's `nv` is 8 but there are only 7 motors. Reducing to
//! independent coordinates here lets the upper layers treat
//!
//! ```text
//! M_r = Gᵀ M G,   h_r = Gᵀ h,   J_r = J G      (G = mimic projection, nv × n)
//! ```
//!
//! as a plain n-DOF arm (`G` is constant, so `J̇_r v_r = J̇ (G v_r)`).
//!
//! # Evaluate only once per cycle
//!
//! [`ArmModel::evaluate`] builds all matrices for one cycle (M, h, g, TCP J,
//! J̇v) at once. Control laws only read [`ArmState`] and never call misarta
//! directly. This avoids running FK several times within one cycle, and makes
//! **the values the control law saw identical to the values recorded**.

use std::collections::BTreeMap;
use std::path::Path;

use misarta::frames::{self, Frame};
use misarta::joint::JointType;
use misarta::model::Model;
use misarta::native::schema::{JointKind, MisaFile};
use nalgebra::{DMatrix, DVector, Isometry3, Translation3, UnitQuaternion, Vector6};

pub use misarta;

pub mod collision;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("cannot load model ({path}): {msg}")]
    Load { path: String, msg: String },
    #[error("link '{0}' not found in model")]
    UnknownLink(String),
    #[error("joint '{0}' is not an independent DOF (fixed, mimic follower, or unknown)")]
    UnknownDof(String),
    #[error("pose '{0}' not found in model")]
    UnknownPose(String),
    #[error("this model is not a fixed-base arm: {0}")]
    NotFixedBase(String),
}

/// Kind of an independent DOF. Kept distinct because the unit differs (rad vs m).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DofKind {
    Revolute,
    Prismatic,
}

/// One independent DOF. Corresponds to one model joint (the mimic leader side).
#[derive(Debug, Clone)]
pub struct Dof {
    /// Model joint name. The only key used to match against the profile.
    pub name: String,
    pub kind: DofKind,
    /// misarta joint index (1-based).
    pub joint_idx: usize,
    /// Position in the full `q` / `v`.
    pub q_idx: usize,
    pub v_idx: usize,
    /// Range of motion [rad or m]. ±∞ if the model declares none.
    pub q_min: f64,
    pub q_max: f64,
    /// Velocity limit [rad/s or m/s]. ∞ if none is declared.
    pub v_max: f64,
    /// Torque (force) limit [N·m or N]. ∞ if none is declared.
    pub effort: f64,
    /// Reflected rotor inertia [kg·m² or kg]. Added to the diagonal of `M`.
    ///
    /// With geared motors, rotor inertia × gear ratio² applies, and on light
    /// distal axes (wrist, gripper) it can exceed the link inertia. Left at 0,
    /// `M` becomes singular for massless links (the B601-DM fingers). Defaults
    /// to the model's `dynamics.armature`; overridden by the profile.
    pub armature: f64,
}

impl Dof {
    pub fn clamp(&self, q: f64) -> f64 {
        q.clamp(self.q_min, self.q_max)
    }

    pub fn within(&self, q: f64) -> bool {
        q >= self.q_min && q <= self.q_max
    }
}

/// Definition of the TCP (control point).
///
/// A frame fixed to a link with an offset. If the URDF has a fixed link such
/// as `end_link`, just pointing at it is enough.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TcpSpec {
    pub link: String,
    /// Position in the link frame [m].
    #[serde(default)]
    pub xyz: [f64; 3],
    /// Orientation in the link frame (roll, pitch, yaw) [rad].
    #[serde(default)]
    pub rpy: [f64; 3],
}

impl TcpSpec {
    pub fn at_link(link: impl Into<String>) -> Self {
        Self {
            link: link.into(),
            xyz: [0.0; 3],
            rpy: [0.0; 3],
        }
    }
}

/// Arm state for one cycle plus the associated matrices. All in independent-DOF order.
#[derive(Debug, Clone)]
pub struct ArmState {
    pub q: DVector<f64>,
    pub v: DVector<f64>,
    /// Mass matrix `Gᵀ M G` (n × n).
    pub mass: DMatrix<f64>,
    /// Nonlinear effects `Gᵀ h(q, v)` (Coriolis, centrifugal, gravity).
    pub nle: DVector<f64>,
    /// Gravity term `Gᵀ g(q)`.
    pub gravity: DVector<f64>,
    /// TCP pose in world coordinates.
    pub tcp_pose: Isometry3<f64>,
    /// TCP spatial velocity `[ω; v]` (world coordinates, linear velocity of the TCP point).
    pub tcp_twist: Vector6<f64>,
    /// TCP Jacobian (6 × n, rows ordered `[ω; v]`, world coordinates).
    pub tcp_jacobian: DMatrix<f64>,
    /// `J̇·v` (6). `a_tcp = J q̈ + J̇v`.
    pub tcp_jdot_v: Vector6<f64>,
}

/// Arm model. Immutable; safe to share across threads.
#[derive(Debug, Clone)]
pub struct ArmModel {
    model: Model<f64>,
    file: MisaFile,
    dofs: Vec<Dof>,
    /// Mimic projection `G` (nv × n).
    g_proj: DMatrix<f64>,
    tcp: Frame<f64>,
    tcp_spec: TcpSpec,
    /// Directory the model was loaded from (mesh paths in the `.misa` are
    /// relative to it). `None` for models built in memory.
    source_dir: Option<std::path::PathBuf>,
}

/// Kinematics of one point fixed to a link: world position, velocity, the
/// 3 × n linear Jacobian (independent DOFs) and its bias `J̇·v`.
#[derive(Debug, Clone)]
pub struct PointState {
    pub p: nalgebra::Vector3<f64>,
    pub v: nalgebra::Vector3<f64>,
    pub jacobian: DMatrix<f64>,
    pub jdot_v: nalgebra::Vector3<f64>,
}

impl ArmModel {
    /// Loads a `.misa` or `.urdf` and builds an arm with `tcp` as the control point.
    pub fn load(path: impl AsRef<Path>, tcp: &TcpSpec) -> Result<Self, ModelError> {
        let path = path.as_ref();
        let load_err = |msg: String| ModelError::Load {
            path: path.display().to_string(),
            msg,
        };
        let dir = path.parent().map(|d| d.to_path_buf());
        let file = match path.extension().and_then(|e| e.to_str()) {
            Some("urdf") | Some("URDF") => {
                misarta_formats::urdf::import(path).map_err(load_err)?.file
            }
            _ => {
                misarta::native::load(path)
                    .map_err(|e| load_err(e.to_string()))?
                    .file
            }
        };
        let mut arm = Self::from_file(file, tcp)?;
        arm.source_dir = dir;
        Ok(arm)
    }

    /// Builds from an already-loaded [`MisaFile`].
    pub fn from_file(file: MisaFile, tcp: &TcpSpec) -> Result<Self, ModelError> {
        let (model, _visual, _collision) =
            misarta::native::build_model(&file).map_err(|e| ModelError::Load {
                path: file.robot.name.clone(),
                msg: e.to_string(),
            })?;

        if model
            .joints
            .iter()
            .any(|j| matches!(j.joint_type, JointType::FreeFlyer))
        {
            return Err(ModelError::NotFixedBase(
                "contains a free-flyer joint (the base must be fixed to the world)".into(),
            ));
        }
        if model.nq != model.nv {
            return Err(ModelError::NotFixedBase(format!(
                "nq={} and nv={} do not match",
                model.nq, model.nv
            )));
        }

        let dofs = collect_dofs(&model, &file);
        let g_proj = misarta::mimic::mimic_projection_matrix(&model);
        debug_assert_eq!(g_proj.ncols(), dofs.len());

        let tcp_frame = make_tcp(&model, tcp)?;
        Ok(Self {
            model,
            file,
            dofs,
            g_proj,
            tcp: tcp_frame,
            tcp_spec: tcp.clone(),
            source_dir: None,
        })
    }

    /// Makes a copy with a different control point (the model is not reloaded).
    pub fn with_tcp(&self, tcp: &TcpSpec) -> Result<Self, ModelError> {
        let mut s = self.clone();
        s.tcp = make_tcp(&s.model, tcp)?;
        s.tcp_spec = tcp.clone();
        Ok(s)
    }

    pub fn name(&self) -> &str {
        &self.model.name
    }

    /// Number of independent DOFs.
    pub fn n(&self) -> usize {
        self.dofs.len()
    }

    pub fn dofs(&self) -> &[Dof] {
        &self.dofs
    }

    pub fn dof(&self, name: &str) -> Result<usize, ModelError> {
        self.dofs
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| ModelError::UnknownDof(name.to_string()))
    }

    /// Independent DOFs that move the TCP (those on the chain from the base to
    /// the TCP's parent link), in ascending order. DOFs hanging beyond the TCP,
    /// such as gripper fingers, are excluded.
    pub fn tcp_chain(&self) -> Vec<usize> {
        let chain = self.model.ancestors_of(self.tcp.parent_joint);
        (0..self.dofs.len())
            .filter(|&i| chain.contains(&self.dofs[i].joint_idx))
            .collect()
    }

    pub fn tcp_spec(&self) -> &TcpSpec {
        &self.tcp_spec
    }

    /// The original misarta model (full coordinates). Needed for visualization and collision checks.
    pub fn raw(&self) -> &Model<f64> {
        &self.model
    }

    /// The loaded `.misa` document (poses, link names, etc.).
    pub fn file(&self) -> &MisaFile {
        &self.file
    }

    /// Independent → full coordinates. Fills in the mimic follower joints.
    pub fn full_q(&self, q: &[f64]) -> Vec<f64> {
        assert_eq!(q.len(), self.n(), "q length differs from the number of independent DOFs");
        let mut full = vec![0.0; self.model.nq];
        for (d, &qi) in self.dofs.iter().zip(q) {
            full[d.q_idx] = qi;
        }
        misarta::mimic::enforce_mimic(&self.model, &full)
    }

    /// Independent → full velocity (`G v`).
    pub fn full_v(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.n(), "v length differs from the number of independent DOFs");
        let v = DVector::from_column_slice(v);
        (&self.g_proj * v).as_slice().to_vec()
    }

    /// Builds all matrices for one cycle at once.
    pub fn evaluate(&self, q: &[f64], v: &[f64]) -> ArmState {
        let qf = self.full_q(q);
        let vf = self.full_v(v);
        let g = &self.g_proj;
        let gt = g.transpose();

        let mass_full = misarta::crba::crba(&self.model, &qf);
        let nle_full = misarta::rnea::nonlinear_effects(&self.model, &qf, &vf);
        let grav_full = misarta::rnea::compute_gravity(&self.model, &qf);

        let data = misarta::fk::forward_kinematics(&self.model, &qf);
        let tcp_pose = frames::compute_frame_placement_from_data(&data, &self.tcp);
        let jac_full = frames::compute_frame_jacobian_from_data(&self.model, &qf, &data, &self.tcp);
        let vf_vec = DVector::from_column_slice(&vf);
        let twist = &jac_full * &vf_vec;
        let jdot_v = self.frame_jdot_v(&qf, &vf);

        let mut mass = &gt * mass_full * g;
        for (i, d) in self.dofs.iter().enumerate() {
            mass[(i, i)] += d.armature;
        }
        ArmState {
            q: DVector::from_column_slice(q),
            v: DVector::from_column_slice(v),
            mass,
            nle: &gt * nle_full,
            gravity: &gt * grav_full,
            tcp_pose,
            tcp_twist: Vector6::from_column_slice(twist.as_slice()),
            tcp_jacobian: jac_full * g,
            tcp_jdot_v: jdot_v,
        }
    }

    /// Gravity torque only (independent coordinates). The default gravity-compensation path.
    pub fn gravity(&self, q: &[f64]) -> DVector<f64> {
        let qf = self.full_q(q);
        self.g_proj.transpose() * misarta::rnea::compute_gravity(&self.model, &qf)
    }

    /// Inverse dynamics `τ = M(q) a + h(q, v)` (independent coordinates).
    ///
    /// Used for computed-torque feedforward. Normally pass the **reference**
    /// (shaped target velocity/acceleration) as `v` and `a`; passing the
    /// measured `v` feeds velocity noise straight into the torque.
    pub fn inverse_dynamics(&self, q: &[f64], v: &[f64], a: &[f64]) -> DVector<f64> {
        let qf = self.full_q(q);
        let vf = self.full_v(v);
        let af = self.full_v(a);
        let mut tau = self.g_proj.transpose() * misarta::rnea::rnea(&self.model, &qf, &vf, &af);
        for (i, d) in self.dofs.iter().enumerate() {
            tau[i] += d.armature * a[i];
        }
        tau
    }

    /// Overrides the reflected rotor inertia (independent DOF name → value).
    pub fn set_armature(&mut self, name: &str, armature: f64) -> Result<(), ModelError> {
        let i = self.dof(name)?;
        self.dofs[i].armature = armature.max(0.0);
        Ok(())
    }

    /// Overrides position/velocity/torque limits. Used when the profile narrows
    /// them relative to the model (cable routing etc. make the real range of
    /// motion narrower than the URDF).
    pub fn set_limits(
        &mut self,
        name: &str,
        q_min: Option<f64>,
        q_max: Option<f64>,
        v_max: Option<f64>,
        effort: Option<f64>,
    ) -> Result<(), ModelError> {
        let i = self.dof(name)?;
        let d = &mut self.dofs[i];
        if let Some(x) = q_min { d.q_min = x; }
        if let Some(x) = q_max { d.q_max = x; }
        if let Some(x) = v_max { d.v_max = x; }
        if let Some(x) = effort { d.effort = x; }
        Ok(())
    }

    /// TCP pose only. For cases like leader-arm FK where the matrices aren't needed.
    pub fn tcp_pose(&self, q: &[f64]) -> Isometry3<f64> {
        let qf = self.full_q(q);
        frames::compute_frame_placement(&self.model, &qf, &self.tcp)
    }

    /// World pose of an arbitrary link (for visualization and rough collision checks).
    pub fn link_pose(&self, q: &[f64], link: &str) -> Result<Isometry3<f64>, ModelError> {
        let idx = self
            .model
            .link_names
            .iter()
            .position(|l| l == link)
            .ok_or_else(|| ModelError::UnknownLink(link.to_string()))?;
        let data = misarta::fk::forward_kinematics(&self.model, &self.full_q(q));
        Ok(data.oMi[idx])
    }

    /// Named pose from the `.misa`. Joints not listed take the `fallback` value;
    /// the result is finally clamped to the range of motion.
    pub fn named_pose(&self, name: &str, fallback: &[f64]) -> Result<DVector<f64>, ModelError> {
        let pose = self
            .file
            .pose
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| ModelError::UnknownPose(name.to_string()))?;
        let mut q = DVector::from_column_slice(fallback);
        for (i, d) in self.dofs.iter().enumerate() {
            if let Some(&v) = pose.angles.get(&d.name) {
                q[i] = v;
            }
            q[i] = d.clamp(q[i]);
        }
        Ok(q)
    }

    pub fn pose_names(&self) -> Vec<&str> {
        self.file.pose.iter().map(|p| p.name.as_str()).collect()
    }

    /// Midpoint of the range of motion (0 for unlimited axes). A default guess at startup.
    pub fn mid_q(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.n(),
            self.dofs.iter().map(|d| {
                if d.q_min.is_finite() && d.q_max.is_finite() {
                    0.5 * (d.q_min + d.q_max)
                } else {
                    0.0
                }
            }),
        )
    }

    /// TCP `J̇·v`. To also work for frames with an offset, takes a central
    /// difference of the frame Jacobian along `v` (same approach as misarta's
    /// joint version).
    /// Directory the model was loaded from, if any.
    pub fn source_dir(&self) -> Option<&std::path::Path> {
        self.source_dir.as_deref()
    }

    /// misarta joint index whose child link is `link`.
    pub fn link_joint(&self, link: &str) -> Result<usize, ModelError> {
        self.model
            .link_names
            .iter()
            .position(|l| l == link)
            .ok_or_else(|| ModelError::UnknownLink(link.to_string()))
    }

    /// Kinematics of the point `local` (in the frame of `link`).
    pub fn point_state(&self, q: &[f64], v: &[f64], link: &str, local: nalgebra::Vector3<f64>) -> Result<PointState, ModelError> {
        let frame = Frame {
            name: format!("{link}+point"),
            parent_joint: self.link_joint(link)?,
            placement: Isometry3::from_parts(Translation3::from(local), UnitQuaternion::identity()),
        };
        let qf = self.full_q(q);
        let vf = self.full_v(v);
        let data = misarta::fk::forward_kinematics(&self.model, &qf);
        let pose = frames::compute_frame_placement_from_data(&data, &frame);
        let jf = frames::compute_frame_jacobian_from_data(&self.model, &qf, &data, &frame);
        let jac = jf.rows(3, 3).into_owned() * &self.g_proj;
        let vv = DVector::from_column_slice(&vf);
        let pv = jf.rows(3, 3) * &vv;
        let eps = 1e-6;
        let qp = misarta::manifold::integrate(&self.model, &qf, &vf, eps);
        let qm = misarta::manifold::integrate(&self.model, &qf, &vf, -eps);
        let vp = frames::compute_frame_jacobian(&self.model, &qp, &frame).rows(3, 3) * &vv;
        let vm = frames::compute_frame_jacobian(&self.model, &qm, &frame).rows(3, 3) * &vv;
        let jdv = (vp - vm) / (2.0 * eps);
        Ok(PointState {
            p: pose.translation.vector,
            v: nalgebra::Vector3::new(pv[0], pv[1], pv[2]),
            jacobian: jac,
            jdot_v: nalgebra::Vector3::new(jdv[0], jdv[1], jdv[2]),
        })
    }

    /// 3 × n linear Jacobian (independent DOFs) of the world point `p` taken
    /// as fixed to the link of misarta joint `joint`, at configuration `q`.
    pub fn point_jacobian_at(&self, q: &[f64], joint: usize, p: &nalgebra::Vector3<f64>) -> DMatrix<f64> {
        let qf = self.full_q(q);
        let data = misarta::fk::forward_kinematics(&self.model, &qf);
        let local = data.oMi[joint].inverse_transform_point(&nalgebra::Point3::from(*p));
        let frame = Frame {
            name: "witness".into(),
            parent_joint: joint,
            placement: Isometry3::from_parts(Translation3::from(local.coords), UnitQuaternion::identity()),
        };
        frames::compute_frame_jacobian_from_data(&self.model, &qf, &data, &frame).rows(3, 3).into_owned() * &self.g_proj
    }

    /// World pose of every misarta joint frame at `q` (index = joint index).
    pub fn joint_poses(&self, q: &[f64]) -> Vec<Isometry3<f64>> {
        misarta::fk::forward_kinematics(&self.model, &self.full_q(q)).oMi
    }

    fn frame_jdot_v(&self, qf: &[f64], vf: &[f64]) -> Vector6<f64> {
        let eps = 1e-6;
        let q_plus = misarta::manifold::integrate(&self.model, qf, vf, eps);
        let q_minus = misarta::manifold::integrate(&self.model, qf, vf, -eps);
        let v = DVector::from_column_slice(vf);
        let jp = frames::compute_frame_jacobian(&self.model, &q_plus, &self.tcp) * &v;
        let jm = frames::compute_frame_jacobian(&self.model, &q_minus, &self.tcp) * &v;
        Vector6::from_column_slice(((jp - jm) / (2.0 * eps)).as_slice())
    }
}

fn positive_or_inf(x: f64) -> f64 {
    if x > 0.0 { x } else { f64::INFINITY }
}

fn collect_dofs(model: &Model<f64>, file: &MisaFile) -> Vec<Dof> {
    let limits: BTreeMap<&str, _> = file
        .joint
        .iter()
        .map(|j| (j.name.as_str(), (j.kind, &j.limit)))
        .collect();
    let armatures: BTreeMap<&str, f64> = file
        .joint
        .iter()
        .map(|j| (j.name.as_str(), j.dynamics.armature))
        .collect();
    let slaves: Vec<usize> = model.mimic.iter().map(|m| m.slave).collect();

    let mut dofs = Vec::new();
    for (idx, j) in model.joints.iter().enumerate().skip(1) {
        let kind = match j.joint_type {
            JointType::Revolute { .. } => DofKind::Revolute,
            JointType::Prismatic { .. } => DofKind::Prismatic,
            _ => continue,
        };
        if slaves.contains(&idx) {
            continue;
        }
        let (mut q_min, mut q_max, mut v_max, mut effort) =
            (f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let armature = armatures.get(j.name.as_str()).copied().unwrap_or(0.0);
        if let Some((jk, l)) = limits.get(j.name.as_str()) {
            v_max = positive_or_inf(l.velocity);
            effort = positive_or_inf(l.effort);
            // continuous has no range of motion. lower >= upper is also read as "not declared".
            if *jk != JointKind::Continuous && l.lower < l.upper {
                q_min = l.lower;
                q_max = l.upper;
            }
        }
        dofs.push(Dof {
            name: j.name.clone(),
            kind,
            joint_idx: idx,
            q_idx: model.q_idx[idx],
            v_idx: model.v_idx[idx],
            q_min,
            q_max,
            v_max,
            effort,
            armature,
        });
    }
    // Mimic projection columns are in ascending independent-v index order. Match the Dof order to it.
    dofs.sort_by_key(|d| d.v_idx);
    dofs
}

fn make_tcp(model: &Model<f64>, spec: &TcpSpec) -> Result<Frame<f64>, ModelError> {
    // link_names[i] is the child link of joints[i]. misarta keeps fixed joints
    // as joints too, so fixed links like `end_link` can be looked up directly.
    let parent_joint = model
        .link_names
        .iter()
        .position(|l| *l == spec.link)
        .ok_or_else(|| ModelError::UnknownLink(spec.link.clone()))?;
    let placement = Isometry3::from_parts(
        Translation3::new(spec.xyz[0], spec.xyz[1], spec.xyz[2]),
        UnitQuaternion::from_euler_angles(spec.rpy[0], spec.rpy[1], spec.rpy[2]),
    );
    Ok(Frame {
        name: "tcp".into(),
        parent_joint,
        placement,
    })
}

#[cfg(test)]
mod tests;
