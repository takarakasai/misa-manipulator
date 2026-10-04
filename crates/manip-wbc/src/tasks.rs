//! Task builders for a fixed-base arm on misa-wbc's [`Dynamics`].
//!
//! Each builder returns a misa-wbc [`Task`]; a controller stacks them into
//! priority levels (`Task + Task` within a level) and solves with
//! [`crate::solve_levels`]. All vectors are in [`ChainState`] order.

use manip_control::{FrictionModel, JointRef};
use misa_wbc::dynamics::{Dynamics, Formulation};
use misa_wbc::tasks::{self, JointLimitCbf};
use misa_wbc::{AsAffine, Task};
use nalgebra::{DMatrix, DVector, Isometry3, Vector3, Vector6};

use crate::chain::{finite_or, ChainState};

/// TCP reference (shaped).
#[derive(Debug, Clone, PartialEq)]
pub struct TcpRef {
    pub pose: Isometry3<f64>,
    /// `[ω; v]`, world coordinates.
    pub twist: Vector6<f64>,
    /// `[α; a]`, world coordinates.
    pub accel: Vector6<f64>,
}

impl TcpRef {
    pub fn at_rest(pose: Isometry3<f64>) -> Self {
        Self {
            pose,
            twist: Vector6::zeros(),
            accel: Vector6::zeros(),
        }
    }
}

/// A second-order control barrier on some scalar `h(q) ≥ 0` (distance to a
/// workspace wall, distance between two links): `ḧ = grad·q̈ + drift`, and the
/// controller keeps `ḧ + (α1+α2)ḣ + α1α2·h ≥ 0`, so `h` decays to 0 no faster
/// than a critically damped approach and never crosses.
#[derive(Debug, Clone)]
pub struct Cbf {
    /// `∂ḣ/∂q̈`, length = independent DOFs (entries off the TCP chain are ignored).
    pub grad: DVector<f64>,
    /// The part of `ḧ` that does not depend on `q̈` (e.g. `n·J̇v`).
    pub drift: f64,
    pub h: f64,
    pub h_dot: f64,
}

/// Pose error `[rotation; translation]` (world coordinates). Rotation is `log(R_ref · Rᵀ)`.
pub fn pose_error(target: &Isometry3<f64>, current: &Isometry3<f64>) -> Vector6<f64> {
    let dp: Vector3<f64> = target.translation.vector - current.translation.vector;
    let dr = (target.rotation * current.rotation.inverse()).scaled_axis();
    Vector6::new(dr.x, dr.y, dr.z, dp.x, dp.y, dp.z)
}

/// Rigid-body dynamics of the chain for the QP. Friction (if any) is folded
/// into `h`: plant `M·q̈ + h = τ − τ_friction` ⇒ plan with `h' = h + τ_friction`.
///
/// Without a reference (OSC) it is evaluated at the measured velocity. With a
/// reference velocity `v_ref` (chain DOFs) it is the smaller of the two
/// compensations where they agree in sign, else 0 ([`agreed_friction`]).
pub fn chain_dynamics(
    c: &ChainState,
    formulation: Formulation,
    friction: Option<&FrictionModel>,
    v_ref: Option<&DVector<f64>>,
) -> Dynamics {
    let n = c.n();
    let nle = match friction {
        Some(f) => {
            let f = f.select(&c.idx);
            let at_meas = f.compensation(&c.v);
            match v_ref {
                Some(vr) => &c.nle + agreed_friction(&f.compensation(vr), &at_meas),
                None => &c.nle + at_meas,
            }
        }
        None => c.nle.clone(),
    };
    Dynamics::new(formulation, &c.mass, &nle, &DMatrix::zeros(0, n), n)
}

/// Friction compensation from its values at the reference (`at_ref`) and the
/// measured (`at_meas`) velocity: per joint the smaller where they agree in
/// sign, else 0 (friction then only damps).
///
/// On the real B601-DM in Mpc mode: at the measured velocity alone an
/// over-estimated Coulomb term is negative damping around v = 0 (slope
/// `coulomb / v_eps` = 28 N·m·s/rad on joint2 vs motor kd 3), and folded
/// joint2/joint4 swung ±1° at 2 Hz; at the reference velocity alone (the plan
/// starts from the measured state, and the arm lags it) fast teleop shook at
/// 3–8 Hz with 2.5–3× the torque swing.
pub fn agreed_friction(at_ref: &DVector<f64>, at_meas: &DVector<f64>) -> DVector<f64> {
    at_ref.zip_map(at_meas, |r, m| if r * m > 0.0 { r.signum() * r.abs().min(m.abs()) } else { 0.0 })
}

/// Joint limits as an exponential CBF on `q̈` (position, velocity) plus an
/// acceleration box.
#[derive(Debug, Clone)]
pub struct JointLimitParams {
    /// Acceleration limit per chain DOF [rad/s²].
    pub a_max: DVector<f64>,
    /// Position barrier poles [1/s].
    pub alpha: f64,
    /// Velocity barrier gain [1/s].
    pub alpha_v: f64,
    /// Velocity limit = model `v_max` × this.
    pub v_scale: f64,
}

/// Level-0 "physics and safety": torque box, joint-limit CBF, and (with
/// `Formulation::Explicit`) the equations of motion.
pub fn physics_and_limits(c: &ChainState, d: &Dynamics, tau_max: &DVector<f64>, jl: &JointLimitParams) -> Task {
    let n = c.n();
    let big = 1e3;
    // The range is widened to include the current position: an arm resting
    // on its stop reads past the limit (the B601-DM shoulder by ~1° when
    // folded), and a barrier that demands pulling back at once is infeasible
    // with the acceleration and velocity bounds. Moving further out is still
    // refused.
    let cbf = JointLimitCbf {
        q_min: DVector::from_iterator(n, c.dofs.iter().zip(c.q.iter()).map(|(d, &q)| finite_or(d.q_min, -big).min(q))),
        q_max: DVector::from_iterator(n, c.dofs.iter().zip(c.q.iter()).map(|(d, &q)| finite_or(d.q_max, big).max(q))),
        v_max: DVector::from_iterator(n, c.dofs.iter().map(|d| finite_or(d.v_max * jl.v_scale, big))),
        a_max: jl.a_max.clone(),
        alpha1: DVector::from_element(n, jl.alpha),
        alpha2: DVector::from_element(n, jl.alpha),
        alpha3: DVector::from_element(n, jl.alpha_v),
    };
    // Equations of motion: an equality task with Explicit; already eliminated
    // with AccelSpace / ForceSpace (with a fixed base every axis is actuated,
    // so no rows remain).
    let mut level = tasks::box_bound(d.tau(), tau_max) + tasks::joint_limit_cbf(d.qddot(), &c.q, &c.v, &cbf);
    if let Some(eom) = d.dynamics_task().filter(|t| t.n_eq() + t.n_iq() > 0) {
        level = eom + level;
    }
    level
}

/// Append safety barriers to `level`:
/// `ḧ + (α1+α2)·ḣ + α1·α2·h ≥ 0` with `ḧ = grad·q̈ + drift` (`α1 = α2 = alpha`).
pub fn with_barriers(mut level: Task, c: &ChainState, d: &Dynamics, cbfs: &[Cbf], alpha: f64) -> Task {
    let n = c.n();
    let (a1, a2) = (alpha, alpha);
    for b in cbfs {
        let g = DMatrix::from_row_slice(1, n, &c.idx.iter().map(|&i| b.grad[i]).collect::<Vec<_>>());
        let expr = &(&g * &d.qddot().as_affine()) + &DVector::from_element(1, b.drift);
        let lb = DVector::from_element(1, -(a1 + a2) * b.h_dot - a1 * a2 * b.h);
        level = level + Task::ge(&expr, &lb);
    }
    level
}

/// TCP pose PD gains (acceleration units: [1/s²], [1/s]).
#[derive(Debug, Clone, Copy)]
pub struct TcpGains {
    pub kp_lin: f64,
    pub kd_lin: f64,
    pub kp_ang: f64,
    pub kd_ang: f64,
}

/// Joint damping blended into the TCP task near singularities (an
/// acceleration-level DLS): as `σ_min(J)` drops from `sigma_hi` to
/// `sigma_lo`, `λ²‖q̈ + kd·v‖²` is smoothly added. misa-wbc's
/// `cartesian_acceleration_damped` penalizes `‖q̈‖²`, which does **not stop** a
/// moving axis (zero acceleration keeps drifting): J4 and J6 kept spinning in
/// opposite directions at a wrist singularity.
#[derive(Debug, Clone, Copy)]
pub struct SingularityParams {
    pub sigma_lo: f64,
    pub sigma_hi: f64,
    pub lambda_sq: f64,
    pub kd: f64,
}

/// The TCP acceleration task and what went into it.
pub struct TcpTask {
    pub task: Task,
    /// TCP error `[rotation; translation]`.
    pub err: Vector6<f64>,
    /// Tracked rows of the 6-D TCP space (`0..6`, or `3..6` for position only).
    pub rows: std::ops::Range<usize>,
    pub jacobian: DMatrix<f64>,
    pub jdot_v: DVector<f64>,
    /// Requested TCP acceleration on `rows`.
    pub a_task: DVector<f64>,
    pub sigma_min: f64,
}

impl TcpTask {
    /// Requested − realized TCP acceleration at `qddot` (6-D, zeros off `rows`).
    pub fn residual(&self, qddot: &DVector<f64>) -> Vector6<f64> {
        let achieved = &self.jacobian * qddot + &self.jdot_v;
        let mut residual = Vector6::zeros();
        for (k, r) in self.rows.clone().enumerate() {
            residual[r] = self.a_task[k] - achieved[k];
        }
        residual
    }
}

/// TCP acceleration `J·q̈ + J̇v = a_ref + Kp·e + Kd·ė` (with singularity damping).
pub fn tcp_acceleration(
    c: &ChainState,
    d: &Dynamics,
    tcp: &TcpRef,
    gains: &TcpGains,
    track_orientation: bool,
    sing: &SingularityParams,
) -> TcpTask {
    let err = pose_error(&tcp.pose, &c.tcp_pose);
    let twist_err = tcp.twist - c.tcp_twist;
    let kp = Vector6::new(gains.kp_ang, gains.kp_ang, gains.kp_ang, gains.kp_lin, gains.kp_lin, gains.kp_lin);
    let kd = Vector6::new(gains.kd_ang, gains.kd_ang, gains.kd_ang, gains.kd_lin, gains.kd_lin, gains.kd_lin);
    let a_ref = tcp.accel + kp.component_mul(&err) + kd.component_mul(&twist_err);
    let rows: std::ops::Range<usize> = if track_orientation { 0..6 } else { 3..6 };
    let j = c.tcp_jacobian.rows(rows.start, rows.len()).into_owned();
    let djv = DVector::from_iterator(rows.len(), rows.clone().map(|r| c.tcp_jdot_v[r]));
    let a_task = DVector::from_iterator(rows.len(), rows.clone().map(|r| a_ref[r]));
    let sigma_min = j.clone().svd(false, false).singular_values.min();
    let span = (sing.sigma_hi - sing.sigma_lo).max(1e-12);
    let ramp = ((sing.sigma_hi - sigma_min) / span).clamp(0.0, 1.0);
    let mut task = tasks::cartesian_acceleration(d.qddot(), &j, &djv, &a_task);
    if ramp > 0.0 {
        task = task + tasks::track(d.qddot(), &(-sing.kd * &c.v)).weight(sing.lambda_sq * ramp * ramp);
    }
    TcpTask {
        task,
        err,
        rows,
        jacobian: j,
        jdot_v: djv,
        a_task,
        sigma_min,
    }
}

/// Joint acceleration from a reference: `a + Kp·(q_ref − q) + Kd·(v_ref − v)`.
pub fn joint_pd_acceleration(c: &ChainState, r: &JointRef, kp: &DVector<f64>, kd: &DVector<f64>) -> DVector<f64> {
    &r.a + kp.component_mul(&(&r.q - &c.q)) + kd.component_mul(&(&r.v - &c.v))
}

/// Track a joint acceleration: `q̈ = a_ref` (least squares).
pub fn joint_acceleration(d: &Dynamics, a_ref: &DVector<f64>) -> Task {
    tasks::track(d.qddot(), a_ref)
}

/// `w·‖τ‖²`.
pub fn torque_regularization(d: &Dynamics, w: f64) -> Task {
    tasks::regularize(d.tau(), &DVector::zeros(d.tau().out_size())).weight(w)
}
