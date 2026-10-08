//! Torque-level nonlinear MPC with iLQR (iterative LQR).
//!
//! State `x = [q; v]` of the TCP-chain DOFs, input `u = τ` held over each
//! interval of length `h`. The forward dynamics `q̈ = M⁻¹(τ − h(q, v))` use
//! manip-model's `M` (with reflected rotor inertia) and `h`; within an interval
//! the acceleration is held (exact double integrator, the same knots as
//! [`JointPlan`]):
//!
//! ```text
//! v' = v + h·q̈,   q' = q + h·v + ½h²·q̈
//! ```
//!
//! Its linearization uses misarta's **analytical** RNEA derivatives:
//! `∂q̈/∂q = −M⁻¹·∂ID/∂q`, `∂q̈/∂v = −M⁻¹·∂ID/∂v` (at `a = q̈`), `∂q̈/∂τ = M⁻¹`.
//!
//! Costs (Gauss–Newton on residuals): TCP pose error (pseudo-Huber: quadratic
//! near the target, linear beyond `e_max_*`, so a far target pulls with a
//! bounded force while every step toward it still lowers the cost), posture, velocity (with the same singularity damping),
//! torque relative to gravity, and quadratic penalties past the joint range,
//! the velocity limit and the workspace box (soft constraints). The torque
//! limit is enforced by clamping in the forward pass.
//!
//! Each plan runs at most `max_iters` iterations from the previous solution
//! shifted to "now" (Levenberg regularization `μ`, backtracking line search).

use manip_model::ArmModel;
use manip_wbc::pose_error;
use nalgebra::{DMatrix, DVector, Isometry3, Vector3};

use crate::ltv::WorkspaceBox;
use crate::plan::JointPlan;
use crate::{MpcError, MpcGoal, MpcReport, Planner};

#[derive(Debug, Clone)]
pub struct IlqrConfig {
    pub horizon: usize,
    pub dt: f64,
    pub w_pos: f64,
    pub w_rot: f64,
    pub track_orientation: bool,
    pub e_max_pos: f64,
    pub e_max_rot: f64,
    pub terminal_scale: f64,
    pub w_posture: f64,
    pub w_vel: f64,
    /// Torque deviation from gravity compensation [1/(N·m)²].
    pub w_tau: f64,
    pub sing_sigma_lo: f64,
    pub sing_sigma_hi: f64,
    pub w_sing: f64,
    /// Penalty weights past the joint range (with `q_margin`), the velocity
    /// limit (model `v_max` × `v_scale`) and the workspace box.
    pub w_limit: f64,
    pub q_margin: f64,
    pub v_scale: f64,
    pub workspace: Option<WorkspaceBox>,
    pub w_workspace: f64,
    /// TCP translational speed limit per axis [m/s] (penalty past it, weight
    /// `w_limit`; the LTV-MPC and the OSC's shaper use 0.3 m/s). `None` = off.
    pub tcp_v_max: Option<f64>,
    /// Torque limit = model effort × this (clamped in the forward pass).
    pub torque_scale: f64,
    pub max_iters: usize,
    /// Levenberg regularization: initial, minimum and maximum `μ`.
    pub mu0: f64,
    pub mu_min: f64,
    pub mu_max: f64,
    /// Stop when the relative cost decrease falls below this.
    pub tol: f64,
}

impl IlqrConfig {
    pub fn defaults() -> Self {
        Self {
            horizon: 25,
            dt: 0.04,
            w_pos: 1e4,
            w_rot: 1e2,
            track_orientation: true,
            e_max_pos: 0.05,
            e_max_rot: 0.3,
            terminal_scale: 10.0,
            w_posture: 1e-2,
            w_vel: 1e-1,
            w_tau: 1e-3,
            sing_sigma_lo: 0.01,
            sing_sigma_hi: 0.06,
            w_sing: 10.0,
            w_limit: 1e4,
            q_margin: 0.02,
            v_scale: 1.0,
            workspace: None,
            w_workspace: 1e6,
            tcp_v_max: Some(0.3),
            torque_scale: 0.9,
            max_iters: 5,
            mu0: 1e-6,
            mu_min: 1e-9,
            mu_max: 1e6,
            tol: 1e-4,
        }
    }
}

/// iLQR planner (implements [`Planner`]).
pub struct IlqrMpc {
    pub cfg: IlqrConfig,
    /// Previous solution (warm start): torques, feedback gains and states.
    prev: Option<Prev>,
    mu: f64,
}

impl IlqrMpc {
    pub fn new(cfg: IlqrConfig) -> Self {
        let mu = cfg.mu0;
        Self { cfg, prev: None, mu }
    }
}

/// The previous plan's policy `u = u_k + K_k·(x − x_k)` from knot time `t0`.
struct Prev {
    t0: f64,
    idx: Vec<usize>,
    u: Vec<DVector<f64>>,
    k: Vec<DMatrix<f64>>,
    x: Vec<DVector<f64>>,
}

/// Fixed data of one plan: chain indices, limits, goal sampling.
struct Problem<'a> {
    arm: &'a ArmModel,
    cfg: &'a IlqrConfig,
    idx: Vec<usize>,
    v_idx: Vec<usize>,
    q_base: Vec<f64>,
    q_lo: DVector<f64>,
    q_hi: DVector<f64>,
    v_lim: DVector<f64>,
    tau_lim: DVector<f64>,
    posture: DVector<f64>,
    targets: Vec<Isometry3<f64>>,
}

/// Model quantities at one knot (chain order).
struct Eval {
    mass: DMatrix<f64>,
    nle: DVector<f64>,
    gravity: DVector<f64>,
    tcp: Isometry3<f64>,
    jac: DMatrix<f64>,
    points: Vec<(Vector3<f64>, DMatrix<f64>)>,
}

impl Problem<'_> {
    fn n(&self) -> usize {
        self.idx.len()
    }

    fn full(&self, q: &DVector<f64>, v: &DVector<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut qf = self.q_base.clone();
        let mut vf = vec![0.0; qf.len()];
        for (k, &i) in self.idx.iter().enumerate() {
            qf[i] = q[k];
            vf[i] = v[k];
        }
        (qf, vf)
    }

    fn eval(&self, q: &DVector<f64>, v: &DVector<f64>) -> Result<Eval, MpcError> {
        let (qf, vf) = self.full(q, v);
        let s = self.arm.evaluate_without_jdot(&qf, &vf);
        let sub = |x: &DVector<f64>| DVector::from_iterator(self.n(), self.idx.iter().map(|&i| x[i]));
        let mut points = Vec::new();
        if let Some(ws) = &self.cfg.workspace {
            for (link, local) in &ws.points {
                let p = self
                    .arm
                    .point_state(&qf, &vf, link, *local)
                    .map_err(|e| MpcError::Setup(format!("workspace point on {link}: {e}")))?;
                points.push((p.p, p.jacobian.select_columns(&self.idx)));
            }
        }
        Ok(Eval {
            mass: s.mass.select_rows(&self.idx).select_columns(&self.idx),
            nle: sub(&s.nle),
            gravity: sub(&s.gravity),
            tcp: s.tcp_pose,
            jac: s.tcp_jacobian.select_columns(&self.idx),
            points,
        })
    }

    /// Forward dynamics `q̈ = M⁻¹(τ − h)` at an evaluated knot.
    fn accel(e: &Eval, u: &DVector<f64>) -> Result<DVector<f64>, MpcError> {
        let chol = e.mass.clone().cholesky().ok_or_else(|| MpcError::Setup("mass matrix not positive definite".into()))?;
        Ok(chol.solve(&(u - &e.nle)))
    }

    fn step(&self, q: &DVector<f64>, v: &DVector<f64>, a: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let h = self.cfg.dt;
        (q + v * h + a * (0.5 * h * h), v + a * h)
    }

    /// State cost at knot `k ≥ 1` (value, gradient, Gauss–Newton Hessian over `[q; v]`).
    fn state_cost(&self, k: usize, q: &DVector<f64>, v: &DVector<f64>, e: &Eval, grad: bool) -> (f64, DVector<f64>, DMatrix<f64>) {
        let cfg = self.cfg;
        let n = self.n();
        let term = if k == self.targets.len() - 1 { cfg.terminal_scale } else { 1.0 };
        let mut val = 0.0;
        let mut lx = DVector::zeros(2 * n);
        let mut lxx = DMatrix::zeros(2 * n, 2 * n);
        // TCP: pseudo-Huber per part, ρ(r) = 2w·c²(√(1 + r²/c²) − 1) ≈ w·r² for
        // r ≪ c; gradient 2w·e/s, Gauss–Newton (IRLS) Hessian 2w/s·JᵀJ with
        // s = √(1 + r²/c²); the residual derivative is −J.
        let err = pose_error(&self.targets[k], &e.tcp);
        let rows: std::ops::Range<usize> = if cfg.track_orientation { 0..6 } else { 3..6 };
        let parts: &[(usize, f64, f64)] = if cfg.track_orientation {
            &[(0, cfg.w_rot, cfg.e_max_rot), (3, cfg.w_pos, cfg.e_max_pos)]
        } else {
            &[(3, cfg.w_pos, cfg.e_max_pos)]
        };
        let j = e.jac.rows(rows.start, rows.len()).into_owned();
        let sigma = j.clone().svd(false, false).singular_values.min();
        for &(r0, w, c) in parts {
            let w = term * w;
            let ep = err.fixed_rows::<3>(r0).into_owned();
            let sq = (1.0 + ep.norm_squared() / (c * c)).sqrt();
            val += 2.0 * w * c * c * (sq - 1.0);
            if grad {
                // Residual derivative: −Jv for position; −J_r⁻¹(φ)·Jω for the
                // rotation error φ = log(R_ref·Rᵀ) (exact, not the small-error −Jω).
                let jp: DMatrix<f64> = if r0 == 0 {
                    let jr = right_jacobian_inv(&ep);
                    DMatrix::from_column_slice(3, 3, jr.as_slice()) * e.jac.rows(0, 3)
                } else {
                    e.jac.rows(r0, 3).into_owned()
                };
                let mut lq = lx.rows_mut(0, n);
                lq -= (2.0 * w / sq) * jp.transpose() * DVector::from_column_slice(ep.as_slice());
                let mut lqq = lxx.view_mut((0, 0), (n, n));
                lqq += (2.0 * w / sq) * jp.transpose() * &jp;
            }
        }
        // Posture.
        let dq = q - &self.posture;
        val += cfg.w_posture * dq.norm_squared();
        // Velocity, damped near singularities.
        let ramp = ((cfg.sing_sigma_hi - sigma) / (cfg.sing_sigma_hi - cfg.sing_sigma_lo).max(1e-12)).clamp(0.0, 1.0);
        let wv = term * cfg.w_vel + cfg.w_sing * ramp * ramp;
        val += wv * v.norm_squared();
        if grad {
            for i in 0..n {
                lx[i] += 2.0 * cfg.w_posture * dq[i];
                lxx[(i, i)] += 2.0 * cfg.w_posture;
                lx[n + i] += 2.0 * wv * v[i];
                lxx[(n + i, n + i)] += 2.0 * wv;
            }
        }
        // Soft limits.
        for i in 0..n {
            let over = (q[i] - self.q_hi[i]).max(0.0) + (q[i] - self.q_lo[i]).min(0.0);
            if over != 0.0 {
                val += cfg.w_limit * over * over;
                if grad {
                    lx[i] += 2.0 * cfg.w_limit * over;
                    lxx[(i, i)] += 2.0 * cfg.w_limit;
                }
            }
            let ov = (v[i] - self.v_lim[i]).max(0.0) + (v[i] + self.v_lim[i]).min(0.0);
            if ov != 0.0 {
                val += cfg.w_limit * ov * ov;
                if grad {
                    lx[n + i] += 2.0 * cfg.w_limit * ov;
                    lxx[(n + i, n + i)] += 2.0 * cfg.w_limit;
                }
            }
        }
        if let Some(vt) = cfg.tcp_v_max {
            // TCP velocity Jv·v (the Jacobian held at the knot).
            let jv = e.jac.rows(3, 3).into_owned();
            let tv = &jv * v;
            for r in 0..3 {
                let over = (tv[r] - vt).max(0.0) + (tv[r] + vt).min(0.0);
                if over != 0.0 {
                    val += cfg.w_limit * over * over;
                    if grad {
                        let row = jv.row(r).transpose();
                        let mut lv = lx.rows_mut(n, n);
                        lv += 2.0 * cfg.w_limit * over * &row;
                        let mut lvv = lxx.view_mut((n, n), (n, n));
                        lvv += 2.0 * cfg.w_limit * &row * row.transpose();
                    }
                }
            }
        }
        if let Some(ws) = &cfg.workspace {
            for (p, jp) in &e.points {
                for r in 0..3 {
                    let over = (p[r] - ws.max[r]).max(0.0) + (p[r] - ws.min[r]).min(0.0);
                    if over != 0.0 {
                        val += cfg.w_workspace * over * over;
                        if grad {
                            let row = jp.row(r).transpose();
                            let mut lq = lx.rows_mut(0, n);
                            lq += 2.0 * cfg.w_workspace * over * &row;
                            let mut lqq = lxx.view_mut((0, 0), (n, n));
                            lqq += 2.0 * cfg.w_workspace * &row * row.transpose();
                        }
                    }
                }
            }
        }
        (val, lx, lxx)
    }

    fn control_cost(&self, u: &DVector<f64>, e: &Eval) -> f64 {
        self.cfg.w_tau * (u - &e.gravity).norm_squared()
    }

    fn clamp_u(&self, u: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(self.n(), (0..self.n()).map(|i| u[i].clamp(-self.tau_lim[i], self.tau_lim[i])))
    }
}

/// A rolled-out trajectory with its cost.
struct Traj {
    q: Vec<DVector<f64>>,
    v: Vec<DVector<f64>>,
    a: Vec<DVector<f64>>,
    u: Vec<DVector<f64>>,
    evals: Vec<Eval>,
    cost: f64,
}

/// Torque at knot `k` from the state `(q, v)` and the model there.
type Policy<'a> = dyn FnMut(usize, &DVector<f64>, &DVector<f64>, &Eval) -> DVector<f64> + 'a;

fn rollout(p: &Problem, q0: &DVector<f64>, v0: &DVector<f64>, u_of: &mut Policy) -> Result<Traj, MpcError> {
    let nh = p.cfg.horizon;
    let (mut q, mut v) = (vec![q0.clone()], vec![v0.clone()]);
    let (mut a, mut u, mut evals) = (Vec::new(), Vec::new(), Vec::new());
    let mut cost = 0.0;
    for k in 0..=nh {
        // A line-search candidate can blow up; never hand non-finite or absurd
        // states to the model (misarta's kinematics may not return on NaN).
        if q[k].iter().chain(v[k].iter()).any(|x| !x.is_finite() || x.abs() > 1e3) {
            return Ok(Traj {
                q,
                v,
                a,
                u,
                evals,
                cost: f64::INFINITY,
            });
        }
        let e = p.eval(&q[k], &v[k])?;
        if k >= 1 {
            cost += p.state_cost(k, &q[k], &v[k], &e, false).0;
        }
        if k < nh {
            let uk = p.clamp_u(&u_of(k, &q[k], &v[k], &e));
            let ak = Problem::accel(&e, &uk)?;
            cost += p.control_cost(&uk, &e);
            let (qn, vn) = p.step(&q[k], &v[k], &ak);
            q.push(qn);
            v.push(vn);
            a.push(ak);
            u.push(uk);
        }
        evals.push(e);
    }
    Ok(Traj { q, v, a, u, evals, cost })
}

impl Planner for IlqrMpc {
    fn q_margin(&self) -> f64 {
        self.cfg.q_margin
    }

    fn reset(&mut self) {
        self.prev = None;
        self.mu = self.cfg.mu0;
    }

    fn plan(&mut self, arm: &ArmModel, q_all: &[f64], v_all: &[f64], t: f64, goal: &MpcGoal) -> Result<(JointPlan, MpcReport), MpcError> {
        let t_start = std::time::Instant::now();
        let cfg = self.cfg.clone();
        let (nh, h) = (cfg.horizon, cfg.dt);
        if nh == 0 || h <= 0.0 {
            return Err(MpcError::Setup("horizon and dt must be positive".into()));
        }
        let idx = arm.tcp_chain();
        let n = idx.len();
        let dofs: Vec<_> = idx.iter().map(|&i| arm.dofs()[i].clone()).collect();
        let pick = |x: &[f64]| DVector::from_iterator(n, idx.iter().map(|&i| x[i]));
        let q0 = pick(q_all);
        let v0 = pick(v_all);
        let p = Problem {
            arm,
            cfg: &cfg,
            v_idx: dofs.iter().map(|d| d.v_idx).collect(),
            q_base: q_all.to_vec(),
            q_lo: DVector::from_iterator(n, (0..n).map(|k| (dofs[k].q_min + cfg.q_margin).min(q0[k]))),
            q_hi: DVector::from_iterator(n, (0..n).map(|k| (dofs[k].q_max - cfg.q_margin).max(q0[k]))),
            v_lim: DVector::from_iterator(n, dofs.iter().map(|d| if d.v_max.is_finite() { d.v_max * cfg.v_scale } else { 1e3 })),
            tau_lim: DVector::from_iterator(n, dofs.iter().map(|d| if d.effort.is_finite() { d.effort * cfg.torque_scale } else { 1e4 })),
            posture: match goal.posture {
                Some(x) => pick(x.as_slice()),
                None => q0.clone(),
            },
            targets: (0..=nh).map(|k| (goal.tcp)(t + k as f64 * h)).collect(),
            idx: idx.clone(),
        };

        // Initial guess: the previous policy (torques + feedback, shifted to
        // now; open-loop torques alone diverge on an arm under gravity), else
        // gravity compensation with light damping.
        // Damping that halves the velocity every interval in the discrete model:
        // τ = g − (½/h)·M·v, so q̈ ≈ −½v/h on every joint. A fixed gain (or one
        // scaled by M's diagonal) is unstable for the light wrist coupled to the
        // heavy elbow at h = 40 ms and made the initial guess blow up.
        let hold = |_: usize, _: &DVector<f64>, v: &DVector<f64>, e: &Eval| &e.gravity - (&e.mass * v) * (0.5 / h);
        let mut traj = match self.prev.as_ref().filter(|pr| pr.idx == idx && !pr.u.is_empty()) {
            Some(pr) => {
                let off = ((t - pr.t0) / h).round().max(0.0) as usize;
                let last = pr.u.len() - 1;
                rollout(&p, &q0, &v0, &mut |k, q, v, e| {
                    let j = k + off;
                    if j > last {
                        return hold(k, q, v, e);
                    }
                    let dx = DVector::from_iterator(2 * n, (q - pr.x[j].rows(0, n)).iter().chain((v - pr.x[j].rows(n, n)).iter()).copied());
                    &pr.u[j] + &pr.k[j] * dx
                })?
            }
            None => rollout(&p, &q0, &v0, &mut |k, q, v, e| hold(k, q, v, e))?,
        };
        // Start from whichever is cheaper: a stale policy can be worse than
        // holding still, and a few iterations from it would still be accepted.
        if self.prev.is_some() {
            let held = rollout(&p, &q0, &v0, &mut |k, q, v, e| hold(k, q, v, e))?;
            if traj.cost.is_nan() || traj.cost > held.cost / 0.9 {
                traj = held;
            }
        }
        if !traj.cost.is_finite() {
            return Err(MpcError::Solve("initial rollout diverged".into()));
        }
        let mut last_k = vec![DMatrix::zeros(n, 2 * n); nh];

        let (mut lin_us, mut bwd_us) = (0.0, 0.0);
        let mut iters = 0;
        let mut status = "max_iters".to_string();
        for _ in 0..cfg.max_iters {
            iters += 1;
            // ── linearize ────────────────────────────────────────────────
            let t_lin = std::time::Instant::now();
            let mut fx = Vec::with_capacity(nh);
            let mut fu = Vec::with_capacity(nh);
            for k in 0..nh {
                let (a, b) = linearize(&p, &traj.q[k], &traj.v[k], &traj.a[k], &traj.evals[k])?;
                fx.push(a);
                fu.push(b);
            }
            lin_us += t_lin.elapsed().as_secs_f64() * 1e6;

            // ── backward pass (with regularization retries) ──────────────
            let t_bwd = std::time::Instant::now();
            let mut gains = None;
            for _ in 0..8 {
                match backward(&p, &traj, &fx, &fu, self.mu) {
                    Some(g) => {
                        gains = Some(g);
                        break;
                    }
                    None => self.mu = (self.mu * 10.0).min(cfg.mu_max),
                }
            }
            let Some((kff, kfb)) = gains else {
                status = "backward pass failed".into();
                break;
            };
            last_k = kfb.clone();
            // ── forward pass with line search ────────────────────────────
            let mut accepted = None;
            for alpha in [1.0, 0.5, 0.25, 0.1, 0.03] {
                let cand = rollout(&p, &q0, &v0, &mut |k, q, v, _| {
                    let dx = DVector::from_iterator(2 * n, (q - &traj.q[k]).iter().chain((v - &traj.v[k]).iter()).copied());
                    &traj.u[k] + &kff[k] * alpha + &kfb[k] * dx
                })?;
                if cand.cost < traj.cost {
                    accepted = Some(cand);
                    break;
                }
            }
            bwd_us += t_bwd.elapsed().as_secs_f64() * 1e6;
            match accepted {
                Some(cand) => {
                    let rel = (traj.cost - cand.cost) / traj.cost.max(1e-12);
                    traj = cand;
                    self.mu = (self.mu * 0.3).max(cfg.mu_min);
                    if rel < cfg.tol {
                        status = "converged".into();
                        break;
                    }
                }
                None => {
                    self.mu = (self.mu * 10.0).min(cfg.mu_max);
                    status = "no decrease".into();
                    break;
                }
            }
        }

        let mut a = traj.a.clone();
        a.push(DVector::zeros(n));
        let plan = JointPlan {
            idx: idx.clone(),
            t0: t,
            dt: h,
            q: traj.q.clone(),
            v: traj.v.clone(),
            a,
        };
        let tcp_now = arm.tcp_pose(q_all);
        let e_now = pose_error(&p.targets[0], &tcp_now);
        let e_now_to_end = pose_error(&p.targets[nh], &tcp_now);
        let e_end = pose_error(&p.targets[nh], &traj.evals[nh].tcp);
        self.prev = Some(Prev {
            t0: t,
            idx,
            u: traj.u.clone(),
            k: last_k,
            x: (0..nh)
                .map(|k| DVector::from_iterator(2 * n, traj.q[k].iter().chain(traj.v[k].iter()).copied()))
                .collect(),
        });
        Ok((
            plan,
            MpcReport {
                cost: traj.cost,
                total_us: t_start.elapsed().as_secs_f64() * 1e6,
                linearize_us: lin_us,
                qp_us: bwd_us,
                qp_iterations: iters,
                status,
                relaxed: false,
                tcp_pos_err_now: e_now.fixed_rows::<3>(3).norm(),
                tcp_pos_err_now_to_end: e_now_to_end.fixed_rows::<3>(3).norm(),
                tcp_pos_err_end: e_end.fixed_rows::<3>(3).norm(),
                tcp_rot_err_end: e_end.fixed_rows::<3>(0).norm(),
            },
        ))
    }
}

type Gains = (Vec<DVector<f64>>, Vec<DMatrix<f64>>);

/// `x' ≈ A·x + B·u` around `(q, v)` with acceleration `a = M⁻¹(u − h)`:
/// `q' = q + h·v + ½h²·q̈`, `v' = v + h·q̈`, `∂q̈/∂q = −M⁻¹·∂ID/∂q|_{a}`,
/// `∂q̈/∂v = −M⁻¹·∂ID/∂v|_{a}`, `∂q̈/∂u = M⁻¹` (misarta's analytical RNEA
/// derivatives; the reflected rotor inertia in `M` does not depend on `q`).
fn linearize(p: &Problem, q: &DVector<f64>, v: &DVector<f64>, acc: &DVector<f64>, e: &Eval) -> Result<(DMatrix<f64>, DMatrix<f64>), MpcError> {
    let arm = p.arm;
    let n = p.n();
    let h = p.cfg.dt;
    let (qf, vf) = p.full(q, v);
    let mut af = vec![0.0; arm.raw().nv];
    for (c, &vi) in p.v_idx.iter().enumerate() {
        af[vi] = acc[c];
    }
    let d = misarta::rnea_derivatives::compute_rnea_derivatives(arm.raw(), &arm.full_q(&qf), &arm.full_v(&vf), &af);
    let dq = d.dtau_dq.select_rows(&p.v_idx).select_columns(&p.v_idx);
    let dv = d.dtau_dv.select_rows(&p.v_idx).select_columns(&p.v_idx);
    let minv = e
        .mass
        .clone()
        .cholesky()
        .ok_or_else(|| MpcError::Setup("mass matrix not positive definite".into()))?
        .inverse();
    let fq = -&minv * dq;
    let fv = -&minv * dv;
    let hh = 0.5 * h * h;
    let eye = DMatrix::<f64>::identity(n, n);
    let mut a = DMatrix::zeros(2 * n, 2 * n);
    a.view_mut((0, 0), (n, n)).copy_from(&(&eye + &fq * hh));
    a.view_mut((0, n), (n, n)).copy_from(&(&eye * h + &fv * hh));
    a.view_mut((n, 0), (n, n)).copy_from(&(&fq * h));
    a.view_mut((n, n), (n, n)).copy_from(&(&eye + &fv * h));
    let mut b = DMatrix::zeros(2 * n, n);
    b.view_mut((0, 0), (n, n)).copy_from(&(&minv * hh));
    b.view_mut((n, 0), (n, n)).copy_from(&(&minv * h));
    Ok((a, b))
}

/// Riccati backward pass. `None` if `Q_uu` is not positive definite.
fn backward(p: &Problem, traj: &Traj, fx: &[DMatrix<f64>], fu: &[DMatrix<f64>], mu: f64) -> Option<Gains> {
    let nh = p.cfg.horizon;
    let n = p.n();
    let (_, mut vx, mut vxx) = p.state_cost(nh, &traj.q[nh], &traj.v[nh], &traj.evals[nh], true);
    let mut kff = vec![DVector::zeros(n); nh];
    let mut kfb = vec![DMatrix::zeros(n, 2 * n); nh];
    for k in (0..nh).rev() {
        let (lx, lxx) = if k >= 1 {
            let (_, lx, lxx) = p.state_cost(k, &traj.q[k], &traj.v[k], &traj.evals[k], true);
            (lx, lxx)
        } else {
            (DVector::zeros(2 * n), DMatrix::zeros(2 * n, 2 * n))
        };
        let lu = 2.0 * p.cfg.w_tau * (&traj.u[k] - &traj.evals[k].gravity);
        let luu = DMatrix::identity(n, n) * (2.0 * p.cfg.w_tau);
        let (a, b) = (&fx[k], &fu[k]);
        let qx = lx + a.transpose() * &vx;
        let qu = lu + b.transpose() * &vx;
        let qxx = lxx + a.transpose() * &vxx * a;
        let vreg = &vxx + DMatrix::identity(2 * n, 2 * n) * mu;
        let quu = luu + b.transpose() * &vreg * b;
        let qux = b.transpose() * &vreg * a;
        let chol = quu.clone().cholesky()?;
        let mut kk = -chol.solve(&qu);
        let mut kmat = -chol.solve(&qux);
        // Torque saturated at the nominal: no feedforward push or feedback
        // further into the limit on that axis.
        for i in 0..n {
            let ui = traj.u[k][i];
            let lim = p.tau_lim[i];
            if (ui >= lim - 1e-9 && kk[i] > 0.0) || (ui <= -lim + 1e-9 && kk[i] < 0.0) {
                kk[i] = 0.0;
                kmat.row_mut(i).fill(0.0);
            }
        }
        vx = &qx + kmat.transpose() * &quu * &kk + kmat.transpose() * &qu + qux.transpose() * &kk;
        vxx = &qxx + kmat.transpose() * &quu * &kmat + kmat.transpose() * &qux + qux.transpose() * &kmat;
        vxx = 0.5 * (&vxx + vxx.transpose());
        kff[k] = kk;
        kfb[k] = kmat;
    }
    Some((kff, kfb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use manip_model::TcpSpec;

    fn arm() -> ArmModel {
        let path = format!("{}/../../models/rebot_b601_dm/rebot_b601_dm.misa", env!("CARGO_MANIFEST_DIR"));
        let mut arm = ArmModel::load(path, &TcpSpec::at_link("end_link")).unwrap();
        let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
        for (i, name) in names.iter().enumerate() {
            arm.set_armature(name, if i < 3 { 0.02 } else if i < 6 { 0.005 } else { 80.0 }).unwrap();
        }
        arm
    }

    /// The analytical A, B match central finite differences of the discrete step.
    #[test]
    fn linearization_matches_finite_difference() {
        let arm = arm();
        let cfg = IlqrConfig::defaults();
        let q_all = [0.3, -1.1, -1.3, 0.4, 0.2, -0.3, 0.01];
        let idx = arm.tcp_chain();
        let n = idx.len();
        let dofs: Vec<_> = idx.iter().map(|&i| arm.dofs()[i].clone()).collect();
        let p = Problem {
            arm: &arm,
            cfg: &cfg,
            v_idx: dofs.iter().map(|d| d.v_idx).collect(),
            q_base: q_all.to_vec(),
            q_lo: DVector::from_element(n, -10.0),
            q_hi: DVector::from_element(n, 10.0),
            v_lim: DVector::from_element(n, 10.0),
            tau_lim: DVector::from_element(n, 1e4),
            posture: DVector::zeros(n),
            targets: vec![Isometry3::identity(); cfg.horizon + 1],
            idx: idx.clone(),
        };
        let q = DVector::from_iterator(n, idx.iter().map(|&i| q_all[i]));
        let v = DVector::from_vec(vec![0.4, -0.3, 0.5, -0.6, 0.2, 0.7]);
        let u = DVector::from_vec(vec![1.0, 3.0, -2.0, 0.3, -0.2, 0.1]);
        let step = |q: &DVector<f64>, v: &DVector<f64>, u: &DVector<f64>| {
            let e = p.eval(q, v).unwrap();
            let a = Problem::accel(&e, u).unwrap();
            let (qn, vn) = p.step(q, v, &a);
            DVector::from_iterator(2 * n, qn.iter().chain(vn.iter()).copied())
        };
        let e = p.eval(&q, &v).unwrap();
        let acc = Problem::accel(&e, &u).unwrap();
        let (a, b) = linearize(&p, &q, &v, &acc, &e).unwrap();
        let eps = 1e-6;
        for j in 0..2 * n {
            let (mut qp, mut vp, mut qm, mut vm) = (q.clone(), v.clone(), q.clone(), v.clone());
            if j < n {
                qp[j] += eps;
                qm[j] -= eps;
            } else {
                vp[j - n] += eps;
                vm[j - n] -= eps;
            }
            let col = (step(&qp, &vp, &u) - step(&qm, &vm, &u)) / (2.0 * eps);
            let diff = (&col - a.column(j)).amax();
            assert!(diff < 1e-5 * (1.0 + col.amax()), "A column {j}: {diff}\nfd {:.6}\nan {:.6}", col.transpose(), a.column(j).transpose());
        }
        for j in 0..n {
            let (mut up, mut um) = (u.clone(), u.clone());
            up[j] += eps;
            um[j] -= eps;
            let col = (step(&q, &v, &up) - step(&q, &v, &um)) / (2.0 * eps);
            let diff = (&col - b.column(j)).amax();
            assert!(diff < 1e-5 * (1.0 + col.amax()), "B column {j}: {diff}");
        }
    }
}

/// Inverse right Jacobian of SO(3): for `φ = log(R_ref·Rᵀ)` and a world
/// angular perturbation `δ` of `R`, `φ' ≈ φ − J_r⁻¹(φ)·δ`.
pub fn right_jacobian_inv(phi: &Vector3<f64>) -> nalgebra::Matrix3<f64> {
    let th = phi.norm();
    let w = phi.cross_matrix();
    let i = nalgebra::Matrix3::identity();
    if th < 1e-6 {
        return i + 0.5 * w + (1.0 / 12.0) * w * w;
    }
    let c = 1.0 / (th * th) - (1.0 + th.cos()) / (2.0 * th * th.sin());
    i + 0.5 * w + c * w * w
}

#[cfg(test)]
mod rot_tests {
    use super::*;
    use nalgebra::{Rotation3, UnitQuaternion};

    /// `log(R_ref·exp(−δ)·Rᵀ…)`: the rotation error moves by −J_r⁻¹(φ)·δ for a
    /// world-frame rotation δ of the current orientation (checked numerically).
    #[test]
    fn rotation_error_derivative() {
        let r_ref = UnitQuaternion::from_euler_angles(0.4, -0.7, 1.1);
        let r = UnitQuaternion::from_euler_angles(-0.3, 0.5, -0.9);
        let err = |r: &UnitQuaternion<f64>| (r_ref * r.inverse()).scaled_axis();
        let phi = err(&r);
        let jinv = right_jacobian_inv(&phi);
        let eps = 1e-6;
        for k in 0..3 {
            let mut d = Vector3::zeros();
            d[k] = eps;
            let rp = UnitQuaternion::from_rotation_matrix(&Rotation3::new(d)) * r;
            let rm = UnitQuaternion::from_rotation_matrix(&Rotation3::new(-d)) * r;
            let fd = (err(&rp) - err(&rm)) / (2.0 * eps);
            let an = -jinv.column(k);
            assert!((fd - an).amax() < 1e-6, "column {k}: fd {fd:?} vs {an:?}");
        }
    }
}
