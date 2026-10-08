//! Linear time-varying MPC at the acceleration level (real-time iteration).
//!
//! State `x = [q; v]` of the TCP-chain DOFs, input `u = q̈` held over each
//! interval (`condense.rs`). Around a nominal trajectory (the previous plan,
//! shifted to "now") every knot is linearized with the full rigid-body model:
//!
//! - TCP pose error `e(q) ≈ ē − J̄·(q − q̄)` (position exact to first order,
//!   rotation `log(R_ref·Rᵀ)` for small errors),
//! - torque `τ ≈ M̄·u + h̄` (inverse dynamics at the nominal state),
//! - workspace points `p(q) ≈ p̄ + J̄p·(q − q̄)`.
//!
//! The condensed problem is one dense QP in `U = [u_0 … u_{N−1}]` solved with
//! misa-wbc's `solve_qp` (warm-started from the shifted previous solution).
//! `sqp_iters > 1` re-linearizes around the new solution.
//!
//! Joint and velocity limits, the torque limit and the workspace box are hard
//! constraints. If the current state already violates a state constraint the
//! QP is infeasible; it is then re-solved with the state constraints dropped
//! (input and torque limits kept) and the plan is marked `relaxed`. The WBC
//! that tracks the plan keeps its own exact limits either way.

use manip_model::{ArmModel, ArmState};
use manip_wbc::pose_error;
use misa_wbc::qp::{solve_qp_warm, QpConfig, QpSolver, QpStatus, QpWorkspace};
use nalgebra::{DMatrix, DVector, Isometry3, Vector3};

use crate::condense::Condensed;
use crate::plan::JointPlan;
use crate::{MpcError, MpcGoal, MpcReport, Planner};

/// Points that must stay inside an axis-aligned box (base frame).
#[derive(Debug, Clone)]
pub struct WorkspaceBox {
    /// `(link name, point in the link frame)`.
    pub points: Vec<(String, Vector3<f64>)>,
    pub min: Vector3<f64>,
    pub max: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct LtvConfig {
    /// Number of intervals `N` and their length [s].
    pub horizon: usize,
    pub dt: f64,
    /// TCP tracking weights: position [1/m²], orientation [1/rad²].
    pub w_pos: f64,
    pub w_rot: f64,
    pub track_orientation: bool,
    /// How far from the current TCP one plan aims [m], [rad] (a trust region:
    /// a far target is approached through nearer ones, which keeps the
    /// linearization valid and the QP's active set small). The goal is
    /// moved toward the current pose, not the error clamped per knot: the
    /// knots of the nominal trajectory (coasting at the measured velocity)
    /// can be far from the goal even when the arm is close, and a clamped
    /// error there under-states what the step must correct (on the real
    /// B601-DM a 0.7 rad/s wrist start sent plans 25° off the goal).
    pub e_max_pos: f64,
    pub e_max_rot: f64,
    /// TCP translational speed limit per axis [m/s] (`|J̄v·v_k| ≤ tcp_v_max`,
    /// linearized; the OSC's TCP shaper uses the same 0.3 m/s). `None` = off.
    pub tcp_v_max: Option<f64>,
    /// Multiplier on the TCP and velocity weights at the last knot.
    pub terminal_scale: f64,
    /// Posture (redundancy) [1/rad²], velocity [s²/rad²], acceleration
    /// [s⁴/rad²] and acceleration change [s⁴/rad²] weights.
    pub w_posture: f64,
    pub w_vel: f64,
    pub w_acc: f64,
    pub w_jerk: f64,
    /// Velocity damping near singularities (as in the OSC): at knots where
    /// `σ_min(J̄)` drops from `sing_sigma_hi` to `sing_sigma_lo`, the velocity
    /// weight ramps up to `w_sing`. Without it, an unreachable target pulls
    /// the arm along the Jacobian's weak directions at full joint speed.
    pub sing_sigma_lo: f64,
    pub sing_sigma_hi: f64,
    pub w_sing: f64,
    /// Acceleration limit per independent DOF [rad/s²].
    pub a_max: DVector<f64>,
    /// Velocity limit = model `v_max` × this.
    pub v_scale: f64,
    /// Keep this far inside the joint range [rad].
    pub q_margin: f64,
    /// Torque limit = model effort × this (≤ 1 leaves room for the WBC's feedback).
    pub torque_scale: f64,
    pub torque_limits: bool,
    pub workspace: Option<WorkspaceBox>,
    /// SQP iterations at most; they stop once a step lowers the merit by less
    /// than `sqp_tol` (relative).
    pub sqp_iters: usize,
    pub sqp_tol: f64,
    pub qp: QpConfig,
    /// Re-solve with this backend if `qp` stops without an optimum
    /// (e.g. the active set hits `max_iters`).
    pub fallback: Option<QpSolver>,
}

impl LtvConfig {
    pub fn defaults(n_all: usize) -> Self {
        Self {
            horizon: 20,
            dt: 0.05,
            w_pos: 1e4,
            w_rot: 1e2,
            track_orientation: true,
            e_max_pos: 0.05,
            e_max_rot: 0.3,
            tcp_v_max: Some(0.3),
            terminal_scale: 10.0,
            w_posture: 1e-2,
            w_vel: 1e-2,
            w_acc: 1e-4,
            w_jerk: 1e-4,
            sing_sigma_lo: 0.01,
            sing_sigma_hi: 0.06,
            w_sing: 10.0,
            a_max: DVector::from_element(n_all, 15.0),
            v_scale: 1.0,
            q_margin: 0.02,
            torque_scale: 0.9,
            torque_limits: true,
            workspace: None,
            sqp_iters: 10,
            sqp_tol: 1e-3,
            qp: QpConfig {
                solver: QpSolver::ActiveSet,
                max_iters: 2000,
                ..QpConfig::default()
            },
            fallback: Some(QpSolver::Clarabel),
        }
    }
}

/// The linear time-varying MPC.
pub struct LtvMpc {
    pub cfg: LtvConfig,
    prev: Option<JointPlan>,
    workspace: QpWorkspace,
    /// Hessian of the state-independent cost terms (posture, velocity,
    /// acceleration, acceleration change), keyed by what it depends on.
    h_const: Option<(Vec<u64>, DMatrix<f64>)>,
}

impl LtvMpc {
    pub fn new(cfg: LtvConfig) -> Self {
        Self {
            cfg,
            prev: None,
            workspace: QpWorkspace::new(),
            h_const: None,
        }
    }

    /// The last plan returned.
    pub fn last_plan(&self) -> Option<&JointPlan> {
        self.prev.as_ref()
    }
}

/// Least-squares terms `Σ w·‖A·U − b‖²` accumulated as `½UᵀHU + gᵀU`.
struct Cost {
    h: DMatrix<f64>,
    g: DVector<f64>,
    c0: f64,
}

impl Cost {
    fn new(nu: usize) -> Self {
        Self {
            h: DMatrix::zeros(nu, nu),
            g: DVector::zeros(nu),
            c0: 0.0,
        }
    }

    fn add(&mut self, a: &DMatrix<f64>, b: &DVector<f64>, w: &DVector<f64>) {
        let wa = DMatrix::from_fn(a.nrows(), a.ncols(), |r, c| a[(r, c)] * w[r]);
        self.h += 2.0 * a.transpose() * &wa;
        self.g -= 2.0 * wa.transpose() * b;
        self.c0 += b.iter().zip(w.iter()).map(|(bi, wi)| wi * bi * bi).sum::<f64>();
    }

    /// [`Self::add`] for a term whose Hessian is already in `h` (only the
    /// gradient and constant depend on `b`).
    fn add_linear(&mut self, a: &DMatrix<f64>, b: &DVector<f64>, w: &DVector<f64>) {
        let wb = b.component_mul(w);
        self.g -= 2.0 * a.transpose() * &wb;
        self.c0 += b.dot(&wb);
    }

    fn value(&self, u: &DVector<f64>) -> f64 {
        0.5 * u.dot(&(&self.h * u)) + self.g.dot(u) + self.c0
    }
}

/// Inequalities `A·U ≤ b`, stacked.
#[derive(Default)]
struct Ineq {
    rows: Vec<(DMatrix<f64>, DVector<f64>)>,
}

impl Ineq {
    /// `lo ≤ A·U + c ≤ hi`.
    fn range(&mut self, a: &DMatrix<f64>, c: &DVector<f64>, lo: &DVector<f64>, hi: &DVector<f64>) {
        self.rows.push((a.clone(), hi - c));
        self.rows.push((-a, -(lo - c)));
    }

    fn stack(&self, nu: usize) -> Option<(DMatrix<f64>, DVector<f64>)> {
        let m: usize = self.rows.iter().map(|(a, _)| a.nrows()).sum();
        if m == 0 {
            return None;
        }
        let mut a = DMatrix::zeros(m, nu);
        let mut b = DVector::zeros(m);
        let mut r = 0;
        for (ai, bi) in &self.rows {
            a.view_mut((r, 0), (ai.nrows(), nu)).copy_from(ai);
            b.rows_mut(r, bi.len()).copy_from(bi);
            r += ai.nrows();
        }
        Some((a, b))
    }
}

/// Model quantities at one knot of the nominal trajectory (chain order).
struct Knot {
    q: DVector<f64>,
    tcp: Isometry3<f64>,
    jac: DMatrix<f64>,
    mass: DMatrix<f64>,
    nle: DVector<f64>,
    points: Vec<(Vector3<f64>, DMatrix<f64>)>,
}

/// Solve with `cfg.qp`, then with `cfg.fallback` if that stopped without an optimum.
fn solve(
    h: &DMatrix<f64>,
    g: &DVector<f64>,
    ineq: Option<&(DMatrix<f64>, DVector<f64>)>,
    x0: Option<&DVector<f64>>,
    cfg: &LtvConfig,
    ws: &mut QpWorkspace,
) -> misa_wbc::QpSolution {
    let run = |qp: &QpConfig, ws: &mut QpWorkspace| {
        solve_qp_warm(h, g, None, None, ineq.map(|(a, _)| a), ineq.map(|(_, b)| b), x0, qp, ws)
    };
    let sol = run(&cfg.qp, ws);
    match (sol.status, cfg.fallback) {
        (QpStatus::Optimal, _) | (_, None) => sol,
        (_, Some(backend)) => {
            *ws = QpWorkspace::new();
            let sol = run(&QpConfig { solver: backend, ..cfg.qp.clone() }, ws);
            *ws = QpWorkspace::new();
            sol
        }
    }
}

/// Scale the translation / rotation parts of a pose error down to the caps.
/// `goal`, moved toward `from` so that it is at most `max_pos` / `max_rot` away.
fn toward(goal: Isometry3<f64>, from: &Isometry3<f64>, max_pos: f64, max_rot: f64) -> Isometry3<f64> {
    let dp = goal.translation.vector - from.translation.vector;
    let p = if dp.norm() > max_pos { from.translation.vector + dp * (max_pos / dp.norm()) } else { goal.translation.vector };
    let dr = goal.rotation * from.rotation.inverse();
    let angle = dr.angle();
    let r = if angle > max_rot { dr.powf(max_rot / angle) * from.rotation } else { goal.rotation };
    Isometry3::from_parts(nalgebra::Translation3::from(p), r)
}

fn select_cols(m: &DMatrix<f64>, idx: &[usize]) -> DMatrix<f64> {
    m.select_columns(idx)
}

impl LtvMpc {
    #[allow(clippy::too_many_arguments)]
    fn knot(
        &self,
        arm: &ArmModel,
        idx: &[usize],
        q_base: &[f64],
        q: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<Knot, MpcError> {
        let mut qf = q_base.to_vec();
        let mut vf = vec![0.0; qf.len()];
        for (k, &i) in idx.iter().enumerate() {
            qf[i] = q[k];
            vf[i] = v[k];
        }
        let s: ArmState = arm.evaluate_without_jdot(&qf, &vf);
        let mut points = Vec::new();
        if let Some(ws) = &self.cfg.workspace {
            for (link, local) in &ws.points {
                let p = arm
                    .point_state(&qf, &vf, link, *local)
                    .map_err(|e| MpcError::Setup(format!("workspace point on {link}: {e}")))?;
                points.push((p.p, select_cols(&p.jacobian, idx)));
            }
        }
        Ok(Knot {
            q: q.clone(),
            tcp: s.tcp_pose,
            jac: select_cols(&s.tcp_jacobian, idx),
            mass: s.mass.select_rows(idx).select_columns(idx),
            nle: DVector::from_iterator(idx.len(), idx.iter().map(|&i| s.nle[i])),
            points,
        })
    }
}

impl Planner for LtvMpc {
    fn q_margin(&self) -> f64 {
        self.cfg.q_margin
    }

    fn reset(&mut self) {
        self.prev = None;
        self.workspace = QpWorkspace::new();
    }

    fn plan(&mut self, arm: &ArmModel, q_all: &[f64], v_all: &[f64], t: f64, goal: &MpcGoal) -> Result<(JointPlan, MpcReport), MpcError> {
        let t_start = std::time::Instant::now();
        let cfg = self.cfg.clone();
        let idx = arm.tcp_chain();
        let n = idx.len();
        let (nh, h) = (cfg.horizon, cfg.dt);
        if nh == 0 || h <= 0.0 {
            return Err(MpcError::Setup("horizon and dt must be positive".into()));
        }
        let nu = nh * n;
        let pick = |x: &[f64]| DVector::from_iterator(n, idx.iter().map(|&i| x[i]));
        let q0 = pick(q_all);
        let v0 = pick(v_all);
        let dofs: Vec<_> = idx.iter().map(|&i| arm.dofs()[i].clone()).collect();
        let a_max = DVector::from_iterator(n, idx.iter().map(|&i| cfg.a_max[i]));
        let v_lim = DVector::from_iterator(n, dofs.iter().map(|d| if d.v_max.is_finite() { d.v_max * cfg.v_scale } else { 1e3 }));
        let tau_lim = DVector::from_iterator(n, dofs.iter().map(|d| if d.effort.is_finite() { d.effort * cfg.torque_scale } else { 1e4 }));
        // Joint range with margin, widened to include the current position (an
        // arm resting on its stop reads slightly past the limit).
        let q_lo = DVector::from_iterator(n, (0..n).map(|k| (dofs[k].q_min + cfg.q_margin).min(q0[k])));
        let q_hi = DVector::from_iterator(n, (0..n).map(|k| (dofs[k].q_max - cfg.q_margin).max(q0[k])));
        let posture = match goal.posture {
            Some(p) => pick(p.as_slice()),
            None => q0.clone(),
        };
        let cond = Condensed::new(&q0, &v0, nh, h);

        // Nominal input: the previous plan's accelerations from "now" on.
        let mut u_nom = DVector::zeros(nu);
        let mut u_prev_applied = DVector::zeros(n);
        if let Some(p) = self.prev.as_ref().filter(|p| p.idx == idx) {
            for k in 0..nh {
                let a = p.sample(t + k as f64 * h).a;
                u_nom.rows_mut(k * n, n).copy_from(&a);
            }
            u_prev_applied = p.sample(t).a;
        }

        // Hessian of the state-independent terms, built once per configuration.
        let key: Vec<u64> = [nh as f64, n as f64, h, cfg.w_posture, cfg.w_vel, cfg.terminal_scale, cfg.w_acc, cfg.w_jerk]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        if self.h_const.as_ref().is_none_or(|(k, _)| *k != key) {
            let mut c = Cost::new(nu);
            for k in 1..=nh {
                let term = if k == nh { cfg.terminal_scale } else { 1.0 };
                c.add(&cond.gq[k], &DVector::zeros(n), &DVector::from_element(n, cfg.w_posture));
                c.add(&cond.gv[k], &DVector::zeros(n), &DVector::from_element(n, term * cfg.w_vel));
            }
            for k in 0..nh {
                let gu = cond.gu(k);
                c.add(&gu, &DVector::zeros(n), &DVector::from_element(n, cfg.w_acc));
                if cfg.w_jerk > 0.0 {
                    let a = if k == 0 { gu } else { &gu - cond.gu(k - 1) };
                    c.add(&a, &DVector::zeros(n), &DVector::from_element(n, cfg.w_jerk));
                }
            }
            self.h_const = Some((key, c.h));
        }
        let h_const = self.h_const.as_ref().map(|(_, m)| m.clone()).expect("built above");

        let mut linearize_us = 0.0;
        let mut qp_us = 0.0;
        let mut iterations = 0;
        let mut relaxed = false;
        let mut cost_value = 0.0;
        let mut status = String::new();
        // Merit of an input sequence on the nonlinear kinematics (the QP's
        // terms, plus range and speed violations): an SQP step is taken only
        // as far as it lowers it. Unchecked, steps re-linearized around a bad
        // solution ran away on the real B601-DM (plans ending 60° from the
        // goal, the arm shaking violently with the leader still).
        let tcp_now = arm.tcp_pose(q_all);
        let aim: Vec<Isometry3<f64>> = (0..=nh)
            .map(|k| toward((goal.tcp)(t + k as f64 * h), &tcp_now, cfg.e_max_pos, cfg.e_max_rot))
            .collect();
        let rows_m: std::ops::Range<usize> = if cfg.track_orientation { 0..6 } else { 3..6 };
        // Workspace points and their box, widened to where they are now (as in the QP).
        let ws_pos = |qf: &[f64]| -> Vec<Vector3<f64>> {
            cfg.workspace
                .as_ref()
                .map(|ws| {
                    ws.points
                        .iter()
                        .filter_map(|(link, local)| arm.link_pose(qf, link).ok().map(|x| (x * nalgebra::Point3::from(*local)).coords))
                        .collect()
                })
                .unwrap_or_default()
        };
        let ws_now = ws_pos(q_all);
        let merit = |u: &DVector<f64>| -> f64 {
            let (qn, vn) = cond.rollout(u);
            let mut m = cfg.w_acc * u.norm_squared();
            let mut qf = q_all.to_vec();
            for k in 1..=nh {
                let term = if k == nh { cfg.terminal_scale } else { 1.0 };
                for (c, &i) in idx.iter().enumerate() {
                    qf[i] = qn[k][c];
                }
                let e = pose_error(&aim[k], &arm.tcp_pose(&qf));
                for r in rows_m.clone() {
                    m += term * if r < 3 { cfg.w_rot } else { cfg.w_pos } * e[r] * e[r];
                }
                m += cfg.w_posture * (&qn[k] - &posture).norm_squared() + term * cfg.w_vel * vn[k].norm_squared();
                if let Some(ws) = &cfg.workspace {
                    for (i, p) in ws_pos(&qf).iter().enumerate() {
                        let now = ws_now.get(i).copied().unwrap_or(*p);
                        for r in 0..3 {
                            let over = (p[r] - ws.max[r].max(now[r])).max(0.0) + (ws.min[r].min(now[r]) - p[r]).max(0.0);
                            m += 1e6 * over * over;
                        }
                    }
                }
                for c in 0..n {
                    let over_q = (qn[k][c] - q_hi[c]).max(0.0) + (q_lo[c] - qn[k][c]).max(0.0);
                    let over_v = (vn[k][c].abs() - v_lim[c]).max(0.0);
                    m += 1e6 * (over_q * over_q + over_v * over_v);
                }
            }
            m
        };
        // Start from the better of the previous plan (shifted) and braking to
        // rest at `a_max`: coasting at the measured velocity for the whole
        // horizon can be 50° from anything sensible, and the first
        // linearization around it sent joints to their limits.
        let mut brake = DVector::zeros(nu);
        let mut vb = v0.clone();
        for k in 0..nh {
            for c in 0..n {
                let a = (-vb[c] / h).clamp(-a_max[c], a_max[c]);
                brake[k * n + c] = a;
                vb[c] += a * h;
            }
        }
        let mut merit_nom = merit(&u_nom);
        let m_brake = merit(&brake);
        if m_brake < merit_nom {
            u_nom = brake;
            merit_nom = m_brake;
        }
        for _ in 0..cfg.sqp_iters.max(1) {
            let t_lin = std::time::Instant::now();
            let (qn, vn) = cond.rollout(&u_nom);
            let knots = (0..=nh)
                .map(|k| self.knot(arm, &idx, q_all, &qn[k], &vn[k]))
                .collect::<Result<Vec<_>, _>>()?;
            let p0: Vec<Vector3<f64>> = knots[0].points.iter().map(|(p, _)| *p).collect();

            let mut cost = Cost::new(nu);
            cost.h = h_const.clone();
            let mut ineq = Ineq::default();
            let rows: std::ops::Range<usize> = if cfg.track_orientation { 0..6 } else { 3..6 };
            for (k, kn) in knots.iter().enumerate() {
                let term = if k == nh { cfg.terminal_scale } else { 1.0 };
                if k >= 1 {
                    // TCP: e(q_k) ≈ ē − J̄(q_free + Gq·U − q̄)  →  ‖J̄·Gq·U − (ē − J̄(q_free − q̄))‖²
                    let e = pose_error(&aim[k], &kn.tcp);
                    let j = kn.jac.rows(rows.start, rows.len()).into_owned();
                    let a = &j * &cond.gq[k];
                    let e_rows = DVector::from_iterator(rows.len(), rows.clone().map(|r| e[r]));
                    let b = e_rows - &j * (&cond.q_free[k] - &kn.q);
                    let w = DVector::from_iterator(rows.len(), rows.clone().map(|r| term * if r < 3 { cfg.w_rot } else { cfg.w_pos }));
                    cost.add(&a, &b, &w);
                    // Posture and velocity.
                    cost.add_linear(&cond.gq[k], &(&posture - &cond.q_free[k]), &DVector::from_element(n, cfg.w_posture));
                    let sigma = j.clone().svd(false, false).singular_values.min();
                    let ramp = ((cfg.sing_sigma_hi - sigma) / (cfg.sing_sigma_hi - cfg.sing_sigma_lo).max(1e-12)).clamp(0.0, 1.0);
                    cost.add_linear(&cond.gv[k], &(-&cond.v_free[k]), &DVector::from_element(n, term * cfg.w_vel));
                    if ramp > 0.0 {
                        cost.add(&cond.gv[k], &(-&cond.v_free[k]), &DVector::from_element(n, cfg.w_sing * ramp * ramp));
                    }
                    // State limits.
                    ineq.range(&cond.gq[k], &cond.q_free[k], &q_lo, &q_hi);
                    ineq.range(&cond.gv[k], &cond.v_free[k], &(-&v_lim), &v_lim);
                    if let Some(vt) = cfg.tcp_v_max {
                        let jv = kn.jac.rows(3, 3).into_owned();
                        let lim = DVector::from_element(3, vt);
                        ineq.range(&(&jv * &cond.gv[k]), &(&jv * &cond.v_free[k]), &(-&lim), &lim);
                    }
                    if let Some(ws) = &cfg.workspace {
                        for (i, (p, jp)) in kn.points.iter().enumerate() {
                            let a = jp * &cond.gq[k];
                            let c = p + jp * (&cond.q_free[k] - &kn.q);
                            let c = DVector::from_column_slice(c.as_slice());
                            let lo = DVector::from_iterator(3, (0..3).map(|r| ws.min[r].min(p0[i][r])));
                            let hi = DVector::from_iterator(3, (0..3).map(|r| ws.max[r].max(p0[i][r])));
                            ineq.range(&a, &c, &lo, &hi);
                        }
                    }
                }
                if k < nh {
                    let gu = cond.gu(k);
                    if cfg.w_jerk > 0.0 && k == 0 {
                        cost.add_linear(&gu, &u_prev_applied, &DVector::from_element(n, cfg.w_jerk));
                    }
                    ineq.range(&gu, &DVector::zeros(n), &(-&a_max), &a_max);
                    if cfg.torque_limits {
                        ineq.range(&(&kn.mass * &gu), &kn.nle, &(-&tau_lim), &tau_lim);
                    }
                }
            }
            linearize_us += t_lin.elapsed().as_secs_f64() * 1e6;

            // Solve; if the state constraints are infeasible, keep only the
            // input and torque limits.
            let mut hmat = cost.h.clone();
            for i in 0..nu {
                hmat[(i, i)] += 1e-9;
            }
            let t_qp = std::time::Instant::now();
            let stacked = ineq.stack(nu);
            let mut sol = solve(&hmat, &cost.g, stacked.as_ref(), Some(&u_nom), &cfg, &mut self.workspace);
            if !matches!(sol.status, QpStatus::Optimal) {
                let mut inputs_only = Ineq::default();
                for (k, kn) in knots.iter().take(nh).enumerate() {
                    let gu = cond.gu(k);
                    inputs_only.range(&gu, &DVector::zeros(n), &(-&a_max), &a_max);
                    if cfg.torque_limits {
                        inputs_only.range(&(&kn.mass * &gu), &kn.nle, &(-&tau_lim), &tau_lim);
                    }
                }
                let st = inputs_only.stack(nu);
                self.workspace = QpWorkspace::new();
                sol = solve(&hmat, &cost.g, st.as_ref(), None, &cfg, &mut self.workspace);
                relaxed = true;
            }
            qp_us += t_qp.elapsed().as_secs_f64() * 1e6;
            iterations += sol.iterations;
            status = format!("{:?}", sol.status);
            if !matches!(sol.status, QpStatus::Optimal) {
                return Err(MpcError::Solve(status));
            }
            cost_value = cost.value(&sol.x);
            let step = &sol.x - &u_nom;
            let mut taken = false;
            let mut converged = false;
            for alpha in [1.0, 0.5, 0.25] {
                let cand = &u_nom + &step * alpha;
                let m = merit(&cand);
                if m < merit_nom {
                    let rel = (merit_nom - m) / merit_nom.max(1e-12);
                    u_nom = cand;
                    merit_nom = m;
                    taken = true;
                    converged = rel < cfg.sqp_tol;
                    break;
                }
            }
            if !taken {
                status = format!("{status} (step rejected)");
                break;
            }
            if converged {
                break;
            }
        }

        let (q, v) = cond.rollout(&u_nom);
        let mut a: Vec<DVector<f64>> = (0..nh).map(|k| u_nom.rows(k * n, n).into_owned()).collect();
        a.push(DVector::zeros(n));
        let plan = JointPlan {
            idx: idx.clone(),
            t0: t,
            dt: h,
            q,
            v,
            a,
        };
        // Predicted TCP error at the end of the horizon (full model, not linearized).
        let mut qf = q_all.to_vec();
        for (k, &i) in idx.iter().enumerate() {
            qf[i] = plan.q[nh][k];
        }
        let tcp_end = arm.tcp_pose(&qf);
        let target_end = (goal.tcp)(t + nh as f64 * h);
        let e_end = pose_error(&target_end, &tcp_end);
        let tcp_now = arm.tcp_pose(q_all);
        let e_now = pose_error(&(goal.tcp)(t), &tcp_now);
        let e_now_to_end = pose_error(&target_end, &tcp_now);
        self.prev = Some(plan.clone());
        Ok((
            plan,
            MpcReport {
                cost: cost_value,
                total_us: t_start.elapsed().as_secs_f64() * 1e6,
                linearize_us,
                qp_us,
                qp_iterations: iterations,
                status,
                relaxed,
                tcp_pos_err_now: e_now.fixed_rows::<3>(3).norm(),
                tcp_pos_err_now_to_end: e_now_to_end.fixed_rows::<3>(3).norm(),
                tcp_pos_err_end: e_end.fixed_rows::<3>(3).norm(),
                tcp_rot_err_end: e_end.fixed_rows::<3>(0).norm(),
            },
        ))
    }
}
