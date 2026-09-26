//! Operational-space control (OSC) solved with misa-wbc's hierarchical QP.
//!
//! The decision variable is `x = [q̈; τ]` (fixed base, no contacts: pass
//! `Jc = 0 × n` to `Formulation::Explicit`). Priorities:
//!
//! | level | contents | role |
//! |---|---|---|
//! | 0 | equations of motion, torque limits, joint-limit CBF | must hold (hard) |
//! | 1 | TCP acceleration (pose PD + reference acceleration) | objective |
//! | 2 | posture (joint PD), torque regularization | use of redundancy |
//!
//! Joint limits are not enforced by slamming torque, but placed at the top
//! level as **bounds on q̈ (CBF)**. Even if the target is outside the range of
//! motion, the TCP task merely "can't be fully achieved" and the joint
//! decelerates and stops short of the limit.
//!
//! # Output
//!
//! `τ` goes out as `τff` with `kp = 0`. If `motor_kd > 0`, motor-side damping
//! is added with `v = v + q̈·dt` as the target velocity (it doesn't act on
//! motion that follows the QP solution, only suppresses high-frequency jitter).
//!
//! # Only DOFs on the TCP chain are solved
//!
//! DOFs that don't move the TCP, like gripper fingers, are kept out of the QP
//! (the caller's `base` command is used as-is). Including them put the
//! fingers' reflected inertia (tens of kg equivalent via rack & pinion) and
//! the wrist (0.003 kg·m²) in the same `M`, pushing the condition number above
//! 10⁴, and Clarabel returned NumericalFailure at level 0. The reaction force
//! of finger acceleration on the arm is ignored (the fingers weigh tens of g).
//!
//! # On failure
//!
//! A cycle where the QP can't be solved optimally returns
//! [`OscError::Degraded`]. The caller should fall back to holding in place
//! (joint impedance + gravity). A single cycle may yield an inconsistent
//! solution (measured on keel in misa-runner), so the magnitude of the solved
//! τ is also checked.

use manip_model::{ArmModel, ArmState};

use misa_wbc::dynamics::{Dynamics, Formulation};
use misa_wbc::solve::{SolveConfig, SolveStatus, Solver};
use misa_wbc::tasks::{self, JointLimitCbf};
use misa_wbc::Task;
use nalgebra::{DMatrix, DVector, Isometry3, Vector3, Vector6};

use crate::command::{AxisCmd, JointCommand};
use crate::shaper::{stack6, JointRef};

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

#[derive(Debug, Clone)]
pub struct OscConfig {
    /// TCP translational PD [1/s², 1/s] (acceleration units).
    pub kp_lin: f64,
    pub kd_lin: f64,
    /// TCP rotational PD.
    pub kp_ang: f64,
    pub kd_ang: f64,
    /// Whether to track orientation too. If false, the TCP tracks position
    /// only (3 DOF) and orientation is left to redundancy (level 2).
    pub track_orientation: bool,
    /// Posture task PD (per joint).
    pub posture_kp: DVector<f64>,
    pub posture_kd: DVector<f64>,
    /// Torque regularization weight (level 2, relative to posture).
    pub torque_reg: f64,
    /// Torque limit = model effort × this factor.
    pub torque_scale: f64,
    /// Joint acceleration limit [rad/s²] (final clamp of the CBF).
    pub a_max: DVector<f64>,
    /// CBF pole [1/s]. How far ahead (in time) to start decelerating when approaching a limit.
    pub cbf_alpha: f64,
    /// CBF gain for the velocity limit [1/s].
    pub cbf_alpha_v: f64,
    /// Joint velocity limit factor (model velocity × this; default 1).
    pub v_scale: f64,
    /// Damping added on the motor side [N·m·s/rad].
    pub motor_kd: DVector<f64>,
    /// Joint damping blended into the TCP task near singularities (an
    /// acceleration-level DLS).
    ///
    /// As `σ_min(J)` drops from `sing_sigma_hi` to `sing_sigma_lo`,
    /// `λ²‖q̈ + sing_kd·v‖²` is smoothly added to level 1. misa-wbc's
    /// `cartesian_acceleration_damped` penalizes `‖q̈‖²`, so it does **not
    /// stop** a moving axis (zero acceleration = keeps drifting at constant
    /// velocity). That is what happened when J4 and J6 kept spinning in
    /// opposite directions at a wrist singularity (the TCP doesn't move, so
    /// level 1 doesn't notice, and the level-2 posture task only acts in the
    /// null space of level 1).
    pub sing_sigma_lo: f64,
    pub sing_sigma_hi: f64,
    pub sing_lambda_sq: f64,
    pub sing_kd: f64,
    /// Solution check: discard if |τ| exceeds this value × the limit.
    pub solution_check: f64,
    /// Choice of decision variables (misa-wbc's Formulation). With a fixed base
    /// and no contacts, `AccelSpace` (τ eliminated, n variables) is smallest.
    pub formulation: Formulation,
    pub solve: SolveConfig,
}

impl OscConfig {
    pub fn defaults(n: usize) -> Self {
        Self {
            kp_lin: 400.0,
            kd_lin: 40.0,
            kp_ang: 200.0,
            kd_ang: 28.0,
            track_orientation: true,
            posture_kp: DVector::from_element(n, 25.0),
            posture_kd: DVector::from_element(n, 10.0),
            torque_reg: 1e-3,
            torque_scale: 1.0,
            a_max: DVector::from_element(n, 50.0),
            cbf_alpha: 10.0,
            cbf_alpha_v: 20.0,
            v_scale: 1.0,
            motor_kd: DVector::zeros(n),
            sing_sigma_lo: 0.01,
            sing_sigma_hi: 0.06,
            sing_lambda_sq: 1e-2,
            sing_kd: 10.0,
            solution_check: 1.05,
            formulation: Formulation::AccelSpace,
            // Against a per-cycle budget of 2 ms (500 Hz), Clarabel takes a
            // median of 0.46 ms and ActiveSet 0.05–0.09 ms (B601-DM, 6 DOF,
            // osc_bench). Tracking results were the same.
            solve: SolveConfig {
                backend: misa_wbc::QpSolver::ActiveSet,
                ..SolveConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OscError {
    #[error("QP could not be solved optimally: {0}")]
    Degraded(String),
    #[error("solution torque exceeds the limit (axis {axis}: {tau:.2} / {limit:.2})")]
    TorqueOutOfRange { axis: usize, tau: f64, limit: f64 },
    #[error("could not set up the QP: {0}")]
    Setup(String),
}

/// Per-cycle breakdown (for logging and tuning).
#[derive(Debug, Clone, Default)]
pub struct OscReport {
    pub qddot: DVector<f64>,
    pub tau: DVector<f64>,
    /// TCP error `[rotation; translation]` (world coordinates).
    pub tcp_error: Vector6<f64>,
    /// How well the TCP acceleration was achieved: requested − realized (at the QP solution).
    pub tcp_accel_residual: Vector6<f64>,
    /// Minimum singular value of the TCP Jacobian (proximity to a singularity).
    pub sigma_min: f64,
    /// Time taken to solve the QP [µs].
    pub solve_us: f64,
    /// If only the posture (level 2) degraded this cycle, its status.
    pub degraded: Option<String>,
}

/// Operational-space controller. Holds the QP warm-start state.
pub struct Osc {
    pub cfg: OscConfig,
    solver: Solver,
}

impl Osc {
    pub fn new(cfg: OscConfig) -> Self {
        Self {
            cfg,
            solver: Solver::new(),
        }
    }

    /// Discards the warm start (on mode switches).
    pub fn reset(&mut self) {
        self.solver.reset();
    }

    /// `base` is a command for all DOFs (joint impedance, etc.). Only the DOFs
    /// on the TCP chain are overwritten with the OSC solution.
    pub fn command(
        &mut self,
        arm: &ArmModel,
        s_full: &ArmState,
        tcp: &TcpRef,
        posture_full: &JointRef,
        dt: f64,
        base: JointCommand,
    ) -> Result<(JointCommand, OscReport), OscError> {
        let idx = arm.tcp_chain();
        let n = idx.len();
        let c = &self.cfg;
        let all = arm.dofs();
        let dofs: Vec<_> = idx.iter().map(|&i| all[i].clone()).collect();
        let sub = |v: &DVector<f64>| DVector::from_iterator(n, idx.iter().map(|&i| v[i]));
        let s = SubState {
            q: sub(&s_full.q),
            v: sub(&s_full.v),
            mass: s_full.mass.select_rows(&idx).select_columns(&idx),
            nle: sub(&s_full.nle),
            tcp_pose: s_full.tcp_pose,
            tcp_twist: s_full.tcp_twist,
            tcp_jacobian: s_full.tcp_jacobian.select_columns(&idx),
            tcp_jdot_v: s_full.tcp_jdot_v,
        };
        let posture = JointRef {
            q: sub(&posture_full.q),
            v: sub(&posture_full.v),
            a: sub(&posture_full.a),
        };
        let pick = |v: &DVector<f64>| sub(v);
        let posture_kp = pick(&c.posture_kp);
        let posture_kd = pick(&c.posture_kd);
        let a_max = pick(&c.a_max);
        let motor_kd = pick(&c.motor_kd);

        let tau_max = DVector::from_iterator(
            n,
            dofs.iter().map(|d| {
                let e = d.effort * c.torque_scale;
                if e.is_finite() { e } else { 1e3 }
            }),
        );

        // ── level 0: physics and safety ────────────────────────────────
        let d = Dynamics::new(c.formulation, &s.mass, &s.nle, &DMatrix::zeros(0, n), n);
        let big = 1e3;
        let cbf = JointLimitCbf {
            q_min: DVector::from_iterator(n, dofs.iter().map(|d| finite_or(d.q_min, -big))),
            q_max: DVector::from_iterator(n, dofs.iter().map(|d| finite_or(d.q_max, big))),
            v_max: DVector::from_iterator(n, dofs.iter().map(|d| finite_or(d.v_max * c.v_scale, big))),
            a_max: a_max.clone(),
            alpha1: DVector::from_element(n, c.cbf_alpha),
            alpha2: DVector::from_element(n, c.cbf_alpha),
            alpha3: DVector::from_element(n, c.cbf_alpha_v),
        };
        // Equations of motion: an equality task with Explicit; already
        // eliminated with AccelSpace / ForceSpace (with a fixed base every axis
        // is actuated, so no rows remain).
        let mut level0 = tasks::box_bound(d.tau(), &tau_max)
            + tasks::joint_limit_cbf(d.qddot(), &s.q, &s.v, &cbf);
        if let Some(eom) = d.dynamics_task().filter(|t| t.n_eq() + t.n_iq() > 0) {
            level0 = eom + level0;
        }

        // ── level 1: TCP ───────────────────────────────────────────────
        let err = pose_error(&tcp.pose, &s.tcp_pose);
        let twist_err = tcp.twist - s.tcp_twist;
        let kp = Vector6::new(c.kp_ang, c.kp_ang, c.kp_ang, c.kp_lin, c.kp_lin, c.kp_lin);
        let kd = Vector6::new(c.kd_ang, c.kd_ang, c.kd_ang, c.kd_lin, c.kd_lin, c.kd_lin);
        let a_ref = tcp.accel + kp.component_mul(&err) + kd.component_mul(&twist_err);
        let rows: std::ops::Range<usize> = if c.track_orientation { 0..6 } else { 3..6 };
        let j = s.tcp_jacobian.rows(rows.start, rows.len()).into_owned();
        let djv = DVector::from_iterator(rows.len(), rows.clone().map(|r| s.tcp_jdot_v[r]));
        let a_task = DVector::from_iterator(rows.len(), rows.clone().map(|r| a_ref[r]));
        let sigma_min = j.clone().svd(false, false).singular_values.min();
        let span = (c.sing_sigma_hi - c.sing_sigma_lo).max(1e-12);
        let ramp = ((c.sing_sigma_hi - sigma_min) / span).clamp(0.0, 1.0);
        let mut level1 = tasks::cartesian_acceleration(d.qddot(), &j, &djv, &a_task);
        if ramp > 0.0 {
            level1 = level1
                + tasks::track(d.qddot(), &(-c.sing_kd * &s.v)).weight(c.sing_lambda_sq * ramp * ramp);
        }

        // ── level 2: posture and regularization ────────────────────────
        let qdd_posture = &posture.a
            + posture_kp.component_mul(&(&posture.q - &s.q))
            + posture_kd.component_mul(&(&posture.v - &s.v));
        let level2 = tasks::track(d.qddot(), &qdd_posture)
            + tasks::regularize(d.tau(), &DVector::zeros(n)).weight(c.torque_reg);

        let levels: [Task; 3] = [level0, level1, level2];
        let t_solve = std::time::Instant::now();
        let sol = self
            .solver
            .solve(&levels, &c.solve)
            .map_err(|e| OscError::Setup(format!("{e:?}")))?;
        let mut degraded = None;
        if let SolveStatus::Degraded { level, status } = &sol.status {
            degraded = Some(format!("level {level}: {status:?}"));
            // Accept degradation of level 2 (posture) alone. Only the use of
            // redundancy is unmet; safety and the objective are solved.
            if *level < 2 {
                return Err(OscError::Degraded(format!("level {level}: {status:?}")));
            }
        }
        let solve_us = t_solve.elapsed().as_secs_f64() * 1e6;
        let ex = d.extract(&sol.x);
        let tau = ex.tau;
        for i in 0..n {
            if !tau[i].is_finite() || tau[i].abs() > tau_max[i] * c.solution_check {
                return Err(OscError::TorqueOutOfRange {
                    axis: i,
                    tau: tau[i],
                    limit: tau_max[i],
                });
            }
        }

        let achieved = &j * &ex.qddot + &djv;
        let mut residual = Vector6::zeros();
        for (k, r) in rows.clone().enumerate() {
            residual[r] = a_task[k] - achieved[k];
        }

        let mut cmd = base;
        assert_eq!(cmd.len(), arm.n(), "base length differs from the number of independent DOFs");
        for (k, &i) in idx.iter().enumerate() {
            cmd.axes[i] = AxisCmd {
                q: s.q[k],
                v: s.v[k] + ex.qddot[k] * dt,
                kp: 0.0,
                kd: motor_kd[k],
                tau: tau[k],
            };
        }
        Ok((
            cmd,
            OscReport {
                qddot: ex.qddot,
                tau,
                tcp_error: err,
                tcp_accel_residual: residual,
                sigma_min,
                solve_us,
                degraded,
            },
        ))
    }
}

/// State restricted to the TCP chain.
struct SubState {
    q: DVector<f64>,
    v: DVector<f64>,
    mass: DMatrix<f64>,
    nle: DVector<f64>,
    tcp_pose: Isometry3<f64>,
    tcp_twist: Vector6<f64>,
    tcp_jacobian: DMatrix<f64>,
    tcp_jdot_v: Vector6<f64>,
}

/// Pose error `[rotation; translation]` (world coordinates). Rotation is `log(R_ref · Rᵀ)`.
pub fn pose_error(target: &Isometry3<f64>, current: &Isometry3<f64>) -> Vector6<f64> {
    let dp: Vector3<f64> = target.translation.vector - current.translation.vector;
    let dr = (target.rotation * current.rotation.inverse()).scaled_axis();
    stack6(&dr, &dp)
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}
