//! Operational-space control (OSC): a TCP target on the hierarchical QP.
//!
//! The decision variable is `x = [q̈; τ]` (fixed base, no contacts). Priorities:
//!
//! | level | contents | role |
//! |---|---|---|
//! | 0 | equations of motion, torque limits, joint-limit CBF, safety barriers | must hold (hard) |
//! | 1 | TCP acceleration (pose PD + reference acceleration) | objective |
//! | 2 | posture (joint PD), torque regularization | use of redundancy |
//!
//! Joint limits are not enforced by slamming torque, but placed at the top
//! level as **bounds on q̈ (CBF)**. Even if the target is outside the range of
//! motion, the TCP task merely "can't be fully achieved" and the joint
//! decelerates and stops short of the limit.
//!
//! A cycle where the QP can't be solved optimally (levels 0–1) returns
//! [`WbcError`]; the caller should fall back to holding in place.

use manip_control::{FrictionModel, JointCommand, JointRef};
use manip_model::{ArmModel, ArmState};
use misa_wbc::dynamics::Formulation;
use misa_wbc::solve::{SolveConfig, Solver};
use misa_wbc::Task;
use nalgebra::{DVector, Vector6};

use crate::chain::ChainState;
use crate::output::{MotorGains, MotorOutput};
use crate::solve::{solve_levels, WbcError};
use crate::tasks::{self, Cbf, JointLimitParams, SingularityParams, TcpGains, TcpRef};

/// Kept for callers written against the OSC-only API.
pub type OscError = WbcError;

#[derive(Debug, Clone)]
pub struct OscConfig {
    /// Friction feedforward, folded into the nonlinear term `h` so the QP plans
    /// the torque that also cancels friction. Evaluated at the measured velocity
    /// (the OSC has no joint reference), so give it a wider `v_eps` than the
    /// joint law. `None` = off.
    pub friction: Option<FrictionModel>,
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
    /// Motor-side stiffness around the integrated QP solution [N·m/rad].
    pub motor_kp: DVector<f64>,
    /// Bound on |integrated reference − measured| [rad].
    pub motor_lead_max: f64,
    /// Singularity damping (see [`SingularityParams`]).
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
            motor_kp: DVector::zeros(n),
            motor_lead_max: 0.03,
            sing_sigma_lo: 0.01,
            sing_sigma_hi: 0.06,
            sing_lambda_sq: 1e-2,
            sing_kd: 10.0,
            solution_check: 1.05,
            formulation: Formulation::AccelSpace,
            friction: None,
            // Against a per-cycle budget of 2 ms (500 Hz), Clarabel takes a
            // median of 0.46 ms and ActiveSet 0.05–0.09 ms (B601-DM, 6 DOF,
            // osc_bench). Tracking results were the same.
            solve: SolveConfig {
                backend: misa_wbc::QpSolver::ActiveSet,
                ..SolveConfig::default()
            },
        }
    }

    /// The joint-limit part (chain order).
    pub fn joint_limits(&self, c: &ChainState) -> JointLimitParams {
        JointLimitParams {
            a_max: c.pick(&self.a_max),
            alpha: self.cbf_alpha,
            alpha_v: self.cbf_alpha_v,
            v_scale: self.v_scale,
        }
    }

    /// The motor-side PD (chain order).
    pub fn motor_gains(&self, c: &ChainState) -> MotorGains {
        MotorGains {
            kp: c.pick(&self.motor_kp),
            kd: c.pick(&self.motor_kd),
            lead_max: self.motor_lead_max,
        }
    }

    pub fn tcp_gains(&self) -> TcpGains {
        TcpGains {
            kp_lin: self.kp_lin,
            kd_lin: self.kd_lin,
            kp_ang: self.kp_ang,
            kd_ang: self.kd_ang,
        }
    }

    pub fn singularity(&self) -> SingularityParams {
        SingularityParams {
            sigma_lo: self.sing_sigma_lo,
            sigma_hi: self.sing_sigma_hi,
            lambda_sq: self.sing_lambda_sq,
            kd: self.sing_kd,
        }
    }
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
    output: MotorOutput,
}

impl Osc {
    pub fn new(cfg: OscConfig) -> Self {
        Self {
            cfg,
            solver: Solver::new(),
            output: MotorOutput::default(),
        }
    }

    /// Discards the warm start and the integrated reference (on mode switches).
    pub fn reset(&mut self) {
        self.solver.reset();
        self.output.reset();
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
    ) -> Result<(JointCommand, OscReport), WbcError> {
        self.command_with(arm, s_full, tcp, posture_full, dt, base, &[])
    }

    /// [`Self::command`] with extra safety constraints in level 0 (workspace
    /// box, self-collision distances; see [`Cbf`]).
    #[allow(clippy::too_many_arguments)]
    pub fn command_with(
        &mut self,
        arm: &ArmModel,
        s_full: &ArmState,
        tcp: &TcpRef,
        posture_full: &JointRef,
        dt: f64,
        base: JointCommand,
        cbfs: &[Cbf],
    ) -> Result<(JointCommand, OscReport), WbcError> {
        let c = ChainState::new(arm, s_full);
        let cfg = &self.cfg;
        let posture = c.pick_ref(posture_full);
        let tau_max = c.torque_limits(cfg.torque_scale);

        let d = tasks::chain_dynamics(&c, cfg.formulation, cfg.friction.as_ref());
        let level0 = tasks::physics_and_limits(&c, &d, &tau_max, &cfg.joint_limits(&c));
        let level0 = tasks::with_barriers(level0, &c, &d, cbfs, cfg.cbf_alpha);
        let tcp_task = tasks::tcp_acceleration(&c, &d, tcp, &cfg.tcp_gains(), cfg.track_orientation, &cfg.singularity());
        let qdd_posture = tasks::joint_pd_acceleration(&c, &posture, &c.pick(&cfg.posture_kp), &c.pick(&cfg.posture_kd));
        let level2 = tasks::joint_acceleration(&d, &qdd_posture) + tasks::torque_regularization(&d, cfg.torque_reg);

        let levels: [Task; 3] = [level0, tcp_task.task.clone(), level2];
        // Accept degradation of level 2 (posture) alone: only the use of
        // redundancy is unmet; safety and the objective are solved.
        let sol = solve_levels(&mut self.solver, &d, &levels, &cfg.solve, 2, &tau_max, cfg.solution_check)?;
        let residual = tcp_task.residual(&sol.qddot);
        let cmd = self.output.command(&c, &sol.qddot, &sol.tau, dt, &cfg.motor_gains(&c), base);
        Ok((
            cmd,
            OscReport {
                qddot: sol.qddot,
                tau: sol.tau,
                tcp_error: tcp_task.err,
                tcp_accel_residual: residual,
                sigma_min: tcp_task.sigma_min,
                solve_us: sol.solve_us,
                degraded: sol.degraded,
            },
        ))
    }
}
