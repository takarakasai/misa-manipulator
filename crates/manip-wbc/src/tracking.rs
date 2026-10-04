//! Joint-trajectory tracking on the hierarchical QP — the WBC half of an
//! MPC + WBC split.
//!
//! A planner (e.g. `manip-mpc`) runs slowly and hands over a joint trajectory
//! `(q, v, a)(t)`; this controller runs every control cycle and turns the
//! sample at "now" into torques while keeping the hard constraints the planner
//! only approximates (exact torque limits with the current `M`, `h`; joint
//! limits as a CBF; workspace / self-collision barriers):
//!
//! | level | contents |
//! |---|---|
//! | 0 | torque limits, joint-limit CBF, safety barriers |
//! | 1 | `q̈ = a + Kp·(q* − q) + Kd·(v* − v)` |
//! | 2 | torque regularization |

use manip_control::{FrictionModel, JointCommand, JointRef};
use manip_model::{ArmModel, ArmState};
use misa_wbc::dynamics::Formulation;
use misa_wbc::solve::{SolveConfig, Solver};
use misa_wbc::Task;
use nalgebra::DVector;

use crate::chain::ChainState;
use crate::osc::OscConfig;
use crate::output::{MotorGains, MotorOutput};
use crate::solve::{solve_levels, WbcError};
use crate::tasks::{self, Cbf, JointLimitParams};

#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Joint feedback on the reference, in acceleration units [1/s², 1/s]
    /// (all independent DOFs; only the chain is used).
    pub kp: DVector<f64>,
    pub kd: DVector<f64>,
    pub torque_reg: f64,
    pub torque_scale: f64,
    pub friction: Option<FrictionModel>,
    pub a_max: DVector<f64>,
    pub cbf_alpha: f64,
    pub cbf_alpha_v: f64,
    pub v_scale: f64,
    pub motor_kp: DVector<f64>,
    pub motor_kd: DVector<f64>,
    pub motor_lead_max: f64,
    pub solution_check: f64,
    pub formulation: Formulation,
    pub solve: SolveConfig,
}

impl TrackingConfig {
    /// Safety, motor and solver settings shared with an OSC configuration;
    /// tracking gains critically damped at `omega` [rad/s].
    pub fn from_osc(osc: &OscConfig, omega: f64) -> Self {
        let n = osc.a_max.len();
        Self {
            kp: DVector::from_element(n, omega * omega),
            kd: DVector::from_element(n, 2.0 * omega),
            torque_reg: osc.torque_reg,
            torque_scale: osc.torque_scale,
            friction: osc.friction.clone(),
            a_max: osc.a_max.clone(),
            cbf_alpha: osc.cbf_alpha,
            cbf_alpha_v: osc.cbf_alpha_v,
            v_scale: osc.v_scale,
            motor_kp: osc.motor_kp.clone(),
            motor_kd: osc.motor_kd.clone(),
            motor_lead_max: osc.motor_lead_max,
            solution_check: osc.solution_check,
            formulation: osc.formulation,
            solve: osc.solve.clone(),
        }
    }
}

/// Per-cycle breakdown.
#[derive(Debug, Clone, Default)]
pub struct TrackingReport {
    /// Requested `q̈` (level 1) and the solution.
    pub qddot_ref: DVector<f64>,
    pub qddot: DVector<f64>,
    pub tau: DVector<f64>,
    pub solve_us: f64,
    pub degraded: Option<String>,
}

/// Joint-trajectory tracking controller. Holds the QP warm-start state.
pub struct JointTracking {
    pub cfg: TrackingConfig,
    solver: Solver,
    output: MotorOutput,
}

impl JointTracking {
    pub fn new(cfg: TrackingConfig) -> Self {
        Self {
            cfg,
            solver: Solver::new(),
            output: MotorOutput::default(),
        }
    }

    pub fn reset(&mut self) {
        self.solver.reset();
        self.output.reset();
    }

    /// Track `r` (all independent DOFs; only the TCP chain is solved, the rest
    /// of `base` is kept).
    pub fn command(
        &mut self,
        arm: &ArmModel,
        s_full: &ArmState,
        r: &JointRef,
        dt: f64,
        base: JointCommand,
        cbfs: &[Cbf],
    ) -> Result<(JointCommand, TrackingReport), WbcError> {
        let c = ChainState::new(arm, s_full);
        let cfg = &self.cfg;
        let tau_max = c.torque_limits(cfg.torque_scale);
        let d = tasks::chain_dynamics(&c, cfg.formulation, cfg.friction.as_ref());
        let jl = JointLimitParams {
            a_max: c.pick(&cfg.a_max),
            alpha: cfg.cbf_alpha,
            alpha_v: cfg.cbf_alpha_v,
            v_scale: cfg.v_scale,
        };
        let level0 = tasks::with_barriers(tasks::physics_and_limits(&c, &d, &tau_max, &jl), &c, &d, cbfs, cfg.cbf_alpha);
        let a_ref = tasks::joint_pd_acceleration(&c, &c.pick_ref(r), &c.pick(&cfg.kp), &c.pick(&cfg.kd));
        let level1 = tasks::joint_acceleration(&d, &a_ref);
        let level2 = tasks::torque_regularization(&d, cfg.torque_reg);
        let levels: [Task; 3] = [level0, level1, level2];
        let sol = solve_levels(&mut self.solver, &d, &levels, &cfg.solve, 2, &tau_max, cfg.solution_check)?;
        let gains = MotorGains {
            kp: c.pick(&cfg.motor_kp),
            kd: c.pick(&cfg.motor_kd),
            lead_max: cfg.motor_lead_max,
        };
        let cmd = self.output.command(&c, &sol.qddot, &sol.tau, dt, &gains, base);
        Ok((
            cmd,
            TrackingReport {
                qddot_ref: a_ref,
                qddot: sol.qddot,
                tau: sol.tau,
                solve_us: sol.solve_us,
                degraded: sol.degraded,
            },
        ))
    }
}
