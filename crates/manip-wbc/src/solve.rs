//! Solve a stack of levels and check the result.

use misa_wbc::dynamics::Dynamics;
use misa_wbc::solve::{SolveConfig, SolveStatus, Solver};
use misa_wbc::Task;
use nalgebra::DVector;

/// Why a cycle has no usable solution. The caller should fall back to holding
/// in place (joint impedance + gravity).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WbcError {
    #[error("QP could not be solved optimally: {0}")]
    Degraded(String),
    #[error("solution torque exceeds the limit (axis {axis}: {tau:.2} / {limit:.2})")]
    TorqueOutOfRange { axis: usize, tau: f64, limit: f64 },
    #[error("could not set up the QP: {0}")]
    Setup(String),
}

/// A solved stack.
#[derive(Debug, Clone)]
pub struct Solved {
    pub qddot: DVector<f64>,
    pub tau: DVector<f64>,
    /// Status of a level at or after `accept_from` that degraded (accepted).
    pub degraded: Option<String>,
    /// Time taken to solve [µs].
    pub solve_us: f64,
}

/// Solve `levels`, accept degradation only of levels `>= accept_from`, and
/// reject solutions whose `|τ|` exceeds `tau_max × check` (a single cycle may
/// yield an inconsistent solution, measured on keel in misa-runner).
///
/// If a level that must hold degrades with the active-set backend, the stack
/// is solved once more with Clarabel: on the real B601-DM, folding back during
/// MPC teleop, misa-wbc's ActiveSet reported level 0 Infeasible for a feasible
/// problem (Clarabel solved it; tolerances, iteration limits and the HQP
/// strategy made no difference). Solutions found the first time are unchanged.
pub fn solve_levels(
    solver: &mut Solver,
    d: &Dynamics,
    levels: &[Task],
    cfg: &SolveConfig,
    accept_from: usize,
    tau_max: &DVector<f64>,
    check: f64,
) -> Result<Solved, WbcError> {
    let t_solve = std::time::Instant::now();
    let mut sol = solver.solve(levels, cfg).map_err(|e| WbcError::Setup(format!("{e:?}")))?;
    if let SolveStatus::Degraded { level, .. } = &sol.status
        && *level < accept_from
        && cfg.backend == misa_wbc::QpSolver::ActiveSet
    {
        let retry = SolveConfig { backend: misa_wbc::QpSolver::Clarabel, ..cfg.clone() };
        let alt = Solver::new().solve(levels, &retry).map_err(|e| WbcError::Setup(format!("{e:?}")))?;
        if !matches!(&alt.status, SolveStatus::Degraded { level, .. } if *level < accept_from) {
            solver.reset();
            sol = alt;
        }
    }
    let mut degraded = None;
    if let SolveStatus::Degraded { level, status } = &sol.status {
        degraded = Some(format!("level {level}: {status:?}"));
        if *level < accept_from {
            return Err(WbcError::Degraded(format!("level {level}: {status:?}")));
        }
    }
    let solve_us = t_solve.elapsed().as_secs_f64() * 1e6;
    let ex = d.extract(&sol.x);
    let tau = ex.tau;
    for i in 0..tau.len() {
        if !tau[i].is_finite() || tau[i].abs() > tau_max[i] * check {
            return Err(WbcError::TorqueOutOfRange {
                axis: i,
                tau: tau[i],
                limit: tau_max[i],
            });
        }
    }
    Ok(Solved {
        qddot: ex.qddot,
        tau,
        degraded,
        solve_us,
    })
}
