//! Runs a manip-mpc planner for the live loop.
//!
//! Each cycle in Mpc mode the loop calls [`MpcDriver::poll`] with the state it
//! just observed and the clock of the coming step. At `rate_hz` the driver
//! requests a plan from that state; a finished plan is returned on the cycle
//! it becomes available and handed to `Policy::step` (and the run log) as an
//! input, so replaying a log reproduces the commands bit for bit whatever the
//! planner's timing was.
//!
//! - Asynchronous (real time): a worker thread plans; results arrive some
//!   cycles later (the plan starts at the time of the state it was asked
//!   from, so the tracker samples it at the right point).
//! - Synchronous (`--fast` simulation): planned inside the cycle, so a
//!   simulation that runs faster than real time sees no planning delay.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;

use manip_model::ArmModel;
use manip_mpc::{JointPlan, MpcError, MpcGoal, MpcReport, Planner};
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

/// What the planner aims at, with enough information to evaluate it over
/// the horizon.
#[derive(Debug, Clone)]
pub enum GoalSpec {
    /// A fixed TCP pose (leader, synthetic targets): no preview.
    Fixed { pose: Isometry3<f64>, posture: DVector<f64> },
    /// `--source circle`: known in advance, so the planner previews it.
    Circle {
        p0: Isometry3<f64>,
        radius: f64,
        freq_hz: f64,
        posture: DVector<f64>,
    },
}

impl GoalSpec {
    pub fn pose_at(&self, t: f64) -> Isometry3<f64> {
        match self {
            GoalSpec::Fixed { pose, .. } => *pose,
            GoalSpec::Circle { p0, radius, freq_hz, .. } => circle_pose(p0, *radius, *freq_hz, t),
        }
    }

    fn posture(&self) -> &DVector<f64> {
        match self {
            GoalSpec::Fixed { posture, .. } | GoalSpec::Circle { posture, .. } => posture,
        }
    }
}

/// The circle `--source circle` draws (y–z plane, through the start pose).
pub fn circle_pose(p0: &Isometry3<f64>, radius: f64, freq_hz: f64, t: f64) -> Isometry3<f64> {
    let w = std::f64::consts::TAU * freq_hz * t;
    let d = Vector3::new(0.0, radius * w.sin(), radius * (w.cos() - 1.0));
    Isometry3::from_parts(Translation3::from(p0.translation.vector + d), p0.rotation)
}

struct Job {
    q: Vec<f64>,
    v: Vec<f64>,
    t: f64,
    goal: GoalSpec,
    reset: bool,
}

type PlanResult = Result<(JointPlan, MpcReport), MpcError>;

fn plan_once(planner: &mut dyn Planner, arm: &ArmModel, job: &Job) -> PlanResult {
    if job.reset {
        planner.reset();
    }
    if log::log_enabled!(log::Level::Trace) {
        let p0 = job.goal.pose_at(job.t).translation.vector;
        let p1 = job.goal.pose_at(job.t + 1.0).translation.vector;
        log::trace!(
            "job t={:.3} q={:?} v={:?} goal now [{:.3} {:.3} {:.3}] +1s [{:.3} {:.3} {:.3}] tcp [{:?}] posture {:?}",
            job.t, job.q, job.v, p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, arm.tcp_pose(&job.q).translation.vector.as_slice(), job.goal.posture().as_slice()
        );
    }
    let tcp = |t: f64| job.goal.pose_at(t);
    planner.plan(arm, &job.q, &job.v, job.t, &MpcGoal { tcp: &tcp, posture: Some(job.goal.posture()) })
}

enum Exec {
    Sync { planner: Box<dyn Planner + Send>, arm: Box<ArmModel> },
    Async { tx: Option<Sender<Job>>, rx: Receiver<PlanResult>, busy: bool, handle: Option<JoinHandle<()>> },
}

pub struct MpcDriver {
    exec: Exec,
    period: f64,
    next_t: f64,
    reset_next: bool,
    /// The planner's distance from the joint limits (joint targets are
    /// clamped by it before they become TCP goals).
    pub q_margin: f64,
    pub plans: u64,
    pub failures: u64,
    pub last_report: Option<MpcReport>,
}

impl MpcDriver {
    pub fn new(planner: Box<dyn Planner + Send>, arm: &ArmModel, rate_hz: f64, synchronous: bool) -> Self {
        let q_margin = planner.q_margin();
        let exec = if synchronous {
            Exec::Sync { planner, arm: Box::new(arm.clone()) }
        } else {
            let (tx, job_rx) = channel::<Job>();
            let (res_tx, rx) = channel::<PlanResult>();
            let arm = arm.clone();
            let mut planner = planner;
            let handle = std::thread::Builder::new()
                .name("mpc".into())
                .spawn(move || {
                    while let Ok(job) = job_rx.recv() {
                        if res_tx.send(plan_once(planner.as_mut(), &arm, &job)).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawn the MPC thread");
            Exec::Async { tx: Some(tx), rx, busy: false, handle: Some(handle) }
        };
        Self {
            exec,
            period: 1.0 / rate_hz.max(1e-3),
            next_t: f64::NEG_INFINITY,
            reset_next: true,
            q_margin,
            plans: 0,
            failures: 0,
            last_report: None,
        }
    }

    /// The next request starts the planner afresh (on entering Mpc mode).
    pub fn restart(&mut self) {
        self.reset_next = true;
        self.next_t = f64::NEG_INFINITY;
    }

    /// Request a plan if one is due, and return a plan that became available.
    pub fn poll(&mut self, t: f64, q: &[f64], v: &[f64], goal: impl FnOnce() -> GoalSpec) -> Option<JointPlan> {
        let due = t >= self.next_t;
        let out = match &mut self.exec {
            Exec::Sync { planner, arm } => {
                if !due {
                    return None;
                }
                self.next_t = t + self.period;
                let job = Job { q: q.to_vec(), v: v.to_vec(), t, goal: goal(), reset: std::mem::take(&mut self.reset_next) };
                Some(plan_once(planner.as_mut(), arm, &job))
            }
            Exec::Async { tx, rx, busy, .. } => {
                let got = match rx.try_recv() {
                    Ok(r) => {
                        *busy = false;
                        Some(r)
                    }
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => {
                        *busy = false;
                        Some(Err(MpcError::Setup("MPC thread stopped".into())))
                    }
                };
                if due && !*busy && let Some(tx) = tx.as_ref() {
                    let job = Job { q: q.to_vec(), v: v.to_vec(), t, goal: goal(), reset: std::mem::take(&mut self.reset_next) };
                    if tx.send(job).is_ok() {
                        *busy = true;
                        self.next_t = t + self.period;
                    }
                }
                got
            }
        };
        match out {
            Some(Ok((plan, report))) => {
                self.plans += 1;
                log::debug!("plan t0={:.3}: {report:?}", plan.t0);
                self.last_report = Some(report);
                Some(plan)
            }
            Some(Err(e)) => {
                self.failures += 1;
                if self.failures <= 3 || self.failures.is_power_of_two() {
                    log::warn!("MPC plan failed ({} so far): {e}", self.failures);
                }
                None
            }
            None => None,
        }
    }
}

impl Drop for MpcDriver {
    fn drop(&mut self) {
        if let Exec::Async { tx, handle, .. } = &mut self.exec {
            tx.take();
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }
    }
}
