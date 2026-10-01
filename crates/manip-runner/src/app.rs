//! The control loop.
//!
//! ```text
//! loop {
//!     s   = arm.evaluate(obs)                  // M, h, g, J, J̇v (once per cycle)
//!     tgt = source.target(t)                   // leader / synthetic / none
//!     cmd = supervisor.tick(s, tgt)            // per-mode control law
//!     gate.apply(cmd)                          // final clamping (recorded)
//!     plant.exchange(cmd, obs)                 // hardware / MuJoCo
//! }
//! ```
//!
//! Observations are one cycle old (misa-core Plant contract): 2 ms at 500 Hz.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use manip_leader::synthetic::SineLeader;
use manip_leader::LeaderThread;
use manip_model::{ArmModel, ArmState};
use misa_core::{AxisId, Command, Observation, Plant};
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

use crate::assemble::TeleopMapping;
use crate::config::RobotProfile;
use crate::policy::Policy;
use crate::record::Recorder;
use crate::replay::{self, LogWriter};
use crate::supervisor::{Mode, Target};

/// Where the target comes from.
pub enum Source {
    None,
    /// Synthetic (joint names are this robot's independent DOFs).
    Sine { leader: SineLeader, dofs: Vec<usize> },
    /// Draw a circle with the TCP (for checking OSC). y-z plane; the TCP at start lies
    /// on the circle.
    Circle { radius: f64, freq_hz: f64, start: Option<Isometry3<f64>> },
    /// Friction identification sweep (`manip hw friction`): per joint in turn,
    /// the others holding where they started, a triangle wave between `lo` and
    /// `hi` at each speed in turn.
    Sweep { plan: SweepPlan, start: Option<(f64, DVector<f64>)> },
    /// Move one joint by `delta` and back (raised cosine over `period_s`),
    /// then hold where it started. For bring-up (`manip hw jog`).
    Jog { dof: usize, delta: f64, period_s: f64, start: Option<(f64, DVector<f64>)> },
    Leader {
        thread: LeaderThread,
        mapping: TeleopMapping,
        timeout: Duration,
        last_seq: u64,
    },
}

impl Source {
    fn target(&mut self, arm: &ArmModel, t: f64, keep: &DVector<f64>, s: &ArmState) -> Target {
        match self {
            Source::None => Target::None,
            Source::Sine { leader, dofs } => {
                let mut q = keep.clone();
                for (v, &i) in leader.at(t).iter().zip(dofs.iter()) {
                    q[i] = *v;
                }
                Target::Joint(q)
            }
            Source::Circle { radius, freq_hz, start } => {
                let p0 = *start.get_or_insert(s.tcp_pose);
                let w = std::f64::consts::TAU * *freq_hz * t;
                let d = Vector3::new(0.0, *radius * w.sin(), *radius * (w.cos() - 1.0));
                let pose = Isometry3::from_parts(Translation3::from(p0.translation.vector + d), p0.rotation);
                let _ = arm;
                Target::Tcp { pose, posture: keep.clone() }
            }
            Source::Sweep { plan, start } => {
                let (t0, q0) = start.get_or_insert_with(|| (t, keep.clone())).clone();
                let mut q = q0.clone();
                if let Some((dof, x)) = plan.at(t - t0, &q0) {
                    q[dof] = x;
                }
                Target::Joint(q)
            }
            Source::Jog { dof, delta, period_s, start } => {
                let (t0, q0) = start.get_or_insert_with(|| (t, keep.clone())).clone();
                let tau = ((t - t0) / *period_s).clamp(0.0, 1.0);
                let mut q = q0;
                q[*dof] += *delta * 0.5 * (1.0 - (std::f64::consts::TAU * tau).cos());
                let _ = arm;
                Target::Joint(q)
            }
            Source::Leader { thread, mapping, timeout, last_seq } => match thread.latest() {
                Some(sample) if sample.at.elapsed() <= *timeout => {
                    *last_seq = sample.seq;
                    Target::Joint(mapping.map(&sample.q, keep))
                }
                _ => Target::None,
            },
        }
    }
}

/// Schedule of a friction sweep. Per joint: settle at `lo`, then for each
/// speed `cycles` round trips `lo -> hi -> lo`, then return to the start pose.
#[derive(Debug, Clone)]
pub struct SweepPlan {
    pub joints: Vec<(usize, f64, f64)>,
    pub speeds: Vec<f64>,
    pub cycles: usize,
    /// Pause at each end [s] (settles the shaper before the next leg).
    pub settle_s: f64,
}

impl SweepPlan {
    fn joint_duration(&self, lo: f64, hi: f64, q0: f64) -> f64 {
        let span = hi - lo;
        let v0 = self.speeds.first().copied().unwrap_or(0.2);
        let approach = (q0 - lo).abs() / v0 + self.settle_s;
        let sweeps: f64 = self.speeds.iter().map(|v| self.cycles as f64 * 2.0 * (span / v + self.settle_s)).sum();
        approach + sweeps + (q0 - lo).abs() / v0 + self.settle_s
    }

    /// Total duration from the start pose `q0`.
    pub fn duration(&self, q0: &DVector<f64>) -> f64 {
        self.joints.iter().map(|&(d, lo, hi)| self.joint_duration(lo, hi, q0[d])).sum()
    }

    /// (joint, target) at time `t` since the sweep started; `None` once done.
    pub fn at(&self, mut t: f64, q0: &DVector<f64>) -> Option<(usize, f64)> {
        for &(d, lo, hi) in &self.joints {
            let total = self.joint_duration(lo, hi, q0[d]);
            if t >= total {
                t -= total;
                continue;
            }
            let v0 = self.speeds.first().copied().unwrap_or(0.2);
            // Approach lo.
            let a = (q0[d] - lo).abs() / v0;
            if t < a {
                return Some((d, q0[d] + (lo - q0[d]) * t / a.max(1e-9)));
            }
            t -= a;
            if t < self.settle_s {
                return Some((d, lo));
            }
            t -= self.settle_s;
            for &v in &self.speeds {
                let leg = (hi - lo) / v;
                for _ in 0..self.cycles {
                    for (from, to) in [(lo, hi), (hi, lo)] {
                        if t < leg {
                            return Some((d, from + (to - from) * t / leg));
                        }
                        t -= leg;
                        if t < self.settle_s {
                            return Some((d, to));
                        }
                        t -= self.settle_s;
                    }
                }
            }
            // Return to the start pose.
            let r = (q0[d] - lo).abs() / v0;
            return Some((d, if t < r { lo + (q0[d] - lo) * t / r.max(1e-9) } else { q0[d] }));
        }
        None
    }
}

pub struct RunOptions {
    pub mode: Mode,
    /// Before entering the requested mode, move to this pose in joint space.
    ///
    /// Starting OSC at the edge of the range of motion (B601's folded pose has the
    /// shoulder and elbow exactly at their upper limits) can make the QP numerically
    /// unsolvable, so move away first.
    pub start_pose: Option<DVector<f64>>,
    pub duration_s: Option<f64>,
    /// Run without waiting for real time (only with sim + synthetic target).
    pub fast: bool,
    pub record: Option<std::path::PathBuf>,
    /// Binary run log for bit-exact replay (`manip replay`), and the profile
    /// path it records.
    pub log: Option<(std::path::PathBuf, std::path::PathBuf)>,
    /// Status display interval [s].
    pub status_every_s: f64,
    /// Publish the measured pose for a viewer: every cycle the values of
    /// [`viewer_joints`] (full model joints, mimic followers included) are
    /// written here; the viewer takes the latest.
    pub monitor: Option<std::sync::Arc<std::sync::Mutex<Option<Vec<f64>>>>>,
}

/// Model joints a viewer poses, with their index in the full `q`: every
/// 1-DOF joint, mimic followers included (so the gripper's second finger moves).
pub fn viewer_joints(arm: &ArmModel) -> Vec<(String, usize)> {
    let m = arm.raw();
    m.joints
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, j)| j.joint_type.nq() == 1)
        .map(|(i, j)| (j.name.clone(), m.q_idx[i]))
        .collect()
}

pub fn run(
    profile: &RobotProfile,
    arm: &ArmModel,
    plant: &mut dyn Plant,
    mut source: Source,
    opts: RunOptions,
) -> Result<(), String> {
    let n = arm.n();
    let dt = 1.0 / profile.control.rate_hz;
    if plant.axes().len() != n {
        return Err(format!("Plant axis count {} ≠ model independent DOFs {n}", plant.axes().len()));
    }
    for (i, d) in arm.dofs().iter().enumerate() {
        let name = plant.axes().name(AxisId::new(i as u16)).unwrap_or("");
        if name != d.name {
            return Err(format!("Plant axis {i} is {name}, model has {}", d.name));
        }
    }

    // ── Read the pose before energizing ──────────────────────────────────
    let mut obs = Observation::empty(n, 0);
    let idle = Command::idle(n);
    let t_read = Instant::now();
    loop {
        plant.exchange(&idle, &mut obs)?;
        if !obs.any_unread() {
            break;
        }
        if t_read.elapsed() > Duration::from_secs(2) {
            let missing: Vec<_> = (0..n)
                .filter(|&i| !obs.axes()[i].health.valid)
                .map(|i| arm.dofs()[i].name.clone())
                .collect();
            return Err(format!("axes not responding: {}", missing.join(", ")));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let (q, _) = read_state(&obs);
    // Don't energize if starting far outside the range of motion. The most likely
    // cause is a **mixed-up zero point or sign** (profile sign / zero, motor zeroing),
    // and energizing anyway makes the shaper drive at full force "into range".
    // Small excursions (e.g. resting pressed against a hard stop) only warn.
    for (i, d) in arm.dofs().iter().enumerate() {
        if d.within(q[i]) {
            continue;
        }
        let span = d.q_max - d.q_min;
        let margin = if span.is_finite() { 0.1 * span } else { f64::INFINITY };
        let off = (q[i] - d.clamp(q[i])).abs();
        if off > margin {
            return Err(format!(
                "{} is {:.3} outside its range ({:.3}, range {:.3} .. {:.3}). \
                 Check the zero point and sign ([hardware] zero / sign)",
                d.name, off, q[i], d.q_min, d.q_max
            ));
        }
        log::warn!(
            "{} starts slightly outside its range: {:.3} ({:.3} .. {:.3}). Targets will be clamped into range",
            d.name, q[i], d.q_min, d.q_max
        );
    }
    log::info!("initial pose {:?}", q.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());

    let mut policy = Policy::new(profile, arm, &q)?;
    let mut rec = match &opts.record {
        Some(p) => Some(Recorder::create(p, arm).map_err(|e| e.to_string())?),
        None => None,
    };
    let mut log = match &opts.log {
        Some((path, profile_path)) => Some(LogWriter::create(path, &replay::header(profile, profile_path, arm, &q)?)?),
        None => None,
    };

    // Ctrl-C: first press -> Park (fold, then release); second -> release immediately.
    let interrupts = Arc::new(AtomicU32::new(0));
    {
        let c = interrupts.clone();
        let _ = ctrlc::set_handler(move || {
            let k = c.fetch_add(1, Ordering::SeqCst) + 1;
            if k == 1 {
                eprintln!("\n[Ctrl-C] folding to rest pose, then releasing (press again to release immediately)");
            } else {
                eprintln!("\n[Ctrl-C] releasing immediately");
            }
        });
    }

    let view = viewer_joints(arm);
    plant.arm()?;
    let t0 = Instant::now();
    let mut t = 0.0;
    let mut next = Instant::now();
    let mut last_status = -1.0;
    let mut requested = false;
    let mut approaching = false;
    let mut overruns = 0u64;
    let mut max_tick_us: f64 = 0.0;
    let mut last_guard_log = f64::NEG_INFINITY;
    let mut last_gate_log = -1.0;
    let result: Result<(), String> = loop {
        let tick_start = Instant::now();
        let s = Policy::state(arm, &obs);

        // Mode requests. Decided here (they depend on time, Ctrl-C and the
        // startup sequence) and handed to the policy as recorded inputs.
        // `pending` is the mode the policy will be in after the requests so far.
        let k = interrupts.load(Ordering::SeqCst);
        if k >= 2 {
            break Ok(());
        }
        if policy.mode() == Mode::Done {
            log::info!("finished folding");
            break Ok(());
        }
        let mut requests = Vec::new();
        let mut pending = policy.mode();
        let push = |m: Mode, requests: &mut Vec<Mode>, pending: &mut Mode| {
            if m != *pending {
                requests.push(m);
                *pending = m;
            }
        };
        if k == 1 && !matches!(pending, Mode::Park | Mode::Done) {
            push(Mode::Park, &mut requests, &mut pending);
        }
        let parking = matches!(pending, Mode::Park | Mode::Done);
        if opts.duration_s.is_some_and(|d| t >= d) && !parking {
            log::info!("{:.1} s elapsed, folding to rest pose", opts.duration_s.unwrap());
            push(Mode::Park, &mut requests, &mut pending);
        }
        if !requested && t >= profile.control.startup_ramp_s && k == 0 && !parking {
            match &opts.start_pose {
                Some(_) if !approaching => {
                    push(Mode::Joint, &mut requests, &mut pending);
                    approaching = true;
                    log::info!("moving to start pose");
                }
                Some(goal) if (&s.q - goal).amax() < profile.control.park_tolerance && s.v.amax() < 0.05 => {
                    push(opts.mode, &mut requests, &mut pending);
                    requested = true;
                }
                Some(_) => {}
                None => {
                    push(opts.mode, &mut requests, &mut pending);
                    requested = true;
                }
            }
        }

        let keep = s.q.clone();
        let target = match (&opts.start_pose, requested) {
            (Some(goal), false) => Target::Joint(goal.clone()),
            _ => source.target(arm, t, &keep, &s),
        };
        let out = policy.step(arm, &obs, &requests, &target);
        if let Some(tr) = &out.info.transition {
            log::info!("{tr}");
        }
        if let Some(why) = out
            .info
            .guard_reason
            .as_ref()
            .filter(|_| t - last_guard_log > 1.0)
        {
            last_guard_log = t;
            log::warn!("guard: reference held at the safety boundary: {why}");
        }
        if !out.verdict.is_clean() && t - last_gate_log > 1.0 {
            log::warn!("SafetyGate: {:?}", out.verdict);
            last_gate_log = t;
        }
        let tick_us = tick_start.elapsed().as_secs_f64() * 1e6;
        max_tick_us = max_tick_us.max(tick_us);
        if let Some(r) = rec.as_mut() {
            r.row(t, policy.mode(), tick_us, &out.state, &out.joint_command, &out.info, &obs)
                .map_err(|e| e.to_string())?;
        }
        if let Some(l) = log.as_mut() {
            l.frame(t, &obs, &requests, &target, policy.command(), &out.verdict)?;
        }
        let s = out.state;
        let cmd = policy.command().clone();

        if let Err(e) = plant.exchange(&cmd, &mut obs) {
            break Err(e);
        }
        if let Some(m) = &opts.monitor {
            let q: Vec<f64> = obs.axes().iter().map(|a| a.position_rad).collect();
            let full = arm.full_q(&q);
            *m.lock().unwrap() = Some(view.iter().map(|&(_, i)| full[i]).collect());
        }
        if obs.axes().iter().any(|a| !a.position_rad.is_finite()) {
            break Err("NaN in observation".into());
        }

        if t - last_status >= opts.status_every_s {
            let p = s.tcp_pose.translation.vector;
            let src = match &source {
                Source::Leader { thread, .. } => {
                    let (hz, errs) = thread.stats();
                    format!(" leader {hz:.0}Hz err={errs} {}", thread.status_line())
                }
                _ => String::new(),
            };
            eprintln!(
                "t={t:6.2} {:?} tcp=[{:+.3} {:+.3} {:+.3}] tick max {:.0}µs overrun {overruns} {}{}",
                policy.mode(),
                p.x,
                p.y,
                p.z,
                max_tick_us,
                plant.status_line(),
                src
            );
            last_status = t;
            max_tick_us = 0.0;
        }

        if opts.fast {
            t += dt;
        } else {
            next += Duration::from_secs_f64(dt);
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            } else {
                overruns += 1;
                next = now;
            }
            t = t0.elapsed().as_secs_f64();
        }
    };

    // Release. **The arm will drop**, so we only get here after folding, when the
    // operator pressed Ctrl-C a second time, or when the Plant failed.
    let _ = plant.exchange(&Command::idle(n), &mut obs);
    plant.disarm()?;
    if let Some(l) = log {
        l.finish()?;
    }
    if policy.osc_failures() > 0 {
        log::warn!("times OSC failed to solve and fell back to hold: {}", policy.osc_failures());
    }
    result
}

fn read_state(obs: &Observation) -> (DVector<f64>, DVector<f64>) {
    let q = DVector::from_iterator(obs.len(), obs.axes().iter().map(|a| a.position_rad));
    let v = DVector::from_iterator(obs.len(), obs.axes().iter().map(|a| a.velocity_rad_s));
    (q, v)
}
