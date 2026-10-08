//! Mode state machine. The only place that builds the command for one cycle.
//!
//! ```text
//!            start
//!              │
//!              ▼            ┌──────────── OSC unsolvable ─────────┐
//!   ┌─────── Hold ◄─────────┤                                      │
//!   │          │ ▲          │                                      │
//!   │  request │ │ request  │                                      │
//!   │          ▼ │          │                                      │
//!   │   Gravity / Joint / Osc ─────────────────────────────────────┘
//!   │          │
//!   │  Ctrl-C  ▼
//!   └─────► Park ──(reached rest)──► Done (runner releases)
//! ```
//!
//! # Never release
//!
//! A failure in any mode falls back to **hold**. The arm drops when released.
//! Release happens only when Park has finished folding (Done) and when the operator
//! presses Ctrl-C a second time.
//!
//! # When the leader drops out
//!
//! In cycles with no target, Joint / Osc **use the current reference as the target**.
//! The shaper decelerates within its acceleration limit and stops (no jump; it stays
//! in place). When the leader returns, the reference moves from where it stopped
//! toward the leader within the speed limit.

use manip_control::{JointCommand, JointGains, JointImpedance, JointRef, JointShaper, ShaperLimits, TcpShaper};
use manip_mpc::JointPlan;
use manip_wbc::{JointTracking, Osc, OscReport, TcpExtras, TcpRef, TrackingReport};
use manip_model::{ArmModel, ArmState};
use nalgebra::{DVector, Isometry3};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum Mode {
    /// Hold in place (hold gains + gravity FF).
    Hold,
    /// Gravity compensation only (can be moved by hand).
    Gravity,
    /// Joint-space tracking (leader joint angles -> joint impedance).
    Joint,
    /// TCP tracking (operational-space control, hierarchical QP).
    Osc,
    /// TCP tracking through a receding-horizon plan (manip-mpc) followed by
    /// the joint-tracking WBC.
    Mpc,
    /// Fold to the rest pose.
    Park,
    /// Finished folding. The runner releases.
    Done,
}

/// Target for this cycle.
#[derive(Debug, Clone)]
pub enum Target {
    /// No target (leader lost or not connected).
    None,
    /// Joint angles (ordered as independent DOFs).
    Joint(DVector<f64>),
    /// TCP pose, plus a posture (joint angles) for the redundancy.
    Tcp {
        pose: Isometry3<f64>,
        posture: DVector<f64>,
    },
    /// A joint trajectory generated outside (motion commands): followed as
    /// is ([`JointShaper::follow`]) instead of shaped toward a point.
    JointRef { r: JointRef, extras: JointExtras },
    /// A TCP trajectory generated outside, with an optional wrench /
    /// impedance (Osc mode).
    TcpRef {
        tcp: TcpRef,
        posture: DVector<f64>,
        extras: Box<TcpExtras>,
    },
}

impl Target {
    /// The joint angles this target asks for, if any (posture of a TCP target).
    fn joints(&self) -> Option<&DVector<f64>> {
        match self {
            Target::None => None,
            Target::Joint(q) => Some(q),
            Target::JointRef { r, .. } => Some(&r.q),
            Target::Tcp { posture, .. } | Target::TcpRef { posture, .. } => Some(posture),
        }
    }
}

/// Joint-space additions to a [`Target::JointRef`] (Joint mode).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointExtras {
    /// Added to the feedforward torque [N·m or N, model units]: joint-space
    /// force control (open loop, like the TCP wrench).
    pub torque: Option<DVector<f64>>,
    /// Replace the tracking gains (motor-side PD) per joint.
    pub kp: Option<DVector<f64>>,
    pub kd: Option<DVector<f64>>,
}

/// Breakdown of one cycle (for recording).
#[derive(Debug, Clone, Default)]
pub struct TickInfo {
    pub reference: Option<JointRef>,
    pub tcp_reference: Option<Isometry3<f64>>,
    pub osc: Option<OscReport>,
    pub mpc: Option<TrackingReport>,
    /// Whether the mode changed this cycle (and if so, why).
    pub transition: Option<String>,
    /// Joint mode: the reference step was refused by the workspace /
    /// self-collision guard and the reference held.
    pub guarded: bool,
    /// Set on the first guarded tick of an episode: what the refused step's
    /// stopping point would have violated (for the operator log).
    pub guard_reason: Option<String>,
    /// The TCP the target asks for (a joint target by its forward
    /// kinematics), in the modes that follow a target. For the record.
    pub tcp_goal: Option<Isometry3<f64>>,
}

pub struct SupervisorConfig {
    pub track: JointGains,
    pub hold: JointGains,
    pub gravity_kd: DVector<f64>,
    pub gravity_scale: DVector<f64>,
    pub feedforward: manip_control::Feedforward,
    pub shaper: ShaperLimits,
    pub park_speed_scale: f64,
    pub park_tolerance: f64,
    pub startup_ramp_s: f64,
    pub tcp_shaper: manip_control::TcpShaperLimits,
    pub rest: DVector<f64>,
    /// Friction feedforward for the tracking law (reference velocity). `None` = off.
    pub friction: Option<manip_control::FrictionModel>,
    /// Workspace box and self-collision (`[safety]`). `None` = unchecked.
    pub safety: Option<crate::guard::SafetyModel>,
    /// Mpc mode: hold if the newest plan is older than this [s].
    pub plan_timeout_s: f64,
}

pub struct Supervisor {
    cfg: SupervisorConfig,
    mode: Mode,
    track: JointImpedance,
    hold: JointImpedance,
    gravity: JointImpedance,
    osc: Osc,
    tracking: JointTracking,
    /// Newest plan from the planner (Mpc mode).
    plan: Option<JointPlan>,
    shaper: JointShaper,
    tcp_shaper: TcpShaper,
    /// Hold reference: the pose when entering the mode.
    hold_q: DVector<f64>,
    /// Time since startup [s] (for the gain ramp-up).
    t: f64,
    pub osc_failures: u64,
    /// The previous Joint-mode tick was guarded (episodes are logged once).
    guarding: bool,
    /// How long the Park reference has been at the rest pose [s].
    park_settle_s: f64,
}

impl Supervisor {
    /// `q` is the **measured pose**. Starts by holding there.
    pub fn new(cfg: SupervisorConfig, osc: Osc, tracking: JointTracking, arm: &ArmModel, q: &DVector<f64>) -> Self {
        let mut track = JointImpedance::new(cfg.track.clone(), cfg.feedforward);
        track.gravity_scale = cfg.gravity_scale.clone();
        track.friction = cfg.friction.clone();
        let mut hold = JointImpedance::new(cfg.hold.clone(), manip_control::Feedforward::Gravity);
        hold.gravity_scale = cfg.gravity_scale.clone();
        let mut gravity = JointImpedance::gravity_comp(arm.n(), 0.0);
        gravity.gains = JointGains::new(DVector::zeros(arm.n()), cfg.gravity_kd.clone());
        gravity.gravity_scale = cfg.gravity_scale.clone();
        let shaper = JointShaper::new(cfg.shaper.clone(), q.clone());
        let tcp_shaper = TcpShaper::new(cfg.tcp_shaper, arm.tcp_pose(q.as_slice()));
        Self {
            cfg,
            mode: Mode::Hold,
            track,
            hold,
            gravity,
            osc,
            tracking,
            plan: None,
            shaper,
            tcp_shaper,
            hold_q: q.clone(),
            t: 0.0,
            osc_failures: 0,
            guarding: false,
            park_settle_s: 0.0,
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Clock of the next tick [s] (a plan requested with the state observed
    /// before that tick starts at this time).
    pub fn time(&self) -> f64 {
        self.t
    }

    /// A new plan from the planner (Mpc mode; ignored in other modes).
    pub fn set_plan(&mut self, plan: JointPlan) {
        self.plan = Some(plan);
    }

    /// Switch modes. On entry, reset the reference to the **current measurement**
    /// (carrying over the previous mode's reference makes it jump at the switch).
    pub fn request(&mut self, mode: Mode, arm: &ArmModel, s: &ArmState) {
        if mode == self.mode {
            return;
        }
        match mode {
            Mode::Hold => self.hold_q = s.q.clone(),
            Mode::Gravity => {}
            Mode::Joint | Mode::Park => self.shaper.reset(s.q.clone()),
            Mode::Osc => {
                self.shaper.reset(s.q.clone());
                self.tcp_shaper.reset(arm.tcp_pose(s.q.as_slice()));
                self.osc.reset();
            }
            Mode::Mpc => {
                self.shaper.reset(s.q.clone());
                self.tracking.reset();
                self.plan = None;
                self.hold_q = s.q.clone();
            }
            Mode::Done => {}
        }
        log::info!("mode {:?} → {:?}", self.mode, mode);
        self.mode = mode;
    }

    pub fn tick(&mut self, arm: &ArmModel, s: &ArmState, target: &Target, dt: f64) -> (JointCommand, TickInfo) {
        let now = self.t;
        self.t += dt;
        let mut info = TickInfo::default();
        if matches!(self.mode, Mode::Joint | Mode::Osc | Mode::Mpc) {
            info.tcp_goal = match target {
                Target::Tcp { pose, .. } => Some(*pose),
                Target::TcpRef { tcp, .. } => Some(tcp.pose),
                Target::Joint(q) | Target::JointRef { r: JointRef { q, .. }, .. } => {
                    Some(arm.tcp_pose(clamp_to_limits(arm, q).as_slice()))
                }
                Target::None => None,
            };
        }
        let ramp = if self.cfg.startup_ramp_s > 0.0 {
            (self.t / self.cfg.startup_ramp_s).min(1.0)
        } else {
            1.0
        };
        let cmd = match self.mode {
            Mode::Hold | Mode::Done => {
                let r = JointRef::at_rest(self.hold_q.clone());
                let mut c = self.hold.command(arm, s, &r);
                scale_gains(&mut c, ramp);
                info.reference = Some(r);
                c
            }
            Mode::Gravity => {
                // Keep the hold point following, so Hold can take over wherever the arm
                // is let go.
                self.hold_q = s.q.clone();
                self.gravity.command(arm, s, &JointRef::at_rest(s.q.clone()))
            }
            Mode::Joint => {
                let prev = self.shaper.current().clone();
                let mut r = match target {
                    Target::JointRef { r, .. } => {
                        let mut want = r.clone();
                        want.q = clamp_to_limits(arm, &want.q);
                        self.shaper.follow(&want, dt).clone()
                    }
                    _ => {
                        let goal = match target.joints() {
                            Some(q) => clamp_to_limits(arm, q),
                            None => self.shaper.current().q.clone(),
                        };
                        self.shaper.step(&goal, dt).clone()
                    }
                };
                if let Some(g) = &self.cfg.safety {
                    // Judge where the reference would come to rest if it braked
                    // now, not where it is, and brake gently (a quarter of the
                    // shaper's acceleration limit). Halting the reference at the
                    // wall, or braking at the full limit, let the arm (which
                    // follows through a lightly damped PD) overshoot by 1–1.5 cm.
                    // A step whose stopping point is worse than the previous
                    // one's is replaced by braking; a step that improves things
                    // is allowed (an arm starting outside can come back).
                    let a_brake: DVector<f64> = self.cfg.shaper.a_max.map(|a| (a * BRAKE_FRACTION).max(1e-6));
                    let stop = |r: &JointRef| {
                        DVector::from_iterator(
                            r.q.len(),
                            (0..r.q.len()).map(|i| r.q[i] + r.v[i] * r.v[i].abs() / (2.0 * a_brake[i])),
                        )
                    };
                    let refused = stop(&r);
                    if g.joint_violation(arm, refused.as_slice()) > g.joint_violation(arm, stop(&prev).as_slice()) + 1e-9 {
                        if !self.guarding {
                            info.guard_reason = Some(g.explain(arm, refused.as_slice()));
                        }
                        r = brake(&prev, &a_brake, dt);
                        self.shaper.restore(r.clone());
                        info.guarded = true;
                    }
                }
                self.guarding = info.guarded;
                let mut c = self.track.command(arm, s, &r);
                if let Target::JointRef { extras, .. } = target {
                    apply_joint_extras(&mut c, extras);
                }
                scale_gains(&mut c, ramp);
                info.reference = Some(r);
                c
            }
            Mode::Osc => {
                let no_extras = TcpExtras::default();
                let (tr, posture, extras) = match target {
                    Target::TcpRef { tcp, posture, extras } => {
                        let want = manip_control::shaper::TcpRefState { pose: tcp.pose, twist: tcp.twist, accel: tcp.accel };
                        (self.tcp_shaper.follow(&want, dt).clone(), clamp_to_limits(arm, posture), &**extras)
                    }
                    _ => {
                        let (pose, posture) = match target {
                            Target::Tcp { pose, posture } => (*pose, clamp_to_limits(arm, posture)),
                            Target::Joint(q) | Target::JointRef { r: JointRef { q, .. }, .. } => {
                                let q = clamp_to_limits(arm, q);
                                (arm.tcp_pose(q.as_slice()), q)
                            }
                            Target::None | Target::TcpRef { .. } => {
                                (self.tcp_shaper.current().pose, self.shaper.current().q.clone())
                            }
                        };
                        (self.tcp_shaper.step(&pose, dt).clone(), posture, &no_extras)
                    }
                };
                let pr = self.shaper.step(&posture, dt).clone();
                let tcp_ref = TcpRef {
                    pose: tr.pose,
                    twist: tr.twist,
                    accel: tr.accel,
                };
                info.tcp_reference = Some(tr.pose);
                // DOFs that don't move the TCP (gripper) are tracked with joint impedance.
                let base = self.track.command(arm, s, &pr);
                let cbfs = self.cfg.safety.as_ref().map(|g| g.cbfs(arm, s)).unwrap_or_default();
                match self.osc.command_ext(arm, s, &tcp_ref, &pr, dt, base, &cbfs, extras) {
                    Ok((c, report)) => {
                        info.osc = Some(report);
                        info.reference = Some(pr);
                        c
                    }
                    Err(e) => {
                        self.osc_failures += 1;
                        info.transition = Some(format!("OSC → Hold: {e}"));
                        log::warn!("OSC unsolvable, falling back to hold: {e}");
                        self.mode = Mode::Hold;
                        self.hold_q = s.q.clone();
                        let r = JointRef::at_rest(self.hold_q.clone());
                        info.reference = Some(r.clone());
                        self.hold.command(arm, s, &r)
                    }
                }
            }
            Mode::Mpc => {
                match target {
                    Target::Tcp { pose, .. } => info.tcp_reference = Some(*pose),
                    Target::TcpRef { tcp, .. } => info.tcp_reference = Some(tcp.pose),
                    _ => {}
                }
                // Posture / gripper reference from the target, as in OSC.
                let posture = match target.joints() {
                    Some(q) => clamp_to_limits(arm, q),
                    None => self.shaper.current().q.clone(),
                };
                let pr = self.shaper.step(&posture, dt).clone();
                let base = self.track.command(arm, s, &pr);
                match self.plan.as_ref().filter(|p| now - p.t0 <= self.cfg.plan_timeout_s) {
                    None if self.plan.is_some() => {
                        // The planner stalled or its plans were rejected: stop
                        // here (not back at the pose Mpc started from, which
                        // can be far away by now) and stay stopped.
                        info.transition = Some("Mpc → Hold: no fresh plan".into());
                        log::warn!("MPC: no fresh plan for {:.0} ms, holding", self.cfg.plan_timeout_s * 1e3);
                        self.mode = Mode::Hold;
                        self.hold_q = s.q.clone();
                        let r = JointRef::at_rest(self.hold_q.clone());
                        info.reference = Some(r.clone());
                        self.hold.command(arm, s, &r)
                    }
                    None => {
                        // No plan yet: hold where Mpc started.
                        let r = JointRef::at_rest(self.hold_q.clone());
                        info.reference = Some(r.clone());
                        let mut c = self.hold.command(arm, s, &r);
                        scale_gains(&mut c, ramp);
                        c
                    }
                    Some(plan) => {
                        let r = plan.sample_full(now, &pr);
                        let cbfs = self.cfg.safety.as_ref().map(|g| g.cbfs(arm, s)).unwrap_or_default();
                        match self.tracking.command(arm, s, &r, dt, base, &cbfs) {
                            Ok((c, report)) => {
                                info.mpc = Some(report);
                                info.reference = Some(r);
                                c
                            }
                            Err(e) => {
                                self.osc_failures += 1;
                                info.transition = Some(format!("Mpc → Hold: {e}"));
                                log::warn!("MPC tracking unsolvable, falling back to hold: {e}");
                                self.mode = Mode::Hold;
                                self.hold_q = s.q.clone();
                                let r = JointRef::at_rest(self.hold_q.clone());
                                info.reference = Some(r.clone());
                                self.hold.command(arm, s, &r)
                            }
                        }
                    }
                }
            }
            Mode::Park => {
                let mut limits = self.cfg.shaper.clone();
                limits.v_max *= self.cfg.park_speed_scale;
                let goal = clamp_to_limits(arm, &self.cfg.rest);
                let r = step_with_limits(&mut self.shaper, &limits, &goal, dt);
                let c = self.hold.command(arm, s, &r);
                let ref_home = (&r.q - &goal).amax() < 1e-4;
                self.park_settle_s = if ref_home { self.park_settle_s + dt } else { 0.0 };
                let off = (&s.q - &goal).amax();
                // Sticking friction can hold a joint just outside the
                // tolerance for good (the wrist at 0.07 rad in the stiction
                // sim, parking forever): after PARK_SETTLE_S at home, settle
                // for 3× the tolerance; farther off, keep holding.
                let reached = ref_home
                    && (off < self.cfg.park_tolerance
                        || (self.park_settle_s > PARK_SETTLE_S && off < 3.0 * self.cfg.park_tolerance));
                info.reference = Some(r);
                if reached {
                    info.transition = Some(if off < self.cfg.park_tolerance {
                        "Park → Done".into()
                    } else {
                        format!("Park → Done ({off:.3} rad from the rest pose, held there by friction)")
                    });
                    self.mode = Mode::Done;
                    self.hold_q = goal;
                }
                c
            }
        };
        (cmd, info)
    }
}

/// Shape with a lowered velocity limit for Park only. The shaper's limits are fixed at
/// construction, so step once with a temporary copy using the other limits, then
/// write the state back.
fn step_with_limits(shaper: &mut JointShaper, limits: &ShaperLimits, goal: &DVector<f64>, dt: f64) -> JointRef {
    let mut tmp = JointShaper::new(limits.clone(), shaper.current().q.clone());
    tmp.restore(shaper.current().clone());
    let r = tmp.step(goal, dt).clone();
    shaper.restore(r.clone());
    r
}

/// Park finishes slightly outside its tolerance after this long at home [s].
const PARK_SETTLE_S: f64 = 2.0;

/// Fraction of the shaper's acceleration limit used when the safety guard
/// brakes the joint reference.
const BRAKE_FRACTION: f64 = 0.25;

/// One step of braking `r` to rest at `a_brake` per joint.
fn brake(r: &JointRef, a_brake: &DVector<f64>, dt: f64) -> JointRef {
    let n = r.q.len();
    let mut out = r.clone();
    for i in 0..n {
        let dv = (a_brake[i] * dt).min(r.v[i].abs());
        let v_new = r.v[i] - dv.copysign(r.v[i]);
        out.a[i] = (v_new - r.v[i]) / dt;
        out.q[i] = r.q[i] + 0.5 * (r.v[i] + v_new) * dt;
        out.v[i] = v_new;
    }
    out
}

/// Torque offsets and gain overrides of a joint command.
fn apply_joint_extras(c: &mut JointCommand, x: &JointExtras) {
    for (i, a) in c.axes.iter_mut().enumerate() {
        if let Some(t) = &x.torque {
            a.tau += t[i];
        }
        if let Some(kp) = &x.kp {
            a.kp = kp[i];
        }
        if let Some(kd) = &x.kd {
            a.kd = kd[i];
        }
    }
}

fn clamp_to_limits(arm: &ArmModel, q: &DVector<f64>) -> DVector<f64> {
    DVector::from_iterator(q.len(), arm.dofs().iter().zip(q.iter()).map(|(d, &x)| d.clamp(x)))
}

fn scale_gains(c: &mut JointCommand, s: f64) {
    if s >= 1.0 {
        return;
    }
    for a in &mut c.axes {
        a.kp *= s;
        a.kd *= s;
    }
}
