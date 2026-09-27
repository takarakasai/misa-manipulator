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

use manip_control::{
    JointCommand, JointGains, JointImpedance, JointRef, JointShaper, Osc, OscReport,
    ShaperLimits, TcpRef, TcpShaper,
};
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
}

/// Breakdown of one cycle (for recording).
#[derive(Debug, Clone, Default)]
pub struct TickInfo {
    pub reference: Option<JointRef>,
    pub tcp_reference: Option<Isometry3<f64>>,
    pub osc: Option<OscReport>,
    /// Whether the mode changed this cycle (and if so, why).
    pub transition: Option<String>,
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
}

pub struct Supervisor {
    cfg: SupervisorConfig,
    mode: Mode,
    track: JointImpedance,
    hold: JointImpedance,
    gravity: JointImpedance,
    osc: Osc,
    shaper: JointShaper,
    tcp_shaper: TcpShaper,
    /// Hold reference: the pose when entering the mode.
    hold_q: DVector<f64>,
    /// Time since startup [s] (for the gain ramp-up).
    t: f64,
    pub osc_failures: u64,
}

impl Supervisor {
    /// `q` is the **measured pose**. Starts by holding there.
    pub fn new(cfg: SupervisorConfig, osc: Osc, arm: &ArmModel, q: &DVector<f64>) -> Self {
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
            shaper,
            tcp_shaper,
            hold_q: q.clone(),
            t: 0.0,
            osc_failures: 0,
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
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
            Mode::Done => {}
        }
        log::info!("mode {:?} → {:?}", self.mode, mode);
        self.mode = mode;
    }

    pub fn tick(&mut self, arm: &ArmModel, s: &ArmState, target: &Target, dt: f64) -> (JointCommand, TickInfo) {
        self.t += dt;
        let mut info = TickInfo::default();
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
                let goal = match target {
                    Target::Joint(q) => clamp_to_limits(arm, q),
                    Target::Tcp { posture, .. } => clamp_to_limits(arm, posture),
                    Target::None => self.shaper.current().q.clone(),
                };
                let r = self.shaper.step(&goal, dt).clone();
                let mut c = self.track.command(arm, s, &r);
                scale_gains(&mut c, ramp);
                info.reference = Some(r);
                c
            }
            Mode::Osc => {
                let (pose, posture) = match target {
                    Target::Tcp { pose, posture } => (*pose, clamp_to_limits(arm, posture)),
                    Target::Joint(q) => {
                        let q = clamp_to_limits(arm, q);
                        (arm.tcp_pose(q.as_slice()), q)
                    }
                    Target::None => (self.tcp_shaper.current().pose, self.shaper.current().q.clone()),
                };
                let tr = self.tcp_shaper.step(&pose, dt).clone();
                let pr = self.shaper.step(&posture, dt).clone();
                let tcp_ref = TcpRef {
                    pose: tr.pose,
                    twist: tr.twist,
                    accel: tr.accel,
                };
                info.tcp_reference = Some(tr.pose);
                // DOFs that don't move the TCP (gripper) are tracked with joint impedance.
                let base = self.track.command(arm, s, &pr);
                match self.osc.command(arm, s, &tcp_ref, &pr, dt, base) {
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
            Mode::Park => {
                let mut limits = self.cfg.shaper.clone();
                limits.v_max *= self.cfg.park_speed_scale;
                let goal = clamp_to_limits(arm, &self.cfg.rest);
                let r = step_with_limits(&mut self.shaper, &limits, &goal, dt);
                let c = self.hold.command(arm, s, &r);
                let reached = (&r.q - &goal).amax() < 1e-4
                    && (&s.q - &goal).amax() < self.cfg.park_tolerance;
                info.reference = Some(r);
                if reached {
                    info.transition = Some("Park → Done".into());
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
