//! Motion commands → the mode requests and target of every cycle.
//!
//! The API (or a script) hands over [`MotionCmd`]s; [`Executive::step`] runs
//! once per control cycle, outside [`crate::policy::Policy`], and produces
//! what the policy takes as **recorded inputs**: mode requests and a
//! [`Target`]. A run log therefore replays bit-exact whatever arrived over
//! the network and when.
//!
//! | command | mode | target |
//! |---|---|---|
//! | joint move / waypoints / velocity / acceleration | Joint | `JointRef` (generated, followed as is) |
//! | TCP move / waypoints / velocity / acceleration | Osc | `TcpRef` (+ force / impedance) |
//! | TCP move `via: mpc` | Mpc | `Tcp` (the planner finds the way) |
//! | teleop | Joint / Osc / Mpc | the leader's joint angles |
//!
//! Motions run one at a time; a new one replaces the active one (or queues
//! behind it), starting from the reference the last one left — so moves chain
//! without a jump. Force, impedance, joint torque and the gripper are
//! settings that apply to whatever runs. Every motion gets a status
//! (queued → active → done / aborted / rejected) on the shared [`Board`].
//!
//! # Safety
//!
//! Everything here only proposes references. The policy still shapes them
//! (no jump), clamps them into the joint range, keeps the workspace box and
//! self-collision barriers, and the SafetyGate clips the commands. Velocity
//! and acceleration streams stop on their own when no command arrives within
//! their timeout. If the controller falls back to Hold (OSC unsolvable, MPC
//! stalled), the active motion is aborted and nothing is re-requested until
//! the next command.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use manip_control::shaper::TcpRefState;
use manip_control::{JointRef, JointShaper, ShaperLimits};
use manip_model::{ArmModel, ArmState};
use manip_wbc::{Compliance, TcpExtras, TcpRef};
use nalgebra::{DVector, Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};
use serde::Serialize;

use crate::supervisor::{JointExtras, Mode, Target};
use crate::traj::{box_excess, Knot, Traj};

pub type MotionId = u64;

/// Frame of a TCP command: the base, or the TCP itself (as it is when the
/// command starts for moves; as it moves, for velocities and forces).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    #[default]
    Base,
    Tool,
}

/// Joint values by independent-DOF index; the others keep their reference.
#[derive(Debug, Clone, PartialEq)]
pub struct JointGoal {
    pub values: Vec<(usize, f64)>,
    pub relative: bool,
}

/// A TCP pose; missing parts keep the current reference.
#[derive(Debug, Clone, PartialEq)]
pub struct TcpGoal {
    pub position: Option<Vector3<f64>>,
    pub rotation: Option<UnitQuaternion<f64>>,
    pub frame: Frame,
    pub relative: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    Osc,
    Mpc,
}

/// A force/moment the TCP exerts (sticky until cleared).
#[derive(Debug, Clone, PartialEq)]
pub struct ForceSpec {
    /// `[moment; force]` in `frame`.
    pub wrench: Vector6<f64>,
    pub frame: Frame,
    /// Axes (`[rx, ry, rz, x, y, z]` of `frame`) left without position
    /// feedback; by default those with a nonzero component. Ignored while an
    /// impedance is set (its zero-stiffness rows are the free ones).
    pub free: [bool; 6],
    /// Ramp in/out time [s].
    pub ramp_s: f64,
    /// Above these speeds along the free axes [m/s, rad/s] the force is
    /// reduced (it pushes into nothing).
    pub max_speed: f64,
    pub max_ang_speed: f64,
}

/// A Cartesian impedance (sticky until cleared), diagonal in `frame`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpedanceSpec {
    /// `[rx, ry, rz, x, y, z]` [N·m/rad, N/m].
    pub stiffness: Vector6<f64>,
    /// [N·m·s/rad, N·s/m]; `None` = critically damped.
    pub damping: Option<Vector6<f64>>,
    pub frame: Frame,
}

/// Joint torque offsets (sticky until cleared), Joint mode.
#[derive(Debug, Clone, PartialEq)]
pub struct JointTorqueSpec {
    pub torque: DVector<f64>,
    /// Scale of the tracking stiffness (0 = torque only, still damped).
    pub stiffness_scale: f64,
    pub ramp_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MotionCmd {
    MoveJoint { goal: JointGoal, duration: Option<f64>, speed: f64 },
    MoveTcp { goal: TcpGoal, duration: Option<f64>, speed: f64, via: Via },
    WaypointsJoint { points: Vec<JointGoal>, times: Option<Vec<f64>>, speed: f64 },
    WaypointsTcp { points: Vec<TcpGoal>, times: Option<Vec<f64>>, speed: f64 },
    /// Joint velocity (`v`) or acceleration (`a`) stream.
    JointStream { v: Option<DVector<f64>>, a: Option<DVector<f64>>, timeout: f64 },
    /// TCP twist / acceleration stream (`[ω; v]`, `[α; a]` in `frame`).
    TcpStream { twist: Option<Vector6<f64>>, accel: Option<Vector6<f64>>, frame: Frame, timeout: f64 },
    Force(Option<ForceSpec>),
    Impedance(Option<ImpedanceSpec>),
    JointTorque(Option<JointTorqueSpec>),
    Gripper { position: f64, speed: f64, max_force: Option<f64> },
    Hold,
    Gravity,
    Park,
    Teleop(Mode),
    Stop,
    Shutdown,
}

impl MotionCmd {
    pub fn kind(&self) -> &'static str {
        match self {
            MotionCmd::MoveJoint { .. } => "move_joint",
            MotionCmd::MoveTcp { .. } => "move_tcp",
            MotionCmd::WaypointsJoint { .. } => "waypoints_joint",
            MotionCmd::WaypointsTcp { .. } => "waypoints_tcp",
            MotionCmd::JointStream { v: Some(_), .. } => "velocity_joint",
            MotionCmd::JointStream { .. } => "accel_joint",
            MotionCmd::TcpStream { twist: Some(_), .. } => "velocity_tcp",
            MotionCmd::TcpStream { .. } => "accel_tcp",
            MotionCmd::Force(_) => "force",
            MotionCmd::Impedance(_) => "impedance",
            MotionCmd::JointTorque(_) => "joint_torque",
            MotionCmd::Gripper { .. } => "gripper",
            MotionCmd::Hold => "hold",
            MotionCmd::Gravity => "gravity",
            MotionCmd::Park => "park",
            MotionCmd::Teleop(_) => "teleop",
            MotionCmd::Stop => "stop",
            MotionCmd::Shutdown => "shutdown",
        }
    }
}

// ── Status board ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Queued,
    Active,
    Done,
    Aborted,
    Rejected,
}

impl State {
    pub fn finished(self) -> bool {
        matches!(self, State::Done | State::Aborted | State::Rejected)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MotionStatus {
    pub id: MotionId,
    pub kind: &'static str,
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Planned duration [s] (trajectories).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}

/// Statuses of the last [`Board::KEEP`] commands, shared with the API.
#[derive(Debug, Default)]
pub struct Board {
    next: MotionId,
    map: BTreeMap<MotionId, MotionStatus>,
}

impl Board {
    const KEEP: usize = 1000;

    pub fn register(&mut self, kind: &'static str) -> MotionId {
        self.next += 1;
        let id = self.next;
        self.map.insert(id, MotionStatus { id, kind, state: State::Queued, message: None, duration: None });
        while self.map.len() > Self::KEEP {
            let first = *self.map.keys().next().unwrap();
            self.map.remove(&first);
        }
        id
    }

    pub fn get(&self, id: MotionId) -> Option<MotionStatus> {
        self.map.get(&id).cloned()
    }

    fn set(&mut self, id: MotionId, state: State, message: Option<String>) {
        if let Some(s) = self.map.get_mut(&id) {
            s.state = state;
            if message.is_some() {
                s.message = message;
            }
        }
    }

    fn set_duration(&mut self, id: MotionId, d: f64) {
        if let Some(s) = self.map.get_mut(&id) {
            s.duration = Some(d);
        }
    }
}

pub type SharedBoard = Arc<Mutex<Board>>;

/// A command and whether it waits behind the running motion.
#[derive(Debug, Clone)]
pub struct Queued {
    pub id: MotionId,
    pub cmd: MotionCmd,
    pub append: bool,
}

// ── Configuration ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MotionConfig {
    pub dt: f64,
    /// Joint speed / acceleration limits (the shaper's).
    pub v_max: DVector<f64>,
    pub a_max: DVector<f64>,
    pub lin_v_max: f64,
    pub lin_a_max: f64,
    pub ang_v_max: f64,
    pub ang_a_max: f64,
    pub time_constant_s: f64,
    /// The gripper DOF (outside the TCP chain), if any.
    pub gripper: Option<usize>,
    /// Joint tracking stiffness (joint-torque stiffness scaling, the gripper
    /// force limit).
    pub track_kp: DVector<f64>,
    /// Joint range per DOF, and what joint streams keep from its ends [rad].
    pub joint_range: Vec<(f64, f64)>,
    pub q_margin: f64,
    /// A TCP stream's reference stays this close to the measured TCP [m, rad].
    pub stream_lead: (f64, f64),
    pub mpc_available: bool,
    /// MPC moves: done within these [m, rad] at rest; aborted after this [s].
    pub mpc_tolerance: (f64, f64),
    pub mpc_timeout_s: f64,
}

/// The leader, for the teleop command.
pub struct Teleop {
    pub thread: manip_leader::LeaderThread,
    pub mapping: crate::assemble::TeleopMapping,
    pub timeout: std::time::Duration,
}

// ── Executive ───────────────────────────────────────────────────────────

/// The reference the last cycle emitted (where the next motion starts).
#[derive(Debug, Clone)]
enum Last {
    None,
    Joint(JointRef),
    Tcp { r: TcpRefState, posture: DVector<f64> },
}

#[derive(Debug, Clone)]
enum Active {
    Joint { traj: Traj, t0: f64 },
    /// Coordinates `[rotation vector of R·R0⁻¹; position]`.
    Tcp { traj: Traj, t0: f64, r0: UnitQuaternion<f64>, posture: DVector<f64> },
    Mpc { pose: Isometry3<f64>, posture: DVector<f64>, t0: f64 },
    JointStream { v: Option<DVector<f64>>, a: Option<DVector<f64>>, last_cmd: f64, timeout: f64, r: JointRef },
    TcpStream { twist: Option<Vector6<f64>>, accel: Option<Vector6<f64>>, frame: Frame, last_cmd: f64, timeout: f64, r: TcpRefState, posture: DVector<f64> },
}

impl Active {
    fn mode(&self) -> Mode {
        match self {
            Active::Joint { .. } | Active::JointStream { .. } => Mode::Joint,
            Active::Tcp { .. } | Active::TcpStream { .. } => Mode::Osc,
            Active::Mpc { .. } => Mode::Mpc,
        }
    }
}

#[derive(Debug, Clone)]
struct Gripper {
    dof: usize,
    shaper: JointShaper,
    goal: f64,
    max_force: Option<f64>,
    id: Option<MotionId>,
    /// The output (possibly held back by the force limit).
    out: JointRef,
    /// How long the finger has been still [s] (done when blocked).
    still_s: f64,
}

#[derive(Debug, Clone)]
struct Ramped<T> {
    spec: T,
    level: f64,
    /// Ramping out (then removed).
    clearing: bool,
}

/// Joint torque offsets stop pushing this close to a limit [rad or m].
const LIMIT_GUARD: f64 = 0.1;

/// What [`Executive::step`] produced for the policy.
pub struct StepOut {
    pub requests: Vec<Mode>,
    pub target: Target,
}

pub struct Executive {
    cfg: MotionConfig,
    rx: Option<Receiver<Queued>>,
    /// Commands delivered at a given time (scripts, tests).
    scripted: VecDeque<(f64, Queued)>,
    inbox: VecDeque<Queued>,
    board: SharedBoard,
    queue: VecDeque<(MotionId, MotionCmd)>,
    active: Option<(MotionId, Active)>,
    last: Last,
    /// The mode our targets are for (what we, or the runner, last asked for).
    want: Mode,
    gripper: Option<Gripper>,
    force: Option<Ramped<ForceSpec>>,
    impedance: Option<ImpedanceSpec>,
    joint_torque: Option<Ramped<JointTorqueSpec>>,
    teleop: Option<Teleop>,
    teleop_on: bool,
    shutdown: bool,
    /// A stop command, done once the motion it brakes is at rest.
    stopping: Option<MotionId>,
    /// A park / shutdown command, done at Done.
    parking: Option<MotionId>,
}

fn push_mode(m: Mode, requests: &mut Vec<Mode>, pending: &mut Mode) {
    if *pending != m {
        requests.push(m);
        *pending = m;
    }
}

impl Executive {
    pub fn new(cfg: MotionConfig, rx: Option<Receiver<Queued>>, board: SharedBoard, teleop: Option<Teleop>) -> Self {
        Self {
            cfg,
            rx,
            scripted: VecDeque::new(),
            inbox: VecDeque::new(),
            board,
            queue: VecDeque::new(),
            active: None,
            last: Last::None,
            want: Mode::Hold,
            gripper: None,
            force: None,
            impedance: None,
            joint_torque: None,
            teleop,
            teleop_on: false,
            shutdown: false,
            stopping: None,
            parking: None,
        }
    }

    /// Deliver `cmd` at time `t` (scripts and tests). Returns its id.
    pub fn schedule(&mut self, t: f64, cmd: MotionCmd, append: bool) -> MotionId {
        let id = self.board.lock().unwrap().register(cmd.kind());
        let at = self.scripted.partition_point(|(x, _)| *x <= t);
        self.scripted.insert(at, (t, Queued { id, cmd, append }));
        id
    }

    /// Asked to shut down (park, then the runner exits).
    pub fn shutting_down(&self) -> bool {
        self.shutdown
    }

    /// Nothing scheduled, queued or running (scripts end here).
    pub fn idle(&self) -> bool {
        self.scripted.is_empty()
            && self.inbox.is_empty()
            && self.queue.is_empty()
            && self.active.is_none()
            && self.gripper.as_ref().is_none_or(|g| g.id.is_none())
            && self.stopping.is_none()
            && self.parking.is_none()
    }

    pub fn summary(&self) -> String {
        let mut s = match &self.active {
            Some((id, a)) => format!("api #{id} {:?}", a.mode()),
            None if self.teleop_on => "api teleop".into(),
            None => "api idle".into(),
        };
        if !self.queue.is_empty() {
            s += &format!(" +{} queued", self.queue.len());
        }
        if self.force.is_some() {
            s += " force";
        }
        if self.impedance.is_some() {
            s += " impedance";
        }
        if self.joint_torque.is_some() {
            s += " joint-torque";
        }
        s
    }

    /// For the state report.
    pub fn report(&self) -> ExecReport {
        ExecReport {
            active: self.active.as_ref().map(|(id, a)| ActiveReport {
                id: *id,
                kind: match a {
                    Active::Joint { .. } => "joint_trajectory",
                    Active::Tcp { .. } => "tcp_trajectory",
                    Active::Mpc { .. } => "mpc",
                    Active::JointStream { .. } => "joint_stream",
                    Active::TcpStream { .. } => "tcp_stream",
                },
            }),
            queued: self.queue.len(),
            teleop: self.teleop_on,
            force: self.force.as_ref().map(|f| ForceReport {
                wrench: f.spec.wrench.as_slice().to_vec(),
                frame: f.spec.frame,
                level: f.level,
            }),
            impedance: self.impedance.as_ref().map(|k| ImpedanceReport {
                stiffness: k.stiffness.as_slice().to_vec(),
                damping: k.damping.map(|d| d.as_slice().to_vec()),
                frame: k.frame,
            }),
            joint_torque: self.joint_torque.as_ref().map(|j| j.spec.torque.as_slice().to_vec()),
            gripper_goal: self.gripper.as_ref().map(|g| g.goal),
        }
    }

    /// One cycle. `mode` is the mode the policy will be in after the
    /// runner's own requests this cycle; `ready` is false during the
    /// runner's startup sequence and while it parks on Ctrl-C (commands wait).
    pub fn step(&mut self, arm: &ArmModel, t: f64, s: &ArmState, mode: Mode, ready: bool) -> StepOut {
        let mut requests = Vec::new();
        let mut pending = mode;
        if let (None, Some(g)) = (&self.gripper, self.cfg.gripper) {
            self.gripper = Some(self.new_gripper(g, s.q[g], 1.0));
        }

        // The controller is not where we drove it (it fell back to Hold, the
        // runner parks, or Park finished): drop what was running.
        if mode != self.want {
            match (self.want, mode) {
                (Mode::Park, Mode::Done) => {
                    if let Some(id) = self.parking.take() {
                        self.set(id, State::Done, None);
                    }
                }
                _ => {
                    let why = format!("controller left {:?} for {:?}", self.want, mode);
                    if self.active.is_some() || !self.queue.is_empty() || self.teleop_on {
                        log::warn!("motion: {why}");
                    }
                    self.abort_all(&why);
                    self.teleop_on = false;
                    self.force = None;
                    self.joint_torque = None;
                }
            }
            self.last = Last::None;
            self.want = mode;
        }

        if ready {
            self.receive(t);
            while let Some(q) = self.inbox.pop_front() {
                self.apply(arm, t, s, &mut pending, &mut requests, q);
            }
            if self.active.is_none()
                && !self.teleop_on
                && self.parking.is_none()
                && let Some((id, cmd)) = self.queue.pop_front()
            {
                self.activate(arm, t, s, &mut pending, &mut requests, id, cmd);
            }
        }
        if pending != self.want {
            self.last = Last::None;
        }
        self.want = pending;
        let target = self.target(arm, t, s, pending);
        StepOut { requests, target }
    }

    fn set(&self, id: MotionId, state: State, message: Option<String>) {
        if let Some(m) = &message {
            match state {
                State::Aborted | State::Rejected => log::warn!("motion #{id} {state:?}: {m}"),
                _ => log::info!("motion #{id} {state:?}: {m}"),
            }
        }
        self.board.lock().unwrap().set(id, state, message);
    }

    fn receive(&mut self, t: f64) {
        while self.scripted.front().is_some_and(|(at, _)| *at <= t) {
            let (_, q) = self.scripted.pop_front().unwrap();
            self.inbox.push_back(q);
        }
        if let Some(rx) = &self.rx {
            while let Ok(q) = rx.try_recv() {
                self.inbox.push_back(q);
            }
        }
    }

    fn abort_all(&mut self, why: &str) {
        if let Some((id, _)) = self.active.take() {
            self.set(id, State::Aborted, Some(why.into()));
        }
        for (id, _) in std::mem::take(&mut self.queue) {
            self.set(id, State::Aborted, Some(why.into()));
        }
        if let Some(id) = self.stopping.take() {
            self.set(id, State::Done, Some(why.into()));
        }
    }

    fn new_gripper(&self, dof: usize, q: f64, speed: f64) -> Gripper {
        let limits = ShaperLimits {
            v_max: DVector::from_element(1, self.cfg.v_max[dof] * speed),
            a_max: DVector::from_element(1, self.cfg.a_max[dof] * speed),
            time_constant_s: self.cfg.time_constant_s,
        };
        Gripper {
            dof,
            shaper: JointShaper::new(limits, DVector::from_element(1, q)),
            goal: q,
            max_force: None,
            id: None,
            out: JointRef::at_rest(DVector::from_element(1, q)),
            still_s: 0.0,
        }
    }

    /// Apply one arriving command.
    fn apply(&mut self, arm: &ArmModel, t: f64, s: &ArmState, pending: &mut Mode, requests: &mut Vec<Mode>, q: Queued) {
        let Queued { id, cmd, append } = q;
        match cmd {
            MotionCmd::Force(spec) => {
                if spec.is_some()
                    && let Err(e) = self.enter_for_setting(Mode::Osc, "a force", pending, requests)
                {
                    self.set(id, State::Rejected, Some(e));
                    return;
                }
                match (spec, self.force.take()) {
                    (Some(spec), prev) => {
                        let level = prev.map(|p| p.level).unwrap_or(0.0);
                        self.force = Some(Ramped { spec, level, clearing: false });
                    }
                    (None, Some(mut prev)) => {
                        prev.clearing = true;
                        self.force = Some(prev);
                    }
                    (None, None) => {}
                }
                self.set(id, State::Done, None);
            }
            MotionCmd::Impedance(spec) => {
                if spec.is_some()
                    && let Err(e) = self.enter_for_setting(Mode::Osc, "an impedance", pending, requests)
                {
                    self.set(id, State::Rejected, Some(e));
                    return;
                }
                self.impedance = spec;
                self.set(id, State::Done, None);
            }
            MotionCmd::JointTorque(spec) => {
                if spec.is_some()
                    && let Err(e) = self.enter_for_setting(Mode::Joint, "a joint torque", pending, requests)
                {
                    self.set(id, State::Rejected, Some(e));
                    return;
                }
                match (spec, self.joint_torque.take()) {
                    (Some(spec), prev) => {
                        let level = prev.map(|p| p.level).unwrap_or(0.0);
                        self.joint_torque = Some(Ramped { spec, level, clearing: false });
                    }
                    (None, Some(mut prev)) => {
                        prev.clearing = true;
                        self.joint_torque = Some(prev);
                    }
                    (None, None) => {}
                }
                self.set(id, State::Done, None);
            }
            MotionCmd::Gripper { position, speed, max_force } => {
                if self.gripper.is_none() {
                    self.set(id, State::Rejected, Some("this arm has no gripper DOF".into()));
                    return;
                }
                // The gripper moves with the arm driven (in Hold it holds too).
                if !matches!(*pending, Mode::Joint | Mode::Osc | Mode::Mpc)
                    && let Err(e) = self.enter_for_setting(Mode::Joint, "the gripper", pending, requests)
                {
                    self.set(id, State::Rejected, Some(e));
                    return;
                }
                let g = self.gripper.take().expect("checked");
                if let Some(old) = g.id {
                    self.set(old, State::Aborted, Some(format!("replaced by #{id}")));
                }
                let d = &arm.dofs()[g.dof];
                let goal = d.clamp(position);
                let mut ng = self.new_gripper(g.dof, g.out.q[0], speed.clamp(0.01, 1.0));
                ng.shaper.restore(g.shaper.current().clone());
                ng.out = g.out;
                ng.goal = goal;
                ng.max_force = max_force;
                ng.id = Some(id);
                ng.still_s = 0.0;
                self.gripper = Some(ng);
                self.set(id, State::Active, (goal != position).then(|| format!("clamped into range: {goal:.4}")));
            }
            MotionCmd::Hold | MotionCmd::Gravity => {
                let m = if cmd == MotionCmd::Hold { Mode::Hold } else { Mode::Gravity };
                self.abort_all(&format!("{} requested", cmd.kind()));
                self.teleop_on = false;
                self.force = None;
                self.joint_torque = None;
                push_mode(m, requests, pending);
                self.set(id, State::Done, None);
            }
            MotionCmd::Park | MotionCmd::Shutdown => {
                self.abort_all(&format!("{} requested", cmd.kind()));
                self.teleop_on = false;
                self.force = None;
                self.joint_torque = None;
                if cmd == MotionCmd::Shutdown {
                    self.shutdown = true;
                }
                if *pending == Mode::Done {
                    self.set(id, State::Done, None);
                } else {
                    push_mode(Mode::Park, requests, pending);
                    self.parking = Some(id);
                    self.set(id, State::Active, None);
                }
            }
            MotionCmd::Teleop(m) => {
                if self.teleop.is_none() {
                    self.set(id, State::Rejected, Some("no leader: start with --leader <profile>".into()));
                    return;
                }
                if m == Mode::Mpc && !self.cfg.mpc_available {
                    self.set(id, State::Rejected, Some("no MPC planner".into()));
                    return;
                }
                self.abort_all("teleop requested");
                self.force = None;
                self.joint_torque = None;
                self.teleop_on = true;
                push_mode(m, requests, pending);
                self.set(id, State::Done, None);
            }
            MotionCmd::Stop => {
                for (qid, _) in std::mem::take(&mut self.queue) {
                    self.set(qid, State::Aborted, Some("stopped".into()));
                }
                if let Some(f) = &mut self.force {
                    f.clearing = true;
                }
                if let Some(j) = &mut self.joint_torque {
                    j.clearing = true;
                }
                self.teleop_on = false;
                if let Some(old) = self.stopping.take() {
                    self.set(old, State::Done, Some(format!("superseded by #{id}")));
                }
                // Brake whatever runs as a stream commanded to rest.
                let braking = match self.active.take() {
                    Some((aid, a)) => {
                        self.set(aid, State::Aborted, Some(format!("stopped by #{id}")));
                        match (a, &self.last) {
                            (Active::Mpc { .. }, _) => {
                                push_mode(Mode::Hold, requests, pending);
                                None
                            }
                            (_, Last::Joint(r)) => Some(Active::JointStream {
                                v: Some(DVector::zeros(r.q.len())),
                                a: None,
                                // Stale from the start: brake to rest.
                                last_cmd: f64::NEG_INFINITY,
                                timeout: 0.0,
                                r: r.clone(),
                            }),
                            (_, Last::Tcp { r, posture }) => Some(Active::TcpStream {
                                twist: Some(Vector6::zeros()),
                                accel: None,
                                frame: Frame::Base,
                                // Stale from the start: brake to rest.
                                last_cmd: f64::NEG_INFINITY,
                                timeout: 0.0,
                                r: r.clone(),
                                posture: posture.clone(),
                            }),
                            (_, Last::None) => None,
                        }
                    }
                    None => None,
                };
                match braking {
                    Some(b) => {
                        self.active = Some((id, b));
                        self.stopping = Some(id);
                        self.set(id, State::Active, None);
                    }
                    None => self.set(id, State::Done, None),
                }
            }
            MotionCmd::JointStream { v, a, timeout } if !append => {
                if let Some((old, Active::JointStream { v: ov, a: oa, last_cmd, timeout: ot, .. })) = &mut self.active
                    && ov.is_some() == v.is_some()
                {
                    let prev = *old;
                    (*ov, *oa, *last_cmd, *ot) = (v, a, t, timeout);
                    *old = id;
                    self.set(prev, State::Done, Some(format!("continued by #{id}")));
                    self.set(id, State::Active, None);
                    return;
                }
                self.replace(id, MotionCmd::JointStream { v, a, timeout });
                let _ = (arm, s);
            }
            MotionCmd::TcpStream { twist, accel, frame, timeout } if !append => {
                if let Some((old, Active::TcpStream { twist: ow, accel: oa, frame: of, last_cmd, timeout: ot, .. })) = &mut self.active
                    && ow.is_some() == twist.is_some()
                {
                    let prev = *old;
                    (*ow, *oa, *of, *last_cmd, *ot) = (twist, accel, frame, t, timeout);
                    *old = id;
                    self.set(prev, State::Done, Some(format!("continued by #{id}")));
                    self.set(id, State::Active, None);
                    return;
                }
                self.replace(id, MotionCmd::TcpStream { twist, accel, frame, timeout });
            }
            cmd if append => self.queue.push_back((id, cmd)),
            cmd => self.replace(id, cmd),
        }
    }

    /// Settings that act in one mode switch to it when the arm is elsewhere,
    /// holding where it is: a motion of another space cannot continue and is
    /// aborted. Teleop and parking refuse (returns the reason).
    fn enter_for_setting(&mut self, want: Mode, what: &str, pending: &mut Mode, requests: &mut Vec<Mode>) -> Result<(), String> {
        if *pending == want {
            return Ok(());
        }
        if self.teleop_on {
            return Err(format!("{what} does not apply during teleop"));
        }
        if self.parking.is_some() || *pending == Mode::Park {
            return Err(format!("{what}: the arm is parking"));
        }
        if let Some((id, _)) = self.active.take() {
            self.set(id, State::Aborted, Some(format!("{what} needs {want:?} mode")));
        }
        for (qid, _) in std::mem::take(&mut self.queue) {
            self.set(qid, State::Aborted, Some(format!("{what} needs {want:?} mode")));
        }
        push_mode(want, requests, pending);
        Ok(())
    }

    /// A motion that replaces the running one (and drops the queue).
    fn replace(&mut self, id: MotionId, cmd: MotionCmd) {
        let why = format!("replaced by #{id}");
        if let Some((old, _)) = self.active.take() {
            self.set(old, State::Aborted, Some(why.clone()));
        }
        for (qid, _) in std::mem::take(&mut self.queue) {
            self.set(qid, State::Aborted, Some(why.clone()));
        }
        if let Some(sid) = self.stopping.take() {
            self.set(sid, State::Done, Some(why));
        }
        self.teleop_on = false;
        self.queue.push_front((id, cmd));
    }

    /// Where a joint motion starts: the last joint reference if we are
    /// already driving Joint mode, else the measurement at rest (after a mode
    /// switch the policy restarts from there too).
    fn joint_anchor(&self, s: &ArmState, pending: Mode) -> JointRef {
        match (&self.last, pending) {
            (Last::Joint(r), Mode::Joint) => r.clone(),
            _ => JointRef::at_rest(s.q.clone()),
        }
    }

    fn tcp_anchor(&self, s: &ArmState, pending: Mode) -> (TcpRefState, DVector<f64>) {
        match (&self.last, pending) {
            (Last::Tcp { r, posture }, Mode::Osc) => (r.clone(), posture.clone()),
            _ => (TcpRefState { pose: s.tcp_pose, twist: Vector6::zeros(), accel: Vector6::zeros() }, s.q.clone()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn activate(&mut self, arm: &ArmModel, t: f64, s: &ArmState, pending: &mut Mode, requests: &mut Vec<Mode>, id: MotionId, cmd: MotionCmd) {
        match self.build(arm, t, s, *pending, id, cmd) {
            Ok((active, note)) => {
                let m = active.mode();
                if *pending != m {
                    push_mode(m, requests, pending);
                }
                if let Active::Joint { traj, .. } | Active::Tcp { traj, .. } = &active {
                    self.board.lock().unwrap().set_duration(id, traj.duration());
                }
                self.set(id, State::Active, note);
                self.active = Some((id, active));
            }
            Err(e) => self.set(id, State::Rejected, Some(e)),
        }
    }

    /// The active motion for `cmd`, anchored where it starts.
    fn build(&mut self, arm: &ArmModel, t: f64, s: &ArmState, pending: Mode, id: MotionId, cmd: MotionCmd) -> Result<(Active, Option<String>), String> {
        let c = &self.cfg;
        let joint_mode = |m: Mode| if m == Mode::Joint { Mode::Joint } else { Mode::Hold };
        let osc_mode = |m: Mode| if m == Mode::Osc { Mode::Osc } else { Mode::Hold };
        match cmd {
            MotionCmd::MoveJoint { goal, duration, speed } => {
                let a = self.joint_anchor(s, joint_mode(pending));
                let (end, clamped) = self.resolve_joint(arm, &a.q, &goal);
                let (traj, stretched) = self.joint_traj(&a, &[end], None, duration, speed);
                Ok((Active::Joint { traj, t0: t }, notes(&[(clamped, "clamped into the joint range"), (stretched, "slowed down to keep the limits")])))
            }
            MotionCmd::WaypointsJoint { points, times, speed } => {
                let a = self.joint_anchor(s, joint_mode(pending));
                let mut prev = a.q.clone();
                let mut ends = Vec::new();
                let mut clamped = false;
                for p in &points {
                    let (q, cl) = self.resolve_joint(arm, &prev, p);
                    clamped |= cl;
                    prev = q.clone();
                    ends.push(q);
                }
                check_times(times.as_deref(), ends.len())?;
                let (traj, stretched) = self.joint_traj(&a, &ends, times.as_deref(), None, speed);
                Ok((Active::Joint { traj, t0: t }, notes(&[(clamped, "clamped into the joint range"), (stretched, "slowed down to keep the limits")])))
            }
            MotionCmd::MoveTcp { goal, duration, speed, via: Via::Osc } => {
                let (a, posture) = self.tcp_anchor(s, osc_mode(pending));
                let end = resolve_tcp(&a.pose, &goal);
                let (traj, stretched) = self.tcp_traj(&a, &[end], None, duration, speed);
                Ok((
                    Active::Tcp { traj, t0: t, r0: a.pose.rotation, posture },
                    notes(&[(stretched, "slowed down to keep the limits")]),
                ))
            }
            MotionCmd::MoveTcp { goal, via: Via::Mpc, .. } => {
                if !c.mpc_available {
                    return Err("no MPC planner".into());
                }
                let start = match &self.last {
                    Last::Tcp { r, .. } if pending == Mode::Mpc => r.pose,
                    _ => s.tcp_pose,
                };
                let pose = resolve_tcp(&start, &goal);
                Ok((Active::Mpc { pose, posture: s.q.clone(), t0: t }, None))
            }
            MotionCmd::WaypointsTcp { points, times, speed } => {
                let (a, posture) = self.tcp_anchor(s, osc_mode(pending));
                let mut prev = a.pose;
                let mut ends = Vec::new();
                for p in &points {
                    prev = resolve_tcp(&prev, p);
                    ends.push(prev);
                }
                check_times(times.as_deref(), ends.len())?;
                let (traj, stretched) = self.tcp_traj(&a, &ends, times.as_deref(), None, speed);
                Ok((
                    Active::Tcp { traj, t0: t, r0: a.pose.rotation, posture },
                    notes(&[(stretched, "slowed down to keep the limits")]),
                ))
            }
            MotionCmd::JointStream { v, a, timeout } => {
                let r = self.joint_anchor(s, joint_mode(pending));
                Ok((Active::JointStream { v, a, last_cmd: t, timeout, r }, None))
            }
            MotionCmd::TcpStream { twist, accel, frame, timeout } => {
                let (r, posture) = self.tcp_anchor(s, osc_mode(pending));
                Ok((Active::TcpStream { twist, accel, frame, last_cmd: t, timeout, r, posture }, None))
            }
            other => Err(format!("#{id}: {} is not a motion", other.kind())),
        }
    }

    /// Joint values of `goal` from `base`, clamped into range (gripper
    /// values go to the gripper channel).
    fn resolve_joint(&mut self, arm: &ArmModel, base: &DVector<f64>, goal: &JointGoal) -> (DVector<f64>, bool) {
        let mut q = base.clone();
        let mut clamped = false;
        for &(i, x) in &goal.values {
            let want = if goal.relative { base[i] + x } else { x };
            if Some(i) == self.cfg.gripper {
                if let Some(g) = &mut self.gripper {
                    g.goal = arm.dofs()[i].clamp(want);
                }
                continue;
            }
            let c = arm.dofs()[i].clamp(want);
            clamped |= c != want;
            q[i] = c;
        }
        (q, clamped)
    }

    fn joint_traj(&self, a: &JointRef, ends: &[DVector<f64>], times: Option<&[f64]>, duration: Option<f64>, speed: f64) -> (Traj, bool) {
        let speed = speed.clamp(0.01, 1.0);
        let vm = &self.cfg.v_max * speed;
        let am = &self.cfg.a_max * speed;
        let ex = box_excess(&vm, &am);
        let start = Knot { p: a.q.clone(), v: a.v.clone(), a: a.a.clone() };
        Traj::through(start, ends, times, duration, &vm, &am, &ex)
    }

    /// A TCP trajectory in the coordinates `[log(R·R0⁻¹); p]`.
    fn tcp_traj(&self, a: &TcpRefState, ends: &[Isometry3<f64>], times: Option<&[f64]>, duration: Option<f64>, speed: f64) -> (Traj, bool) {
        let speed = speed.clamp(0.01, 1.0);
        let c = &self.cfg;
        let (lv, la, av, aa) = (c.lin_v_max * speed, c.lin_a_max * speed, c.ang_v_max * speed, c.ang_a_max * speed);
        let vm = DVector::from_vec(vec![av, av, av, lv, lv, lv]);
        let am = DVector::from_vec(vec![aa, aa, aa, la, la, la]);
        let ex = move |v: &DVector<f64>, acc: &DVector<f64>| {
            let n = |x: &DVector<f64>, r: usize| x.rows(r, 3).norm();
            (n(v, 0) / av).max(n(v, 3) / lv).max((n(acc, 0) / aa).sqrt()).max((n(acc, 3) / la).sqrt())
        };
        let r0 = a.pose.rotation;
        let coords = |p: &Isometry3<f64>| {
            let r = (p.rotation * r0.inverse()).scaled_axis();
            let x = p.translation.vector;
            DVector::from_vec(vec![r.x, r.y, r.z, x.x, x.y, x.z])
        };
        let start = Knot {
            p: coords(&a.pose),
            v: DVector::from_column_slice(a.twist.as_slice()),
            a: DVector::from_column_slice(a.accel.as_slice()),
        };
        let pts: Vec<DVector<f64>> = ends.iter().map(coords).collect();
        Traj::through(start, &pts, times, duration, &vm, &am, &ex)
    }
}

/// TCP pose `goal` resolved against `from`.
pub fn resolve_tcp(from: &Isometry3<f64>, goal: &TcpGoal) -> Isometry3<f64> {
    let p0 = from.translation.vector;
    let r0 = from.rotation;
    let (p, r) = match (goal.frame, goal.relative) {
        (Frame::Base, false) => (goal.position.unwrap_or(p0), goal.rotation.unwrap_or(r0)),
        (Frame::Base, true) => (p0 + goal.position.unwrap_or_else(Vector3::zeros), goal.rotation.unwrap_or_else(UnitQuaternion::identity) * r0),
        (Frame::Tool, _) => (p0 + r0 * goal.position.unwrap_or_else(Vector3::zeros), r0 * goal.rotation.unwrap_or_else(UnitQuaternion::identity)),
    };
    Isometry3::from_parts(Translation3::from(p), r)
}

fn check_times(times: Option<&[f64]>, n: usize) -> Result<(), String> {
    if let Some(ts) = times {
        if ts.len() != n {
            return Err(format!("{} times for {n} points", ts.len()));
        }
        if ts.windows(2).any(|w| w[1] <= w[0]) || ts.first().is_some_and(|&t| t <= 0.0) {
            return Err("times must be increasing and positive".into());
        }
    }
    Ok(())
}

fn notes(items: &[(bool, &str)]) -> Option<String> {
    let v: Vec<&str> = items.iter().filter(|(on, _)| *on).map(|(_, s)| *s).collect();
    (!v.is_empty()).then(|| v.join("; "))
}

/// Left Jacobian of SO(3) at the rotation vector `r`: `ω = J_l(r)·ṙ` for
/// `R = exp(r)·R0`.
fn left_jacobian(r: &Vector3<f64>) -> Matrix3<f64> {
    let th = r.norm();
    let k = r.cross_matrix();
    if th < 1e-6 {
        return Matrix3::identity() + 0.5 * k;
    }
    Matrix3::identity() + (1.0 - th.cos()) / (th * th) * k + (th - th.sin()) / (th * th * th) * k * k
}

fn block_rotate(r: &UnitQuaternion<f64>, m: &Matrix6<f64>) -> Matrix6<f64> {
    let rm = r.to_rotation_matrix().into_inner();
    let mut b = Matrix6::zeros();
    b.fixed_view_mut::<3, 3>(0, 0).copy_from(&rm);
    b.fixed_view_mut::<3, 3>(3, 3).copy_from(&rm);
    b * m * b.transpose()
}

fn rotate6(r: &UnitQuaternion<f64>, x: &Vector6<f64>) -> Vector6<f64> {
    let a = r * Vector3::new(x[0], x[1], x[2]);
    let b = r * Vector3::new(x[3], x[4], x[5]);
    Vector6::new(a.x, a.y, a.z, b.x, b.y, b.z)
}

// ── The per-cycle target ────────────────────────────────────────────────
impl Executive {
    fn target(&mut self, arm: &ArmModel, t: f64, s: &ArmState, mode: Mode) -> Target {
        let dt = self.cfg.dt;
        let driving = matches!(mode, Mode::Joint | Mode::Osc | Mode::Mpc);
        self.ramp_settings(dt);
        let gripper = self.step_gripper(arm, s, driving);
        if !driving {
            return Target::None;
        }
        if self.teleop_on {
            let Some(tp) = &self.teleop else { return Target::None };
            return match tp.thread.latest() {
                Some(sample) if sample.at.elapsed() <= tp.timeout => Target::Joint(tp.mapping.map(&sample.q, &s.q)),
                _ => Target::None,
            };
        }
        match mode {
            Mode::Joint => {
                let mut r = self.joint_reference(t, s);
                self.last = Last::Joint(r.clone());
                if let (Some(g), Some(out)) = (self.cfg.gripper, &gripper) {
                    r.q[g] = out.q[0];
                    r.v[g] = out.v[0];
                    r.a[g] = out.a[0];
                }
                let extras = self.joint_extras(s);
                Target::JointRef { r, extras }
            }
            Mode::Osc => {
                let (mut r, mut posture) = self.tcp_reference(t, s);
                let extras = self.tcp_extras(s, &mut r);
                self.last = Last::Tcp { r: r.clone(), posture: posture.clone() };
                if let (Some(g), Some(out)) = (self.cfg.gripper, &gripper) {
                    posture[g] = out.q[0];
                }
                Target::TcpRef { tcp: TcpRef { pose: r.pose, twist: r.twist, accel: r.accel }, posture, extras: Box::new(extras) }
            }
            Mode::Mpc => {
                let (pose, mut posture) = match &self.active {
                    Some((id, Active::Mpc { pose, posture, t0 })) => {
                        let (id, pose, posture, t0) = (*id, *pose, posture.clone(), *t0);
                        let e = manip_wbc::pose_error(&pose, &s.tcp_pose);
                        let (tp, tr) = self.cfg.mpc_tolerance;
                        let at_rest = s.tcp_twist.norm() < 0.02;
                        if e.fixed_rows::<3>(3).norm() < tp && e.fixed_rows::<3>(0).norm() < tr && at_rest {
                            self.set(id, State::Done, Some(format!("reached within {:.1} mm", e.fixed_rows::<3>(3).norm() * 1e3)));
                            self.active = None;
                        } else if t - t0 > self.cfg.mpc_timeout_s {
                            self.set(id, State::Aborted, Some(format!("not reached after {:.0} s ({:.1} mm away)", self.cfg.mpc_timeout_s, e.fixed_rows::<3>(3).norm() * 1e3)));
                            self.active = None;
                        }
                        (pose, posture)
                    }
                    _ => match &self.last {
                        Last::Tcp { r, posture } => (r.pose, posture.clone()),
                        _ => (s.tcp_pose, s.q.clone()),
                    },
                };
                self.last = Last::Tcp { r: TcpRefState { pose, twist: Vector6::zeros(), accel: Vector6::zeros() }, posture: posture.clone() };
                if let (Some(g), Some(out)) = (self.cfg.gripper, &gripper) {
                    posture[g] = out.q[0];
                }
                Target::Tcp { pose, posture }
            }
            _ => Target::None,
        }
    }

    fn ramp_settings(&mut self, dt: f64) {
        if let Some(f) = &mut self.force {
            let step = dt / f.spec.ramp_s.max(dt);
            f.level = if f.clearing { (f.level - step).max(0.0) } else { (f.level + step).min(1.0) };
            if f.clearing && f.level == 0.0 {
                self.force = None;
            }
        }
        if let Some(j) = &mut self.joint_torque {
            let step = dt / j.spec.ramp_s.max(dt);
            j.level = if j.clearing { (j.level - step).max(0.0) } else { (j.level + step).min(1.0) };
            if j.clearing && j.level == 0.0 {
                self.joint_torque = None;
            }
        }
    }

    /// The gripper's reference this cycle (follows the measurement while the
    /// arm is not driven by motions).
    fn step_gripper(&mut self, arm: &ArmModel, s: &ArmState, driving: bool) -> Option<JointRef> {
        let g = self.gripper.as_mut()?;
        let dof = g.dof;
        if !driving || self.teleop_on {
            if let Some(id) = g.id.take() {
                self.board.lock().unwrap().set(id, State::Aborted, Some("the arm is not driven by motion commands".into()));
            }
            let q = s.q[dof];
            g.shaper.reset(DVector::from_element(1, q));
            g.goal = q;
            g.out = JointRef::at_rest(DVector::from_element(1, q));
            return None;
        }
        let r = g.shaper.step(&DVector::from_element(1, g.goal), self.cfg.dt).clone();
        let mut out = r.clone();
        if let Some(f) = g.max_force {
            // Hold the reference within F/kp of the finger, so the position
            // loop squeezes with at most F.
            let lead = f / self.cfg.track_kp[dof].max(1e-9);
            let m = s.q[dof];
            let c = r.q[0].clamp(m - lead, m + lead);
            if c != r.q[0] {
                out = JointRef::at_rest(DVector::from_element(1, c));
            }
        }
        g.out = out.clone();
        // Done once the reference has arrived and the finger either has too
        // or has stopped (an object, or the force limit, holds it back).
        g.still_s = if s.v[dof].abs() < 1e-3 { g.still_s + self.cfg.dt } else { 0.0 };
        let arrived = (r.q[0] - g.goal).abs() < 1e-6 && r.v[0].abs() < 1e-6;
        if let Some(id) = g.id
            && arrived
            && ((s.q[dof] - g.goal).abs() < 1e-3 || g.still_s > 0.3)
        {
            let m = s.q[dof];
            let msg = if (m - g.goal).abs() > 0.002 {
                format!("holding at {m:.4} (blocked {:.1} mm short)", (g.goal - m).abs() * 1e3)
            } else {
                format!("at {m:.4}")
            };
            g.id = None;
            self.board.lock().unwrap().set(id, State::Done, Some(msg));
        }
        let _ = arm;
        Some(out)
    }

    /// Joint torque offsets for this cycle. A joint within [`LIMIT_GUARD`]
    /// of a limit gets no torque toward it: with little stiffness left the
    /// torque alone would drive it into the stop.
    fn joint_extras(&self, s: &ArmState) -> JointExtras {
        let Some(j) = &self.joint_torque else { return JointExtras::default() };
        let scale = 1.0 - j.level * (1.0 - j.spec.stiffness_scale);
        let mut kp = &self.cfg.track_kp * scale;
        if let Some(g) = self.cfg.gripper {
            kp[g] = self.cfg.track_kp[g];
        }
        let mut torque = &j.spec.torque * j.level;
        for (i, &(lo, hi)) in self.cfg.joint_range.iter().enumerate() {
            if (torque[i] > 0.0 && s.q[i] > hi - LIMIT_GUARD) || (torque[i] < 0.0 && s.q[i] < lo + LIMIT_GUARD) {
                torque[i] = 0.0;
            }
        }
        JointExtras { torque: Some(torque), kp: Some(kp), kd: None }
    }

    /// This cycle's joint reference, finishing the motion that produced it.
    fn joint_reference(&mut self, t: f64, s: &ArmState) -> JointRef {
        let hold = match &self.last {
            Last::Joint(r) => JointRef::at_rest(r.q.clone()),
            _ => JointRef::at_rest(s.q.clone()),
        };
        let dt = self.cfg.dt;
        let (vmax, amax, margin) = (self.cfg.v_max.clone(), self.cfg.a_max.clone(), self.cfg.q_margin);
        let range = self.cfg.joint_range.clone();
        let Some((id, active)) = &mut self.active else { return hold };
        let id = *id;
        match active {
            Active::Joint { traj, t0 } => {
                let k = traj.sample(t - *t0);
                let done = t - *t0 >= traj.duration();
                let r = JointRef { q: k.p, v: k.v, a: k.a };
                if done {
                    self.active = None;
                    self.set(id, State::Done, None);
                }
                r
            }
            Active::JointStream { v, a, last_cmd, timeout, r } => {
                let fresh = t - *last_cmd <= *timeout;
                let n = r.q.len();
                let mut moving = false;
                for i in 0..n {
                    let vt = match (fresh, v.as_ref(), a.as_ref()) {
                        (true, Some(v), _) => v[i],
                        (true, None, Some(a)) => r.v[i] + a[i] * dt,
                        _ => 0.0,
                    }
                    .clamp(-vmax[i], vmax[i]);
                    let (lo, hi) = range[i];
                    let stop = r.v[i] * r.v[i].abs() / (2.0 * amax[i].max(1e-9));
                    let vt = if (vt > 0.0 && r.q[i] + stop.max(0.0) >= hi - margin) || (vt < 0.0 && r.q[i] + stop.min(0.0) <= lo + margin) {
                        0.0
                    } else {
                        vt
                    };
                    let dv = (vt - r.v[i]).clamp(-amax[i] * dt, amax[i] * dt);
                    let v1 = r.v[i] + dv;
                    r.q[i] += 0.5 * (r.v[i] + v1) * dt;
                    r.a[i] = dv / dt;
                    r.v[i] = v1;
                    moving |= v1 != 0.0;
                }
                let out = r.clone();
                if !fresh && !moving {
                    let why = if self.stopping == Some(id) { "stopped" } else { "no command within the timeout: stopped" };
                    self.active = None;
                    if self.stopping == Some(id) {
                        self.stopping = None;
                    }
                    self.set(id, State::Done, Some(why.into()));
                    return JointRef::at_rest(out.q);
                }
                out
            }
            _ => hold,
        }
    }

    /// This cycle's TCP reference and posture.
    fn tcp_reference(&mut self, t: f64, s: &ArmState) -> (TcpRefState, DVector<f64>) {
        let hold = match &self.last {
            Last::Tcp { r, posture } => (TcpRefState { pose: r.pose, twist: Vector6::zeros(), accel: Vector6::zeros() }, posture.clone()),
            _ => (TcpRefState { pose: s.tcp_pose, twist: Vector6::zeros(), accel: Vector6::zeros() }, s.q.clone()),
        };
        let c = self.cfg.clone();
        let dt = c.dt;
        let Some((id, active)) = &mut self.active else { return hold };
        let id = *id;
        match active {
            Active::Tcp { traj, t0, r0, posture } => {
                let k = traj.sample(t - *t0);
                let done = t - *t0 >= traj.duration();
                let rv = Vector3::new(k.p[0], k.p[1], k.p[2]);
                let rd = Vector3::new(k.v[0], k.v[1], k.v[2]);
                let rdd = Vector3::new(k.a[0], k.a[1], k.a[2]);
                let jl = left_jacobian(&rv);
                let w = jl * rd;
                // α = J_l·r̈ + J̇_l·ṙ (J̇_l by a central difference along ṙ).
                let h = 1e-4;
                let jdot = (left_jacobian(&(rv + rd * h)) - left_jacobian(&(rv - rd * h))) / (2.0 * h);
                let alpha = jl * rdd + jdot * rd;
                let r = TcpRefState {
                    pose: Isometry3::from_parts(
                        Translation3::new(k.p[3], k.p[4], k.p[5]),
                        UnitQuaternion::from_scaled_axis(rv) * *r0,
                    ),
                    twist: Vector6::new(w.x, w.y, w.z, k.v[3], k.v[4], k.v[5]),
                    accel: Vector6::new(alpha.x, alpha.y, alpha.z, k.a[3], k.a[4], k.a[5]),
                };
                let posture = posture.clone();
                if done {
                    self.active = None;
                    let e = manip_wbc::pose_error(&r.pose, &s.tcp_pose);
                    self.set(id, State::Done, Some(format!("TCP {:.1} mm, {:.2}° from the goal", e.fixed_rows::<3>(3).norm() * 1e3, e.fixed_rows::<3>(0).norm().to_degrees())));
                }
                (r, posture)
            }
            Active::TcpStream { twist, accel, frame, last_cmd, timeout, r, posture } => {
                let fresh = t - *last_cmd <= *timeout;
                let rot = r.pose.rotation;
                let to_world = |x: &Vector6<f64>| if *frame == Frame::Tool { rotate6(&rot, x) } else { *x };
                let (w0, v0) = (r.twist.fixed_rows::<3>(0).into_owned(), r.twist.fixed_rows::<3>(3).into_owned());
                let target = match (fresh, twist.as_ref(), accel.as_ref()) {
                    (true, Some(tw), _) => to_world(tw),
                    (true, None, Some(acc)) => r.twist + to_world(acc) * dt,
                    _ => Vector6::zeros(),
                };
                let limit = |x: Vector3<f64>, max: f64| if x.norm() > max { x * (max / x.norm()) } else { x };
                let wt = limit(target.fixed_rows::<3>(0).into_owned(), c.ang_v_max);
                let vt = limit(target.fixed_rows::<3>(3).into_owned(), c.lin_v_max);
                let w1 = w0 + limit(wt - w0, c.ang_a_max * dt);
                let v1 = v0 + limit(vt - v0, c.lin_a_max * dt);
                let mut p = r.pose.translation.vector + 0.5 * (v0 + v1) * dt;
                let mut q = UnitQuaternion::from_scaled_axis(0.5 * (w0 + w1) * dt) * rot;
                // Keep the reference near the arm (a wall or a joint limit
                // stops the arm; the reference must not run on beyond it).
                let (lead_p, lead_r) = c.stream_lead;
                let dp = p - s.tcp_pose.translation.vector;
                if dp.norm() > lead_p {
                    p = s.tcp_pose.translation.vector + dp * (lead_p / dp.norm());
                }
                let dr = (q * s.tcp_pose.rotation.inverse()).scaled_axis();
                if dr.norm() > lead_r {
                    q = UnitQuaternion::from_scaled_axis(dr * (lead_r / dr.norm())) * s.tcp_pose.rotation;
                }
                r.pose = Isometry3::from_parts(Translation3::from(p), q);
                let (a_w, a_v) = ((w1 - w0) / dt, (v1 - v0) / dt);
                r.twist = Vector6::new(w1.x, w1.y, w1.z, v1.x, v1.y, v1.z);
                r.accel = Vector6::new(a_w.x, a_w.y, a_w.z, a_v.x, a_v.y, a_v.z);
                let out = (r.clone(), posture.clone());
                if !fresh && r.twist.norm() == 0.0 {
                    let why = if self.stopping == Some(id) { "stopped" } else { "no command within the timeout: stopped" };
                    if self.stopping == Some(id) {
                        self.stopping = None;
                    }
                    self.active = None;
                    self.set(id, State::Done, Some(why.into()));
                }
                out
            }
            _ => hold,
        }
    }

    /// Force / impedance for this cycle; with a force on free axes, the
    /// reference slides with the arm along them (so it holds where the arm
    /// ends up once the force is gone).
    fn tcp_extras(&mut self, s: &ArmState, r: &mut TcpRefState) -> TcpExtras {
        let mut extras = TcpExtras::default();
        let rot_ref = r.pose.rotation;
        let frame_rot = |f: Frame| if f == Frame::Tool { rot_ref } else { UnitQuaternion::identity() };
        if let Some(k) = &self.impedance {
            let fr = frame_rot(k.frame);
            extras.compliance = Some(Compliance {
                stiffness: block_rotate(&fr, &Matrix6::from_diagonal(&k.stiffness)),
                damping: k.damping.map(|d| block_rotate(&fr, &Matrix6::from_diagonal(&d))),
            });
        }
        if let Some(f) = &self.force {
            let fr = frame_rot(f.spec.frame);
            let sel = Vector6::from_fn(|i, _| if f.spec.free[i] { 1.0 } else { 0.0 });
            let p = block_rotate(&fr, &Matrix6::from_diagonal(&sel));
            // Speed along the free axes: reduce the force above the limit.
            let v = p * s.tcp_twist;
            let over_lin = v.fixed_rows::<3>(3).norm() / f.spec.max_speed.max(1e-6);
            let over_ang = v.fixed_rows::<3>(0).norm() / f.spec.max_ang_speed.max(1e-6);
            let slow = (2.0 - over_lin.max(over_ang)).clamp(0.0, 1.0);
            extras.wrench = Some(rotate6(&fr, &f.spec.wrench) * (f.level * slow));
            if self.impedance.is_none() {
                extras.free = Some(p);
                let pl = p.fixed_view::<3, 3>(3, 3).into_owned();
                let pa = p.fixed_view::<3, 3>(0, 0).into_owned();
                let dp = s.tcp_pose.translation.vector - r.pose.translation.vector;
                let x = r.pose.translation.vector + pl * dp;
                let dr = (s.tcp_pose.rotation * r.pose.rotation.inverse()).scaled_axis();
                let q = UnitQuaternion::from_scaled_axis(pa * dr) * r.pose.rotation;
                r.pose = Isometry3::from_parts(Translation3::from(x), q);
                r.twist -= p * r.twist;
                r.accel -= p * r.accel;
            }
        }
        extras
    }
}

/// What the executive reports in the state.
#[derive(Debug, Clone, Serialize)]
pub struct ExecReport {
    pub active: Option<ActiveReport>,
    pub queued: usize,
    pub teleop: bool,
    pub force: Option<ForceReport>,
    pub impedance: Option<ImpedanceReport>,
    pub joint_torque: Option<Vec<f64>>,
    pub gripper_goal: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveReport {
    pub id: MotionId,
    pub kind: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForceReport {
    pub wrench: Vec<f64>,
    pub frame: Frame,
    pub level: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImpedanceReport {
    pub stiffness: Vec<f64>,
    pub damping: Option<Vec<f64>>,
    pub frame: Frame,
}
