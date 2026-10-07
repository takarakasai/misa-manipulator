//! HTTP/JSON API: motion commands in, state and command status out.
//!
//! A small synchronous server (`tiny_http`) on its own thread. Requests are
//! checked and turned into [`MotionCmd`]s here, so a malformed command is
//! refused with a message before it reaches the control loop; accepted ones
//! go over a channel to the [`crate::motion::Executive`], which runs them in
//! the loop. Every command gets an id whose status (`queued`, `active`,
//! `done`, `aborted`, `rejected`) is kept on a shared board;
//! `?wait=true` blocks until it finishes. The state is a snapshot the loop
//! publishes every cycle.
//!
//! Listening anywhere but loopback requires a token (`Authorization: Bearer
//! <token>`). See `doc/api.md` for the endpoints.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use manip_model::{ArmModel, ArmState, DofKind};
use nalgebra::{DVector, UnitQuaternion, Vector3, Vector6};
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::motion::{
    ExecReport, ForceSpec, Frame, ImpedanceSpec, JointGoal, JointTorqueSpec, MotionCmd, MotionStatus, Queued, SharedBoard,
    TcpGoal, Via,
};
use crate::supervisor::Mode;

/// Default timeouts and speeds.
const DEFAULT_SPEED: f64 = 0.5;
const DEFAULT_STREAM_TIMEOUT: f64 = 0.2;
const DEFAULT_RAMP: f64 = 0.3;
const DEFAULT_FORCE_MAX_SPEED: f64 = 0.05;
const DEFAULT_FORCE_MAX_ANG_SPEED: f64 = 0.5;

/// What the API knows about the arm (to check and convert requests).
#[derive(Debug, Clone, Serialize)]
pub struct ArmInfo {
    pub robot: String,
    pub joints: Vec<JointInfo>,
    pub tcp_link: String,
    pub gripper: Option<String>,
    pub poses: BTreeMap<String, Vec<f64>>,
    pub teleop: bool,
    pub mpc: bool,
    pub rate_hz: f64,
    #[serde(skip)]
    gripper_idx: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JointInfo {
    pub name: String,
    /// "revolute" (rad) or "prismatic" (m).
    pub kind: &'static str,
    pub min: f64,
    pub max: f64,
    pub v_max: f64,
    pub effort: f64,
}

impl ArmInfo {
    pub fn new(
        robot: &str,
        arm: &ArmModel,
        gripper: Option<usize>,
        poses: BTreeMap<String, Vec<f64>>,
        teleop: bool,
        mpc: bool,
        rate_hz: f64,
    ) -> Self {
        Self {
            robot: robot.into(),
            joints: arm
                .dofs()
                .iter()
                .map(|d| JointInfo {
                    name: d.name.clone(),
                    kind: match d.kind {
                        DofKind::Revolute => "revolute",
                        DofKind::Prismatic => "prismatic",
                    },
                    min: d.q_min,
                    max: d.q_max,
                    v_max: d.v_max,
                    effort: d.effort,
                })
                .collect(),
            tcp_link: arm.tcp_spec().link.clone(),
            gripper: gripper.map(|g| arm.dofs()[g].name.clone()),
            poses,
            teleop,
            mpc,
            rate_hz,
            gripper_idx: gripper,
        }
    }

    fn n(&self) -> usize {
        self.joints.len()
    }
}

// ── State snapshot ──────────────────────────────────────────────────────

/// What `GET /v1/state` returns.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Snapshot {
    /// Seconds since the run started.
    pub t: f64,
    pub mode: String,
    pub q: Vec<f64>,
    pub v: Vec<f64>,
    /// Torque each motor reports [N·m or N] (null where it reports none).
    pub tau_measured: Vec<Option<f64>>,
    pub tcp: TcpState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gripper: Option<GripperState>,
    /// Rough wrench the TCP exerts on its surroundings, `[mx, my, mz, fx,
    /// fy, fz]` (base frame), from the reported torques minus gravity: no
    /// friction or dynamics, so only meaningful at rest, and only to within
    /// the joint friction (~2 N·m on the B601 shoulder/elbow).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrench_estimate: Option<Vec<f64>>,
    pub motion: Option<ExecReport>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TcpState {
    pub position: [f64; 3],
    /// `[x, y, z, w]`.
    pub quat: [f64; 4],
    /// Roll, pitch, yaw [rad] (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`).
    pub rpy: [f64; 3],
    /// `[ωx, ωy, ωz, vx, vy, vz]`.
    pub twist: [f64; 6],
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GripperState {
    /// Finger displacement [m] (one finger; the other mirrors it).
    pub position: f64,
    /// Distance between the fingers [m] (2 × position).
    pub width: f64,
}

pub type SharedSnapshot = Arc<Mutex<Snapshot>>;

impl Snapshot {
    pub fn build(t: f64, mode: Mode, s: &ArmState, obs: &misa_core::Observation, info: &ArmInfo, motion: Option<ExecReport>) -> Self {
        let p = s.tcp_pose.translation.vector;
        let c = s.tcp_pose.rotation.coords;
        let (r, pi, y) = s.tcp_pose.rotation.euler_angles();
        let tau: Vec<Option<f64>> = obs.axes().iter().map(|a| a.torque_nm).collect();
        let wrench_estimate = if tau.iter().all(|x| x.is_some()) {
            let ext = DVector::from_iterator(tau.len(), tau.iter().map(|x| x.unwrap())) - &s.gravity;
            let jt = s.tcp_jacobian.transpose();
            jt.svd(true, true).solve(&ext, 1e-6).ok().map(|w| w.as_slice().to_vec())
        } else {
            None
        };
        Self {
            t,
            mode: format!("{mode:?}").to_lowercase(),
            q: s.q.as_slice().to_vec(),
            v: s.v.as_slice().to_vec(),
            tau_measured: tau,
            tcp: TcpState {
                position: [p.x, p.y, p.z],
                quat: [c.x, c.y, c.z, c.w],
                rpy: [r, pi, y],
                twist: s.tcp_twist.into(),
            },
            gripper: info.gripper_idx.map(|g| GripperState { position: s.q[g], width: 2.0 * s.q[g] }),
            wrench_estimate,
            motion,
        }
    }
}

// ── Request parsing ─────────────────────────────────────────────────────

type Obj = Map<String, Value>;

fn allow(o: &Obj, keys: &[&str]) -> Result<(), String> {
    for k in o.keys() {
        if !keys.contains(&k.as_str()) && k != "queue" {
            return Err(format!("unknown field \"{k}\" (allowed: {})", keys.join(", ")));
        }
    }
    Ok(())
}

fn num(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{what}: expected a number, got {v}"))
}

fn opt_num(o: &Obj, k: &str) -> Result<Option<f64>, String> {
    o.get(k).filter(|v| !v.is_null()).map(|v| num(v, k)).transpose()
}

fn positive(o: &Obj, k: &str) -> Result<Option<f64>, String> {
    match opt_num(o, k)? {
        Some(x) if x <= 0.0 => Err(format!("{k} must be > 0")),
        x => Ok(x),
    }
}

fn flag(o: &Obj, k: &str) -> Result<bool, String> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(format!("{k}: expected true/false, got {v}")),
    }
}

fn vec3(v: &Value, what: &str) -> Result<Vector3<f64>, String> {
    let a = v.as_array().filter(|a| a.len() == 3).ok_or_else(|| format!("{what}: expected [x, y, z]"))?;
    Ok(Vector3::new(num(&a[0], what)?, num(&a[1], what)?, num(&a[2], what)?))
}

/// `[x, y, z]` or a single number for all three.
fn vec3_or_scalar(v: &Value, what: &str) -> Result<Vector3<f64>, String> {
    match v.as_f64() {
        Some(x) => Ok(Vector3::from_element(x)),
        None => vec3(v, what),
    }
}

fn frame(o: &Obj) -> Result<Frame, String> {
    match o.get("frame").and_then(|v| v.as_str()) {
        None => Ok(Frame::Base),
        Some("base") => Ok(Frame::Base),
        Some("tool") => Ok(Frame::Tool),
        Some(other) => Err(format!("frame: \"{other}\" (base | tool)")),
    }
}

fn speed(o: &Obj) -> Result<f64, String> {
    match opt_num(o, "speed")? {
        None => Ok(DEFAULT_SPEED),
        Some(s) if s > 0.0 && s <= 1.0 => Ok(s),
        Some(s) => Err(format!("speed {s}: a fraction of the limits, 0 < speed ≤ 1")),
    }
}

impl ArmInfo {
    fn dof(&self, name: &str) -> Result<usize, String> {
        self.joints
            .iter()
            .position(|j| j.name == name)
            .ok_or_else(|| format!("no joint \"{name}\" (joints: {})", self.joints.iter().map(|j| j.name.as_str()).collect::<Vec<_>>().join(", ")))
    }

    /// Joint values: an array (all joints, or the TCP chain when one shorter
    /// and there is a gripper) or `{name: value}`. `deg` converts revolute
    /// joints (prismatic ones stay in m).
    fn joint_values(&self, v: &Value, deg: bool, what: &str) -> Result<Vec<(usize, f64)>, String> {
        let conv = |i: usize, x: f64| if deg && self.joints[i].kind == "revolute" { x.to_radians() } else { x };
        match v {
            Value::Array(a) => {
                let n = self.n();
                let idx: Vec<usize> = if a.len() == n {
                    (0..n).collect()
                } else if self.gripper_idx.is_some() && a.len() == n - 1 {
                    (0..n).filter(|&i| Some(i) != self.gripper_idx).collect()
                } else {
                    return Err(format!("{what}: {} values for {n} joints", a.len()));
                };
                idx.iter().zip(a).map(|(&i, x)| Ok((i, conv(i, num(x, what)?)))).collect()
            }
            Value::Object(m) => m.iter().map(|(k, x)| {
                let i = self.dof(k)?;
                Ok((i, conv(i, num(x, what)?)))
            }).collect(),
            other => Err(format!("{what}: expected an array or {{joint: value}}, got {other}")),
        }
    }

    fn joint_vector(&self, v: &Value, deg: bool, what: &str) -> Result<DVector<f64>, String> {
        let mut out = DVector::zeros(self.n());
        for (i, x) in self.joint_values(v, deg, what)? {
            out[i] = x;
        }
        Ok(out)
    }

    fn rotation(&self, o: &Obj, deg: bool) -> Result<Option<UnitQuaternion<f64>>, String> {
        let a = |x: f64| if deg { x.to_radians() } else { x };
        match (o.get("quat"), o.get("rpy"), o.get("axis_angle")) {
            (None, None, None) => Ok(None),
            (Some(q), None, None) => {
                let q = q.as_array().filter(|q| q.len() == 4).ok_or("quat: expected [x, y, z, w]")?;
                let q: Vec<f64> = q.iter().map(|x| num(x, "quat")).collect::<Result<_, _>>()?;
                let n = (q.iter().map(|x| x * x).sum::<f64>()).sqrt();
                if n < 1e-6 {
                    return Err("quat: zero".into());
                }
                Ok(Some(UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))))
            }
            (None, Some(r), None) => {
                let r = vec3(r, "rpy")?;
                Ok(Some(UnitQuaternion::from_euler_angles(a(r.x), a(r.y), a(r.z))))
            }
            (None, None, Some(r)) => {
                let r = vec3(r, "axis_angle")?;
                Ok(Some(UnitQuaternion::from_scaled_axis(r.map(a))))
            }
            _ => Err("give one of quat, rpy, axis_angle".into()),
        }
    }

    fn tcp_goal(&self, o: &Obj, frame: Frame, relative: bool, deg: bool) -> Result<TcpGoal, String> {
        if frame == Frame::Tool && !relative {
            return Err("frame \"tool\" moves relative to the TCP: add \"relative\": true".into());
        }
        let position = o.get("position").map(|p| vec3(p, "position")).transpose()?;
        let rotation = self.rotation(o, deg)?;
        if position.is_none() && rotation.is_none() {
            return Err("give a position and/or an orientation (quat, rpy, axis_angle)".into());
        }
        Ok(TcpGoal { position, rotation, frame, relative })
    }

    /// `[ang; lin]` from `{"linear": [..], "angular": [..]}` (either may be missing).
    fn six(&self, o: &Obj, lin: &str, ang: &str, deg: bool) -> Result<Vector6<f64>, String> {
        let l = o.get(lin).map(|v| vec3(v, lin)).transpose()?.unwrap_or_else(Vector3::zeros);
        let mut a = o.get(ang).map(|v| vec3(v, ang)).transpose()?.unwrap_or_else(Vector3::zeros);
        if deg {
            a = a.map(f64::to_radians);
        }
        Ok(Vector6::new(a.x, a.y, a.z, l.x, l.y, l.z))
    }

    /// The command for `POST /v1/<path>` with JSON `body`, and whether it
    /// queues behind the running motion.
    pub fn parse(&self, path: &str, body: &Value) -> Result<(MotionCmd, bool), String> {
        let empty = Map::new();
        let o = match body {
            Value::Object(m) => m,
            Value::Null => &empty,
            other => return Err(format!("body: expected a JSON object, got {other}")),
        };
        let queue = flag(o, "queue")?;
        let deg = flag(o, "deg")?;
        let cmd = match path {
            "move/joint" => {
                allow(o, &["q", "relative", "deg", "duration", "speed"])?;
                let q = o.get("q").ok_or("missing \"q\"")?;
                MotionCmd::MoveJoint {
                    goal: JointGoal { values: self.joint_values(q, deg, "q")?, relative: flag(o, "relative")? },
                    duration: positive(o, "duration")?,
                    speed: speed(o)?,
                }
            }
            "move/pose" => {
                allow(o, &["name", "duration", "speed"])?;
                let name = o.get("name").and_then(|v| v.as_str()).ok_or("missing \"name\"")?;
                let q = self.poses.get(name).ok_or_else(|| {
                    format!("no pose \"{name}\" (poses: {})", self.poses.keys().cloned().collect::<Vec<_>>().join(", "))
                })?;
                MotionCmd::MoveJoint {
                    goal: JointGoal {
                        values: q.iter().enumerate().filter(|(i, _)| Some(*i) != self.gripper_idx).map(|(i, &x)| (i, x)).collect(),
                        relative: false,
                    },
                    duration: positive(o, "duration")?,
                    speed: speed(o)?,
                }
            }
            "move/tcp" => {
                allow(o, &["position", "quat", "rpy", "axis_angle", "frame", "relative", "deg", "duration", "speed", "via"])?;
                let via = match o.get("via").and_then(|v| v.as_str()) {
                    None | Some("osc") => Via::Osc,
                    Some("mpc") if self.mpc => Via::Mpc,
                    Some("mpc") => return Err("no MPC planner in this run".into()),
                    Some(other) => return Err(format!("via: \"{other}\" (osc | mpc)")),
                };
                MotionCmd::MoveTcp {
                    goal: self.tcp_goal(o, frame(o)?, flag(o, "relative")?, deg)?,
                    duration: positive(o, "duration")?,
                    speed: speed(o)?,
                    via,
                }
            }
            "waypoints/joint" => {
                allow(o, &["points", "times", "relative", "deg", "speed"])?;
                let pts = o.get("points").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or("missing \"points\" (a non-empty array)")?;
                let relative = flag(o, "relative")?;
                let points = pts
                    .iter()
                    .enumerate()
                    .map(|(k, p)| {
                        let q = match p {
                            Value::Object(m) if m.contains_key("q") => &m["q"],
                            other => other,
                        };
                        Ok(JointGoal { values: self.joint_values(q, deg, &format!("points[{k}]"))?, relative })
                    })
                    .collect::<Result<_, String>>()?;
                MotionCmd::WaypointsJoint { points, times: times(o)?, speed: speed(o)? }
            }
            "waypoints/tcp" => {
                allow(o, &["points", "times", "frame", "relative", "deg", "speed"])?;
                let pts = o.get("points").and_then(|v| v.as_array()).filter(|a| !a.is_empty()).ok_or("missing \"points\" (a non-empty array)")?;
                let (fr, relative) = (frame(o)?, flag(o, "relative")?);
                let points = pts
                    .iter()
                    .enumerate()
                    .map(|(k, p)| {
                        let m = p.as_object().ok_or_else(|| format!("points[{k}]: expected {{position, quat|rpy}}"))?;
                        allow(m, &["position", "quat", "rpy", "axis_angle"]).map_err(|e| format!("points[{k}]: {e}"))?;
                        self.tcp_goal(m, fr, relative, deg).map_err(|e| format!("points[{k}]: {e}"))
                    })
                    .collect::<Result<_, String>>()?;
                MotionCmd::WaypointsTcp { points, times: times(o)?, speed: speed(o)? }
            }
            "velocity/joint" | "accel/joint" => {
                let key = if path == "velocity/joint" { "v" } else { "a" };
                allow(o, &[key, "deg", "timeout"])?;
                let x = self.joint_vector(o.get(key).ok_or_else(|| format!("missing \"{key}\""))?, deg, key)?;
                let timeout = positive(o, "timeout")?.unwrap_or(DEFAULT_STREAM_TIMEOUT);
                if key == "v" {
                    MotionCmd::JointStream { v: Some(x), a: None, timeout }
                } else {
                    MotionCmd::JointStream { v: None, a: Some(x), timeout }
                }
            }
            "velocity/tcp" | "accel/tcp" => {
                allow(o, &["linear", "angular", "frame", "deg", "timeout"])?;
                let x = self.six(o, "linear", "angular", deg)?;
                let timeout = positive(o, "timeout")?.unwrap_or(DEFAULT_STREAM_TIMEOUT);
                let fr = frame(o)?;
                if path == "velocity/tcp" {
                    MotionCmd::TcpStream { twist: Some(x), accel: None, frame: fr, timeout }
                } else {
                    MotionCmd::TcpStream { twist: None, accel: Some(x), frame: fr, timeout }
                }
            }
            "force" => {
                allow(o, &["force", "moment", "frame", "free", "ramp", "max_speed", "max_angular_speed", "clear"])?;
                if flag(o, "clear")? || !(o.contains_key("force") || o.contains_key("moment")) {
                    MotionCmd::Force(None)
                } else {
                    let wrench = self.six(o, "force", "moment", false)?;
                    let free = match o.get("free") {
                        None | Some(Value::Null) => std::array::from_fn(|i| wrench[i] != 0.0),
                        Some(Value::Array(a)) => {
                            let mut f = [false; 6];
                            for x in a {
                                let i = ["rx", "ry", "rz", "x", "y", "z"]
                                    .iter()
                                    .position(|n| Some(*n) == x.as_str())
                                    .ok_or_else(|| format!("free: {x} (axes: x, y, z, rx, ry, rz)"))?;
                                f[i] = true;
                            }
                            f
                        }
                        Some(other) => return Err(format!("free: expected a list of axes, got {other}")),
                    };
                    MotionCmd::Force(Some(ForceSpec {
                        wrench,
                        frame: frame(o)?,
                        free,
                        ramp_s: positive(o, "ramp")?.unwrap_or(DEFAULT_RAMP),
                        max_speed: positive(o, "max_speed")?.unwrap_or(DEFAULT_FORCE_MAX_SPEED),
                        max_ang_speed: positive(o, "max_angular_speed")?.unwrap_or(DEFAULT_FORCE_MAX_ANG_SPEED),
                    }))
                }
            }
            "impedance" => {
                allow(o, &["stiffness", "damping", "frame", "clear"])?;
                if flag(o, "clear")? || !o.contains_key("stiffness") {
                    MotionCmd::Impedance(None)
                } else {
                    let six = |v: &Value, what: &str| -> Result<Vector6<f64>, String> {
                        let m = v.as_object().ok_or_else(|| format!("{what}: expected {{linear, angular}}"))?;
                        allow(m, &["linear", "angular"]).map_err(|e| format!("{what}: {e}"))?;
                        let l = m.get("linear").map(|x| vec3_or_scalar(x, "linear")).transpose()?.ok_or_else(|| format!("{what}: missing linear"))?;
                        let a = m.get("angular").map(|x| vec3_or_scalar(x, "angular")).transpose()?.ok_or_else(|| format!("{what}: missing angular"))?;
                        let v = Vector6::new(a.x, a.y, a.z, l.x, l.y, l.z);
                        if v.iter().any(|x| *x < 0.0) {
                            return Err(format!("{what}: must be ≥ 0"));
                        }
                        Ok(v)
                    };
                    MotionCmd::Impedance(Some(ImpedanceSpec {
                        stiffness: six(&o["stiffness"], "stiffness")?,
                        damping: o.get("damping").filter(|v| !v.is_null()).map(|v| six(v, "damping")).transpose()?,
                        frame: frame(o)?,
                    }))
                }
            }
            "joint/torque" => {
                allow(o, &["torque", "stiffness_scale", "ramp", "clear"])?;
                if flag(o, "clear")? || !o.contains_key("torque") {
                    MotionCmd::JointTorque(None)
                } else {
                    let scale = opt_num(o, "stiffness_scale")?.unwrap_or(1.0);
                    if !(0.0..=1.0).contains(&scale) {
                        return Err("stiffness_scale: 0 ≤ x ≤ 1".into());
                    }
                    MotionCmd::JointTorque(Some(JointTorqueSpec {
                        torque: self.joint_vector(&o["torque"], false, "torque")?,
                        stiffness_scale: scale,
                        ramp_s: positive(o, "ramp")?.unwrap_or(DEFAULT_RAMP),
                    }))
                }
            }
            "gripper" => {
                allow(o, &["position", "width", "open", "close", "speed", "max_force"])?;
                let g = self.gripper_idx.ok_or("this arm has no gripper")?;
                let j = &self.joints[g];
                let position = match (opt_num(o, "position")?, opt_num(o, "width")?, flag(o, "open")?, flag(o, "close")?) {
                    (Some(p), None, false, false) => p,
                    (None, Some(w), false, false) => 0.5 * w,
                    (None, None, true, false) => j.max,
                    (None, None, false, true) => j.min,
                    _ => return Err("give one of position, width, open, close".into()),
                };
                MotionCmd::Gripper { position, speed: speed(o)?, max_force: positive(o, "max_force")? }
            }
            "mode" => {
                allow(o, &["mode", "control"])?;
                match o.get("mode").and_then(|v| v.as_str()) {
                    Some("hold") => MotionCmd::Hold,
                    Some("gravity") => MotionCmd::Gravity,
                    Some("park") => MotionCmd::Park,
                    Some("teleop") => {
                        if !self.teleop {
                            return Err("no leader in this run (start with --leader <profile>)".into());
                        }
                        MotionCmd::Teleop(match o.get("control").and_then(|v| v.as_str()) {
                            None | Some("joint") => Mode::Joint,
                            Some("osc") => Mode::Osc,
                            Some("mpc") if self.mpc => Mode::Mpc,
                            Some(other) => return Err(format!("control: \"{other}\" (joint | osc{})", if self.mpc { " | mpc" } else { "" })),
                        })
                    }
                    other => return Err(format!("mode: {other:?} (hold | gravity | park | teleop)")),
                }
            }
            "stop" => {
                allow(o, &[])?;
                MotionCmd::Stop
            }
            "shutdown" => {
                allow(o, &[])?;
                MotionCmd::Shutdown
            }
            other => return Err(format!("no endpoint /v1/{other}")),
        };
        Ok((cmd, queue))
    }
}

fn times(o: &Obj) -> Result<Option<Vec<f64>>, String> {
    o.get("times")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_array()
                .ok_or("times: expected an array of seconds from the start")?
                .iter()
                .map(|x| num(x, "times"))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
}

// ── Server ──────────────────────────────────────────────────────────────

pub struct ApiOptions {
    pub bind: SocketAddr,
    pub token: Option<String>,
}

#[derive(Clone)]
struct Ctx {
    info: Arc<ArmInfo>,
    board: SharedBoard,
    snapshot: SharedSnapshot,
    tx: Sender<Queued>,
    token: Option<String>,
}

/// Start the server thread.
pub fn serve(opts: ApiOptions, info: ArmInfo, board: SharedBoard, snapshot: SharedSnapshot, tx: Sender<Queued>) -> Result<(), String> {
    if !opts.bind.ip().is_loopback() && opts.token.is_none() {
        return Err(format!(
            "--api-bind {}: listening beyond this machine needs a token (--api-token-file, or MANIP_API_TOKEN)",
            opts.bind
        ));
    }
    let server = tiny_http::Server::http(opts.bind).map_err(|e| format!("API on {}: {e}", opts.bind))?;
    log::info!("API listening on http://{}{}", opts.bind, if opts.token.is_some() { " (token required)" } else { "" });
    let ctx = Ctx { info: Arc::new(info), board, snapshot, tx, token: opts.token };
    std::thread::Builder::new()
        .name("api".into())
        .spawn(move || {
            for rq in server.incoming_requests() {
                let ctx = ctx.clone();
                // A thread per request: `?wait` blocks only its own caller.
                let _ = std::thread::Builder::new().name("api-req".into()).spawn(move || handle(&ctx, rq));
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn respond(rq: tiny_http::Request, code: u16, body: &Value) {
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let text = serde_json::to_string_pretty(body).unwrap_or_default();
    let _ = rq.respond(tiny_http::Response::from_string(text).with_status_code(code).with_header(header));
}

fn handle(ctx: &Ctx, mut rq: tiny_http::Request) {
    if let Some(token) = &ctx.token {
        let ok = rq
            .headers()
            .iter()
            .any(|h| h.field.equiv("Authorization") && h.value.as_str() == format!("Bearer {token}"));
        if !ok {
            return respond(rq, 401, &json!({"error": "missing or wrong token (Authorization: Bearer <token>)"}));
        }
    }
    let url = rq.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    let path = path.trim_end_matches('/');
    let Some(path) = path.strip_prefix("/v1/") else {
        return respond(rq, 404, &json!({"error": "endpoints are under /v1/ (see GET /v1/info)"}));
    };
    let params: BTreeMap<&str, &str> = query.split('&').filter_map(|kv| kv.split_once('=')).collect();
    match (rq.method(), path) {
        (tiny_http::Method::Get, "state") => {
            let s = ctx.snapshot.lock().unwrap().clone();
            respond(rq, 200, &serde_json::to_value(s).unwrap_or_default())
        }
        (tiny_http::Method::Get, "info") => respond(rq, 200, &serde_json::to_value(&*ctx.info).unwrap_or_default()),
        (tiny_http::Method::Get, p) if p.starts_with("motions/") => {
            match p["motions/".len()..].parse::<u64>().ok().and_then(|id| ctx.board.lock().unwrap().get(id)) {
                Some(st) => respond(rq, 200, &serde_json::to_value(st).unwrap_or_default()),
                None => respond(rq, 404, &json!({"error": "no such motion"})),
            }
        }
        (tiny_http::Method::Post, p) => {
            let mut text = String::new();
            if let Err(e) = rq.as_reader().read_to_string(&mut text) {
                return respond(rq, 400, &json!({"error": e.to_string()}));
            }
            let body: Value = if text.trim().is_empty() {
                Value::Null
            } else {
                match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => return respond(rq, 400, &json!({"error": format!("JSON: {e}")})),
                }
            };
            let (cmd, append) = match ctx.info.parse(p, &body) {
                Ok(x) => x,
                Err(e) => return respond(rq, if e.starts_with("no endpoint") { 404 } else { 400 }, &json!({"error": e})),
            };
            let id = ctx.board.lock().unwrap().register(cmd.kind());
            if ctx.tx.send(Queued { id, cmd, append }).is_err() {
                return respond(rq, 503, &json!({"error": "the control loop has ended"}));
            }
            let wait = matches!(params.get("wait").copied(), Some("1" | "true"));
            let timeout = params.get("timeout").and_then(|t| t.parse::<f64>().ok()).unwrap_or(60.0);
            let st = if wait { wait_for(&ctx.board, id, Duration::from_secs_f64(timeout)) } else { ctx.board.lock().unwrap().get(id) };
            respond(rq, 200, &serde_json::to_value(st).unwrap_or_default())
        }
        (m, _) => {
            let msg = format!("no endpoint {m} /v1/{path}");
            respond(rq, 404, &json!({"error": msg}))
        }
    }
}

/// Block until motion `id` has finished (or `timeout`).
pub fn wait_for(board: &SharedBoard, id: u64, timeout: Duration) -> Option<MotionStatus> {
    let t0 = Instant::now();
    loop {
        let st = board.lock().unwrap().get(id);
        match &st {
            Some(s) if s.state.finished() => return st,
            None => return None,
            _ if t0.elapsed() > timeout => return st,
            _ => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

// ── Scripts ─────────────────────────────────────────────────────────────

/// A motion script: `[{"t": seconds, "path": "move/tcp", "body": {...}}, ...]`,
/// the same requests the API takes, delivered at `t` after the start.
pub fn load_script(path: &std::path::Path, info: &ArmInfo) -> Result<Vec<(f64, MotionCmd, bool)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let items = v.as_array().ok_or("a script is a JSON array of {t, path, body}")?;
    items
        .iter()
        .enumerate()
        .map(|(k, it)| {
            let o = it.as_object().ok_or_else(|| format!("script[{k}]: expected {{t, path, body}}"))?;
            let t = o.get("t").and_then(|x| x.as_f64()).ok_or_else(|| format!("script[{k}]: missing t"))?;
            let p = o.get("path").and_then(|x| x.as_str()).ok_or_else(|| format!("script[{k}]: missing path"))?;
            let p = p.trim_start_matches("/v1/").trim_start_matches('/');
            let (cmd, append) = info.parse(p, o.get("body").unwrap_or(&Value::Null)).map_err(|e| format!("script[{k}] {p}: {e}"))?;
            Ok((t, cmd, append))
        })
        .collect()
}

// ── Client (`manip cmd`) ────────────────────────────────────────────────

/// `key=value` arguments to a JSON object: values are JSON when they parse
/// (`3`, `true`, `[1,2]`, `{"a":1}`), comma lists become number arrays,
/// anything else a string; `a.b=1` nests.
pub fn args_to_json(args: &[String]) -> Result<Value, String> {
    let mut root = Map::new();
    for a in args {
        let (k, v) = a.split_once('=').ok_or_else(|| format!("\"{a}\": expected key=value"))?;
        let val = match serde_json::from_str::<Value>(v) {
            Ok(x) => x,
            Err(_) if v.contains(',') => Value::Array(
                v.split(',')
                    .map(|x| x.trim().parse::<f64>().map(|f| json!(f)).map_err(|_| format!("{k}: \"{x}\" is not a number")))
                    .collect::<Result<_, _>>()?,
            ),
            Err(_) => Value::String(v.into()),
        };
        let mut obj = &mut root;
        let parts: Vec<&str> = k.split('.').collect();
        for p in &parts[..parts.len() - 1] {
            obj = obj
                .entry(p.to_string())
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .ok_or_else(|| format!("{k}: {p} is not an object"))?;
        }
        obj.insert(parts[parts.len() - 1].to_string(), val);
    }
    Ok(Value::Object(root))
}

/// Send one request and print the reply. GET for `state`, `info` and
/// `motions/<id>`; POST otherwise.
pub fn client(url: &str, token: Option<&str>, path: &str, args: &[String], wait: bool, timeout: f64) -> Result<(), String> {
    let path = path.trim_start_matches('/').trim_start_matches("v1/");
    let get = matches!(path, "state" | "info") || path.starts_with("motions/");
    let mut full = format!("{}/v1/{path}", url.trim_end_matches('/'));
    if wait && !get {
        full += &format!("?wait=true&timeout={timeout}");
    }
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs_f64(timeout + 5.0)))
        .http_status_as_error(false)
        .build()
        .new_agent();
    let auth = token.map(|t| format!("Bearer {t}"));
    let resp = if get {
        let mut r = agent.get(&full);
        if let Some(a) = &auth {
            r = r.header("Authorization", a);
        }
        r.call()
    } else {
        let body = args_to_json(args)?;
        let mut r = agent.post(&full);
        if let Some(a) = &auth {
            r = r.header("Authorization", a);
        }
        r.send_json(&body)
    };
    let mut resp = resp.map_err(|e| format!("{full}: {e}"))?;
    let status = resp.status();
    let text = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
    println!("{text}");
    if !status.is_success() {
        return Err(format!("HTTP {}", status.as_u16()));
    }
    if let Ok(v) = serde_json::from_str::<Value>(&text)
        && matches!(v.get("state").and_then(|s| s.as_str()), Some("aborted" | "rejected"))
    {
        return Err(format!("motion {}", v["state"].as_str().unwrap()));
    }
    Ok(())
}
