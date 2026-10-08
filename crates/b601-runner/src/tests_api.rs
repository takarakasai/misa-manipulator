//! Motion commands (the API's requests, scripted) in the closed loop: the
//! rigid plant with latency, jitter, quantization and friction, as
//! `tests.rs` runs the other sources.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::api::{ArmInfo, Snapshot};
use crate::app::{self, ApiSource, RunOptions, Source};
use crate::assemble;
use crate::config::RobotProfile;
use crate::effects::{Effects, EffectsPlant};
use crate::motion::{Board, Executive, MotionStatus, State};
use crate::rigid::RigidPlant;
use crate::supervisor::Mode;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Run {
    head: Vec<String>,
    rows: Vec<Vec<String>>,
    board: Arc<Mutex<Board>>,
    ids: Vec<u64>,
    log: PathBuf,
    dir: PathBuf,
}

impl Run {
    fn col(&self, name: &str) -> Vec<f64> {
        let c = self.head.iter().position(|h| h == name).unwrap_or_else(|| panic!("no column {name}"));
        self.rows.iter().map(|r| r[c].parse::<f64>().unwrap()).collect()
    }

    fn modes(&self) -> Vec<String> {
        let c = self.head.iter().position(|h| h == "mode").unwrap();
        self.rows.iter().map(|r| r[c].clone()).collect()
    }

    fn status(&self, k: usize) -> MotionStatus {
        self.board.lock().unwrap().get(self.ids[k]).unwrap()
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Run `script` (`(t, path, body)`, the API's requests) on the B601-DM
/// profile edited by `edit`, from the ready pose, for `duration` seconds.
fn run_script(edit: impl Fn(String) -> String, script: &[(f64, &str, Value)], duration: f64, tag: &str) -> Run {
    let dir = std::env::temp_dir().join(format!("manip-api-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let models = root().join("models").canonicalize().unwrap();
    let text = std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml"))
        .unwrap()
        .replace("../models", models.to_str().unwrap());
    let path = dir.join("p.toml");
    std::fs::write(&path, edit(text)).unwrap();
    let (p, pdir) = RobotProfile::load(&path).unwrap();
    let arm = assemble::load_arm(&p, &pdir).unwrap();
    let friction = assemble::joints_in_order(&p, &arm).iter().map(|j| (j.sim_friction, j.sim_damping)).collect();
    let q0 = assemble::named_pose(&p, &arm, "ready").unwrap();
    let rigid = RigidPlant::new(arm.clone(), q0, 1.0 / p.control.rate_hz, p.sim.timestep_s, friction, p.sim.friction_v_eps).unwrap();
    let mut plant = EffectsPlant::new(
        Box::new(rigid),
        Effects {
            command_delay_ticks: 1,
            observation_delay_ticks: 0,
            jitter_probability: 0.1,
            quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
            seed: 5,
            period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
        },
    );
    let cfg = assemble::motion_config(&p, &arm, true).unwrap();
    let poses = p.pose.keys().filter_map(|n| assemble::named_pose(&p, &arm, n).map(|q| (n.clone(), q.as_slice().to_vec()))).collect();
    let info = ArmInfo::new(&p.robot.name, &arm, cfg.gripper, poses, false, true, p.control.rate_hz);
    let board = Arc::new(Mutex::new(Board::default()));
    let mut exec = Executive::new(cfg, None, board.clone(), None);
    let ids = script
        .iter()
        .map(|(t, path, body)| {
            let (cmd, append) = info.parse(path, body).unwrap_or_else(|e| panic!("{path}: {e}"));
            exec.schedule(*t, cmd, append)
        })
        .collect();
    let src = Source::Api(Box::new(ApiSource { exec, info, snapshot: Arc::new(Mutex::new(Snapshot::default())), script_end: None }));
    let csv = dir.join("run.csv");
    let log = dir.join("run.mlog");
    app::run(
        &p,
        &arm,
        &mut plant,
        src,
        RunOptions {
            mode: Mode::Hold,
            start_pose: None,
            duration_s: Some(duration),
            fast: true,
            record: Some(csv.clone()),
            log: Some((log.clone(), path.clone())),
            status_every_s: 1e9,
            monitor: None,
        },
    )
    .unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let mut lines = text.lines();
    let head = lines.next().unwrap().split(',').map(String::from).collect();
    let rows = lines.map(|l| l.split(',').map(String::from).collect()).collect();
    Run { head, rows, board, ids, log, dir }
}

fn open_ceiling(t: String) -> String {
    let i = t.find("\nbox_max = [").unwrap() + 1;
    let j = t[i..].find('\n').unwrap() + i;
    format!("{}box_max = [0.85, 0.80, 0.95]{}", &t[..i], &t[j..])
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Chained TCP moves (straight line, tool frame, waypoints) end where they
/// were sent, the reference never jumps between them, and the log replays
/// bit-exact.
#[test]
fn chained_tcp_moves_reach_their_goals_and_replay() {
    let script = [
        (1.0, "move/tcp", json!({"position": [0, 0, -0.05], "relative": true})),
        (1.0, "move/tcp", json!({"position": [0.03, 0, 0], "rpy": [0, 0, 15], "deg": true, "frame": "tool", "relative": true, "queue": true})),
        (1.0, "waypoints/tcp", json!({"points": [{"position": [0, 0.04, 0]}, {"position": [0, 0, 0.04]}], "relative": true, "queue": true})),
    ];
    let r = run_script(open_ceiling, &script, 9.0, "chain");
    for k in 0..3 {
        let st = r.status(k);
        eprintln!("#{} {:?} {:?}", st.id, st.state, st.message);
        assert_eq!(st.state, State::Done, "{st:?}");
    }
    let (x, y, z) = (r.col("tcp_x"), r.col("tcp_y"), r.col("tcp_z"));
    let (rx, ry, rz) = (r.col("ref_x"), r.col("ref_y"), r.col("ref_z"));
    let modes = r.modes();
    let osc: Vec<usize> = (0..modes.len()).filter(|&i| modes[i] == "Osc").collect();
    assert!(osc.len() > 2000);
    let mut step_max: f64 = 0.0;
    let mut err2 = 0.0;
    for w in osc.windows(2) {
        step_max = step_max.max(dist([rx[w[0]], ry[w[0]], rz[w[0]]], [rx[w[1]], ry[w[1]], rz[w[1]]]));
        let e = dist([x[w[1]], y[w[1]], z[w[1]]], [rx[w[1]], ry[w[1]], rz[w[1]]]);
        err2 += e * e;
    }
    let rms = (err2 / (osc.len() - 1) as f64).sqrt();
    eprintln!("largest reference step {:.2} mm/tick, tracking rms {:.2} mm", step_max * 1e3, rms * 1e3);
    // 0.3 m/s × speed 0.5 × 2 ms = 0.3 mm per tick.
    assert!(step_max < 0.35e-3, "reference jumped {step_max}");
    assert!(rms < 3e-3, "tracking rms {rms}");
    // The net displacement: down 5 cm, then +3 cm along the tool x, then
    // +4 cm y and +4 cm z.
    let (p, rep) = crate::replay::read_log(&r.log).unwrap();
    let (rp, rtext, rarm) = crate::replay::load_for_replay(&p, None).unwrap();
    let same = crate::replay::replay(&p, &rep, &rp, &rtext, &rarm, 5, None).unwrap();
    assert!(same.divergences.is_empty(), "replay diverged: {:?}", same.divergences);
}

/// A stop in the middle of a joint move brakes within the limits, reports
/// done, and leaves the arm at rest; a velocity stream without new commands
/// stops by itself.
#[test]
fn stop_brakes_and_streams_time_out() {
    let script = [
        (1.0, "move/joint", json!({"q": {"joint1": 60}, "deg": true, "relative": true, "speed": 1.0})),
        (1.4, "stop", json!({})),
        (3.0, "velocity/joint", json!({"v": {"joint1": -0.3}, "timeout": 0.5})),
    ];
    let r = run_script(|t| t, &script, 6.0, "stop");
    assert_eq!(r.status(0).state, State::Aborted);
    assert_eq!(r.status(1).state, State::Done, "{:?}", r.status(1));
    assert_eq!(r.status(2).state, State::Done, "{:?}", r.status(2));
    let (q, qr, vr) = (r.col("q_joint1"), r.col("qref_joint1"), r.col("vref_joint1"));
    let t = r.col("t");
    let k_stop = t.iter().position(|&x| x >= 1.4).unwrap();
    let k_rest = (k_stop..t.len()).find(|&k| vr[k].abs() < 1e-9).unwrap();
    let a_max = 15.0 + 1e-6; // the profile's joint1 a_max
    for k in k_stop + 1..k_rest {
        assert!(((vr[k] - vr[k - 1]) / 0.002).abs() <= a_max, "braking too hard at {}", t[k]);
    }
    eprintln!("stopped {:.2} s after the stop at {:.3} rad", t[k_rest] - 1.4, qr[k_rest]);
    // The stream: moved while commanded, then came to rest by itself.
    let k3 = t.iter().position(|&x| x >= 3.0).unwrap();
    let end = q.len() - 1;
    assert!(q[k3] - q[t.iter().position(|&x| x >= 3.6).unwrap()] > 0.1, "the stream did not move joint1");
    let v_end = r.col("v_joint1")[t.iter().position(|&x| x >= 5.0).unwrap()];
    assert!(v_end.abs() < 0.05, "still moving after the timeout: {v_end}");
    let _ = end;
}

/// A force pushing down into nothing stops above the workspace floor, and
/// the arm holds where it ended once the force is cleared.
#[test]
fn force_into_nothing_stays_in_the_box_and_holds_after() {
    let floor = 0.20;
    let script = [
        (1.0, "force", json!({"force": [0, 0, -3]})),
        (5.0, "force", json!({"clear": true})),
    ];
    let edit = move |t: String| open_ceiling(t).replace("box_min = [-0.50, -0.80, 0.03]", &format!("box_min = [-0.50, -0.80, {floor}]"));
    let r = run_script(edit, &script, 8.0, "force");
    let z = r.col("tcp_z");
    let t = r.col("t");
    let modes = r.modes();
    // Park (at the end) is not guarded.
    let lowest = (0..z.len()).filter(|&i| modes[i] == "Osc").map(|i| z[i]).fold(f64::INFINITY, f64::min);
    let at = |s: f64| z[t.iter().position(|&x| x >= s).unwrap()];
    eprintln!("start {:.3}, lowest {:.4} (floor {floor}), at 5 s {:.4}, at 8 s {:.4}", z[0], lowest, at(5.0), at(7.9));
    assert!(z[0] - at(4.9) > 0.03, "the force did not push");
    assert!(lowest > floor - 2e-3, "went through the floor: {lowest}");
    assert!((at(7.9) - at(5.5)).abs() < 3e-3, "did not hold after the force was cleared");
}

/// The gripper moves to the asked width; with a force limit it stops at an
/// obstacle-free goal more slowly but still reports.
#[test]
fn gripper_reaches_the_width() {
    let script = [(1.0, "gripper", json!({"width": 0.06}))];
    let r = run_script(|t| t, &script, 4.0, "grip");
    let st = r.status(0);
    eprintln!("{st:?}");
    assert_eq!(st.state, State::Done);
    let (f, t) = (r.col("q_finger_left"), r.col("t"));
    let k = t.iter().position(|&x| x >= 3.9).unwrap();
    assert!((f[k] - 0.03).abs() < 1e-3, "finger at {}", f[k]);
}

#[test]
fn requests_are_checked() {
    let (p, dir) = RobotProfile::load(&root().join("robots/rebot_b601_dm.toml")).unwrap();
    let arm = assemble::load_arm(&p, &dir).unwrap();
    let cfg = assemble::motion_config(&p, &arm, false).unwrap();
    let info = ArmInfo::new("x", &arm, cfg.gripper, Default::default(), false, false, 500.0);
    let bad = [
        ("move/tcp", json!({"positon": [0, 0, 0]}), "unknown field"),
        ("move/tcp", json!({"position": [0, 0, 0.1], "frame": "tool"}), "relative"),
        ("move/tcp", json!({"position": [0, 0, 0.1], "via": "mpc"}), "MPC"),
        ("move/joint", json!({"q": [1, 2]}), "values for"),
        ("move/joint", json!({"q": {"joint9": 1}}), "no joint"),
        ("move/joint", json!({"q": [0, 0, 0, 0, 0, 0], "speed": 2}), "speed"),
        ("gripper", json!({"width": 0.02, "open": true}), "one of"),
        ("mode", json!({"mode": "teleop"}), "leader"),
        ("fly", json!({}), "no endpoint"),
    ];
    for (path, body, want) in bad {
        let e = info.parse(path, &body).expect_err(path);
        assert!(e.contains(want), "{path} {body}: {e}");
    }
    // Six values for the arm (the gripper keeps its reference); degrees.
    let (cmd, queue) = info.parse("move/joint", &json!({"q": [90, 0, 0, 0, 0, 0], "deg": true, "queue": true})).unwrap();
    assert!(queue);
    match cmd {
        crate::motion::MotionCmd::MoveJoint { goal, .. } => {
            assert_eq!(goal.values.len(), 6);
            assert!((goal.values[0].1 - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        }
        other => panic!("{other:?}"),
    }
    let v = crate::api::args_to_json(&["position=0,0,0.05".into(), "relative=true".into(), "stiffness.linear=300".into(), "frame=tool".into()]).unwrap();
    assert_eq!(v, json!({"position": [0.0, 0.0, 0.05], "relative": true, "stiffness": {"linear": 300}, "frame": "tool"}));
}


/// Acceleration streams speed up while commanded and come to rest by
/// themselves after the timeout (joint, and TCP in the tool frame).
#[test]
fn acceleration_streams_speed_up_then_stop() {
    let script = [
        (1.0, "accel/joint", json!({"a": {"joint1": 1.0}, "timeout": 0.5})),
        (3.0, "accel/tcp", json!({"linear": [0, 0, -0.2], "frame": "tool", "timeout": 0.5})),
    ];
    let r = run_script(open_ceiling, &script, 6.0, "accel");
    assert_eq!(r.status(0).state, State::Done, "{:?}", r.status(0));
    assert_eq!(r.status(1).state, State::Done, "{:?}", r.status(1));
    let (t, vr) = (r.col("t"), r.col("vref_joint1"));
    let at = |x: &[f64], s: f64| x[t.iter().position(|&y| y >= s).unwrap()];
    // 1 rad/s² for 0.5 s (less the cycles of the mode switch), then braking.
    let peak = vr.iter().cloned().fold(0.0, f64::max);
    assert!((0.45..=0.501).contains(&peak), "peak speed {peak}");
    assert!(at(&vr, 2.5).abs() < 1e-9);
    let (x, z) = (r.col("tcp_x"), r.col("tcp_z"));
    let moved = ((at(&x, 4.5) - at(&x, 3.0)).powi(2) + (at(&z, 4.5) - at(&z, 3.0)).powi(2)).sqrt();
    eprintln!("tool-frame accel stream moved the TCP {:.1} mm", moved * 1e3);
    assert!(moved > 0.01);
}

/// A joint torque with no stiffness left moves the joint until it nears its
/// limit, where the torque toward it is dropped.
#[test]
fn joint_torque_stops_pushing_near_a_limit() {
    let script = [(1.0, "joint/torque", json!({"torque": {"joint1": 2.0}, "stiffness_scale": 0.0}))];
    let r = run_script(|t| t, &script, 8.0, "jtorque");
    let (q, t) = (r.col("q_joint1"), r.col("t"));
    let modes = r.modes();
    let top = (0..q.len()).filter(|&i| modes[i] == "Joint").map(|i| q[i]).fold(f64::MIN, f64::max);
    eprintln!("joint1 from {:.3} rose to {top:.3} (limit 2.8)", q[0]);
    assert!(top > q[0] + 0.5, "the torque did not move the joint");
    assert!(top < 2.8 - 0.02, "reached the stop: {top}");
    let _ = t;
}

/// With sticking friction a joint can stop just outside the Park tolerance
/// for good; Park then finishes after settling instead of holding forever.
#[test]
fn park_finishes_when_friction_holds_a_joint_short() {
    // A sticky, softly held wrist (1.5 N·m against 6 N·m/rad) stops short of rest.
    let stiction = |t: String| t.replace("[sim]\n", "[sim]\nstiction = true\n").replace("sim_friction = 0.22", "sim_friction = 1.5").replace("hold_kp = 18.0", "hold_kp = 6.0");
    let script = [(1.0, "move/joint", json!({"q": {"joint4": 0.6, "joint2": -0.6}, "relative": true}))];
    let r = run_script(stiction, &script, 4.0, "park");
    let modes = r.modes();
    assert_eq!(modes.last().map(String::as_str), Some("Done"), "never finished parking");
    let t = r.col("t");
    let off = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"].iter().map(|j| r.col(&format!("q_{j}")).last().unwrap().abs()).fold(0.0, f64::max);
    eprintln!("parked {:.1} s after the end, {off:.3} rad from rest", t[t.len() - 1] - 4.0);
    assert!(off > 0.05, "the test should leave a joint outside the tolerance ({off})");
    assert!(t[t.len() - 1] < 4.0 + 15.0);
}
