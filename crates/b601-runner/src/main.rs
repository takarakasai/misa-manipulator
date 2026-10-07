//! `manip` — model-based manipulator control.
//!
//! ```sh
//! # In MuJoCo, track a synthetic target with joint impedance
//! manip run --robot robots/rebot_b601_dm.toml --plant sim --source sine --mode joint
//! # In MuJoCo, draw a TCP circle with OSC
//! manip run --robot robots/rebot_b601_dm.toml --plant sim --source circle --mode osc
//! # Drive a MuJoCo follower with a Star Arm 102 as leader
//! manip run --robot robots/rebot_b601_dm.toml --plant sim --source leader --leader leaders/stararm102.toml
//! # Real hardware (CAN)
//! manip run --robot robots/rebot_b601_dm.toml --plant can --source leader --leader leaders/stararm102.toml
//! ```

mod api;
mod app;
mod assemble;
mod config;
mod effects;
mod guard;
mod hw;
mod motion;
mod mpc_driver;
mod policy;
mod record;
mod replay;
mod rigid;
mod supervisor;
mod traj;
mod virtual_arm;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_api;

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use manip_leader::synthetic::SineLeader;
use misa_core::Plant;

use crate::app::{RunOptions, Source};
use crate::config::RobotProfile;
use crate::supervisor::Mode;

#[derive(Parser)]
#[command(name = "manip", about = "Model-based manipulator control")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Run under SCHED_FIFO at this priority (1–99), inherited by the bus and
    /// leader threads. Needs CAP_SYS_NICE on the binary or an rtprio limit
    /// (`/etc/security/limits.d`); `chrt` cannot pass the binary's capability on.
    #[arg(long, global = true)]
    rt_priority: Option<i32>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the control loop.
    Run {
        #[arg(long)]
        robot: PathBuf,
        #[arg(long, value_enum, default_value_t = PlantKind::Sim)]
        plant: PlantKind,
        #[arg(long, value_enum, default_value_t = SourceKind::None)]
        source: SourceKind,
        /// Leader profile (`--source leader`), or run log (`--source log`).
        #[arg(long)]
        leader: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = Mode::Joint)]
        mode: Mode,
        /// Pose to move to before entering the requested mode (name of a [pose.*] in the
        /// profile).
        #[arg(long)]
        start_pose: Option<String>,
        /// Fold up and exit after this many seconds.
        #[arg(long)]
        duration: Option<f64>,
        /// Don't wait for real time (only with sim + synthetic target).
        #[arg(long)]
        fast: bool,
        /// Record to CSV.
        #[arg(long)]
        record: Option<PathBuf>,
        /// Write a binary run log for bit-exact replay (`manip replay`).
        #[arg(long)]
        log: Option<PathBuf>,
        /// Circle radius [m] and frequency [Hz] (`--source circle`).
        #[arg(long, default_value_t = 0.05)]
        radius: f64,
        #[arg(long, default_value_t = 0.25)]
        freq: f64,
        /// Run the simulated plant ideal: ignore `[sim.effects]` (no latency,
        /// jitter or quantization).
        #[arg(long)]
        ideal: bool,
        /// Override `[sim.effects] command_delay_ticks`.
        #[arg(long)]
        delay_ticks: Option<usize>,
        /// Override `[sim.effects] jitter_probability`.
        #[arg(long)]
        jitter: Option<f64>,
        /// Show the arm live in MuJoCo's viewer (needs `--features sim`).
        /// Works with every plant: it poses a display model from the measured
        /// joint angles.
        #[arg(long)]
        viewer: bool,
        /// `--source api`: listen for HTTP/JSON motion commands here.
        /// Anything but loopback needs a token.
        #[arg(long, default_value = "127.0.0.1:8080")]
        api_bind: std::net::SocketAddr,
        /// File holding the API token (default: the MANIP_API_TOKEN variable).
        #[arg(long)]
        api_token_file: Option<PathBuf>,
        /// `--source script`: a JSON motion script, `[{"t": s, "path":
        /// "move/tcp", "body": {...}}, ...]` (the API's requests at given times).
        #[arg(long)]
        script: Option<PathBuf>,
    },
    /// Re-run a binary run log through the current code and compare every
    /// command bit for bit. Exits non-zero if anything diverged.
    Replay {
        log: PathBuf,
        /// Use this profile instead of the one recorded in the log.
        #[arg(long)]
        robot: Option<PathBuf>,
        /// Maximum number of divergences to print.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Also write every replayed cycle to this CSV (the `run --record` columns).
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// Bring-up: check the arm before trusting the control loop with it
    /// (scan / monitor / sign never energize the motors).
    Hw {
        #[arg(long)]
        robot: PathBuf,
        /// `can` for the real arm, `virtual-can` to rehearse without it.
        #[arg(long, value_enum, default_value_t = PlantKind::Can)]
        plant: PlantKind,
        #[command(subcommand)]
        cmd: HwCmd,
    },
    /// Send one request to a running `manip run --source api`, print the
    /// reply. GET for `state`, `info`, `motions/<id>`; POST otherwise.
    /// Example: `manip cmd move/tcp position=0,0,0.05 relative=true --wait`.
    #[command(name = "cmd")]
    Client {
        /// Endpoint under /v1/ (`move/joint`, `move/tcp`, `gripper`, `state`, ...).
        path: String,
        /// Body as key=value: JSON values (`3`, `true`, `[1,2]`), `1,2,3` for
        /// arrays, `a.b=1` to nest.
        args: Vec<String>,
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// API token (default: the MANIP_API_TOKEN variable).
        #[arg(long)]
        token: Option<String>,
        /// Wait until the motion has finished (exit status 1 if it was
        /// aborted or rejected).
        #[arg(long)]
        wait: bool,
        #[arg(long, default_value_t = 60.0)]
        timeout: f64,
    },
    /// Only display leader values (does not move the follower).
    Leader {
        #[arg(long)]
        leader: PathBuf,
        #[arg(long, default_value_t = 10.0)]
        duration: f64,
        /// Read servo angles at the zero pose and print the profile's `offset_deg`
        /// (does not write to the servos).
        #[arg(long)]
        zero: bool,
    },
}

#[derive(Subcommand)]
enum HwCmd {
    /// Every motor answers and the pose is inside the range of motion.
    Scan,
    /// Live joint angles (model frame) while moving the arm by hand.
    Monitor {
        #[arg(long, default_value_t = 30.0)]
        duration: f64,
    },
    /// Per joint, move it by hand as instructed; reports OK / REVERSED signs.
    Sign {
        /// Seconds to wait for each joint to move.
        #[arg(long, default_value_t = 15.0)]
        timeout: f64,
    },
    /// Identify joint friction: sweep each joint at constant speeds, fit
    /// `Fc·sign(v) + Fv·v` to (commanded torque − rigid-body model). Runs on
    /// any plant (a sim has a known answer to check against).
    Friction {
        /// Joints to identify (default: the arm joints on the TCP chain).
        #[arg(long, value_delimiter = ',')]
        joints: Vec<String>,
        /// Sweep speeds [rad/s].
        #[arg(long, value_delimiter = ',', default_value = "0.2,0.5,1.0")]
        speeds: Vec<f64>,
        /// Half-width of the sweep around the start pose [rad] (clipped to the range).
        #[arg(long, default_value_t = 0.5)]
        amplitude: f64,
        #[arg(long, default_value_t = 2)]
        cycles: usize,
        /// Scale on the position gains `kp` during the sweep. The MIT position
        /// target is held for a whole control period while the joint moves, so
        /// the motor's PD torque saws by up to `kp·v·dt` within each period and
        /// a sampled torque is biased by an amount proportional to speed (read
        /// as viscous friction). Softer `kp` shrinks that bias in proportion.
        #[arg(long, default_value_t = 0.1)]
        kp_scale: f64,
        /// Write the result into the profile's `friction` / `viscous`.
        #[arg(long)]
        write: bool,
        /// Where to keep the sweep's CSV.
        #[arg(long, default_value = "logs/friction.csv")]
        record: PathBuf,
    },
    /// Energize, hold, move one joint by `delta` degrees and back, fold, release.
    Jog {
        #[arg(long)]
        joint: String,
        /// Degrees (mm for a prismatic joint). Flipped if it would leave the range.
        #[arg(long, default_value_t = 5.0, allow_hyphen_values = true)]
        delta: f64,
        /// Seconds for the out-and-back move.
        #[arg(long, default_value_t = 3.0)]
        period: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PlantKind {
    /// MuJoCo (`--features sim`).
    Sim,
    /// Rigid-body integration without MuJoCo (no contact, no friction). For checks in CI
    /// or on an SBC.
    Rigid,
    /// Real hardware (CAN).
    Can,
    /// The real CAN plant (bus threads, frame conversion, arm/disarm) driving
    /// a virtual arm instead of CAN motors. Runs in real time.
    VirtualCan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SourceKind {
    None,
    Sine,
    Circle,
    Leader,
    /// The targets of a run log (`--leader <run.mlog>`), cycle by cycle: a
    /// real teleop session replayed against a simulated plant.
    Log,
    /// Motion commands over HTTP/JSON (`--api-bind`; `manip cmd`). With
    /// `--leader <profile>`, the `teleop` mode command hands over to it.
    Api,
    /// The API's commands from a file at given times (`--script`), then fold
    /// up and exit.
    Script,
}

fn main() {
    let cli = Cli::parse();
    // Replay re-runs every mode transition; keep its output to the verdict.
    let level = if matches!(cli.cmd, Cmd::Replay { .. }) { "warn" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();
    if let Err(e) = cli.rt_priority.map_or(Ok(()), set_realtime) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    if let Err(e) = real_main(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// SCHED_FIFO for this thread; threads spawned afterwards inherit it.
fn set_realtime(prio: i32) -> Result<(), String> {
    if !(1..=99).contains(&prio) {
        return Err(format!("--rt-priority {prio}: must be 1..=99"));
    }
    let param = libc::sched_param { sched_priority: prio };
    // SAFETY: plain syscall on the calling thread with a valid sched_param.
    if unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) } != 0 {
        let e = std::io::Error::last_os_error();
        return Err(format!(
            "--rt-priority {prio}: {e}. Give the binary CAP_SYS_NICE \
             (`sudo setcap cap_sys_nice+ep target/release/manip`, lost on every rebuild) \
             or the user an rtprio limit in /etc/security/limits.d (new login)"
        ));
    }
    log::info!("running under SCHED_FIFO priority {prio}");
    Ok(())
}

fn real_main(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Run {
            robot,
            plant,
            source,
            leader,
            mode,
            start_pose,
            duration,
            fast,
            record,
            log,
            radius,
            freq,
            ideal,
            delay_ticks,
            jitter,
            viewer,
            api_bind,
            api_token_file,
            script,
        } => {
            let (profile, dir) = RobotProfile::load(&robot)?;
            let arm = assemble::load_arm(&profile, &dir)?;
            log::info!("{} ({}): {} DOF, TCP = {}", profile.robot.name, arm.name(), arm.n(), profile.robot.tcp.link);
            if fast && (matches!(plant, PlantKind::Can | PlantKind::VirtualCan) || matches!(source, SourceKind::Leader | SourceKind::Api)) {
                return Err("--fast is only allowed with sim + synthetic target".into());
            }
            // Motion commands start from holding where the arm is.
            let mode = if matches!(source, SourceKind::Api | SourceKind::Script) { Mode::Hold } else { mode };
            let start_pose = match &start_pose {
                Some(name) => Some(
                    assemble::named_pose(&profile, &arm, name)
                        .ok_or_else(|| format!("pose {name} not found"))?,
                ),
                None => None,
            };
            let src = match source {
                SourceKind::Api | SourceKind::Script => {
                    let token = match &api_token_file {
                        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?.trim().to_string()),
                        None => std::env::var("MANIP_API_TOKEN").ok().filter(|t| !t.is_empty()),
                    };
                    make_api_source(&profile, &arm, source, leader.as_deref(), api_bind, token, script.as_deref())?
                }
                _ => make_source(&profile, &arm, source, leader.as_deref(), radius, freq)?,
            };
            // Effects wrap only the physics sims; virtual-can already has the
            // real bus-thread timing.
            let simulated = matches!(plant, PlantKind::Sim | PlantKind::Rigid);
            if !simulated && (delay_ticks.is_some() || jitter.is_some()) {
                return Err("--delay-ticks / --jitter only apply to simulated plants".into());
            }
            let log_opt = log.map(|l| (l, robot.clone()));
            let misa_path = dir.join(&profile.robot.model);
            let view_names: Vec<String> = app::viewer_joints(&arm).into_iter().map(|(n, _)| n).collect();
            // Everything the control loop needs, owned, so it can move to a
            // worker thread when the viewer takes the main one. The plant is
            // built inside (a MuJoCo sim cannot cross threads).
            let run = move |monitor| -> Result<(), String> {
                let mut plant = make_plant(plant, &profile, &dir, &arm)?;
                if simulated && !ideal {
                    plant = wrap_effects(plant, &profile, &arm, delay_ticks, jitter)?;
                }
                app::run(
                    &profile,
                    &arm,
                    plant.as_mut(),
                    src,
                    RunOptions {
                        mode,
                        start_pose,
                        duration_s: duration,
                        fast,
                        record,
                        log: log_opt,
                        status_every_s: 1.0,
                        monitor,
                    },
                )
            };
            if viewer {
                run_with_viewer(run, &misa_path, view_names)
            } else {
                run(None)
            }
        }
        Cmd::Hw { robot, plant, cmd } => {
            if !matches!(plant, PlantKind::Can | PlantKind::VirtualCan) && !matches!(cmd, HwCmd::Friction { .. }) {
                return Err("hw commands run on --plant can or virtual-can".into());
            }
            let (profile, dir) = RobotProfile::load(&robot)?;
            let arm = assemble::load_arm(&profile, &dir)?;
            let mut p = make_plant(plant, &profile, &dir, &arm)?;
            match cmd {
                HwCmd::Scan => {
                    if hw::scan(p.as_mut(), &arm)? { Ok(()) } else { Err("scan found problems".into()) }
                }
                HwCmd::Monitor { duration } => hw::monitor(p.as_mut(), &arm, Duration::from_secs_f64(duration)),
                HwCmd::Sign { timeout } => {
                    if hw::sign_check(p.as_mut(), &arm, Duration::from_secs_f64(timeout))? {
                        Ok(())
                    } else {
                        Err("sign check found problems".into())
                    }
                }
                HwCmd::Friction { joints, speeds, amplitude, cycles, kp_scale, write, record } => {
                    let dofs: Vec<usize> = if joints.is_empty() {
                        arm.tcp_chain()
                    } else {
                        joints.iter().map(|j| arm.dof(j).map_err(|e| e.to_string())).collect::<Result<_, _>>()?
                    };
                    let simulated = matches!(plant, PlantKind::Sim | PlantKind::Rigid);
                    if simulated {
                        p = wrap_effects(p, &profile, &arm, None, None)?;
                    }
                    let opts = hw::FrictionSweep { dofs, speeds, amplitude, cycles, kp_scale, fast: simulated };
                    let fits = hw::friction_sweep(&profile, &arm, p.as_mut(), &opts, &record)?;
                    let js = assemble::joints_in_order(&profile, &arm);
                    println!("{:<14} {:>9} {:>9} {:>8} {:>9}   (profile now: friction / viscous)", "joint", "Fc[N·m]", "Fv", "samples", "resid");
                    for f in &fits {
                        let j = js.iter().find(|j| j.name == f.joint).unwrap();
                        println!(
                            "{:<14} {:>9.4} {:>9.4} {:>8} {:>9.4}   ({:.4} / {:.4})",
                            f.joint, f.coulomb, f.viscous, f.samples, f.residual, j.friction, j.viscous
                        );
                    }
                    if write {
                        let text = std::fs::read_to_string(&robot).map_err(|e| e.to_string())?;
                        std::fs::write(&robot, hw::write_friction(&text, &fits)).map_err(|e| e.to_string())?;
                        println!("wrote friction / viscous into {}", robot.display());
                    }
                    Ok(())
                }
                HwCmd::Jog { joint, delta, period } => {
                    let dof = arm.dof(&joint).map_err(|e| e.to_string())?;
                    let d = &arm.dofs()[dof];
                    let mut delta = match d.kind {
                        manip_model::DofKind::Revolute => delta.to_radians(),
                        manip_model::DofKind::Prismatic => delta * 1e-3,
                    };
                    let q0 = hw::read_passive(p.as_mut(), arm.n(), Duration::from_millis(300))?.axes()[dof].position_rad;
                    if !d.within(q0 + delta) {
                        log::warn!("{joint}: {q0:.3} + {delta:.3} leaves the range; jogging the other way");
                        delta = -delta;
                    }
                    let duration = profile.control.startup_ramp_s + period + 1.0;
                    app::run(
                        &profile,
                        &arm,
                        p.as_mut(),
                        Source::Jog { dof, delta, period_s: period, start: None },
                        RunOptions {
                            mode: Mode::Joint,
                            start_pose: None,
                            duration_s: Some(duration),
                            fast: false,
                            record: None,
                            log: None,
                            status_every_s: 0.5,
                            monitor: None,
                        },
                    )
                }
            }
        }
        Cmd::Client { path, args, url, token, wait, timeout } => {
            let token = token.or_else(|| std::env::var("MANIP_API_TOKEN").ok().filter(|t| !t.is_empty()));
            api::client(&url, token.as_deref(), &path, &args, wait, timeout)
        }
        Cmd::Replay { log, robot, limit, record } => {
            let (header, frames) = replay::read_log(&log)?;
            let (profile, text, arm) = replay::load_for_replay(&header, robot.as_deref())?;
            let mut rec = match &record {
                Some(p) => Some(record::Recorder::create(p, &arm).map_err(|e| e.to_string())?),
                None => None,
            };
            let r = replay::replay(&header, &frames, &profile, &text, &arm, limit, rec.as_mut())?;
            if r.profile_changed {
                eprintln!("note: the profile differs from the one recorded in the log");
            }
            if r.divergences.is_empty() {
                println!("{} frames: identical", r.frames);
                Ok(())
            } else {
                for d in &r.divergences {
                    let name = header.axes.get(d.axis.index()).map(String::as_str).unwrap_or("?");
                    println!("seq {:>7} {:<14} {:<16} recorded {:+.9e} replayed {:+.9e}", d.seq, name, d.field, d.left, d.right);
                }
                Err(format!("{} frames: diverged (first at seq {})", r.frames, r.divergences[0].seq))
            }
        }
        Cmd::Leader { leader, duration, zero } => {
            if zero {
                return leader_zero(&leader);
            }
            let (thread, rate) = open_leader(&leader)?;
            let t0 = std::time::Instant::now();
            while t0.elapsed().as_secs_f64() < duration {
                std::thread::sleep(Duration::from_millis(200));
                let (hz, errs) = thread.stats();
                match thread.latest() {
                    Some(s) => {
                        let vals: Vec<String> = thread
                            .names()
                            .iter()
                            .zip(&s.q)
                            .map(|(n, q)| format!("{n}={:+7.2}°", q.to_degrees()))
                            .collect();
                        eprintln!("{hz:5.0}Hz err={errs} age={:>4}ms  {}", s.at.elapsed().as_millis(), vals.join(" "));
                    }
                    None => eprintln!("(nothing read yet) {rate:.0}Hz requested, err={errs}"),
                }
            }
            Ok(())
        }
    }
}

fn make_source(
    profile: &RobotProfile,
    arm: &manip_model::ArmModel,
    kind: SourceKind,
    leader: Option<&std::path::Path>,
    radius: f64,
    freq: f64,
) -> Result<Source, String> {
    Ok(match kind {
        SourceKind::None => Source::None,
        SourceKind::Api | SourceKind::Script => return Err("built by make_api_source".into()),
        SourceKind::Sine => {
            if profile.sine.is_empty() {
                return Err("profile has no [[sine]]".into());
            }
            let dofs = profile
                .sine
                .iter()
                .map(|s| arm.dof(&s.name).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            Source::Sine {
                leader: SineLeader::new(profile.sine.clone()),
                dofs,
            }
        }
        SourceKind::Circle => Source::Circle {
            radius,
            freq_hz: freq,
            start: None,
        },
        SourceKind::Log => {
            let path = leader.ok_or("--source log requires --leader <run.mlog>")?;
            let (_, frames) = crate::replay::read_log(path)?;
            Source::Recorded {
                targets: frames.iter().map(|f| (&f.target).into()).collect(),
                dt: 1.0 / profile.control.rate_hz,
            }
        }
        SourceKind::Leader => {
            let path = leader.ok_or("--source leader requires --leader <profile>")?;
            let (thread, _) = open_leader(path)?;
            let mapping = assemble::TeleopMapping::new(&profile.teleop, arm, thread.names())?;
            Source::Leader {
                thread,
                mapping,
                timeout: Duration::from_secs_f64(profile.control.leader_timeout_s),
                last_seq: 0,
            }
        }
    })
}

/// The executive with the HTTP server (`api`) or a script.
fn make_api_source(
    profile: &RobotProfile,
    arm: &manip_model::ArmModel,
    kind: SourceKind,
    leader: Option<&std::path::Path>,
    bind: std::net::SocketAddr,
    token: Option<String>,
    script: Option<&std::path::Path>,
) -> Result<Source, String> {
    use std::sync::{Arc, Mutex};
    let cfg = assemble::motion_config(profile, arm, true)?;
    let poses = profile
        .pose
        .keys()
        .filter_map(|n| assemble::named_pose(profile, arm, n).map(|q| (n.clone(), q.as_slice().to_vec())))
        .collect();
    let teleop = match (kind, leader) {
        (SourceKind::Api, Some(path)) => {
            let (thread, _) = open_leader(path)?;
            let mapping = assemble::TeleopMapping::new(&profile.teleop, arm, thread.names())?;
            Some(motion::Teleop { thread, mapping, timeout: Duration::from_secs_f64(profile.control.leader_timeout_s) })
        }
        _ => None,
    };
    let info = api::ArmInfo::new(&profile.robot.name, arm, cfg.gripper, poses, teleop.is_some(), true, profile.control.rate_hz);
    let board = Arc::new(Mutex::new(motion::Board::default()));
    let snapshot = Arc::new(Mutex::new(api::Snapshot::default()));
    let (exec, script_end) = match kind {
        SourceKind::Api => {
            let (tx, rx) = std::sync::mpsc::channel();
            api::serve(api::ApiOptions { bind, token }, info.clone(), board.clone(), snapshot.clone(), tx)?;
            (motion::Executive::new(cfg, Some(rx), board, teleop), None)
        }
        _ => {
            let path = script.ok_or("--source script requires --script <file.json>")?;
            let items = api::load_script(path, &info)?;
            let end = items.iter().map(|(t, _, _)| *t).fold(0.0, f64::max);
            let mut exec = motion::Executive::new(cfg, None, board, None);
            for (t, cmd, append) in items {
                exec.schedule(t, cmd, append);
            }
            (exec, Some(end))
        }
    };
    Ok(Source::Api(Box::new(app::ApiSource { exec, info, snapshot, script_end })))
}

#[cfg(feature = "fashionstar")]
fn open_leader(path: &std::path::Path) -> Result<(manip_leader::LeaderThread, f64), String> {
    let (cfg, rate) = manip_leader::fashionstar::LeaderProfile::load(path)?;
    let leader = manip_leader::fashionstar::FashionStarLeader::open(cfg).map_err(|e| e.to_string())?;
    let thread = manip_leader::LeaderThread::spawn(Box::new(leader), Duration::from_secs_f64(1.0 / rate));
    Ok((thread, rate))
}

/// Read 50 times at the zero pose, average, and print `offset_deg`.
#[cfg(feature = "fashionstar")]
fn leader_zero(path: &std::path::Path) -> Result<(), String> {
    use manip_leader::fashionstar::{FashionStarLeader, LeaderProfile};
    let (cfg, _) = LeaderProfile::load(path)?;
    let mut l = FashionStarLeader::open(cfg).map_err(|e| e.to_string())?;
    eprintln!("Put the leader in the zero pose (folded, gripper closed) and press Enter");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    let n = l.joints().len();
    let mut sum = vec![0.0; n];
    let mut cnt = vec![0usize; n];
    for _ in 0..50 {
        for (i, a) in l.read_raw().map_err(|e| e.to_string())?.iter().enumerate() {
            if let Some(a) = a {
                // The multi-turn counter has been reset, so wrap into ±180° and average.
                sum[i] += manip_leader::unwrap_to_window(*a, -std::f64::consts::PI, std::f64::consts::PI);
                cnt[i] += 1;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    println!("# Servo angles at the zero pose for {}. Copy into each [[joint]] offset_deg.", path.display());
    for (i, j) in l.joints().iter().enumerate() {
        if cnt[i] == 0 {
            println!("# {}: no response", j.name);
        } else {
            println!("{:<14} offset_deg = {:.2}", j.name, (sum[i] / cnt[i] as f64).to_degrees());
        }
    }
    Ok(())
}

#[cfg(not(feature = "fashionstar"))]
fn leader_zero(_path: &std::path::Path) -> Result<(), String> {
    Err("build with --features fashionstar".into())
}

#[cfg(not(feature = "fashionstar"))]
fn open_leader(_path: &std::path::Path) -> Result<(manip_leader::LeaderThread, f64), String> {
    Err("build with --features fashionstar to use a leader".into())
}

fn make_plant(
    kind: PlantKind,
    profile: &RobotProfile,
    dir: &std::path::Path,
    arm: &manip_model::ArmModel,
) -> Result<Box<dyn Plant>, String> {
    match kind {
        PlantKind::Can => {
            let hw = profile
                .hardware
                .as_ref()
                .ok_or("profile has no [hardware]")?;
            let p = manip_plant_can::CanArmPlant::open(manip_plant_can::CanOptions {
                buses: hw.bus.clone(),
                joints: arm.dofs().iter().map(|d| d.name.clone()).collect(),
                stale_after: Duration::from_secs_f64(hw.stale_after_s),
            })?;
            Ok(Box::new(p))
        }
        PlantKind::Sim => make_sim(profile, dir, arm),
        PlantKind::VirtualCan => {
            let hw = profile.hardware.as_ref().ok_or("profile has no [hardware]")?;
            let q0 = match &profile.sim.initial_pose {
                Some(name) => assemble::named_pose(profile, arm, name)
                    .ok_or_else(|| format!("[sim] initial_pose = \"{name}\" not found"))?,
                None => nalgebra::DVector::zeros(arm.n()),
            };
            let friction = assemble::joints_in_order(profile, arm)
                .iter()
                .map(|j| (j.sim_friction, j.sim_damping))
                .collect();
            let motors = virtual_arm::virtual_motors(
                arm,
                &hw.bus,
                q0,
                friction,
                profile.sim.friction_v_eps,
                profile.sim.timestep_s,
                virtual_arm::DEFAULT_TRANSACTION,
            )?;
            let p = manip_plant_can::CanArmPlant::with_actuators(
                manip_plant_can::CanOptions {
                    buses: hw.bus.clone(),
                    joints: arm.dofs().iter().map(|d| d.name.clone()).collect(),
                    stale_after: Duration::from_secs_f64(hw.stale_after_s),
                },
                motors,
            )?;
            Ok(Box::new(p))
        }
        PlantKind::Rigid => {
            let q0 = match &profile.sim.initial_pose {
                Some(name) => assemble::named_pose(profile, arm, name)
                    .ok_or_else(|| format!("[sim] initial_pose = \"{name}\" not found"))?,
                None => nalgebra::DVector::zeros(arm.n()),
            };
            let friction = assemble::joints_in_order(profile, arm)
                .iter()
                .map(|j| (j.sim_friction, j.sim_damping))
                .collect();
            Ok(Box::new(rigid::RigidPlant::new(
                arm.clone(),
                q0,
                1.0 / profile.control.rate_hz,
                profile.sim.timestep_s,
                friction,
                profile.sim.friction_v_eps,
            )?
            .with_stiction(profile.sim.stiction)))
        }
    }
}

/// Run the control loop on a worker thread and MuJoCo's viewer on this (main)
/// thread, which winit requires for its event loop.
#[cfg(feature = "sim")]
fn run_with_viewer(
    run: impl FnOnce(Option<std::sync::Arc<std::sync::Mutex<Option<Vec<f64>>>>>) -> Result<(), String> + Send + 'static,
    misa_path: &std::path::Path,
    names: Vec<String>,
) -> Result<(), String> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    let slot = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));
    let (slot2, done2) = (slot.clone(), done.clone());
    let worker = std::thread::Builder::new()
        .name("control".into())
        .spawn(move || {
            let r = run(Some(slot2));
            done2.store(true, Ordering::Relaxed);
            r
        })
        .map_err(|e| e.to_string())?;
    let title = format!("manip — {}", misa_path.file_stem().and_then(|s| s.to_str()).unwrap_or("arm"));
    let shown = manip_plant_mujoco::viewer::run_viewer(&misa_path.display().to_string(), &names, slot, &done, &title);
    if let Err(e) = &shown {
        // The control loop keeps running without a view (no display, GL failure).
        log::warn!("{e}; continuing without the viewer");
        while !done.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    worker.join().map_err(|_| "control thread panicked".to_string())?
}

#[cfg(not(feature = "sim"))]
fn run_with_viewer(
    _run: impl FnOnce(Option<std::sync::Arc<std::sync::Mutex<Option<Vec<f64>>>>>) -> Result<(), String>,
    _misa_path: &std::path::Path,
    _names: Vec<String>,
) -> Result<(), String> {
    Err("--viewer needs a build with --features sim (MuJoCo)".into())
}

/// Wrap a simulated plant in `[sim.effects]` (latency, jitter, quantization).
/// Without `[sim.effects]` the plant stays ideal unless the CLI asks for delay.
fn wrap_effects(
    plant: Box<dyn Plant>,
    profile: &RobotProfile,
    arm: &manip_model::ArmModel,
    delay_ticks: Option<usize>,
    jitter: Option<f64>,
) -> Result<Box<dyn Plant>, String> {
    let fx = match (&profile.sim.effects, delay_ticks, jitter) {
        (None, None, None) => return Ok(plant),
        (cfg, _, _) => cfg.clone().unwrap_or(config::EffectsSection {
            command_delay_ticks: 0,
            observation_delay_ticks: 0,
            jitter_probability: 0.0,
            quantize: false,
            seed: 1,
        }),
    };
    let quantization = if fx.quantize {
        Some(assemble::feedback_quantization(profile, arm)?)
    } else {
        None
    };
    let e = effects::Effects {
        command_delay_ticks: delay_ticks.unwrap_or(fx.command_delay_ticks),
        observation_delay_ticks: fx.observation_delay_ticks,
        jitter_probability: jitter.unwrap_or(fx.jitter_probability),
        quantization,
        seed: fx.seed,
        period: Duration::from_secs_f64(1.0 / profile.control.rate_hz),
    };
    log::info!(
        "sim effects: command delay {} tick, observation delay {} tick, jitter {:.0}%, quantized {}",
        e.command_delay_ticks,
        e.observation_delay_ticks,
        e.jitter_probability * 100.0,
        e.quantization.is_some()
    );
    Ok(Box::new(effects::EffectsPlant::new(plant, e)))
}

#[cfg(feature = "sim")]
fn make_sim(
    profile: &RobotProfile,
    dir: &std::path::Path,
    arm: &manip_model::ArmModel,
) -> Result<Box<dyn Plant>, String> {
    use manip_plant_mujoco::{MujocoArmPlant, SimAxis, SimMimic, SimOptions};
    let js = assemble::joints_in_order(profile, arm);
    let initial = match &profile.sim.initial_pose {
        Some(name) => assemble::named_pose(profile, arm, name)
            .ok_or_else(|| format!("[sim] initial_pose = \"{name}\" not found"))?,
        None => nalgebra::DVector::zeros(arm.n()),
    };
    let mimics = arm
        .file()
        .mimic
        .iter()
        .map(|m| {
            let armature = arm
                .dof(&m.source)
                .map(|i| arm.dofs()[i].armature)
                .unwrap_or(0.01);
            SimMimic {
                joint: m.joint.clone(),
                source: m.source.clone(),
                multiplier: m.multiplier,
                offset: m.offset,
                armature,
            }
        })
        .collect();
    let p = MujocoArmPlant::new(SimOptions {
        misa_path: dir.join(&profile.robot.model).display().to_string(),
        axes: arm
            .dofs()
            .iter()
            .zip(&js)
            .map(|(d, j)| SimAxis {
                joint: d.name.clone(),
                armature: d.armature,
                damping: j.sim_damping,
                effort: if d.effort.is_finite() { d.effort } else { 0.0 },
                friction: j.sim_friction,
            })
            .collect(),
        mimics,
        control_period_s: 1.0 / profile.control.rate_hz,
        timestep_s: profile.sim.timestep_s,
        initial_q: initial.as_slice().to_vec(),
        base_pos: [0.0, 0.0, 0.0],
        ground: profile.sim.ground,
        joint_limits: profile.sim.joint_limits,
        self_collision: profile.sim.self_collision,
        friction_v_eps: profile.sim.friction_v_eps,
    })?;
    Ok(Box::new(p))
}

#[cfg(not(feature = "sim"))]
fn make_sim(
    _profile: &RobotProfile,
    _dir: &std::path::Path,
    _arm: &manip_model::ArmModel,
) -> Result<Box<dyn Plant>, String> {
    Err("build with --features sim to use the sim (requires MuJoCo 3.8)".into())
}
