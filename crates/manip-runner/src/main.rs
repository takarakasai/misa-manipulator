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

mod app;
mod assemble;
mod config;
mod effects;
mod record;
mod rigid;
mod supervisor;
#[cfg(test)]
mod tests;

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
        /// Leader profile (`--source leader`).
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PlantKind {
    /// MuJoCo (`--features sim`).
    Sim,
    /// Rigid-body integration without MuJoCo (no contact, no friction). For checks in CI
    /// or on an SBC.
    Rigid,
    /// Real hardware (CAN).
    Can,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SourceKind {
    None,
    Sine,
    Circle,
    Leader,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    match Cli::parse().cmd {
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
            radius,
            freq,
            ideal,
            delay_ticks,
            jitter,
        } => {
            let (profile, dir) = RobotProfile::load(&robot)?;
            let arm = assemble::load_arm(&profile, &dir)?;
            log::info!("{} ({}): {} DOF, TCP = {}", profile.robot.name, arm.name(), arm.n(), profile.robot.tcp.link);
            if fast && (plant == PlantKind::Can || source == SourceKind::Leader) {
                return Err("--fast is only allowed with sim + synthetic target".into());
            }
            let start_pose = match &start_pose {
                Some(name) => Some(
                    assemble::named_pose(&profile, &arm, name)
                        .ok_or_else(|| format!("pose {name} not found"))?,
                ),
                None => None,
            };
            let src = make_source(&profile, &arm, source, leader.as_deref(), radius, freq)?;
            let simulated = plant != PlantKind::Can;
            let mut plant = make_plant(plant, &profile, &dir, &arm)?;
            if simulated && !ideal {
                plant = wrap_effects(plant, &profile, &arm, delay_ticks, jitter)?;
            } else if !simulated && (delay_ticks.is_some() || jitter.is_some()) {
                return Err("--delay-ticks / --jitter only apply to simulated plants".into());
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
                    status_every_s: 1.0,
                },
            )
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
            )?))
        }
    }
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
