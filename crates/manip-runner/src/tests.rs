//! Profile assembly, and the closed loop without MuJoCo.

use std::path::{Path, PathBuf};

use manip_leader::fashionstar::LeaderProfile;
use nalgebra::DVector;

use crate::app::{self, RunOptions, Source};
use crate::assemble::{self, TeleopMapping};
use crate::config::RobotProfile;
use crate::effects::{Effects, EffectsPlant};
use crate::rigid::RigidPlant;
use crate::supervisor::Mode;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn robot(name: &str) -> (RobotProfile, manip_model::ArmModel) {
    let (p, dir) = RobotProfile::load(&root().join(format!("robots/{name}.toml"))).unwrap();
    let arm = assemble::load_arm(&p, &dir).unwrap();
    (p, arm)
}

#[test]
fn all_profiles_assemble() {
    for name in ["rebot_b601_dm", "rebot_b601_rs"] {
        let (p, arm) = robot(name);
        assert_eq!(arm.n(), 7, "{name}");
        assemble::supervisor_config(&p, &arm).unwrap();
        assemble::osc_config(&p, &arm).unwrap();
        assert_eq!(assemble::safety_config(&p, &arm).axes.len(), 7);
        // The gripper is outside the TCP chain (not part of OSC).
        assert_eq!(arm.tcp_chain(), vec![0, 1, 2, 3, 4, 5], "{name}");
        let hw = p.hardware.as_ref().unwrap();
        assert_eq!(hw.bus[0].motor.len(), 7);
        assert!(assemble::named_pose(&p, &arm, "ready").is_some());
    }
}

/// From the same leader pose, the DM and RS TCPs end up at nearly the same place.
///
/// The RS URDF has every axis reversed relative to DM, and the −1 in the mapping
/// cancels that. If some sign were mixed up, the TCP would be off by tens of cm (the
/// arm lengths are nearly the same: upper arm 264/236 mm, forearm 243/228 mm).
#[test]
fn leader_maps_dm_and_rs_to_the_same_physical_pose() {
    let (lp, _) = LeaderProfile::load(&root().join("leaders/stararm102.toml")).unwrap();
    let names: Vec<String> = lp.joint.iter().map(|j| j.name.clone()).collect();
    let (pd, dm) = robot("rebot_b601_dm");
    let (pr, rs) = robot("rebot_b601_rs");
    let md = TeleopMapping::new(&pd.teleop, &dm, &names).unwrap();
    let mr = TeleopMapping::new(&pr.teleop, &rs, &names).unwrap();
    let deg = |v: [f64; 7]| v.map(f64::to_radians).to_vec();
    for leader in [
        deg([0.0, -60.0, -60.0, 20.0, 0.0, 0.0, -100.0]),
        deg([30.0, -90.0, -45.0, -30.0, 40.0, 60.0, -200.0]),
        deg([-45.0, -120.0, -100.0, 45.0, -30.0, -80.0, 0.0]),
    ] {
        let qd = md.map(&leader, &DVector::zeros(7));
        let qr = mr.map(&leader, &DVector::zeros(7));
        for (arm, q) in [(&dm, &qd), (&rs, &qr)] {
            for (d, x) in arm.dofs().iter().zip(q.iter()) {
                assert!(d.within(*x), "{} {} = {x:.3} out of [{:.2}, {:.2}]", arm.name(), d.name, d.q_min, d.q_max);
            }
        }
        let pd = dm.tcp_pose(qd.as_slice()).translation.vector;
        let pr = rs.tcp_pose(qr.as_slice()).translation.vector;
        eprintln!("DM-RS tcp distance {:.1} mm", (pd - pr).norm() * 1e3);
        assert!((pd - pr).norm() < 0.08, "DM tcp {pd:?} vs RS tcp {pr:?} for leader {leader:?}");
    }
}

/// With the rigid Plant, "startup -> start pose -> tracking -> Park -> release"
/// completes, both ideal and with the hardware effects of `[sim.effects]`
/// (latency, jitter, MIT feedback quantization, friction).
#[test]
fn rigid_closed_loop_runs_to_done() {
    for (name, source, mode, with_effects) in [
        ("rebot_b601_dm", "sine", Mode::Joint, false),
        ("rebot_b601_rs", "circle", Mode::Osc, false),
        ("rebot_b601_dm", "sine", Mode::Joint, true),
        ("rebot_b601_dm", "circle", Mode::Osc, true),
    ] {
        let (p, arm) = robot(name);
        let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
        let friction = assemble::joints_in_order(&p, &arm)
            .iter()
            .map(|j| (j.sim_friction, j.sim_damping))
            .collect();
        let rigid = RigidPlant::new(
            arm.clone(),
            q0,
            1.0 / p.control.rate_hz,
            p.sim.timestep_s,
            friction,
            p.sim.friction_v_eps,
        )
        .unwrap();
        let mut plant: Box<dyn misa_core::Plant> = if with_effects {
            let fx = p.sim.effects.as_ref().expect("profile has [sim.effects]");
            Box::new(EffectsPlant::new(
                Box::new(rigid),
                Effects {
                    command_delay_ticks: fx.command_delay_ticks,
                    observation_delay_ticks: fx.observation_delay_ticks,
                    jitter_probability: fx.jitter_probability,
                    quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
                    seed: fx.seed,
                    period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
                },
            ))
        } else {
            Box::new(rigid)
        };
        let src = match source {
            "sine" => Source::Sine {
                leader: manip_leader::synthetic::SineLeader::new(p.sine.clone()),
                dofs: p.sine.iter().map(|s| arm.dof(&s.name).unwrap()).collect(),
            },
            _ => Source::Circle {
                radius: 0.04,
                freq_hz: 0.3,
                start: None,
            },
        };
        app::run(
            &p,
            &arm,
            plant.as_mut(),
            src,
            RunOptions {
                mode,
                start_pose: assemble::named_pose(&p, &arm, "ready"),
                duration_s: Some(5.0),
                fast: true,
                record: None,
                log: None,
                status_every_s: 1e9,
                monitor: None,
            },
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// Record a run (rigid plant + hardware effects, OSC), replay it: every command
/// must match bit for bit. Replaying with a profile that differs by one gain
/// must be caught.
#[test]
fn replay_is_bit_exact_and_catches_changes() {
    let dir = std::env::temp_dir().join(format!("manip-replay-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let models = root().join("models").canonicalize().unwrap();
    let text = std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml"))
        .unwrap()
        .replace("../models", models.to_str().unwrap());
    let profile_path = dir.join("dm.toml");
    std::fs::write(&profile_path, &text).unwrap();
    let changed_path = dir.join("dm_changed.toml");
    std::fs::write(&changed_path, text.replace("kp_lin = 400.0", "kp_lin = 401.0")).unwrap();
    assert_ne!(std::fs::read_to_string(&changed_path).unwrap(), text, "kp_lin not found in profile");

    let (p, pdir) = RobotProfile::load(&profile_path).unwrap();
    let arm = assemble::load_arm(&p, &pdir).unwrap();
    let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
    let friction = assemble::joints_in_order(&p, &arm)
        .iter()
        .map(|j| (j.sim_friction, j.sim_damping))
        .collect();
    let rigid = RigidPlant::new(arm.clone(), q0, 1.0 / p.control.rate_hz, p.sim.timestep_s, friction, p.sim.friction_v_eps)
        .unwrap();
    let fx = p.sim.effects.as_ref().unwrap();
    let mut plant = EffectsPlant::new(
        Box::new(rigid),
        Effects {
            command_delay_ticks: fx.command_delay_ticks,
            observation_delay_ticks: 0,
            jitter_probability: fx.jitter_probability,
            quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
            seed: 7,
            period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
        },
    );
    let log = dir.join("run.mrec");
    app::run(
        &p,
        &arm,
        &mut plant,
        Source::Circle { radius: 0.04, freq_hz: 0.3, start: None },
        RunOptions {
            mode: Mode::Osc,
            start_pose: assemble::named_pose(&p, &arm, "ready"),
            duration_s: Some(4.0),
            fast: true,
            record: None,
            log: Some((log.clone(), profile_path.clone())),
            status_every_s: 1e9,
            monitor: None,
        },
    )
    .unwrap();

    let (header, frames) = crate::replay::read_log(&log).unwrap();
    assert!(frames.len() > 1000, "{} frames", frames.len());
    assert!(frames.iter().any(|f| f.requests.contains(&Mode::Osc)), "never entered OSC");

    let (rp, rtext, rarm) = crate::replay::load_for_replay(&header, None).unwrap();
    let same = crate::replay::replay(&header, &frames, &rp, &rtext, &rarm, 10).unwrap();
    assert!(!same.profile_changed);
    assert!(same.divergences.is_empty(), "{:?}", same.divergences);

    let (cp, ctext, carm) = crate::replay::load_for_replay(&header, Some(&changed_path)).unwrap();
    let changed = crate::replay::replay(&header, &frames, &cp, &ctext, &carm, 10).unwrap();
    assert!(changed.profile_changed);
    assert!(!changed.divergences.is_empty(), "a changed OSC gain went unnoticed");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Run the DM arm on the rigid plant with friction and the hardware effects,
/// recording CSV; return (rms TCP error [m] in Osc, max |v| of the arm joints
/// in Osc). `edit` rewrites the profile text first.
fn osc_run_with(edit: impl Fn(String) -> String, source: &str, tag: &str) -> (f64, f64) {
    let dir = std::env::temp_dir().join(format!("manip-friction-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let models = root().join("models").canonicalize().unwrap();
    let text = std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml"))
        .unwrap()
        .replace("../models", models.to_str().unwrap());
    let path = dir.join("p.toml");
    std::fs::write(&path, edit(text)).unwrap();
    let (p, pdir) = RobotProfile::load(&path).unwrap();
    let arm = assemble::load_arm(&p, &pdir).unwrap();
    let friction = assemble::joints_in_order(&p, &arm)
        .iter()
        .map(|j| (j.sim_friction, j.sim_damping))
        .collect();
    let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
    let rigid = RigidPlant::new(arm.clone(), q0, 1.0 / p.control.rate_hz, p.sim.timestep_s, friction, p.sim.friction_v_eps)
        .unwrap();
    let fx = p.sim.effects.as_ref().unwrap();
    let mut plant = EffectsPlant::new(
        Box::new(rigid),
        Effects {
            command_delay_ticks: fx.command_delay_ticks,
            observation_delay_ticks: 0,
            jitter_probability: fx.jitter_probability,
            quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
            seed: 3,
            period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
        },
    );
    let csv = dir.join("run.csv");
    let src = match source {
        "circle" => Source::Circle { radius: 0.05, freq_hz: 0.25, start: None },
        _ => Source::None,
    };
    app::run(
        &p,
        &arm,
        &mut plant,
        src,
        RunOptions {
            mode: Mode::Osc,
            start_pose: assemble::named_pose(&p, &arm, "ready"),
            duration_s: Some(8.0),
            fast: true,
            record: Some(csv.clone()),
            log: None,
            status_every_s: 1e9,
            monitor: None,
        },
    )
    .unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap();
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).filter(|r: &Vec<&str>| r[1] == "Osc").collect();
    let rows = &rows[rows.len() / 3..];
    let f = |r: &Vec<&str>, n: &str| r[col(n)].parse::<f64>().unwrap();
    let mut se = 0.0;
    let mut vmax: f64 = 0.0;
    for r in rows {
        let e = [("tcp_x", "ref_x"), ("tcp_y", "ref_y"), ("tcp_z", "ref_z")]
            .iter()
            .map(|(a, b)| (f(r, a) - f(r, b)).powi(2))
            .sum::<f64>();
        se += e;
        for j in 1..=6 {
            vmax = vmax.max(f(r, &format!("v_joint{j}")).abs());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    ((se / rows.len() as f64).sqrt(), vmax)
}

/// Friction feedforward in the OSC: tracking gets better when compensated,
/// and a static hold stays still (no limit cycle from compensating on the
/// quantized measured velocity).
#[test]
fn osc_friction_compensation_helps_and_holds_still() {
    let off = |t: String| t.replace("\nfriction = ", "\nfriction_off = ").replace("friction_off", "#friction");
    let (err_off, _) = osc_run_with(off, "circle", "off");
    let (err_on, _) = osc_run_with(|t| t, "circle", "on");
    eprintln!("osc rms: off {:.2} mm, on {:.2} mm", err_off * 1e3, err_on * 1e3);
    eprintln!("osc rms: off {:.2} mm, on {:.2} mm", err_off * 1e3, err_on * 1e3);
    assert!(err_on < 0.6 * err_off, "compensated {err_on} vs uncompensated {err_off}");
    let (_, vmax_hold) = osc_run_with(|t| t, "none", "hold");
    eprintln!("hold |v| max {vmax_hold:.5} rad/s");
    assert!(vmax_hold < 0.01, "static hold moves: |v| max {vmax_hold}");
}

/// The real CAN plant (bus threads, model <-> motor frame conversion incl. the
/// gripper's m/rad ratio and sign, arm/disarm) driving a virtual arm: startup,
/// start pose, joint tracking, Park and release complete, and tracking is sane.
/// A wrong conversion (sign, zero, ratio, ratio² on the gains) would make the
/// arm sag or run away here. Real time (~5 s).
#[test]
fn virtual_can_arm_runs_the_hardware_path() {
    for name in ["rebot_b601_dm", "rebot_b601_rs"] {
        let (p, arm) = robot(name);
        let hw = p.hardware.as_ref().unwrap();
        let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
        let friction = assemble::joints_in_order(&p, &arm)
            .iter()
            .map(|j| (j.sim_friction, j.sim_damping))
            .collect();
        let motors = crate::virtual_arm::virtual_motors(
            &arm,
            &hw.bus,
            q0,
            friction,
            p.sim.friction_v_eps,
            p.sim.timestep_s,
            crate::virtual_arm::DEFAULT_TRANSACTION,
        )
        .unwrap();
        let mut plant = manip_plant_can::CanArmPlant::with_actuators(
            manip_plant_can::CanOptions {
                buses: hw.bus.clone(),
                joints: arm.dofs().iter().map(|d| d.name.clone()).collect(),
                stale_after: std::time::Duration::from_secs_f64(hw.stale_after_s),
            },
            motors,
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("manip-vcan-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("run.csv");
        app::run(
            &p,
            &arm,
            &mut plant,
            Source::Sine {
                leader: manip_leader::synthetic::SineLeader::new(p.sine.clone()),
                dofs: p.sine.iter().map(|s| arm.dof(&s.name).unwrap()).collect(),
            },
            RunOptions {
                mode: Mode::Joint,
                start_pose: assemble::named_pose(&p, &arm, "ready"),
                duration_s: Some(3.0),
                fast: false,
                record: Some(csv.clone()),
                log: None,
                status_every_s: 1e9,
                monitor: None,
            },
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        // rms |q - qref| over all DOFs while tracking.
        let text = std::fs::read_to_string(&csv).unwrap();
        let mut lines = text.lines();
        let head: Vec<&str> = lines.next().unwrap().split(',').collect();
        let pairs: Vec<(usize, usize)> = arm
            .dofs()
            .iter()
            .map(|d| {
                let c = |k: &str| head.iter().position(|h| *h == format!("{k}_{}", d.name)).unwrap();
                (c("q"), c("qref"))
            })
            .collect();
        let (mut se, mut n) = (0.0, 0usize);
        for l in lines {
            let r: Vec<&str> = l.split(',').collect();
            if r[1] != "Joint" {
                continue;
            }
            for &(q, qr) in &pairs[..6] {
                let e = r[q].parse::<f64>().unwrap() - r[qr].parse::<f64>().unwrap();
                se += e * e;
                n += 1;
            }
        }
        let rms = (se / n as f64).sqrt();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(rms < 0.03, "{name}: joint tracking rms {rms:.4} rad through the virtual CAN arm");
    }
}

/// `hw sign` hints come from the Jacobian. DM and RS have mirrored URDF axes,
/// so at the same physical pose each arm joint's "positive" hint must point
/// the opposite way.
#[test]
fn sign_hints_are_mirrored_between_dm_and_rs() {
    let (_, dm) = robot("rebot_b601_dm");
    let (_, rs) = robot("rebot_b601_rs");
    // Same physical pose: RS angles are the negated DM ones (see teleop maps).
    let qd = [0.4, -1.2, -1.1, 0.3, 0.5, 0.2, 0.0];
    let qr: Vec<f64> = qd.iter().enumerate().map(|(i, x)| if i < 6 { -x } else { 0.0 }).collect();
    let flip = |h: &str| h.replace('+', "?").replace('-', "+").replace('?', "-");
    for i in 0..6 {
        let hd = crate::hw::positive_hint(&dm, &qd, i);
        let hr = crate::hw::positive_hint(&rs, &qr, i);
        assert_eq!(hr, flip(&hd), "joint{}: DM '{hd}' vs RS '{hr}'", i + 1);
    }
}


/// Friction identification recovers the simulated plant's friction (rigid
/// plant with the hardware effects), and --write puts it into the profile.
#[test]
fn friction_sweep_recovers_the_plant() {
    let (p, arm) = robot("rebot_b601_dm");
    let dofs = vec![arm.dof("joint1").unwrap(), arm.dof("joint4").unwrap()];
    let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
    let truth: Vec<(f64, f64)> = assemble::joints_in_order(&p, &arm)
        .iter()
        .map(|j| (j.sim_friction, j.sim_damping))
        .collect();
    let rigid = RigidPlant::new(arm.clone(), q0, 1.0 / p.control.rate_hz, p.sim.timestep_s, truth.clone(), p.sim.friction_v_eps)
        .unwrap();
    let fx = p.sim.effects.as_ref().unwrap();
    let mut plant = EffectsPlant::new(
        Box::new(rigid),
        Effects {
            command_delay_ticks: fx.command_delay_ticks,
            observation_delay_ticks: 0,
            jitter_probability: fx.jitter_probability,
            quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
            seed: 5,
            period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
        },
    );
    let csv = std::env::temp_dir().join(format!("manip-friction-id-{}.csv", std::process::id()));
    let fits = crate::hw::friction_sweep(
        &p,
        &arm,
        &mut plant,
        &crate::hw::FrictionSweep { dofs: dofs.clone(), speeds: vec![0.2, 0.5, 1.0], amplitude: 0.5, cycles: 2, kp_scale: 0.1, fast: true },
        &csv,
    )
    .unwrap();
    let _ = std::fs::remove_file(&csv);
    for (f, &d) in fits.iter().zip(&dofs) {
        let (fc, fv) = truth[d];
        assert!((f.coulomb - fc).abs() < 0.1 * fc, "{}: Fc {:.4} vs {fc}", f.joint, f.coulomb);
        assert!((f.viscous - fv).abs() < 0.1 * fv, "{}: Fv {:.4} vs {fv}", f.joint, f.viscous);
    }
    // Writing keeps the rest of the profile and replaces only these joints' values.
    let text = std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml")).unwrap();
    let out = crate::hw::write_friction(&text, &fits);
    let reparsed: RobotProfile = toml::from_str(&out).unwrap();
    let j1 = reparsed.joint.iter().find(|j| j.name == "joint1").unwrap();
    assert!((j1.friction - fits[0].coulomb).abs() < 1e-3 && (j1.viscous - fits[0].viscous).abs() < 1e-3);
    let j2 = reparsed.joint.iter().find(|j| j.name == "joint2").unwrap();
    assert_eq!(j2.friction, 0.3, "untouched joint changed");
    assert_eq!(out.lines().count(), text.lines().count() + 2, "only a viscous line per written joint should be added");
}

/// The profiles' safety box contains the rest and ready poses (Park and the
/// OSC start must not begin in violation), and self-collision is clear there.
#[test]
fn safety_box_contains_rest_and_ready() {
    for name in ["rebot_b601_dm", "rebot_b601_rs"] {
        let (p, arm) = robot(name);
        let cfg = p.safety.as_ref().expect("profile has [safety]");
        let rest = assemble::named_pose(&p, &arm, "rest").unwrap();
        let g = crate::guard::SafetyModel::build(cfg, &arm, Some(rest.as_slice())).unwrap();
        for pose in ["rest", "ready"] {
            let q = assemble::named_pose(&p, &arm, pose).unwrap();
            assert_eq!(g.violation(&arm, q.as_slice()), 0.0, "{name} {pose} violates [safety]");
            assert_eq!(
                g.joint_violation(&arm, q.as_slice()),
                0.0,
                "{name} {pose} violates [safety] with the joint margin: {}",
                g.explain(&arm, q.as_slice())
            );
        }
    }
}

/// Run the DM arm on the rigid plant (with friction and hardware effects) from
/// [pose.ready] using a profile whose text is rewritten by `edit`; return the
/// CSV as (header, rows).
fn run_rigid_csv(
    edit: impl Fn(String) -> String,
    source: impl FnOnce(&RobotProfile, &manip_model::ArmModel) -> Source,
    mode: Mode,
    duration: f64,
    tag: &str,
) -> (Vec<String>, Vec<Vec<String>>, manip_model::ArmModel) {
    let dir = std::env::temp_dir().join(format!("manip-safety-{}-{tag}", std::process::id()));
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
    let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
    let rigid = RigidPlant::new(arm.clone(), q0, 1.0 / p.control.rate_hz, p.sim.timestep_s, friction, p.sim.friction_v_eps)
        .unwrap();
    let mut plant = EffectsPlant::new(
        Box::new(rigid),
        Effects {
            command_delay_ticks: 1,
            observation_delay_ticks: 0,
            jitter_probability: 0.1,
            quantization: Some(assemble::feedback_quantization(&p, &arm).unwrap()),
            seed: 11,
            period: std::time::Duration::from_secs_f64(1.0 / p.control.rate_hz),
        },
    );
    let csv = dir.join("run.csv");
    let src = source(&p, &arm);
    app::run(
        &p,
        &arm,
        &mut plant,
        src,
        RunOptions {
            mode,
            start_pose: assemble::named_pose(&p, &arm, "ready"),
            duration_s: Some(duration),
            fast: true,
            record: Some(csv.clone()),
            log: None,
            status_every_s: 1e9,
            monitor: None,
        },
    )
    .unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let mut lines = text.lines();
    let head = lines.next().unwrap().split(',').map(String::from).collect();
    let rows = lines.map(|l| l.split(',').map(String::from).collect()).collect();
    (head, rows, arm)
}

fn col_f(head: &[String], rows: &[Vec<String>], name: &str, mode: &str) -> Vec<f64> {
    let c = head.iter().position(|h| h == name).unwrap();
    let m = head.iter().position(|h| h == "mode").unwrap();
    rows.iter().filter(|r| r[m] == mode).map(|r| r[c].parse::<f64>().unwrap()).collect()
}

fn with_floor(z: f64) -> impl Fn(String) -> String {
    move |t: String| t.replace("box_min = [-0.50, -0.80, 0.03]", &format!("box_min = [-0.50, -0.80, {z:.4}]"))
}

fn no_safety(t: String) -> String {
    let i = t.find("[safety]").unwrap();
    let j = t[i..].find("# ── OSC").unwrap() + i;
    format!("{}{}", &t[..i], &t[j..])
}

/// The OSC stops the TCP at a floor its target circle goes through.
#[test]
fn osc_stops_at_the_workspace_floor() {
    let (_, arm) = robot("rebot_b601_dm");
    let ready = [0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.0];
    let z0 = arm.tcp_pose(&ready).translation.z;
    let floor = z0 - 0.04; // the 5 cm circle dips to z0 − 0.10
    let circle = |_: &RobotProfile, _: &manip_model::ArmModel| Source::Circle { radius: 0.05, freq_hz: 0.25, start: None };
    let (h, r, _) = run_rigid_csv(no_safety, circle, Mode::Osc, 8.0, "osc-free");
    let free = col_f(&h, &r, "tcp_z", "Osc").into_iter().fold(f64::INFINITY, f64::min);
    let (h, r, _) = run_rigid_csv(with_floor(floor), circle, Mode::Osc, 8.0, "osc-floor");
    let guarded = col_f(&h, &r, "tcp_z", "Osc").into_iter().fold(f64::INFINITY, f64::min);
    eprintln!("floor {floor:.4}: min tcp z free {free:.4}, with [safety] {guarded:.4}");
    assert!(free < floor - 0.04, "the circle should cross the floor without [safety]");
    assert!(guarded > floor - 0.002, "OSC went through the floor: {guarded:.4} < {floor:.4}");
}

/// Joint tracking refuses reference steps that would take the TCP below the floor.
#[test]
fn joint_guard_stops_at_the_workspace_floor() {
    let sine = |p: &RobotProfile, arm: &manip_model::ArmModel| Source::Sine {
        leader: manip_leader::synthetic::SineLeader::new(p.sine.clone()),
        dofs: p.sine.iter().map(|s| arm.dof(&s.name).unwrap()).collect(),
    };
    let (h, r, _) = run_rigid_csv(no_safety, sine, Mode::Joint, 10.0, "joint-free");
    let free = col_f(&h, &r, "tcp_z", "Joint").into_iter().fold(f64::INFINITY, f64::min);
    let floor = free + 0.05;
    let (h, r, _) = run_rigid_csv(with_floor(floor), sine, Mode::Joint, 10.0, "joint-floor");
    let guarded = col_f(&h, &r, "tcp_z", "Joint").into_iter().fold(f64::INFINITY, f64::min);
    eprintln!("floor {floor:.4}: min tcp z free {free:.4}, guarded {guarded:.4}");
    assert!(guarded > floor, "joint tracking went through the floor: {guarded:.4} < {floor:.4}");
}

/// Joint tracking toward a self-colliding pose stops before the links touch.
#[test]
fn joint_guard_prevents_self_collision() {
    let (p, arm) = robot("rebot_b601_dm");
    let cfg = p.safety.as_ref().unwrap();
    let rest = assemble::named_pose(&p, &arm, "rest").unwrap();
    let mut sc = manip_model::collision::SelfCollision::build(&arm).unwrap();
    sc.exclude_close_at(&arm, rest.as_slice(), cfg.collision_margin);
    sc.exclude_pairs(
        &cfg.exclude_pairs
            .iter()
            .map(|[a, b]| (a.clone(), b.clone()))
            .collect::<Vec<_>>(),
    );
    let g = crate::guard::SafetyModel::build(cfg, &arm, Some(rest.as_slice())).unwrap();
    // Deterministic search for a pose that is inside the box but self-colliding,
    // near the ready pose (so the arm can head there from ready).
    let ready = assemble::named_pose(&p, &arm, "ready").unwrap();
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let target = (0..20000)
        .find_map(|_| {
            let q: Vec<f64> = arm
                .dofs()
                .iter()
                .enumerate()
                .map(|(i, d)| if i < 6 { d.clamp(ready[i] + (rnd() - 0.5) * 3.0) } else { 0.0 })
                .collect();
            let dmin = sc.min_distance(&arm, &q);
            (dmin < -0.01 && g.violation(&arm, &q) - (cfg.collision_margin - dmin) < 1e-9).then_some(q)
        })
        .expect("no self-colliding pose found");
    eprintln!("target {:?}: distance {:.4}", target.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), sc.min_distance(&arm, &target));
    let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    let hold = |_: &RobotProfile, arm: &manip_model::ArmModel| Source::Sine {
        leader: manip_leader::synthetic::SineLeader::new(
            names
                .iter()
                .zip(&target)
                .map(|(n, &c)| manip_leader::synthetic::SineJoint { name: n.clone(), center: c, amp: 0.0, freq_hz: 0.1, phase: 0.0 })
                .collect(),
        ),
        dofs: (0..arm.n()).collect(),
    };
    let (h, r, arm) = run_rigid_csv(|t| t, hold, Mode::Joint, 8.0, "selfcol");
    let cols: Vec<Vec<f64>> = names.iter().map(|n| col_f(&h, &r, &format!("q_{n}"), "Joint")).collect();
    let dmin = (0..cols[0].len())
        .step_by(10)
        .map(|k| sc.min_distance(&arm, &cols.iter().map(|c| c[k]).collect::<Vec<_>>()))
        .fold(f64::INFINITY, f64::min);
    eprintln!("closest approach while tracking: {dmin:.4} m");
    assert!(dmin > 0.0, "links touched: {dmin:.4}");
}

/// Diagnostic: for each checked pair, the share of uniformly sampled poses
/// (all arm joints, within limits) where it is under the margin, and the
/// joints between the two links. `cargo test -- --ignored --nocapture pair_survey`
#[test]
#[ignore]
fn pair_survey() {
    for name in ["rebot_b601_dm", "rebot_b601_rs"] {
        let (p, arm) = robot(name);
        let g = p.safety.as_ref().unwrap();
        let rest = assemble::named_pose(&p, &arm, "rest").unwrap();
        let mut sc = manip_model::collision::SelfCollision::build(&arm).unwrap();
        sc.exclude_close_at(&arm, rest.as_slice(), g.collision_margin);
        sc.exclude_pairs(
            &g.exclude_pairs
                .iter()
                .map(|[a, b]| (a.clone(), b.clone()))
                .collect::<Vec<_>>(),
        );
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 20000;
        let mut hits = std::collections::BTreeMap::<String, usize>::new();
        let mut floor = std::collections::BTreeMap::<String, f64>::new();
        for _ in 0..n {
            let q: Vec<f64> = arm
                .dofs()
                .iter()
                .map(|d| d.q_min + (d.q_max - d.q_min) * rnd())
                .collect();
            let mut seen = std::collections::BTreeSet::new();
            for x in sc.close_pairs(&arm, &q, 0.03) {
                let k = format!("{} – {}", x.link_a, x.link_b);
                let m = floor.entry(k.clone()).or_insert(f64::INFINITY);
                *m = m.min(x.distance);
                if x.distance < g.collision_margin {
                    seen.insert(k);
                }
            }
            for k in seen {
                *hits.entry(k).or_default() += 1;
            }
        }
        println!("{name}:");
        for (k, m) in floor {
            let c = hits.get(&k).copied().unwrap_or(0);
            println!(
                "  {k}: min {:.1} mm, under the margin {:.1} %",
                m * 1e3,
                100.0 * c as f64 / n as f64
            );
        }
    }
}

/// The DM wrist hulls (link3 and link5, across joint4 and joint5) overlap
/// within the joint range, so the profile excludes the pair: the guard must
/// allow the whole wrist range from the ready pose.
#[test]
fn dm_wrist_range_is_not_guarded() {
    let (p, arm) = robot("rebot_b601_dm");
    let rest = assemble::named_pose(&p, &arm, "rest").unwrap();
    let ready = assemble::named_pose(&p, &arm, "ready").unwrap();
    let g =
        crate::guard::SafetyModel::build(p.safety.as_ref().unwrap(), &arm, Some(rest.as_slice()))
            .unwrap();
    let (d4, d5) = (&arm.dofs()[3], &arm.dofs()[4]);
    for a in 0..=12 {
        for b in 0..=12 {
            let mut q = ready.clone();
            q[3] = d4.q_min + (d4.q_max - d4.q_min) * a as f64 / 12.0;
            q[4] = d5.q_min + (d5.q_max - d5.q_min) * b as f64 / 12.0;
            assert_eq!(
                g.joint_violation(&arm, q.as_slice()),
                0.0,
                "{}",
                g.explain(&arm, q.as_slice())
            );
        }
    }
}
