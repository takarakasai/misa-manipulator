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


/// At the folded rest pose the B601's joint2/joint3 sit on their upper limit,
/// where the positive hint cannot be followed: `hw sign` asks for the
/// negative direction there, described as the mirror of the positive hint.
#[test]
fn sign_check_asks_for_the_free_direction_at_a_limit() {
    let (_, dm) = robot("rebot_b601_dm");
    let q = vec![0.0; dm.n()];
    for name in ["joint2", "joint3"] {
        let i = dm.dof(name).unwrap();
        let d = &dm.dofs()[i];
        assert_eq!(crate::hw::test_direction(q[i], d.q_min, d.q_max, 0.26), -1.0, "{name}");
        let flip = |h: &str| h.replace('+', "?").replace('-', "+").replace('?', "-");
        assert_eq!(crate::hw::move_hint(&dm, &q, i, -1.0), flip(&crate::hw::positive_hint(&dm, &q, i)), "{name}");
    }
    // Mid-range joints keep the positive direction.
    let i = dm.dof("joint1").unwrap();
    let d = &dm.dofs()[i];
    assert_eq!(crate::hw::test_direction(q[i], d.q_min, d.q_max, 0.26), 1.0);
}

/// Friction identification recovers the simulated plant's friction (rigid
/// plant with the hardware effects), and --write puts it into the profile.
#[test]
fn friction_sweep_recovers_the_plant() {
    let (mut p, arm) = robot("rebot_b601_dm");
    // The rigid plant's gravity is the unscaled model; a profile gravity_scale
    // (fitted to the real arm) would be a model mismatch this test is not about.
    p.joint.iter_mut().for_each(|j| j.gravity_scale = 1.0);
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
    let j2_before = p.joint.iter().find(|j| j.name == "joint2").unwrap();
    assert_eq!(j2.friction, j2_before.friction, "untouched joint changed");
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
    move |t: String| open_ceiling(t).replace("box_min = [-0.50, -0.80, 0.03]", &format!("box_min = [-0.50, -0.80, {z:.4}]"))
}

/// The profile's box ceiling is set for the bench the arm stands on; tests
/// that are not about the ceiling put it out of reach.
fn open_ceiling(t: String) -> String {
    let i = t.find("\nbox_max = [").unwrap() + 1;
    let j = t[i..].find('\n').unwrap() + i;
    format!("{}box_max = [0.85, 0.80, 0.95]{}", &t[..i], &t[j..])
}

fn no_safety(t: String) -> String {
    let i = t.find("[safety]").unwrap();
    let j = t[i..].find("# ── OSC").unwrap() + i;
    format!("{}{}", &t[..i], &t[j..])
}

/// OSC with the profile's motor-side PD (`osc_kp` / `osc_kd`, around the
/// integrated QP solution) tracks the circle at least as well as torque-only
/// OSC and holds still. On the real arm torque-only OSC chattered at ~30 Hz.
#[test]
fn osc_with_motor_pd_tracks_and_holds() {
    let torque_only = |t: String| {
        t.lines()
            .filter(|l| !l.starts_with("osc_kp = ") && !l.starts_with("osc_kd = "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml")).unwrap().contains("\nosc_kp = "),
        "the profile is expected to set osc_kp"
    );
    let (err_tau, _) = osc_run_with(torque_only, "circle", "pd-off");
    let (err_pd, _) = osc_run_with(|t| t, "circle", "pd-on");
    eprintln!("osc rms: torque only {:.2} mm, motor PD {:.2} mm", err_tau * 1e3, err_pd * 1e3);
    assert!(err_pd < 1.1 * err_tau, "motor PD {err_pd} vs torque only {err_tau}");
    let (_, vmax_hold) = osc_run_with(|t| t, "none", "pd-hold");
    assert!(vmax_hold < 0.01, "static hold moves: |v| max {vmax_hold}");
}

/// `hw scan` accepts a joint resting slightly past its limit on the stop
/// (the B601-DM shoulder at +0.14° when folded) but not a zero/sign error.
#[test]
fn scan_tolerates_resting_on_a_stop() {
    let (_, arm) = robot("rebot_b601_dm");
    let j2 = &arm.dofs()[arm.dof("joint2").unwrap()];
    let finger = &arm.dofs()[arm.dof("finger_left").unwrap()];
    assert_eq!(crate::hw::range_verdict(j2, -0.5), Ok(None));
    let past = crate::hw::range_verdict(j2, j2.q_max + 0.14f64.to_radians()).unwrap().unwrap();
    assert!((past - 0.14f64.to_radians()).abs() < 1e-12);
    assert_eq!(crate::hw::range_verdict(j2, j2.q_max + 5f64.to_radians()), Err(()));
    assert!(crate::hw::range_verdict(finger, finger.q_max + 0.0005).unwrap().is_some());
    assert_eq!(crate::hw::range_verdict(finger, finger.q_min - 0.003), Err(()));
}

/// The pre-identification friction (0.3 N·m shoulder/elbow, 0.08 N·m wrist),
/// exactly compensated. The barrier tests check the barrier itself: with the
/// friction identified on the arm (1.75 / 0.22 N·m) stick-slip carries the TCP
/// 2–6 mm past it even when the feedforward matches the plant, ~1 cm at the
/// profile's 0.8 compensation.
fn light_friction(t: String) -> String {
    let mut nominal: Option<f64> = None;
    let mut out: Vec<String> = Vec::new();
    for line in t.lines() {
        if line.starts_with('[') {
            nominal = None;
        }
        match line {
            "name = \"joint1\"" | "name = \"joint2\"" | "name = \"joint3\"" => nominal = Some(0.3),
            "name = \"joint4\"" | "name = \"joint5\"" | "name = \"joint6\"" => nominal = Some(0.08),
            _ => {}
        }
        let k = ["sim_friction = ", "friction = "].into_iter().find(|k| line.starts_with(k));
        out.push(match (k, nominal) {
            (Some(k), Some(v)) => format!("{k}{v}"),
            _ => line.to_string(),
        });
    }
    out.join("\n")
}

/// The OSC stops the TCP at a floor its target circle goes through.
#[test]
fn osc_stops_at_the_workspace_floor() {
    let (_, arm) = robot("rebot_b601_dm");
    let ready = [0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.0];
    let z0 = arm.tcp_pose(&ready).translation.z;
    let floor = z0 - 0.04; // the 5 cm circle dips to z0 − 0.10
    let circle = |_: &RobotProfile, _: &manip_model::ArmModel| Source::Circle { radius: 0.05, freq_hz: 0.25, start: None };
    let (h, r, _) = run_rigid_csv(|t| no_safety(light_friction(t)), circle, Mode::Osc, 8.0, "osc-free");
    let free = col_f(&h, &r, "tcp_z", "Osc").into_iter().fold(f64::INFINITY, f64::min);
    let (h, r, _) = run_rigid_csv(|t| with_floor(floor)(light_friction(t)), circle, Mode::Osc, 8.0, "osc-floor");
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
    let (mut p, arm) = robot("rebot_b601_dm");
    // Raising the wrist from ready reaches above a low bench ceiling; this is about the hulls.
    p.safety.as_mut().unwrap().box_max[2] = 0.95;
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

/// Profile text with an `[mpc]` planner selected.
fn with_planner(planner: &'static str) -> impl Fn(String) -> String {
    move |t: String| format!("{t}\n[mpc]\nplanner = \"{planner}\"\n")
}

/// Mpc mode (planner + joint-tracking WBC) follows a 3 cm circle better than
/// the OSC does, with either planner, and never falls back to Hold.
#[test]
fn mpc_mode_tracks_circle_with_both_planners() {
    let circle = |_: &RobotProfile, _: &manip_model::ArmModel| Source::Circle { radius: 0.03, freq_hz: 0.2, start: None };
    for planner in ["ltv", "ilqr"] {
        let (h, r, _) = run_rigid_csv(with_planner(planner), circle, Mode::Mpc, 8.0, &format!("mpc-{planner}"));
        let m = h.iter().position(|x| x == "mode").unwrap();
        let t = col_f(&h, &r, "t", "Mpc");
        let refs: Vec<Vec<f64>> = ["ref_x", "ref_y", "ref_z"].iter().map(|c| col_f(&h, &r, c, "Mpc")).collect();
        let tcp: Vec<Vec<f64>> = ["tcp_x", "tcp_y", "tcp_z"].iter().map(|c| col_f(&h, &r, c, "Mpc")).collect();
        let (mut se, mut n) = (0.0, 0);
        for k in 0..t.len() {
            if t[k] > 4.0 {
                se += (0..3).map(|i| (tcp[i][k] - refs[i][k]).powi(2)).sum::<f64>();
                n += 1;
            }
        }
        let rms = (se / n as f64).sqrt();
        let first = r.iter().position(|row| row[m] == "Mpc").unwrap();
        let fell_back = r[first..].iter().any(|row| row[m] == "Hold");
        eprintln!("{planner}: circle rms {:.2} mm over {n} ticks, fell back to Hold: {fell_back}", rms * 1e3);
        assert!(n > 1000, "{planner}: Mpc did not run long enough ({n})");
        assert!(!fell_back, "{planner}: fell back to Hold");
        assert!(rms < 1.5e-3, "{planner}: rms {rms}");
    }
}

/// A run in Mpc mode records the plans it received; replaying feeds them to
/// the policy at the same cycles, so the commands match bit for bit.
#[test]
fn mpc_log_replays_bit_exact() {
    let dir = std::env::temp_dir().join(format!("manip-mpc-replay-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let models = root().join("models").canonicalize().unwrap();
    let text = with_planner("ltv")(std::fs::read_to_string(root().join("robots/rebot_b601_dm.toml")).unwrap().replace("../models", models.to_str().unwrap()));
    let profile_path = dir.join("dm.toml");
    std::fs::write(&profile_path, &text).unwrap();
    let (p, pdir) = RobotProfile::load(&profile_path).unwrap();
    let arm = assemble::load_arm(&p, &pdir).unwrap();
    let q0 = assemble::named_pose(&p, &arm, "rest").unwrap();
    let friction = assemble::joints_in_order(&p, &arm).iter().map(|j| (j.sim_friction, j.sim_damping)).collect();
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
    let log = dir.join("run.mlog");
    app::run(
        &p,
        &arm,
        &mut plant,
        Source::Circle { radius: 0.03, freq_hz: 0.2, start: None },
        RunOptions {
            mode: Mode::Mpc,
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
    let with_plans = frames.iter().filter(|f| f.plan.is_some()).count();
    let (rp, rtext, rarm) = crate::replay::load_for_replay(&header, None).unwrap();
    let r = crate::replay::replay(&header, &frames, &rp, &rtext, &rarm, 5).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("{} frames, {with_plans} with a plan, {} divergences", r.frames, r.divergences.len());
    assert!(with_plans > 50, "plans recorded: {with_plans}");
    assert!(r.divergences.is_empty(), "{:?}", r.divergences.first());
}

/// Format-1 logs (before plans were recorded) still load, without plans.
#[test]
fn log_format_1_is_still_read() {
    let dir = std::env::temp_dir().join(format!("manip-log-v1-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v1.mlog");
    let header = crate::replay::LogHeader {
        format: 1,
        robot: "x".into(),
        profile_path: dir.clone(),
        profile_text: String::new(),
        axes: vec!["a".into()],
        rate_hz: 500.0,
        q0: vec![0.0],
    };
    let frame = crate::replay::LogFrameV1 {
        frame: misa_core::Frame {
            seq: 0,
            time: misa_core::Time::from_secs_f64(0.0),
            intent: misa_core::Intent::default(),
            observation: misa_core::Observation::empty(1, 0),
            command: misa_core::Command::idle(1),
            verdict: Default::default(),
        },
        requests: vec![Mode::Hold],
        target: crate::replay::TargetRec::None,
    };
    let mut bytes = Vec::new();
    for rec in [postcard::to_allocvec(&header).unwrap(), postcard::to_allocvec(&frame).unwrap()] {
        bytes.extend_from_slice(&(rec.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&rec);
    }
    std::fs::write(&path, bytes).unwrap();
    let (h, frames) = crate::replay::read_log(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(h.format, 1);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].plan.is_none());
    assert_eq!(frames[0].requests, vec![Mode::Hold]);
}

/// Without a target, Mpc holds the TCP where it started (an earlier version
/// aimed every plan at the current pose and let the real arm drift 9 cm).
#[test]
fn mpc_mode_without_target_holds_the_tcp() {
    let none = |_: &RobotProfile, _: &manip_model::ArmModel| Source::None;
    let (h, r, _) = run_rigid_csv(with_planner("ltv"), none, Mode::Mpc, 6.0, "mpc-hold");
    let xyz: Vec<Vec<f64>> = ["tcp_x", "tcp_y", "tcp_z"].iter().map(|c| col_f(&h, &r, c, "Mpc")).collect();
    let spread = (0..xyz[0].len())
        .map(|k| (0..3).map(|i| (xyz[i][k] - xyz[i][0]).powi(2)).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    eprintln!("Mpc hold: TCP spread {:.2} mm over {} ticks", spread * 1e3, xyz[0].len());
    assert!(xyz[0].len() > 1000);
    assert!(spread < 3e-3, "TCP drifted {spread}");
}

/// The state where MPC teleop fell back to Hold on the real arm (folding
/// back: shoulder 0.19° below its upper limit, moving toward it). misa-wbc's
/// ActiveSet calls level 0 Infeasible here although Clarabel solves it; the
/// WBC now retries with Clarabel and gets a solution that brakes the shoulder.
#[test]
fn wbc_solves_the_state_activeset_called_infeasible() {
    let (p, arm) = robot("rebot_b601_dm");
    let q = [-0.035668373107910156, -0.00324249267578125, -0.02193450927734375, -0.055886268615722656, 0.02193450927734375, 0.02231597900390625, 0.00014971160888671873];
    let v = [-0.0073261260986328125, 0.0610504150390625, -0.0024423599243164063, -0.40293121337890625, 0.007328033447265625, 0.036632537841796875, 6.183250427246093e-5];
    let s = arm.evaluate(&q, &v);
    let cbfs = assemble::supervisor_config(&p, &arm).unwrap().safety.unwrap().cbfs(&arm, &s);
    let mut wbc = manip_wbc::JointTracking::new(assemble::tracking_config(&p, &arm).unwrap());
    let r = manip_control::JointRef::at_rest(s.q.clone());
    let base = manip_control::JointCommand { axes: vec![Default::default(); arm.n()] };
    let (_, rep) = wbc.command(&arm, &s, &r, 0.002, base, &cbfs).expect("solvable");
    let j2 = arm.dof("joint2").unwrap();
    assert!(rep.qddot[j2] < -0.89, "shoulder must brake toward its limit: {}", rep.qddot[j2]);
}
