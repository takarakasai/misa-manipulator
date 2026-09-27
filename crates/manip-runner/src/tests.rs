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
        assemble::supervisor_config(&p, &arm);
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
