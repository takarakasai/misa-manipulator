//! MPC (50 Hz) + WBC joint tracking (500 Hz) on the real B601-DM model,
//! rigid-body integration at 1 kHz with the motor MIT law evaluated every
//! physics step (the command is held between control cycles).

use manip_control::{Feedforward, JointCommand, JointGains, JointImpedance, JointRef};
use manip_model::{ArmModel, TcpSpec};
use manip_mpc::{LtvConfig, LtvMpc, MpcGoal, Planner, WorkspaceBox};
use manip_wbc::{JointTracking, OscConfig, TrackingConfig};
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

const PHYS_DT: f64 = 0.001;
const CTRL_EVERY: usize = 2;
const MPC_EVERY: usize = 20;

fn model() -> ArmModel {
    let path = format!("{}/../../models/rebot_b601_dm/rebot_b601_dm.misa", env!("CARGO_MANIFEST_DIR"));
    let mut arm = ArmModel::load(path, &TcpSpec::at_link("end_link")).unwrap();
    let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    for (i, name) in names.iter().enumerate() {
        let a = if i < 3 { 0.02 } else if i < 6 { 0.005 } else { 80.0 };
        arm.set_armature(name, a).unwrap();
        arm.set_limits(name, None, None, Some(if i < 6 { 4.0 } else { 0.1 }), None).unwrap();
    }
    arm
}

fn step(arm: &ArmModel, q: &mut DVector<f64>, v: &mut DVector<f64>, cmd: &JointCommand) -> DVector<f64> {
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let tau = cmd.torque_at(q, v);
    let qdd = s.mass.clone().cholesky().unwrap().solve(&(&tau - &s.nle));
    *v += qdd * PHYS_DT;
    *q += &*v * PHYS_DT;
    tau
}

struct Run {
    q: DVector<f64>,
    v: DVector<f64>,
    min_point_z: f64,
    max_tau_ratio: f64,
    relaxed: usize,
    failures: usize,
    max_plan_us: f64,
    /// Over the last second: max joint speed and TCP travel.
    last_v_max: f64,
    last_tcp_travel: f64,
}

/// Run MPC + WBC toward `goal` for `steps` physics steps.
fn run(arm: &ArmModel, q0: &DVector<f64>, goal: Isometry3<f64>, mpc_cfg: LtvConfig, steps: usize, floor_point: Option<(&str, Vector3<f64>)>) -> Run {
    let n = arm.n();
    let mut mpc = LtvMpc::new(mpc_cfg);
    let mut wbc = JointTracking::new(TrackingConfig::from_osc(&OscConfig::defaults(n), 20.0));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let hold = JointRef::at_rest(q0.clone());
    let target = move |_t: f64| goal;
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = JointCommand::default();
    let mut plan = None;
    let mut out = Run {
        q: q.clone(),
        v: v.clone(),
        min_point_z: f64::INFINITY,
        max_tau_ratio: 0.0,
        relaxed: 0,
        failures: 0,
        max_plan_us: 0.0,
        last_v_max: 0.0,
        last_tcp_travel: 0.0,
    };
    let mut tcp_mark = None;
    for k in 0..steps {
        let t = k as f64 * PHYS_DT;
        if k % MPC_EVERY == 0 {
            let g = MpcGoal { tcp: &target, posture: Some(q0) };
            match mpc.plan(arm, q.as_slice(), v.as_slice(), t, &g) {
                Ok((p, rep)) => {
                    out.relaxed += rep.relaxed as usize;
                    out.max_plan_us = out.max_plan_us.max(rep.total_us);
                    plan = Some(p);
                }
                Err(e) => {
                    if out.failures < 3 {
                        eprintln!("plan failed at t={t:.3}: {e}");
                    }
                    out.failures += 1
                }
            }
        }
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let r = plan.as_ref().map(|p| p.sample_full(t, &hold)).unwrap_or_else(|| hold.clone());
            let base = grip.command(arm, &s, &hold);
            match wbc.command(arm, &s, &r, PHYS_DT * CTRL_EVERY as f64, base.clone(), &[]) {
                Ok((c, _)) => cmd = c,
                Err(_) => {
                    out.failures += 1;
                    cmd = grip.command(arm, &s, &JointRef::at_rest(q.clone()));
                }
            }
        }
        let tau = step(arm, &mut q, &mut v, &cmd);
        if k + 1000 >= steps {
            out.last_v_max = out.last_v_max.max(v.rows(0, 6).amax());
            let p = arm.tcp_pose(q.as_slice()).translation.vector;
            let p0 = *tcp_mark.get_or_insert(p);
            out.last_tcp_travel = out.last_tcp_travel.max((p - p0).norm());
        }
        if std::env::var("MPC_TRACE").is_ok() && k % 250 == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let p = s.tcp_pose.translation.vector;
            let rot = s.tcp_pose.rotation.angle_to(&goal.rotation);
            eprintln!("t={t:.2} tcp=[{:.3} {:.3} {:.3}] rot_err={rot:.2} q={:?} v={:?}", p.x, p.y, p.z, q.as_slice()[..6].iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), v.as_slice()[..6].iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
        }
        for (i, d) in arm.dofs().iter().enumerate() {
            assert!(q[i] >= d.q_min - 0.02 && q[i] <= d.q_max + 0.02, "{} out of range: {}", d.name, q[i]);
            if i < 6 {
                out.max_tau_ratio = out.max_tau_ratio.max(tau[i].abs() / d.effort);
            }
        }
        if let Some((link, local)) = floor_point {
            let p = arm.point_state(q.as_slice(), v.as_slice(), link, local).unwrap().p;
            out.min_point_z = out.min_point_z.min(p.z);
        }
    }
    out.q = q;
    out.v = v;
    out
}

fn ready() -> DVector<f64> {
    DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01])
}

/// The MPC carries the TCP to a target 10 cm away and holds it there.
#[test]
fn mpc_wbc_reaches_tcp_target() {
    let arm = model();
    let q0 = ready();
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; arm.n()]);
    let goal = Isometry3::from_parts(
        Translation3::from(s0.tcp_pose.translation.vector + Vector3::new(0.05, 0.08, -0.04)),
        s0.tcp_pose.rotation,
    );
    let r = run(&arm, &q0, goal, LtvConfig::defaults(arm.n()), 3000, None);
    let s = arm.evaluate(r.q.as_slice(), r.v.as_slice());
    let pos_err = (s.tcp_pose.translation.vector - goal.translation.vector).norm();
    let rot_err = s.tcp_pose.rotation.angle_to(&goal.rotation);
    eprintln!("pos {pos_err:.5} m, rot {rot_err:.5} rad, |v| {:.4}, plan max {:.0} µs, failures {}", r.v.amax(), r.max_plan_us, r.failures);
    assert_eq!(r.failures, 0);
    assert_eq!(r.relaxed, 0);
    assert!(pos_err < 2e-3, "position error {pos_err}");
    assert!(rot_err < 1e-2, "rotation error {rot_err}");
    assert!(r.v.amax() < 0.02, "still moving {}", r.v.amax());
}

/// A target out of reach (straight above the base: also degenerate in yaw):
/// the arm stays inside the joint range and does not run away. It does not
/// come fully to rest: with a TCP error that cannot vanish, replanning at
/// 50 Hz leaves a ~0.1 rad/s jitter in the TCP's null space (the OSC's
/// strict posture level removes it; the MPC's posture is a soft cost).
#[test]
fn mpc_stays_bounded_toward_unreachable_target() {
    let arm = model();
    let q0 = ready();
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; arm.n()]);
    let goal = Isometry3::from_parts(Translation3::new(0.0, 0.0, 1.5), s0.tcp_pose.rotation);
    let r = run(&arm, &q0, goal, LtvConfig::defaults(arm.n()), 6000, None);
    eprintln!("last second: |v| max {:.3}, TCP travel {:.4} m, failures {}", r.last_v_max, r.last_tcp_travel, r.failures);
    assert_eq!(r.failures, 0);
    assert!(r.last_v_max < 0.3, "joints still fast: {}", r.last_v_max);
    assert!(r.last_tcp_travel < 0.01, "TCP still travelling: {}", r.last_tcp_travel);
}

/// A target below a floor: the TCP stops on the workspace box.
#[test]
fn mpc_respects_workspace_floor() {
    let arm = model();
    let q0 = ready();
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; arm.n()]);
    let z0 = s0.tcp_pose.translation.z;
    let floor = z0 - 0.06;
    let goal = Isometry3::from_parts(
        Translation3::from(s0.tcp_pose.translation.vector + Vector3::new(0.0, 0.0, -0.15)),
        s0.tcp_pose.rotation,
    );
    let mut cfg = LtvConfig::defaults(arm.n());
    cfg.workspace = Some(WorkspaceBox {
        points: vec![("end_link".into(), Vector3::zeros())],
        min: Vector3::new(-2.0, -2.0, floor),
        max: Vector3::new(2.0, 2.0, 2.0),
    });
    let r = run(&arm, &q0, goal, cfg, 3000, Some(("end_link", Vector3::zeros())));
    eprintln!("floor {floor:.4}, min z {:.4}", r.min_point_z);
    assert!(r.min_point_z > floor - 0.003, "went through the floor: {} < {floor}", r.min_point_z);
    assert!(r.min_point_z < floor + 0.01, "did not reach the floor: {}", r.min_point_z);
}

/// Every planned interval satisfies the torque, acceleration, velocity and
/// joint limits (checked with the full model at the plan's knots).
#[test]
fn plan_respects_limits() {
    let arm = model();
    let n = arm.n();
    let q0 = ready();
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let goal = Isometry3::from_parts(
        Translation3::from(s0.tcp_pose.translation.vector + Vector3::new(0.15, -0.15, 0.05)),
        s0.tcp_pose.rotation,
    );
    let mut cfg = LtvConfig::defaults(n);
    cfg.torque_scale = 0.5;
    // Re-linearize around the solution so the torque model matches the plan.
    cfg.sqp_iters = 3;
    let a_max = cfg.a_max.clone();
    let mut mpc = LtvMpc::new(cfg);
    let target = move |_t: f64| goal;
    let (plan, rep) = mpc
        .plan(&arm, q0.as_slice(), &vec![0.0; n], 0.0, &MpcGoal { tcp: &target, posture: None })
        .unwrap();
    eprintln!("{rep:?}");
    for k in 0..plan.horizon() {
        let mut qf = q0.as_slice().to_vec();
        let mut vf = vec![0.0; n];
        for (j, &i) in plan.idx.iter().enumerate() {
            qf[i] = plan.q[k][j];
            vf[i] = plan.v[k][j];
        }
        let s = arm.evaluate(&qf, &vf);
        for (j, &i) in plan.idx.iter().enumerate() {
            let d = &arm.dofs()[i];
            assert!(plan.a[k][j].abs() <= a_max[i] + 1e-6, "accel {k}/{j}");
            assert!(plan.v[k + 1][j].abs() <= d.v_max + 1e-6, "velocity {k}/{j}");
            assert!(plan.q[k + 1][j] >= d.q_min && plan.q[k + 1][j] <= d.q_max, "position {k}/{j}");
            // Torque at the knot with the full model (the plan's τ = M̄·u + h̄
            // is linearized at the last iterate).
            let tau: f64 = (0..plan.idx.len()).map(|c| s.mass[(i, plan.idx[c])] * plan.a[k][c]).sum::<f64>() + s.nle[i];
            assert!(tau.abs() <= 0.5 * d.effort * 1.03, "torque {k}/{j}: {tau} vs {}", 0.5 * d.effort);
        }
    }
}

/// A moving target (3 cm circle at 0.2 Hz): the MPC sees the target over its
/// horizon (preview), so the TCP follows with millimetre error.
#[test]
fn mpc_follows_moving_target_with_preview() {
    let arm = model();
    let n = arm.n();
    let q0 = ready();
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let (p0, rot) = (s0.tcp_pose.translation.vector, s0.tcp_pose.rotation);
    let w = 2.0 * std::f64::consts::PI * 0.2;
    let circle = move |t: f64| {
        let ramp = (t / 1.0).min(1.0);
        Isometry3::from_parts(Translation3::from(p0 + Vector3::new(0.0, 0.03 * (w * t).sin(), 0.03 * ((w * t).cos() - 1.0)) * ramp), rot)
    };
    let mut mpc = LtvMpc::new(LtvConfig::defaults(n));
    let mut wbc = JointTracking::new(TrackingConfig::from_osc(&OscConfig::defaults(n), 20.0));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let hold = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = JointCommand::default();
    let mut plan = None;
    let (mut se, mut cnt, mut emax) = (0.0, 0, 0.0f64);
    for k in 0..7000 {
        let t = k as f64 * PHYS_DT;
        if k % MPC_EVERY == 0 {
            let (p, _) = mpc.plan(&arm, q.as_slice(), v.as_slice(), t, &MpcGoal { tcp: &circle, posture: Some(&q0) }).unwrap();
            plan = Some(p);
        }
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let r = plan.as_ref().unwrap().sample_full(t, &hold);
            cmd = wbc.command(&arm, &s, &r, PHYS_DT * CTRL_EVERY as f64, grip.command(&arm, &s, &hold), &[]).unwrap().0;
            if t > 2.0 {
                let e = (s.tcp_pose.translation.vector - circle(t).translation.vector).norm();
                se += e * e;
                cnt += 1;
                emax = emax.max(e);
            }
        }
        step(&arm, &mut q, &mut v, &cmd);
    }
    let rms = (se / cnt as f64).sqrt();
    eprintln!("circle tracking: rms {:.2} mm, max {:.2} mm", rms * 1e3, emax * 1e3);
    assert!(rms < 1.5e-3, "rms {rms}");
}
