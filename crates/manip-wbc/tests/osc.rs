//! OSC on the real B601 model (rigid-body integration, no MuJoCo).

mod common;
use common::{model, step, CTRL_EVERY, PHYS_DT};

use manip_control::{Feedforward, JointGains, JointImpedance, JointRef};
use manip_wbc::{Osc, OscConfig, TcpRef};
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

/// OSC carries the TCP to the target position, keeps orientation, and respects joint limits.
#[test]
fn osc_reaches_tcp_target_within_limits() {
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut osc = Osc::new(OscConfig::defaults(n));
    // Outside the TCP chain (the gripper) is held by joint impedance.
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let goal = Isometry3::from_parts(
        Translation3::from(s0.tcp_pose.translation.vector + Vector3::new(0.05, 0.08, -0.04)),
        s0.tcp_pose.rotation,
    );
    let tcp_ref = TcpRef::at_rest(goal);
    let posture = JointRef::at_rest(q0.clone());

    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    for k in 0..3000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let (c, _) = osc
                .command(&arm, &s, &tcp_ref, &posture, PHYS_DT * CTRL_EVERY as f64, grip.command(&arm, &s, &posture))
                .unwrap();
            cmd = c;
        }
        step(&arm, &mut q, &mut v, &cmd);
        for (i, d) in arm.dofs().iter().enumerate() {
            assert!(q[i] >= d.q_min - 1e-3 && q[i] <= d.q_max + 1e-3, "{} out of range: {}", d.name, q[i]);
        }
    }
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let pos_err = (s.tcp_pose.translation.vector - goal.translation.vector).norm();
    let rot_err = s.tcp_pose.rotation.angle_to(&goal.rotation);
    assert!(pos_err < 1e-3, "position error {pos_err}");
    assert!(rot_err < 5e-3, "rotation error {rot_err}");
}

/// Even with a target outside the range of motion, the CBF stops the joints short of the limit.
#[test]
fn osc_cbf_stops_at_joint_limit() {
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut osc = Osc::new(OscConfig::defaults(n));
    // Outside the TCP chain (the gripper) is held by joint impedance.
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    // Unreachably far (1.5 m straight above the base).
    let goal = Isometry3::from_parts(Translation3::new(0.0, 0.0, 1.5), s0.tcp_pose.rotation);
    let tcp_ref = TcpRef::at_rest(goal);
    let posture = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    let mut degraded = 0;
    for k in 0..4000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            match osc.command(&arm, &s, &tcp_ref, &posture, PHYS_DT * CTRL_EVERY as f64, grip.command(&arm, &s, &posture)) {
                Ok((c, _)) => cmd = c,
                Err(_) => degraded += 1,
            }
        }
        step(&arm, &mut q, &mut v, &cmd);
        for (i, d) in arm.dofs().iter().enumerate() {
            assert!(q[i] >= d.q_min - 0.02 && q[i] <= d.q_max + 0.02, "{} out of range: {}", d.name, q[i]);
        }
    }
    eprintln!("final q = {:.3}\nfinal v = {:.3}\ndegraded = {degraded}", q.transpose(), v.transpose());
    assert!(v.amax() < 0.05, "still moving: {}", v.amax());
    assert!(degraded < 20, "too many degraded ticks: {degraded}");
}
