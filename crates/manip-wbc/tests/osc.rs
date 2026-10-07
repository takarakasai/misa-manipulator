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

/// An arm resting slightly past a joint limit (the folded B601-DM shoulder
/// reads ~1° over) still gets a solution, and the joint does not go further out.
#[test]
fn wbc_starts_with_a_joint_past_its_limit() {
    use manip_wbc::{JointTracking, TrackingConfig};
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut q0 = DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let j2 = arm.dof("joint2").unwrap();
    q0[j2] = arm.dofs()[j2].q_max + 1f64.to_radians();
    let mut wbc = JointTracking::new(TrackingConfig::from_osc(&OscConfig::defaults(n), 20.0));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let r = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    for k in 0..1000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            cmd = wbc.command(&arm, &s, &r, PHYS_DT * CTRL_EVERY as f64, grip.command(&arm, &s, &r), &[]).expect("solvable past the limit").0;
        }
        step(&arm, &mut q, &mut v, &cmd);
        assert!(q[j2] <= q0[j2] + 1e-3, "joint2 went further out: {}", q[j2]);
    }
}

/// Friction compensation with a reference: at the reference velocity
/// (also from standstill, to break away), except where the joint clearly
/// moves the other way.
#[test]
fn gated_friction_follows_the_reference_unless_the_joint_opposes_it() {
    use manip_control::FrictionModel;
    use manip_wbc::tasks::gated_friction;
    use nalgebra::DVector;
    let f = FrictionModel { coulomb: DVector::from_element(4, 1.0), viscous: DVector::zeros(4), v_eps: 0.05 };
    let v_ref = DVector::from_vec(vec![1.0, 1.0, 1.0, 0.0]);
    let v = DVector::from_vec(vec![0.0, -0.03, -0.2, 0.5]);
    let g = gated_friction(&f, &v_ref, &v);
    let full = (1.0f64 / 0.05).tanh();
    assert_eq!(g.as_slice(), &[full, full, 0.0, 0.0]);
}

/// One physics step with an external wrench `[moment; force]` (world) on the TCP.
fn step_with_wrench(
    arm: &manip_model::ArmModel,
    q: &mut DVector<f64>,
    v: &mut DVector<f64>,
    cmd: &manip_control::JointCommand,
    wrench: &nalgebra::Vector6<f64>,
) {
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let tau = cmd.torque_at(q, v) + s.tcp_jacobian.transpose() * wrench;
    let qdd = s.mass.clone().cholesky().unwrap().solve(&(tau - &s.nle));
    *v += qdd * PHYS_DT;
    *q += &*v * PHYS_DT;
}

fn diag6(rot: f64, lin: [f64; 3]) -> nalgebra::Matrix6<f64> {
    nalgebra::Matrix6::from_diagonal(&nalgebra::Vector6::new(rot, rot, rot, lin[0], lin[1], lin[2]))
}

/// With x left free (zero stiffness) and a commanded +x force, the TCP moves
/// to a stiff wall 2 cm ahead and presses on it with that force (frictionless
/// model: the only error is the wall's damping and the PD on the other axes).
#[test]
fn osc_presses_on_a_wall_with_the_commanded_force() {
    use manip_wbc::{Compliance, TcpExtras};
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut osc = Osc::new(OscConfig::defaults(n));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let x_wall = s0.tcp_pose.translation.x + 0.02;
    let (k_wall, d_wall) = (2.0e4, 100.0);
    let push = 5.0;
    let extras = TcpExtras {
        wrench: Some(nalgebra::Vector6::new(0.0, 0.0, 0.0, push, 0.0, 0.0)),
        free: None,
        compliance: Some(Compliance { stiffness: diag6(10.0, [0.0, 500.0, 500.0]), damping: None }),
    };
    let tcp_ref = TcpRef::at_rest(s0.tcp_pose);
    let posture = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    let mut contact = Vec::new();
    for k in 0..4000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let base = grip.command(&arm, &s, &posture);
            cmd = osc.command_ext(&arm, &s, &tcp_ref, &posture, PHYS_DT * CTRL_EVERY as f64, base, &[], &extras).unwrap().0;
        }
        let s = arm.evaluate(q.as_slice(), v.as_slice());
        let pen = s.tcp_pose.translation.x - x_wall;
        let f = if pen > 0.0 { -(k_wall * pen + d_wall * s.tcp_twist[3]).max(0.0) } else { 0.0 };
        step_with_wrench(&arm, &mut q, &mut v, &cmd, &nalgebra::Vector6::new(0.0, 0.0, 0.0, f, 0.0, 0.0));
        if k >= 3000 {
            contact.push(-f);
        }
    }
    let mean = contact.iter().sum::<f64>() / contact.len() as f64;
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let off = s.tcp_pose.translation.vector - s0.tcp_pose.translation.vector;
    eprintln!("contact force {mean:.3} N (commanded {push}), TCP moved {:.1?} mm", (off * 1e3).as_slice());
    assert!((mean - push).abs() < 0.05 * push, "contact force {mean}");
    assert!(off.y.abs() < 2e-3 && off.z.abs() < 2e-3, "drifted off the pushing axis: {off:?}");
}

/// A commanded impedance yields to a sudden external push by F/K. The
/// rotational stiffness asked for (10 N·m/rad) is above what the light wrist
/// holds at 500 Hz and gets scaled down; the translation is exact.
#[test]
fn osc_compliance_yields_like_the_commanded_spring() {
    use manip_wbc::{Compliance, TcpExtras};
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut osc = Osc::new(OscConfig::defaults(n));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let k_z = 500.0;
    let extras = TcpExtras {
        wrench: None,
        free: None,
        compliance: Some(Compliance { stiffness: diag6(10.0, [500.0, 500.0, k_z]), damping: None }),
    };
    let mut scale = [0.0; 2];
    let tcp_ref = TcpRef::at_rest(s0.tcp_pose);
    let posture = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    let load = -8.0;
    for k in 0..4000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let base = grip.command(&arm, &s, &posture);
            let (c, rep) = osc.command_ext(&arm, &s, &tcp_ref, &posture, PHYS_DT * CTRL_EVERY as f64, base, &[], &extras).unwrap();
            cmd = c;
            scale = rep.compliance_scale;
        }
        step_with_wrench(&arm, &mut q, &mut v, &cmd, &nalgebra::Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, load));
    }
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let dz = s.tcp_pose.translation.z - s0.tcp_pose.translation.z;
    eprintln!("yielded {:.2} mm (F/K = {:.2} mm), cap scale {scale:?}", dz * 1e3, load / k_z * 1e3);
    assert!((dz - load / k_z).abs() < 0.05 * (load / k_z).abs(), "dz {dz}");
    assert!(v.amax() < 1e-3, "still moving {}", v.amax());
}

/// The profile's gains with the force axis left free (no impedance given).
#[test]
fn osc_free_axis_presses_with_profile_gains_elsewhere() {
    use manip_wbc::TcpExtras;
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let mut osc = Osc::new(OscConfig::defaults(n));
    let grip = JointImpedance::new(JointGains::uniform(n, 200.0, 5.0), Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let x_wall = s0.tcp_pose.translation.x + 0.02;
    let push = 5.0;
    let mut free = nalgebra::Matrix6::zeros();
    free[(3, 3)] = 1.0;
    let extras = TcpExtras { wrench: Some(nalgebra::Vector6::new(0.0, 0.0, 0.0, push, 0.0, 0.0)), free: Some(free), compliance: None };
    let tcp_ref = TcpRef::at_rest(s0.tcp_pose);
    let posture = JointRef::at_rest(q0.clone());
    let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
    let mut cmd = Default::default();
    let mut contact = Vec::new();
    for k in 0..4000 {
        if k % CTRL_EVERY == 0 {
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            let base = grip.command(&arm, &s, &posture);
            cmd = osc.command_ext(&arm, &s, &tcp_ref, &posture, PHYS_DT * CTRL_EVERY as f64, base, &[], &extras).unwrap().0;
        }
        let s = arm.evaluate(q.as_slice(), v.as_slice());
        let pen = s.tcp_pose.translation.x - x_wall;
        let f = if pen > 0.0 { -(2.0e4 * pen + 100.0 * s.tcp_twist[3]).max(0.0) } else { 0.0 };
        step_with_wrench(&arm, &mut q, &mut v, &cmd, &nalgebra::Vector6::new(0.0, 0.0, 0.0, f, 0.0, 0.0));
        if k >= 3000 {
            contact.push(-f);
        }
    }
    let mean = contact.iter().sum::<f64>() / contact.len() as f64;
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let off = s.tcp_pose.translation.vector - s0.tcp_pose.translation.vector;
    eprintln!("free axis: contact {mean:.3} N, TCP moved {:.1?} mm, tilt {:.4} rad", (off * 1e3).as_slice(), s.tcp_pose.rotation.angle_to(&s0.tcp_pose.rotation));
    assert!((mean - push).abs() < 0.05 * push, "contact force {mean}");
}
