use super::*;
use approx::assert_relative_eq;

/// A 3-DOF arm + two fingers with a mimic follower. Same shape as the reBot B601 gripper.
const ARM_URDF: &str = r#"
<robot name="toy">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l1"><inertial><origin xyz="0 0 0.1"/><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.002" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l2"><inertial><origin xyz="0.15 0 0"/><mass value="0.8"/><inertia ixx="0.002" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="l3"><inertial><origin xyz="0.1 0 0"/><mass value="0.5"/><inertia ixx="0.001" iyy="0.004" izz="0.004" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="ee"><inertial><mass value="0.2"/><inertia ixx="0.0005" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="fl"><inertial><mass value="0.05"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="fr"><inertial><mass value="0.05"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0"/><axis xyz="0 1 0"/><limit lower="-2.5" upper="2.5" effort="10" velocity="5"/></joint>
  <joint name="tip" type="fixed"><parent link="l3"/><child link="ee"/><origin xyz="0.2 0 0"/></joint>
  <joint name="finger_left" type="prismatic"><parent link="ee"/><child link="fl"/><axis xyz="0 1 0"/><limit lower="0" upper="0.03" effort="8" velocity="0.1"/></joint>
  <joint name="finger_right" type="prismatic"><parent link="ee"/><child link="fr"/><axis xyz="0 1 0"/><limit lower="-0.03" upper="0" effort="8" velocity="0.1"/><mimic joint="finger_left" multiplier="-1" offset="0"/></joint>
</robot>
"#;

fn toy(tcp: &TcpSpec) -> ArmModel {
    let file = misarta_formats::urdf::import_str(ARM_URDF).unwrap().file;
    ArmModel::from_file(file, tcp).unwrap()
}

fn tcp_offset() -> TcpSpec {
    TcpSpec {
        link: "ee".into(),
        xyz: [0.05, 0.01, -0.02],
        rpy: [0.1, -0.2, 0.3],
    }
}

#[test]
fn mimic_slave_is_not_a_dof() {
    let m = toy(&TcpSpec::at_link("ee"));
    let names: Vec<_> = m.dofs().iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["j1", "j2", "j3", "finger_left"]);
    assert_eq!(m.raw().nv, 5);
    let q = m.full_q(&[0.0, 0.0, 0.0, 0.02]);
    let fr = m.raw().joints.iter().position(|j| j.name == "finger_right").unwrap();
    assert_relative_eq!(q[m.raw().q_idx[fr]], -0.02);
    let fl = &m.dofs()[3];
    assert_eq!((fl.q_min, fl.q_max, fl.effort), (0.0, 0.03, 8.0));
}

#[test]
fn reduced_mass_is_symmetric_positive() {
    let m = toy(&tcp_offset());
    let s = m.evaluate(&[0.3, -0.4, 0.9, 0.01], &[0.0; 4]);
    assert_relative_eq!(s.mass.clone(), s.mass.transpose(), epsilon = 1e-12);
    assert!(s.mass.clone().cholesky().is_some());
}

/// Gravity torque is the gradient of potential energy. Checked against finite differences in independent coordinates.
#[test]
fn gravity_matches_potential_gradient() {
    let m = toy(&tcp_offset());
    let q = [0.3, -0.4, 0.9, 0.01];
    let g = m.gravity(&q);
    let potential = |q: &[f64]| -> f64 {
        let qf = m.full_q(q);
        let data = misarta::fk::forward_kinematics(m.raw(), &qf);
        let mut u = 0.0;
        for (i, inertia) in m.raw().inertias.iter().enumerate() {
            let com = data.oMi[i] * nalgebra::Point3::from(inertia.center_of_mass);
            u += inertia.mass * 9.81 * com.z;
        }
        u
    };
    for i in 0..q.len() {
        let h = 1e-6;
        let mut qp = q;
        let mut qm = q;
        qp[i] += h;
        qm[i] -= h;
        let du = (potential(&qp) - potential(&qm)) / (2.0 * h);
        assert_relative_eq!(g[i], du, epsilon = 1e-5);
    }
}

/// `J·v` matches the time derivative of the TCP pose (with an offset TCP).
#[test]
fn tcp_jacobian_matches_finite_difference() {
    let m = toy(&tcp_offset());
    let q = [0.3, -0.4, 0.9, 0.01];
    let v = [0.7, -0.2, 0.5, 0.02];
    let s = m.evaluate(&q, &v);
    let dt = 1e-6;
    let step = |sign: f64| -> Isometry3<f64> {
        let qq: Vec<f64> = q.iter().zip(&v).map(|(a, b)| a + sign * dt * b).collect();
        m.tcp_pose(&qq)
    };
    let (p, n) = (step(1.0), step(-1.0));
    let lin = (p.translation.vector - n.translation.vector) / (2.0 * dt);
    let dr = p.rotation * n.rotation.inverse();
    let ang = dr.scaled_axis() / (2.0 * dt);
    assert_relative_eq!(s.tcp_twist.fixed_rows::<3>(3).into_owned(), lin, epsilon = 1e-6);
    assert_relative_eq!(s.tcp_twist.fixed_rows::<3>(0).into_owned(), ang, epsilon = 1e-6);
}

/// `J̇v` is the rate of change of the TCP spatial velocity when moving with q̈ = 0.
#[test]
fn jdot_v_matches_finite_difference() {
    let m = toy(&tcp_offset());
    let q = [0.3, -0.4, 0.9, 0.01];
    let v = [0.7, -0.2, 0.5, 0.02];
    let s = m.evaluate(&q, &v);
    let dt = 1e-5;
    let twist_at = |sign: f64| -> Vector6<f64> {
        let qq: Vec<f64> = q.iter().zip(&v).map(|(a, b)| a + sign * dt * b).collect();
        m.evaluate(&qq, &v).tcp_twist
    };
    let fd = (twist_at(1.0) - twist_at(-1.0)) / (2.0 * dt);
    assert_relative_eq!(s.tcp_jdot_v, fd, epsilon = 1e-4);
}

#[test]
fn unknown_tcp_link_is_rejected() {
    let file = misarta_formats::urdf::import_str(ARM_URDF).unwrap().file;
    let e = ArmModel::from_file(file, &TcpSpec::at_link("nope")).unwrap_err();
    assert!(matches!(e, ModelError::UnknownLink(_)));
}

#[test]
fn self_collision_on_the_b601() {
    use crate::collision::SelfCollision;
    let path = format!("{}/../../models/rebot_b601_dm/rebot_b601_dm.misa", env!("CARGO_MANIFEST_DIR"));
    let arm = ArmModel::load(path, &TcpSpec::at_link("end_link")).unwrap();
    let mut sc = SelfCollision::build(&arm).unwrap();
    let before = sc.pair_count();
    let rest = vec![0.0; arm.n()];
    let dropped = sc.exclude_close_at(&arm, &rest, 0.005);
    eprintln!("pairs {before} -> {} ; dropped at rest: {dropped:?}", sc.pair_count());
    let ready = [0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01];
    let t0 = std::time::Instant::now();
    let d_ready = sc.min_distance(&arm, &ready);
    eprintln!("min distance at ready {d_ready:.4} m ({:?})", t0.elapsed());
    for p in sc.close_pairs(&arm, &ready, 0.05) {
        eprintln!("  ready: {} - {} {:.4}", p.link_a, p.link_b, p.distance);
    }
    assert!(d_ready > 0.005, "ready pose should be clear: {d_ready}");
    // Elbow folded hard with the wrist bent back toward the upper arm / base.
    let tight = [0.0, -0.3, -2.9, -1.8, 0.0, 0.0, 0.0];
    let close = sc.close_pairs(&arm, &tight, 0.05);
    eprintln!("tight pose: {:?}", close.iter().take(3).map(|p| (&p.link_a, &p.link_b, p.distance)).collect::<Vec<_>>());
}
