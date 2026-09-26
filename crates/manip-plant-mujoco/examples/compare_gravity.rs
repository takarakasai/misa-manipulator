//! Compares misarta's (manip-model) gravity term with MuJoCo's `qfrc_bias` at the same pose.
//!
//! Checks that the controller's model and the sim's model agree. If they don't, gains and
//! gravity compensation tuned in sim are already off inside the sim, before ever reaching
//! hardware (cf. the incident where MuJoCo synthesized mass for zero-mass links from
//! volume × water density).
//!
//! ```sh
//! cd crates/manip-plant-mujoco
//! cargo run --release --example compare_gravity -- ../../models/rebot_b601_dm/rebot_b601_dm.misa end_link
//! ```
//!
//! Axes with a mimic-dependent joint (the gripper) cannot be compared: misarta reports the
//! independent-coordinate value (leader + follower summed), MuJoCo only the leader joint's.
use articara::mjcf::MjcfExportOptions;
use articara::mujoco_sim::MujocoSim;
use articara::rbd::model::ActuatorMode;
use articara::robot::RobotModel;
use manip_model::{ArmModel, TcpSpec};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 {
        eprintln!("usage: {} <model.misa> <tcp-link>", a[0]);
        std::process::exit(2);
    }
    let arm = ArmModel::load(&a[1], &TcpSpec::at_link(&a[2])).unwrap();
    let mut robot = RobotModel::from_misa(std::path::Path::new(&a[1])).unwrap();
    for j in robot.joints.iter_mut() {
        j.actuator_mode = ActuatorMode::Torque;
    }
    let mut sim = MujocoSim::new(
        &robot,
        MjcfExportOptions {
            base_pos: Some([0.0; 3]),
            base_locked_axes: [true; 6],
            add_actuators: true,
            bake_joint_position_limits: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut worst: f64 = 0.0;
    for q in [
        vec![0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.0],
        vec![0.5, -0.5, -2.0, -0.8, 0.7, 1.0, 0.01],
        vec![-1.0, 1.0, 1.5, 0.5, -0.4, -1.2, 0.02],
    ] {
        let q: Vec<f64> = q.into_iter().take(arm.n()).collect();
        let qf = arm.full_q(&q);
        for (i, j) in arm.raw().joints.iter().enumerate().skip(1) {
            if j.joint_type.nq() == 0 {
                continue;
            }
            let adr = sim.joint_dof_adr(&j.name).expect("joint in mujoco");
            sim.mj_data_mut().qpos_mut()[adr] = qf[arm.raw().q_idx[i]];
        }
        sim.mj_data_mut().qvel_mut().iter_mut().for_each(|v| *v = 0.0);
        sim.mj_data_mut().forward();
        let bias = sim.qfrc_bias();
        let g = arm.gravity(&q);
        let chain = arm.tcp_chain();
        println!("q = {q:?}");
        for (k, d) in arm.dofs().iter().enumerate() {
            let adr = sim.joint_dof_adr(&d.name).unwrap();
            let note = if chain.contains(&k) {
                worst = worst.max((g[k] - bias[adr]).abs());
                ""
            } else {
                "  (mimic: not compared)"
            };
            println!("  {:<14} misarta {:+9.4}  mujoco {:+9.4}{note}", d.name, g[k], bias[adr]);
        }
    }
    println!("max |Δ| on the TCP chain = {worst:.2e} N·m");
}
