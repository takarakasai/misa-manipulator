//! Shared harness: the real B601 models, rigid-body integration with the
//! motor MIT law evaluated every physics step (1 kHz) while the controller
//! runs at 500 Hz (the command is held between control cycles, as on the
//! hardware).
#![allow(dead_code)]

use manip_model::{ArmModel, TcpSpec};
use nalgebra::DVector;

pub const PHYS_DT: f64 = 0.001;
pub const CTRL_EVERY: usize = 2;

pub fn model(name: &str, tcp: &str) -> ArmModel {
    let path = format!(
        "{}/../../models/{name}/{name}.misa",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut arm = ArmModel::load(path, &TcpSpec::at_link(tcp)).unwrap();
    // Rough reflected rotor inertia of geared motors (DM4340P / DM4310 class).
    // Keeps M non-singular even in a model whose finger links are massless.
    let arm_names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    for (i, name) in arm_names.iter().enumerate() {
        // The gripper gets a large, hardware-like reflected inertia (also checks that it's excluded from OSC).
        let a = if i < 3 { 0.02 } else if i < 6 { 0.005 } else { 80.0 };
        arm.set_armature(name, a).unwrap();
        // The URDF velocity (50 / 200 rad/s) is an order of magnitude above the
        // real motors. Use the same values the profile narrows it to.
        arm.set_limits(name, None, None, Some(if i < 6 { 4.0 } else { 0.1 }), None).unwrap();
    }
    arm
}

/// One physics step: evaluate MIT and integrate forward dynamics (semi-implicit Euler).
pub fn step(arm: &ArmModel, q: &mut DVector<f64>, v: &mut DVector<f64>, cmd: &manip_control::JointCommand) {
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let tau = cmd.torque_at(q, v);
    let qdd = s.mass.clone().cholesky().unwrap().solve(&(tau - &s.nle));
    *v += qdd * PHYS_DT;
    *q += &*v * PHYS_DT;
}

