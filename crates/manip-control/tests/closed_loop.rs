//! Runs closed loops on the real B601 model (rigid-body integration, no MuJoCo).
//!
//! The motor MIT law `τ = kp(q* − q) + kd(v* − v) + τff` is evaluated every
//! physics step (1 kHz) while the control law runs at 500 Hz. Same structure
//! as the hardware: "the command is held between control cycles and the PD
//! runs inside the motor".

use manip_control::{Feedforward, JointGains, JointImpedance, JointRef, JointShaper, ShaperLimits};
use manip_model::{ArmModel, TcpSpec};
use nalgebra::DVector;

const PHYS_DT: f64 = 0.001;
const CTRL_EVERY: usize = 2;

fn model(name: &str, tcp: &str) -> ArmModel {
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
fn step(arm: &ArmModel, q: &mut DVector<f64>, v: &mut DVector<f64>, cmd: &manip_control::JointCommand) {
    let s = arm.evaluate(q.as_slice(), v.as_slice());
    let tau = cmd.torque_at(q, v);
    let qdd = s.mass.clone().cholesky().unwrap().solve(&(tau - &s.nle));
    *v += qdd * PHYS_DT;
    *q += &*v * PHYS_DT;
}

/// With gravity compensation alone in a bent posture, the arm stays in place.
#[test]
fn gravity_comp_holds_bent_pose() {
    for (name, tcp, q0) in [
        ("rebot_b601_dm", "end_link", [0.3, -1.2, -1.0, 0.4, 0.2, 0.0, 0.01]),
        ("rebot_b601_rs", "gripper_end", [0.3, 1.2, 1.0, 0.4, 0.2, 0.0, 0.01]),
    ] {
        let arm = model(name, tcp);
        let ctl = JointImpedance::gravity_comp(arm.n(), 0.0);
        let mut q = DVector::from_row_slice(&q0);
        let mut v = DVector::zeros(arm.n());
        let r = JointRef::at_rest(q.clone());
        let mut cmd = Default::default();
        for k in 0..2000 {
            if k % CTRL_EVERY == 0 {
                let s = arm.evaluate(q.as_slice(), v.as_slice());
                cmd = ctl.command(&arm, &s, &r);
            }
            step(&arm, &mut q, &mut v, &cmd);
        }
        let drift = (&q - DVector::from_row_slice(&q0)).amax();
        assert!(drift < 1e-6, "{name}: drift {drift}");
    }
}

/// Joint impedance + inverse-dynamics FF tracks a shaped reference with small error.
#[test]
fn joint_impedance_tracks_shaped_reference() {
    let arm = model("rebot_b601_dm", "end_link");
    let n = arm.n();
    let gains = JointGains::new(
        DVector::from_row_slice(&[60.0, 60.0, 60.0, 15.0, 15.0, 15.0, 200.0]),
        DVector::from_row_slice(&[3.0, 3.0, 3.0, 0.8, 0.8, 0.8, 5.0]),
    );
    let ctl = JointImpedance::new(gains.clone(), Feedforward::InverseDynamics);
    let ctl_g = JointImpedance::new(gains, Feedforward::Gravity);
    let q0 = DVector::from_row_slice(&[0.0, -0.5, -0.8, 0.0, 0.0, 0.0, 0.0]);
    let target = DVector::from_row_slice(&[0.8, -1.4, -1.6, 0.5, -0.4, 1.0, 0.02]);
    let limits = ShaperLimits {
        v_max: DVector::from_element(n, 1.5),
        a_max: DVector::from_element(n, 6.0),
        time_constant_s: 0.05,
    };

    let run = |ctl: &JointImpedance| -> f64 {
        let mut shaper = JointShaper::new(limits.clone(), q0.clone());
        let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
        let mut cmd = Default::default();
        let mut worst: f64 = 0.0;
        for k in 0..3000 {
            if k % CTRL_EVERY == 0 {
                let r = shaper.step(&target, PHYS_DT * CTRL_EVERY as f64).clone();
                let s = arm.evaluate(q.as_slice(), v.as_slice());
                cmd = ctl.command(&arm, &s, &r);
                worst = worst.max((&r.q - &q).rows(0, 6).amax());
            }
            step(&arm, &mut q, &mut v, &cmd);
        }
        assert!((&q - &target).amax() < 2e-3, "did not settle: {}", (&q - &target).amax());
        worst
    };
    let err_id = run(&ctl);
    let err_g = run(&ctl_g);
    // Inverse-dynamics FF tracks better than gravity-only (evidence the model helps).
    assert!(err_id < err_g, "ID {err_id} vs gravity-only {err_g}");
    assert!(err_id < 0.02, "tracking error {err_id}");
}
