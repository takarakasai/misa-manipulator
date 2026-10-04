//! Compares OSC per-cycle computation time across formulation × QP backend.
//!
//! Has the B601-DM TCP draw a circle in a rigid-body closed loop (no MuJoCo)
//! and prints the per-cycle `Osc::command` time and TCP tracking error.
//!
//! ```sh
//! cargo run --release -p manip-control --example osc_bench
//! ```
use std::time::Instant;

use manip_control::{Feedforward, JointGains, JointImpedance, JointRef};
use manip_wbc::{Osc, OscConfig, TcpRef};
use manip_model::{ArmModel, TcpSpec};
use misa_wbc::dynamics::Formulation;
use misa_wbc::QpSolver;
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

fn main() {
    let path = format!("{}/../../models/rebot_b601_dm/rebot_b601_dm.misa", env!("CARGO_MANIFEST_DIR"));
    let mut arm = ArmModel::load(path, &TcpSpec::at_link("end_link")).unwrap();
    let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    for (i, n) in names.iter().enumerate() {
        arm.set_armature(n, [0.02, 0.02, 0.02, 0.003, 0.003, 0.003, 80.0][i]).unwrap();
        arm.set_limits(n, None, None, Some(if i < 6 { 4.0 } else { 0.05 }), None).unwrap();
    }
    let n = arm.n();
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    for (form, backend) in [
        (Formulation::Explicit, QpSolver::Clarabel),
        (Formulation::AccelSpace, QpSolver::Clarabel),
        (Formulation::Explicit, QpSolver::ActiveSet),
        (Formulation::AccelSpace, QpSolver::ActiveSet),
        (Formulation::ForceSpace, QpSolver::ActiveSet),
    ] {
        let mut cfg = OscConfig::defaults(n);
        cfg.formulation = form;
        cfg.solve.backend = backend;
        let mut osc = Osc::new(cfg);
        let grip = JointImpedance::new(JointGains::uniform(n, 2e5, 5e3), Feedforward::Gravity);
        let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
        let p0 = arm.tcp_pose(q0.as_slice());
        let posture = JointRef::at_rest(q0.clone());
        let (dt, sub) = (0.002, 4);
        let mut times = Vec::new();
        let mut errs = Vec::new();
        let mut fails = 0;
        let mut solve = Vec::new();
        let mut degr = std::collections::BTreeMap::<String, usize>::new();
        let mut cmd = grip.command(&arm, &arm.evaluate(q.as_slice(), v.as_slice()), &posture);
        for k in 0..3000 {
            let t = k as f64 * dt;
            let w = std::f64::consts::TAU * 0.25 * t;
            let d = Vector3::new(0.0, 0.05 * w.sin(), 0.05 * (w.cos() - 1.0));
            let dd = Vector3::new(0.0, 0.05 * w.cos(), -0.05 * w.sin()) * (std::f64::consts::TAU * 0.25);
            let mut r = TcpRef::at_rest(Isometry3::from_parts(Translation3::from(p0.translation.vector + d), p0.rotation));
            r.twist.fixed_rows_mut::<3>(3).copy_from(&dd);
            let s = arm.evaluate(q.as_slice(), v.as_slice());
            if k > 500 {
                errs.push((s.tcp_pose.translation.vector - r.pose.translation.vector).norm());
            }
            let t0 = Instant::now();
            match osc.command(&arm, &s, &r, &posture, dt, grip.command(&arm, &s, &posture)) {
                Ok((c, rep)) => {
                    cmd = c;
                    if rep.solve_us > 1000.0 && std::env::var("VERBOSE").is_ok() {
                        println!("    slow k={k} {:.0}µs degraded={:?}", rep.solve_us, rep.degraded);
                    }
                    if rep.degraded.is_some() { *degr.entry(rep.degraded.clone().unwrap()).or_insert(0) += 1; }
                    solve.push(rep.solve_us);
                }
                Err(_) => fails += 1,
            }
            times.push(t0.elapsed().as_secs_f64() * 1e6);
            for _ in 0..sub {
                let s = arm.evaluate(q.as_slice(), v.as_slice());
                let tau = cmd.torque_at(&q, &v);
                let qdd = s.mass.clone().cholesky().unwrap().solve(&(tau - &s.nle));
                v += qdd * (dt / sub as f64);
                q += &v * (dt / sub as f64);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        solve.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = solve.len();
        println!("  degraded: {degr:?}");
        println!("  solve only: p50 {:.0} p99 {:.0} max {:.0}", solve[ms / 2], solve[ms * 99 / 100], solve[ms - 1]);
        let m = times.len();
        let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
        println!(
            "{form:?}+{backend:?}: p50 {:.0} p99 {:.0} max {:.0} µs, fails {fails}, tcp rms {:.2} mm",
            times[m / 2], times[m * 99 / 100], times[m - 1], rms * 1e3
        );
    }
}
