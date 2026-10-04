//! Wall time of one plan (LTV-MPC, iLQR) on the B601-DM (warm-started, replanning
//! every 20 ms toward a target 10 cm away), per horizon length.
//!
//!     cargo run --release -p manip-mpc --example mpc_bench

use manip_model::{ArmModel, TcpSpec};
use manip_mpc::{IlqrConfig, IlqrMpc, LtvConfig, LtvMpc, MpcGoal, Planner};
use nalgebra::{DVector, Isometry3, Translation3, Vector3};

fn main() {
    let path = format!("{}/../../models/rebot_b601_dm/rebot_b601_dm.misa", env!("CARGO_MANIFEST_DIR"));
    let arm = ArmModel::load(path, &TcpSpec::at_link("end_link")).unwrap();
    let n = arm.n();
    let q0 = DVector::from_row_slice(&[0.0, -1.2, -1.2, 0.3, 0.0, 0.0, 0.01]);
    let s0 = arm.evaluate(q0.as_slice(), &vec![0.0; n]);
    let goal = Isometry3::from_parts(
        Translation3::from(s0.tcp_pose.translation.vector + Vector3::new(0.05, 0.08, -0.04)),
        s0.tcp_pose.rotation,
    );
    let target = move |_t: f64| goal;
    println!("{:>6} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10}", "", "horizon", "dt", "median us", "p95 us", "lin us", "solve us");
    let mut planners: Vec<(&str, usize, f64, Box<dyn Planner>)> = Vec::new();
    for (horizon, dt) in [(10, 0.05), (20, 0.05), (30, 0.05), (20, 0.025)] {
        let mut cfg = LtvConfig::defaults(n);
        cfg.horizon = horizon;
        cfg.dt = dt;
        planners.push(("ltv", horizon, dt, Box::new(LtvMpc::new(cfg))));
    }
    for (horizon, dt) in [(15, 0.04), (25, 0.04)] {
        let mut cfg = IlqrConfig::defaults();
        cfg.horizon = horizon;
        cfg.dt = dt;
        planners.push(("ilqr", horizon, dt, Box::new(IlqrMpc::new(cfg))));
    }
    for (name, horizon, dt, mut mpc) in planners {
        let (mut q, mut v) = (q0.clone(), DVector::zeros(n));
        let mut times = Vec::new();
        let (mut lin, mut qp) = (0.0, 0.0);
        for k in 0..100 {
            let t = k as f64 * 0.02;
            let (plan, rep) = mpc.plan(&arm, q.as_slice(), v.as_slice(), t, &MpcGoal { tcp: &target, posture: None }).unwrap();
            // Follow the plan exactly (no plant) to exercise the warm start.
            let r = plan.sample_full(t + 0.02, &manip_control::JointRef::at_rest(q.clone()));
            q = r.q;
            v = r.v;
            if k >= 5 {
                times.push(rep.total_us);
                lin += rep.linearize_us;
                qp += rep.qp_us;
            }
        }
        let m = times.len() as f64;
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:>6} {:>8} {:>6.3} {:>10.0} {:>10.0} {:>10.0} {:>10.0}",
            name,
            horizon,
            dt,
            times[times.len() / 2],
            times[(times.len() * 95) / 100],
            lin / m,
            qp / m
        );
    }
}
