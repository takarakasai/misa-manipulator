//! Arm model inspection. Prints independent DOFs, ranges of motion, the TCP and
//! maximum gravity torque.
//!
//! ```sh
//! manip-inspect models/rebot_b601_dm/rebot_b601_dm.misa --tcp end_link [--q 0,-1,-1,0,0,0,0]
//! ```
//!
//! The maximum gravity torque comes from a coarse sweep over the range of
//! motion. It is meant to be compared against motor ratings to see "how much
//! is eaten just by holding", not an exact maximum.

use clap::Parser;
use manip_model::{ArmModel, TcpSpec};

#[derive(Parser)]
struct Args {
    model: std::path::PathBuf,
    /// Link to use as the TCP.
    #[arg(long)]
    tcp: String,
    /// Also print values at these joint angles (independent-DOF order, comma-separated).
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    q: Option<Vec<f64>>,
    /// Number of steps per axis for the gravity torque sweep.
    #[arg(long, default_value_t = 7)]
    steps: usize,
}

fn main() -> Result<(), String> {
    let a = Args::parse();
    let arm = ArmModel::load(&a.model, &TcpSpec::at_link(&a.tcp)).map_err(|e| e.to_string())?;
    println!("robot: {}  n={}  tcp={}", arm.name(), arm.n(), a.tcp);
    println!("{:<16} {:>9} {:>9} {:>8} {:>8}", "dof", "min", "max", "v_max", "effort");
    for d in arm.dofs() {
        println!(
            "{:<16} {:>9.4} {:>9.4} {:>8.2} {:>8.2}   {:?}",
            d.name, d.q_min, d.q_max, d.v_max, d.effort, d.kind
        );
    }
    let mut masses = 0.0;
    for (name, inertia) in arm.raw().link_names.iter().zip(&arm.raw().inertias) {
        masses += inertia.mass;
        println!("  link {name:<22} m={:.4}", inertia.mass);
    }
    println!("total mass {masses:.3} kg");

    let show = |label: &str, q: &[f64]| {
        let s = arm.evaluate(q, &vec![0.0; q.len()]);
        let p = s.tcp_pose.translation.vector;
        let (r, pi, y) = s.tcp_pose.rotation.euler_angles();
        println!("\n[{label}] q = {q:?}");
        println!("  tcp xyz = [{:.4}, {:.4}, {:.4}] rpy = [{r:.3}, {pi:.3}, {y:.3}]", p.x, p.y, p.z);
        println!("  gravity = {:?}", s.gravity.iter().map(|g| (g * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        let svd = s.tcp_jacobian.clone().svd(false, false);
        println!("  J singular values = {:?}", svd.singular_values.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
    };
    let zero = vec![0.0; arm.n()];
    show("zero", &zero);
    show("mid", arm.mid_q().as_slice());
    if let Some(q) = &a.q {
        show("given", q);
    }
    for name in arm.pose_names() {
        let q = arm.named_pose(name, &zero).map_err(|e| e.to_string())?;
        show(name, q.as_slice());
    }

    // Maximum gravity torque: sweep each axis over its range of motion (a grid of
    // combinations; with many axes that gets heavy, so the grid covers only the
    // 3 shoulder-to-elbow axes and the rest take 0 and both ends).
    let n = arm.n();
    let grid = |i: usize| -> Vec<f64> {
        let d = &arm.dofs()[i];
        let (lo, hi) = if d.q_min.is_finite() { (d.q_min, d.q_max) } else { (-std::f64::consts::PI, std::f64::consts::PI) };
        let k = if i < 4 { a.steps } else { 3 };
        (0..k).map(|j| lo + (hi - lo) * j as f64 / (k - 1) as f64).collect()
    };
    let grids: Vec<Vec<f64>> = (0..n).map(grid).collect();
    let mut idx = vec![0usize; n];
    let mut max_g = vec![0.0f64; n];
    loop {
        let q: Vec<f64> = idx.iter().enumerate().map(|(i, &k)| grids[i][k]).collect();
        let g = arm.gravity(&q);
        for i in 0..n {
            max_g[i] = max_g[i].max(g[i].abs());
        }
        let mut i = 0;
        loop {
            if i == n {
                println!("\nmax |gravity| over the sweep (vs effort):");
                for (d, g) in arm.dofs().iter().zip(&max_g) {
                    println!("  {:<16} {:>7.3} / {:>6.2}  ({:>5.1} %)", d.name, g, d.effort, 100.0 * g / d.effort);
                }
                return Ok(());
            }
            idx[i] += 1;
            if idx[i] < grids[i].len() {
                break;
            }
            idx[i] = 0;
            i += 1;
        }
    }
}
