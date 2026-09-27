//! Bring-up commands: talk to the arm before trusting the control loop with it.
//!
//! Order on a freshly assembled arm (see doc/handover.md, bring-up):
//!
//! 1. `hw scan`: every motor answers, and the measured pose (model frame) is
//!    inside the range of motion. Nothing is energized.
//! 2. `hw monitor`: live angles while moving the arm by hand. Nothing is energized.
//! 3. `hw sign`: per joint, move it by hand the way the model says is positive
//!    (the hint comes from the model's Jacobian); a reversed sign in
//!    `[hardware]` shows up as REVERSED.
//! 4. `hw jog`: energize, hold with gravity feedforward, move one joint a few
//!    degrees and back, then fold and release.
//!
//! 1-3 never energize the motors, so they are safe on an arm whose zero,
//! sign or ID assignment is still unknown.

use std::time::{Duration, Instant};

use manip_model::ArmModel;
use misa_core::{Command, Observation, Plant};

/// Read the arm a few times without energizing. Returns the last observation.
pub fn read_passive(plant: &mut dyn Plant, n: usize, dur: Duration) -> Result<Observation, String> {
    let mut obs = Observation::empty(n, 0);
    let idle = Command::idle(n);
    let t0 = Instant::now();
    while t0.elapsed() < dur {
        plant.exchange(&idle, &mut obs)?;
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(obs)
}

/// One line per joint: responded, angle, range check. Returns whether all passed.
pub fn scan(plant: &mut dyn Plant, arm: &ArmModel) -> Result<bool, String> {
    let obs = read_passive(plant, arm.n(), Duration::from_millis(500))?;
    let mut ok = true;
    println!("{:<16} {:>8} {:>12} {:>22}  verdict", "joint", "reply", "angle", "range");
    for (i, d) in arm.dofs().iter().enumerate() {
        let a = obs.axes()[i];
        let (val, unit) = display(d, a.position_rad);
        let (lo, _) = display(d, d.q_min);
        let (hi, _) = display(d, d.q_max);
        let verdict = if !a.health.valid {
            ok = false;
            "NO REPLY (check id / wiring / power)"
        } else if !d.within(a.position_rad) {
            ok = false;
            "OUT OF RANGE (check zero / sign in [hardware])"
        } else {
            "ok"
        };
        let reply = if a.health.valid { format!("{:.0}ms", a.health.age.as_secs_f64() * 1e3) } else { "-".into() };
        println!(
            "{:<16} {:>8} {:>9.2}{unit:<3} [{lo:>8.2}, {hi:>8.2}]{unit:<3}  {verdict}",
            d.name, reply, val
        );
    }
    println!("{}", plant.status_line());
    Ok(ok)
}

/// Live angles without energizing, until `dur` elapses.
pub fn monitor(plant: &mut dyn Plant, arm: &ArmModel, dur: Duration) -> Result<(), String> {
    let n = arm.n();
    let mut obs = Observation::empty(n, 0);
    let idle = Command::idle(n);
    let t0 = Instant::now();
    let mut last = Instant::now() - Duration::from_secs(1);
    while t0.elapsed() < dur {
        plant.exchange(&idle, &mut obs)?;
        if last.elapsed() >= Duration::from_millis(200) {
            let cols: Vec<String> = arm
                .dofs()
                .iter()
                .zip(obs.axes())
                .map(|(d, a)| {
                    let (v, u) = display(d, a.position_rad);
                    if a.health.valid { format!("{}={v:+.1}{u}", d.name) } else { format!("{}=--", d.name) }
                })
                .collect();
            eprintln!("{}", cols.join(" "));
            last = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

/// How to move joint `i` by hand in its model-positive direction, at pose `q`.
///
/// From the TCP Jacobian column: the dominant world axis of the TCP's linear
/// velocity if the joint moves the TCP noticeably, otherwise the axis it
/// rotates the TCP about. Joints outside the TCP chain (the gripper) are told
/// "toward the upper limit".
pub fn positive_hint(arm: &ArmModel, q: &[f64], i: usize) -> String {
    let s = arm.evaluate(q, &vec![0.0; q.len()]);
    let col = s.tcp_jacobian.column(i);
    let lin = nalgebra::Vector3::new(col[3], col[4], col[5]);
    let ang = nalgebra::Vector3::new(col[0], col[1], col[2]);
    let axis = |v: &nalgebra::Vector3<f64>| {
        let k = v.iamax();
        let sign = if v[k] >= 0.0 { "+" } else { "-" };
        format!("{sign}{}", ["x", "y", "z"][k])
    };
    let d = &arm.dofs()[i];
    if lin.norm() > 0.02 {
        format!("so the gripper tip moves toward {} (world)", axis(&lin))
    } else if ang.norm() > 0.5 {
        format!("so the gripper turns about {} (world, right hand rule)", axis(&ang))
    } else {
        let (hi, u) = display(d, d.q_max);
        format!("toward its upper limit ({hi:.2}{u}; for a gripper finger that is usually 'open')")
    }
}

/// Verdict for one joint from the angle change seen after the operator moved
/// it in the hinted (model-positive) direction.
pub fn sign_verdict(delta: f64, threshold: f64) -> Option<bool> {
    if delta.abs() < threshold { None } else { Some(delta > 0.0) }
}

/// Interactive sign check. For each joint: print the hint, wait until the
/// joint moved by more than `threshold` (or `timeout`), report OK / REVERSED.
pub fn sign_check(plant: &mut dyn Plant, arm: &ArmModel, timeout: Duration) -> Result<bool, String> {
    let n = arm.n();
    let mut obs = read_passive(plant, n, Duration::from_millis(300))?;
    let idle = Command::idle(n);
    let mut all_ok = true;
    for (i, d) in arm.dofs().iter().enumerate() {
        let q: Vec<f64> = obs.axes().iter().map(|a| a.position_rad).collect();
        let threshold = if d.kind == manip_model::DofKind::Prismatic { 0.003 } else { 5f64.to_radians() };
        eprintln!("\n[{}] move it by hand {}", d.name, positive_hint(arm, &q, i));
        let start = q[i];
        let t0 = Instant::now();
        let verdict = loop {
            plant.exchange(&idle, &mut obs)?;
            if let Some(v) = sign_verdict(obs.axes()[i].position_rad - start, threshold) {
                break Some(v);
            }
            if t0.elapsed() > timeout {
                break None;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        match verdict {
            Some(true) => println!("{:<16} OK", d.name),
            Some(false) => {
                all_ok = false;
                println!("{:<16} REVERSED: flip `sign` of this motor in [hardware]", d.name)
            }
            None => {
                all_ok = false;
                println!("{:<16} no movement seen (skipped)", d.name)
            }
        }
        // Let the operator put it back before the next joint.
        std::thread::sleep(Duration::from_millis(1500));
        obs = read_passive(plant, n, Duration::from_millis(200))?;
    }
    Ok(all_ok)
}

/// Degrees for revolute joints, millimetres for prismatic ones.
fn display(d: &manip_model::Dof, x: f64) -> (f64, &'static str) {
    match d.kind {
        manip_model::DofKind::Revolute => (x.to_degrees(), "°"),
        manip_model::DofKind::Prismatic => (x * 1e3, "mm"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_needs_a_clear_move() {
        assert_eq!(sign_verdict(0.01, 0.05), None);
        assert_eq!(sign_verdict(0.2, 0.05), Some(true));
        assert_eq!(sign_verdict(-0.2, 0.05), Some(false));
    }
}
