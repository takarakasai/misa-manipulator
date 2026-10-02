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
    move_hint(arm, q, i, 1.0)
}

/// Like [`positive_hint`], for the direction `dir` (+1 model-positive, -1 negative).
pub fn move_hint(arm: &ArmModel, q: &[f64], i: usize, dir: f64) -> String {
    let s = arm.evaluate(q, &vec![0.0; q.len()]);
    let col = s.tcp_jacobian.column(i);
    let lin = nalgebra::Vector3::new(col[3], col[4], col[5]) * dir;
    let ang = nalgebra::Vector3::new(col[0], col[1], col[2]) * dir;
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
    } else if dir > 0.0 {
        let (hi, u) = display(d, d.q_max);
        format!("toward its upper limit ({hi:.2}{u}; for a gripper finger that is usually 'open')")
    } else {
        let (lo, u) = display(d, d.q_min);
        format!("toward its lower limit ({lo:.2}{u}; for a gripper finger that is usually 'closed')")
    }
}

/// Direction (+1 / -1) to ask the operator to move a joint at `q`: positive,
/// unless the joint sits within `room` of its upper limit (the folded rest
/// pose puts the B601's joint2/joint3 exactly there) and has more room below.
pub fn test_direction(q: f64, q_min: f64, q_max: f64, room: f64) -> f64 {
    if q_max - q < room && q - q_min > q_max - q { -1.0 } else { 1.0 }
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
        let dir = test_direction(q[i], d.q_min, d.q_max, 3.0 * threshold);
        eprintln!("\n[{}] move it by hand {}", d.name, move_hint(arm, &q, i, dir));
        let start = q[i];
        let t0 = Instant::now();
        let verdict = loop {
            plant.exchange(&idle, &mut obs)?;
            if let Some(v) = sign_verdict((obs.axes()[i].position_rad - start) * dir, threshold) {
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

/// Friction fit for one joint: `τ_friction = coulomb·sign(v) + viscous·v`.
#[derive(Debug, Clone)]
pub struct FrictionFit {
    pub joint: String,
    pub coulomb: f64,
    pub viscous: f64,
    pub samples: usize,
    /// rms of the fit residual [N·m].
    pub residual: f64,
}

/// Fit friction from a `manip run --record` CSV of a friction sweep.
///
/// For each constant-velocity sample of joint `j` (reference speed above
/// `v_min`, reference acceleration ~0, the other joints still), the friction
/// torque is what the motor applied minus what the rigid-body model says the
/// motion needs: `τ_applied − ID(q, v, 0)`. That is fitted with least squares
/// to `Fc·sign(v_ref) + Fv·v`.
///
/// `τ_applied` is the motor's **reported** torque (`taum`). The commanded
/// torque evaluated at the observed state (`tau`) is biased: the observation
/// lags by the loop delay, so at speed `v` the motor PD sees a position error
/// `kp·v·delay` larger than the logged one, which shows up as extra viscous
/// friction (+0.16 N·m·s/rad on a kp = 60 joint at 2.7 ms in the sim). `tau`
/// is used only if the plant reports no torque.
///
/// Run the sweep with soft position gains (`hw friction --kp-scale`): the
/// sampled torque still carries a `kp·v·dt` sawtooth from the zero-order-held
/// MIT position target, which with the tracking gains read as −12 % viscous
/// friction in the sim.
pub fn fit_friction(csv: &std::path::Path, arm: &ArmModel, joints: &[usize], v_min: f64) -> Result<Vec<FrictionFit>, String> {
    let text = std::fs::read_to_string(csv).map_err(|e| format!("{}: {e}", csv.display()))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().ok_or("empty CSV")?.split(',').collect();
    let col = |k: &str| head.iter().position(|h| *h == k).ok_or_else(|| format!("column {k} missing"));
    let n = arm.n();
    let names: Vec<&str> = arm.dofs().iter().map(|d| d.name.as_str()).collect();
    let cq: Vec<usize> = names.iter().map(|d| col(&format!("q_{d}"))).collect::<Result<_, _>>()?;
    let cv: Vec<usize> = names.iter().map(|d| col(&format!("v_{d}"))).collect::<Result<_, _>>()?;
    let cvr: Vec<usize> = names.iter().map(|d| col(&format!("vref_{d}"))).collect::<Result<_, _>>()?;
    let ct: Vec<usize> = names.iter().map(|d| col(&format!("tau_{d}"))).collect::<Result<_, _>>()?;
    let ctm: Vec<usize> = names.iter().map(|d| col(&format!("taum_{d}"))).collect::<Result<_, _>>()?;
    let (c_t, c_mode) = (col("t")?, col("mode")?);

    // (sign(vref), v, residual) per joint.
    let mut data: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); n];
    let mut prev: Option<(f64, Vec<f64>)> = None;
    for line in lines {
        let r: Vec<&str> = line.split(',').collect();
        if r.len() < head.len() || r[c_mode] != "Joint" {
            prev = None;
            continue;
        }
        let f = |c: usize| r[c].parse::<f64>().unwrap_or(f64::NAN);
        let t = f(c_t);
        let vref: Vec<f64> = cvr.iter().map(|&c| f(c)).collect();
        if let Some((tp, vp)) = &prev {
            let dt = (t - tp).max(1e-6);
            let q: Vec<f64> = cq.iter().map(|&c| f(c)).collect();
            let v: Vec<f64> = cv.iter().map(|&c| f(c)).collect();
            let id = arm.inverse_dynamics(&q, &v, &vec![0.0; n]);
            for &j in joints {
                let accel = (vref[j] - vp[j]).abs() / dt;
                let others_still = (0..n).all(|k| k == j || vref[k].abs() < 1e-6);
                // The joint must also have caught up with the reference: right
                // after the reference reaches its cruise speed the joint is
                // still accelerating, and that inertial torque (not in ID with
                // a = 0) biased Fv low by ~30 % in the sim.
                let caught_up = (v[j] - vref[j]).abs() < 0.05 * vref[j].abs();
                if vref[j].abs() >= v_min && accel < 0.05 && others_still && caught_up {
                    let applied = if f(ctm[j]).is_finite() { f(ctm[j]) } else { f(ct[j]) };
                    data[j].push((vref[j].signum(), v[j], applied - id[j]));
                }
            }
        }
        prev = Some((t, vref));
    }

    joints
        .iter()
        .map(|&j| {
            let d = &data[j];
            if d.len() < 20 {
                return Err(format!("{}: only {} constant-velocity samples", names[j], d.len()));
            }
            // Normal equations for [Fc, Fv].
            let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &(s, v, r) in d {
                a11 += s * s;
                a12 += s * v;
                a22 += v * v;
                b1 += s * r;
                b2 += v * r;
            }
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-12 {
                return Err(format!("{}: sweep speeds too similar to separate Coulomb and viscous", names[j]));
            }
            let fc = (b1 * a22 - b2 * a12) / det;
            let fv = (a11 * b2 - a12 * b1) / det;
            let rms = (d.iter().map(|&(s, v, r)| (r - fc * s - fv * v).powi(2)).sum::<f64>() / d.len() as f64).sqrt();
            Ok(FrictionFit {
                joint: names[j].to_string(),
                coulomb: fc,
                viscous: fv,
                samples: d.len(),
                residual: rms,
            })
        })
        .collect()
}

/// Rewrite `friction = ` / `viscous = ` of the given joints in a profile's
/// text, keeping everything else (comments included) untouched.
pub fn write_friction(text: &str, fits: &[FrictionFit]) -> String {
    let mut out = String::new();
    let mut current: Option<String> = None;
    // Only `[[joint]]` tables: `[[sine]]` entries also have `name = "joint1"`.
    let mut in_joint = false;
    let mut pending: Vec<&FrictionFit> = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let flush = |out: &mut String, pending: &mut Vec<&FrictionFit>| {
        for f in pending.drain(..) {
            out.push_str(&format!("friction = {:.4}\nviscous = {:.4}\n", f.coulomb.max(0.0), f.viscous.max(0.0)));
        }
    };
    for line in lines {
        let t = line.trim();
        if t.starts_with('[') {
            flush(&mut out, &mut pending);
            current = None;
            in_joint = t == "[[joint]]";
        }
        if let Some(name) = t.strip_prefix("name = \"").and_then(|x| x.strip_suffix('"')).filter(|_| in_joint) {
            current = Some(name.to_string());
            if let Some(f) = fits.iter().find(|f| f.joint == name) {
                pending.push(f);
            }
        }
        let is_ours = current.as_deref().is_some_and(|c| fits.iter().any(|f| f.joint == c));
        if is_ours && (t.starts_with("friction =") || t.starts_with("viscous =")) {
            continue; // replaced by the flushed pair
        }
        out.push_str(line);
        out.push('\n');
    }
    flush(&mut out, &mut pending);
    out
}

/// Options of a friction sweep (`manip hw friction`).
pub struct FrictionSweep {
    pub dofs: Vec<usize>,
    /// Sweep speeds [rad/s].
    pub speeds: Vec<f64>,
    /// Half-width of the sweep around `[pose.ready]` [rad], clipped to the range.
    pub amplitude: f64,
    pub cycles: usize,
    /// Scale on the tracking `kp` during the sweep (see [`fit_friction`]).
    pub kp_scale: f64,
    /// Don't wait for real time (simulated plants only).
    pub fast: bool,
}

/// Run a friction sweep from `[pose.ready]` on `plant`, record it to `csv`,
/// and fit every swept joint.
pub fn friction_sweep(
    profile: &crate::config::RobotProfile,
    arm: &ArmModel,
    plant: &mut dyn Plant,
    o: &FrictionSweep,
    csv: &std::path::Path,
) -> Result<Vec<FrictionFit>, String> {
    let mut profile = profile.clone();
    for j in &mut profile.joint {
        j.kp *= o.kp_scale;
    }
    let base = crate::assemble::named_pose(&profile, arm, "ready").ok_or("friction sweep starts from [pose.ready]")?;
    let margin = 0.05;
    let plan = crate::app::SweepPlan {
        joints: o
            .dofs
            .iter()
            .map(|&d| {
                let dof = &arm.dofs()[d];
                ((d), (base[d] - o.amplitude).max(dof.q_min + margin), (base[d] + o.amplitude).min(dof.q_max - margin))
            })
            .collect(),
        speeds: o.speeds.clone(),
        cycles: o.cycles,
        settle_s: 0.3,
    };
    let duration = profile.control.startup_ramp_s + 6.0 + plan.duration(&base);
    log::info!("friction sweep: {} joints, about {:.0} s", o.dofs.len(), duration);
    crate::app::run(
        &profile,
        arm,
        plant,
        crate::app::Source::Sweep { plan, start: None },
        crate::app::RunOptions {
            mode: crate::supervisor::Mode::Joint,
            start_pose: Some(base),
            duration_s: Some(duration),
            fast: o.fast,
            record: Some(csv.to_path_buf()),
            log: None,
            status_every_s: 2.0,
            monitor: None,
        },
    )?;
    let v_min = 0.8 * o.speeds.iter().cloned().fold(f64::INFINITY, f64::min);
    fit_friction(csv, arm, &o.dofs, v_min)
}
