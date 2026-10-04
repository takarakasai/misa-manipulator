//! Assembles components from the profile. **Robot-specific values reach the control
//! laws only through here.**

use std::path::Path;
use std::time::Duration;

use manip_control::{JointGains, ShaperLimits, TcpShaperLimits};
use manip_wbc::OscConfig;
use manip_model::ArmModel;
use misa_core::{AxisLimits, SafetyConfig};
use nalgebra::DVector;

use crate::config::{JointSection, RobotProfile, TeleopMap};
use crate::supervisor::SupervisorConfig;

/// Load the model and apply the profile's overrides (inertia, range of motion, torque
/// limit).
pub fn load_arm(p: &RobotProfile, dir: &Path) -> Result<ArmModel, String> {
    let path = dir.join(&p.robot.model);
    let mut arm = ArmModel::load(&path, &p.robot.tcp).map_err(|e| e.to_string())?;
    // Check that independent DOFs and profile joints correspond one-to-one.
    let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    for n in &names {
        if !p.joint.iter().any(|j| &j.name == n) {
            return Err(format!("profile has no [[joint]] name = \"{n}\""));
        }
    }
    for j in &p.joint {
        if !names.contains(&j.name) {
            return Err(format!(
                "[[joint]] {} is not an independent DOF of the model ({})",
                j.name,
                names.join(", ")
            ));
        }
        if let Some(a) = j.armature {
            arm.set_armature(&j.name, a).map_err(|e| e.to_string())?;
        }
        arm.set_limits(&j.name, j.min, j.max, Some(j.v_max), j.effort)
            .map_err(|e| e.to_string())?;
    }
    Ok(arm)
}

/// Look up the profile's joint settings in the order of the model's independent DOFs.
pub fn joints_in_order<'a>(p: &'a RobotProfile, arm: &ArmModel) -> Vec<&'a JointSection> {
    arm.dofs()
        .iter()
        .map(|d| p.joint.iter().find(|j| j.name == d.name).expect("checked in load_arm"))
        .collect()
}

fn col(js: &[&JointSection], f: impl Fn(&JointSection) -> f64) -> DVector<f64> {
    DVector::from_iterator(js.len(), js.iter().map(|j| f(j)))
}

/// Named pose. The profile's `[pose.*]` takes precedence, then the model's pose, else
/// `None`. Unlisted joints are 0.
pub fn named_pose(p: &RobotProfile, arm: &ArmModel, name: &str) -> Option<DVector<f64>> {
    let zero = vec![0.0; arm.n()];
    if let Some(tbl) = p.pose.get(name) {
        let q = DVector::from_iterator(
            arm.n(),
            arm.dofs().iter().map(|d| d.clamp(tbl.get(&d.name).copied().unwrap_or(0.0))),
        );
        return Some(q);
    }
    arm.named_pose(name, &zero).ok()
}

pub fn supervisor_config(p: &RobotProfile, arm: &ArmModel) -> Result<SupervisorConfig, String> {
    let js = joints_in_order(p, arm);
    let c = &p.control;
    Ok(SupervisorConfig {
        track: JointGains::new(col(&js, |j| j.kp), col(&js, |j| j.kd)),
        hold: JointGains::new(
            col(&js, |j| j.hold_kp.unwrap_or(j.kp)),
            col(&js, |j| j.hold_kd.unwrap_or(j.kd)),
        ),
        gravity_kd: col(&js, |j| j.gravity_kd),
        gravity_scale: col(&js, |j| j.gravity_scale),
        feedforward: c.feedforward,
        shaper: ShaperLimits {
            v_max: col(&js, |j| j.v_max),
            a_max: col(&js, |j| j.a_max),
            time_constant_s: c.shaper_time_constant_s,
        },
        park_speed_scale: c.park_speed_scale,
        park_tolerance: c.park_tolerance,
        startup_ramp_s: c.startup_ramp_s,
        tcp_shaper: TcpShaperLimits {
            lin_v_max: p.osc.lin_v_max,
            lin_a_max: p.osc.lin_a_max,
            ang_v_max: p.osc.ang_v_max,
            ang_a_max: p.osc.ang_a_max,
            time_constant_s: c.shaper_time_constant_s,
        },
        rest: named_pose(p, arm, "rest").unwrap_or_else(|| DVector::zeros(arm.n())),
        friction: friction_model(p, arm, p.control.friction_v_eps),
        safety: match &p.safety {
            Some(cfg) => {
                let exclude = named_pose(p, arm, &cfg.exclude_at_pose);
                Some(crate::guard::SafetyModel::build(cfg, arm, exclude.as_ref().map(|q| q.as_slice()))?)
            }
            None => None,
        },
    })
}

pub fn osc_config(p: &RobotProfile, arm: &ArmModel) -> Result<OscConfig, String> {
    let n = arm.n();
    let o = &p.osc;
    let mut c = OscConfig::defaults(n);
    c.kp_lin = o.kp_lin;
    c.kd_lin = o.kd_lin;
    c.kp_ang = o.kp_ang;
    c.kd_ang = o.kd_ang;
    c.track_orientation = o.track_orientation;
    c.posture_kp = DVector::from_element(n, o.posture_kp);
    c.posture_kd = DVector::from_element(n, o.posture_kd);
    c.torque_scale = o.torque_scale;
    c.a_max = DVector::from_element(n, o.a_max);
    c.cbf_alpha = o.cbf_alpha;
    let js = joints_in_order(p, arm);
    c.motor_kd = col(&js, |j| j.osc_kd.unwrap_or(o.motor_kd));
    c.motor_kp = col(&js, |j| j.osc_kp.unwrap_or(0.0));
    c.motor_lead_max = o.motor_lead_max;
    c.friction = friction_model(p, arm, o.friction_v_eps);
    c.solve.backend = match o.backend.as_str() {
        "active_set" => misa_wbc::QpSolver::ActiveSet,
        "clarabel" => misa_wbc::QpSolver::Clarabel,
        other => return Err(format!("[osc] backend = \"{other}\" is not supported (active_set | clarabel)")),
    };
    c.formulation = match o.formulation.as_str() {
        "accel_space" => misa_wbc::Formulation::AccelSpace,
        "explicit" => misa_wbc::Formulation::Explicit,
        "force_space" => misa_wbc::Formulation::ForceSpace,
        other => return Err(format!("[osc] formulation = \"{other}\" is not supported")),
    };
    Ok(c)
}

/// SafetyGate config. Looser than the shaper (references produced by the shaper pass
/// through unclamped; it only catches anomalies that bypass the shaper).
pub fn safety_config(p: &RobotProfile, arm: &ArmModel) -> SafetyConfig {
    let js = joints_in_order(p, arm);
    SafetyConfig {
        axes: arm
            .dofs()
            .iter()
            .zip(&js)
            .map(|(d, j)| AxisLimits {
                min_rad: d.q_min,
                max_rad: d.q_max,
                max_target_rate_rad_s: 1.5 * j.v_max,
                max_torque_nm: if d.effort.is_finite() { d.effort } else { 0.0 },
                max_torque_rate_nm_s: j.max_torque_rate,
                max_velocity_rad_s: 1.5 * j.v_max,
            })
            .collect(),
        max_observation_age: Duration::from_secs_f64(p.control.max_observation_age_s),
        max_tilt_rad: 0.0,
    }
}

/// Leader neutral space -> this robot's joints.
#[derive(Debug, Clone)]
pub struct TeleopMapping {
    /// Per independent DOF: (leader index, scale, offset). `None` for unmapped axes.
    per_dof: Vec<Option<(usize, f64, f64)>>,
}

impl TeleopMapping {
    pub fn new(maps: &[TeleopMap], arm: &ArmModel, leader_names: &[String]) -> Result<Self, String> {
        let mut per_dof = vec![None; arm.n()];
        for m in maps {
            let i = arm.dof(&m.joint).map_err(|e| e.to_string())?;
            let li = leader_names
                .iter()
                .position(|n| *n == m.from)
                .ok_or_else(|| format!("leader has no joint {} ({})", m.from, leader_names.join(", ")))?;
            per_dof[i] = Some((li, m.scale, m.offset));
        }
        Ok(Self { per_dof })
    }

    /// Build this robot's target from leader values. Unmapped axes take the `keep` value.
    pub fn map(&self, leader: &[f64], keep: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(
            self.per_dof.len(),
            self.per_dof.iter().enumerate().map(|(i, m)| match m {
                Some((li, s, o)) => o + s * leader[*li],
                None => keep[i],
            }),
        )
    }
}

/// Per-DOF feedback LSB `[position, velocity, torque]` in model units, from the
/// motor models in `[hardware]` (so the simulator quantizes exactly like the
/// real MIT status frames).
pub fn feedback_quantization(p: &RobotProfile, arm: &ArmModel) -> Result<Vec<[f64; 3]>, String> {
    let hw = p
        .hardware
        .as_ref()
        .ok_or("[sim.effects] quantize = true needs [hardware] to know the motor models")?;
    arm.dofs()
        .iter()
        .map(|d| {
            hw.bus
                .iter()
                .find_map(|b| b.motor.iter().find(|m| m.joint == d.name).map(|m| (b.vendor, m)))
                .ok_or_else(|| format!("no motor in [hardware] for joint {}", d.name))
                .and_then(|(v, m)| manip_plant_can::feedback_resolution(v, m))
        })
        .collect()
}

/// The controller's friction estimate from `[[joint]] friction / viscous`;
/// `None` when every joint is 0 (compensation off).
pub fn friction_model(p: &RobotProfile, arm: &ArmModel, v_eps: f64) -> Option<manip_control::FrictionModel> {
    let js = joints_in_order(p, arm);
    let m = manip_control::FrictionModel {
        coulomb: col(&js, |j| j.friction),
        viscous: col(&js, |j| j.viscous),
        v_eps,
    };
    (!m.is_zero()).then_some(m)
}
