//! Profile (TOML). One robot = model + this file.
//!
//! Paths are relative to the profile's location (so they point at the same model no
//! matter where it is launched from).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use manip_control::Feedforward;
use manip_leader::synthetic::SineJoint;
use manip_model::TcpSpec;
use manip_plant_can::BusSpec;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotProfile {
    pub robot: RobotSection,
    pub control: ControlSection,
    /// Per-independent-DOF settings. **List every independent DOF of the model**
    /// (startup aborts if any is missing, so an axis with zero gains can't silently
    /// slip in).
    pub joint: Vec<JointSection>,
    #[serde(default)]
    pub osc: OscSection,
    /// Named poses (joint name -> value). `rest` is where Park goes.
    #[serde(default)]
    pub pose: BTreeMap<String, BTreeMap<String, f64>>,
    #[serde(default)]
    pub hardware: Option<HardwareSection>,
    #[serde(default)]
    pub sim: SimSection,
    /// Mapping from leader neutral space to this robot's joints.
    #[serde(default)]
    pub teleop: Vec<TeleopMap>,
    /// Workspace box and self-collision (`guard.rs`). Absent = no checks.
    #[serde(default)]
    pub safety: Option<SafetySection>,
    /// Synthetic target (`--source sine`).
    #[serde(default)]
    pub sine: Vec<SineJoint>,
    /// Model predictive control (`--mode mpc`).
    #[serde(default)]
    pub mpc: MpcSection,
}

/// `[mpc]`: a planner (manip-mpc) replanning at `rate_hz` on a worker thread,
/// its joint plan tracked every cycle by manip-wbc's `JointTracking` (with the
/// OSC's motor-side PD, safety barriers and solver settings from `[osc]`).
/// The workspace box and points come from `[safety]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MpcSection {
    /// "ltv" (acceleration-level QP) or "ilqr" (torque-level).
    #[serde(default = "default_planner")]
    pub planner: String,
    #[serde(default = "d50")]
    pub rate_hz: f64,
    /// Horizon (intervals) and interval [s]; 0 = the planner's default.
    #[serde(default)]
    pub horizon: usize,
    #[serde(default)]
    pub dt: f64,
    /// Joint-trajectory tracking bandwidth of the WBC [rad/s] (Kp = ω², Kd = 2ω).
    #[serde(default = "d20")]
    pub track_omega: f64,
    /// TCP speed limit [m/s].
    #[serde(default = "d03")]
    pub tcp_v_max: f64,
    /// Hold if the newest plan is older than this [s] (planner stalled).
    #[serde(default = "d03")]
    pub plan_timeout_s: f64,
    /// Joint targets (leader) are extrapolated at their estimated velocity for
    /// up to this long over the horizon [s]; 0 = held fixed (each plan
    /// arrives and stops). Not yet shown to help on the real arm.
    #[serde(default)]
    pub target_lookahead_s: f64,
    /// Low-pass time constant of that velocity estimate [s].
    #[serde(default = "d005")]
    pub target_v_tau_s: f64,
    /// Continue plans from the tracked plan while the measurement is within
    /// `[|Δq| rad, |Δv| rad/s]` of it (`MpcDriver::from_reference`); empty =
    /// always from the measurement.
    #[serde(default)]
    pub replan_from_reference: Vec<f64>,
    /// LTV: QP solves per plan, re-linearizing around each solution; 0 = the
    /// planner's default. One step from the previous plan can be far off near
    /// a singularity: at the folded B601-DM, every other plan sent joint2 to
    /// its limit and the arm swung ±1° at rest; 3 converge to the target.
    #[serde(default)]
    pub sqp_iters: usize,
    /// Integral action of the tracking WBC [N·m/(rad·s)]; 0 = off.
    #[serde(default)]
    pub track_ki: f64,
    /// Its clamp per joint, as a multiple of the joint's `friction`.
    #[serde(default = "d15")]
    pub track_i_max_friction: f64,
}

impl Default for MpcSection {
    fn default() -> Self {
        Self {
            planner: default_planner(),
            rate_hz: 50.0,
            horizon: 0,
            dt: 0.0,
            track_omega: 20.0,
            tcp_v_max: 0.3,
            plan_timeout_s: 0.3,
            target_lookahead_s: 0.0,
            target_v_tau_s: 0.05,
            replan_from_reference: Vec::new(),
            sqp_iters: 0,
            track_ki: 0.0,
            track_i_max_friction: 1.5,
        }
    }
}

fn default_planner() -> String {
    "ltv".into()
}

fn d20() -> f64 {
    20.0
}

fn d005() -> f64 {
    0.05
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotSection {
    pub name: String,
    /// `.misa` / `.urdf`, relative to the profile.
    pub model: PathBuf,
    pub tcp: TcpSpec,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlSection {
    pub rate_hz: f64,
    #[serde(default)]
    pub feedforward: Feedforward,
    /// At startup, ramp hold gains up from 0 over this time [s].
    #[serde(default = "default_ramp")]
    pub startup_ramp_s: f64,
    /// Reference shaping time constant [s].
    #[serde(default = "default_tc")]
    pub shaper_time_constant_s: f64,
    /// Fall back to hold if the leader value is older than this [s].
    #[serde(default = "default_leader_timeout")]
    pub leader_timeout_s: f64,
    /// SafetyGate freezes the target if the observation is older than this [s].
    #[serde(default = "default_obs_age")]
    pub max_observation_age_s: f64,
    /// Speed factor for Park (folding at exit), multiplied into `v_max`.
    #[serde(default = "default_park_speed")]
    pub park_speed_scale: f64,
    /// Error at which Park counts as "arrived" [rad].
    #[serde(default = "default_park_tol")]
    pub park_tolerance: f64,
    /// Velocity over which the Coulomb part of the friction feedforward ramps in,
    /// for the joint law (evaluated at the clean reference velocity) [rad/s].
    #[serde(default = "default_ff_eps_joint")]
    pub friction_v_eps: f64,
}

fn default_ff_eps_joint() -> f64 {
    0.05
}

fn default_ramp() -> f64 {
    1.0
}
fn default_tc() -> f64 {
    0.05
}
fn default_leader_timeout() -> f64 {
    0.1
}
fn default_obs_age() -> f64 {
    0.05
}
fn default_park_speed() -> f64 {
    0.3
}
fn default_park_tol() -> f64 {
    0.03
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSection {
    pub name: String,
    /// Tracking (teleop) gains.
    pub kp: f64,
    pub kd: f64,
    /// Hold gains (Hold / Park / leader lost). Default to kp, kd.
    #[serde(default)]
    pub hold_kp: Option<f64>,
    #[serde(default)]
    pub hold_kd: Option<f64>,
    /// Damping in gravity compensation (hand-guided mode).
    #[serde(default)]
    pub gravity_kd: f64,
    /// Reference velocity / acceleration limits (used by both the shaper and SafetyGate).
    pub v_max: f64,
    pub a_max: f64,
    /// Reflected rotor inertia. Overrides the model value.
    #[serde(default)]
    pub armature: Option<f64>,
    /// To narrow the range of motion / torque limit below the model's.
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub effort: Option<f64>,
    /// Torque command rate limit [N·m/s] (0 = unlimited).
    #[serde(default)]
    pub max_torque_rate: f64,
    /// Gravity term scale (an escape hatch for model-vs-hardware mismatch).
    #[serde(default = "one")]
    pub gravity_scale: f64,
    /// Motor-side PD in OSC around the integrated QP solution (MIT kp / kd).
    /// Defaults: kp 0, kd `[osc] motor_kd`. Feedback closed only in the 500 Hz
    /// PC loop chattered on the real arm; motor-side damping removes it.
    #[serde(default)]
    pub osc_kp: Option<f64>,
    #[serde(default)]
    pub osc_kd: Option<f64>,
    /// Joint damping in the sim [N·m·s/rad].
    #[serde(default)]
    pub sim_damping: f64,
    /// Coulomb friction in the sim [N·m] ([N] for prismatic): the "true" plant.
    /// An estimate until identified on hardware (misa-sysid).
    #[serde(default)]
    pub sim_friction: f64,
    /// Coulomb friction the **controller** compensates [N·m] ([N] for prismatic).
    /// Kept separate from `sim_friction` so a wrong estimate can be simulated.
    /// 0 = no compensation.
    #[serde(default)]
    pub friction: f64,
    /// Viscous friction the controller compensates [N·m·s/rad].
    #[serde(default)]
    pub viscous: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OscSection {
    #[serde(default = "d400")]
    pub kp_lin: f64,
    #[serde(default = "d40")]
    pub kd_lin: f64,
    #[serde(default = "d200")]
    pub kp_ang: f64,
    #[serde(default = "d28")]
    pub kd_ang: f64,
    #[serde(default = "yes")]
    pub track_orientation: bool,
    #[serde(default = "d25")]
    pub posture_kp: f64,
    #[serde(default = "d10")]
    pub posture_kd: f64,
    #[serde(default = "one")]
    pub torque_scale: f64,
    #[serde(default = "d50")]
    pub a_max: f64,
    #[serde(default = "d10")]
    pub cbf_alpha: f64,
    /// Damping added on the motor side (joints without `osc_kd`).
    #[serde(default)]
    pub motor_kd: f64,
    /// How far the integrated OSC reference may lead the measured joint [rad]
    /// (bounds the motor-side `osc_kp` torque when the arm is held back).
    #[serde(default = "d003")]
    pub motor_lead_max: f64,
    /// TCP reference shaping (translation m/s, m/s²; rotation rad/s, rad/s²).
    #[serde(default = "d03")]
    pub lin_v_max: f64,
    #[serde(default = "d2")]
    pub lin_a_max: f64,
    #[serde(default = "d15")]
    pub ang_v_max: f64,
    #[serde(default = "d8")]
    pub ang_a_max: f64,
    /// Caps on a commanded TCP impedance (API), in acceleration units: the
    /// fastest mode's ω² [1/s²] and damping [1/s], and the damping floor.
    /// Above them the rotational stiffness (then the translational) is
    /// scaled down. Not yet tried on the real arm.
    #[serde(default = "d1600")]
    pub compliance_kp_max: f64,
    #[serde(default = "d80")]
    pub compliance_kd_max: f64,
    #[serde(default = "d4")]
    pub compliance_kd_min: f64,
    /// QP backend (`active_set` | `clarabel`).
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Formulation (`accel_space` | `explicit` | `force_space`).
    #[serde(default = "default_formulation")]
    pub formulation: String,
    /// Friction feedforward in the OSC (folded into `h`, evaluated at the
    /// **measured** velocity) [rad/s]. Keep it at least ~3 LSB of the motors'
    /// velocity feedback: in MuJoCo with DAMIAO quantization (0.015 rad/s per
    /// LSB), 0.05 held still, 0.02 turned a static hold into a limit cycle
    /// (0.13 rad/s, 0.6 mm), and 0.2 under-compensated the circle (2.4 mm vs
    /// 1.3 mm at 0.05).
    #[serde(default = "default_ff_eps_osc")]
    pub friction_v_eps: f64,
}

fn default_ff_eps_osc() -> f64 {
    0.05
}

fn default_backend() -> String {
    "active_set".into()
}
fn default_formulation() -> String {
    "accel_space".into()
}

impl Default for OscSection {
    fn default() -> Self {
        toml::from_str("").unwrap()
    }
}

fn d400() -> f64 { 400.0 }
fn d40() -> f64 { 40.0 }
fn d200() -> f64 { 200.0 }
fn d28() -> f64 { 28.0 }
fn d25() -> f64 { 25.0 }
fn d10() -> f64 { 10.0 }
fn d50() -> f64 { 50.0 }
fn d03() -> f64 { 0.3 }
fn d003() -> f64 { 0.03 }
fn d2() -> f64 { 2.0 }
fn d15() -> f64 { 1.5 }
fn d8() -> f64 { 8.0 }
fn d1600() -> f64 { 1600.0 }
fn d80() -> f64 { 80.0 }
fn d4() -> f64 { 4.0 }
fn yes() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSection {
    /// When the control loop stalls, time until the bus thread drops position stiffness
    /// [s].
    #[serde(default = "default_stale")]
    pub stale_after_s: f64,
    pub bus: Vec<BusSpec>,
}

fn default_stale() -> f64 {
    0.1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "sim"), allow(dead_code))]
pub struct SimSection {
    #[serde(default = "default_timestep")]
    pub timestep_s: f64,
    /// Initial pose (name of a `[pose.*]`). All axes 0 if absent.
    #[serde(default)]
    pub initial_pose: Option<String>,
    #[serde(default)]
    pub ground: bool,
    #[serde(default = "yes")]
    pub joint_limits: bool,
    /// Link-link contact (off by default; see SimOptions in manip-plant-mujoco).
    #[serde(default)]
    pub self_collision: bool,
    /// `--plant rigid`: dry friction that sticks (`RigidPlant::with_stiction`)
    /// instead of the tanh ramp.
    #[serde(default)]
    pub stiction: bool,
    /// Velocity [rad/s] over which Coulomb friction ramps up (tanh smoothing).
    #[serde(default = "default_friction_v_eps")]
    pub friction_v_eps: f64,
    /// Hardware non-idealities added around the simulated plant. Absent = ideal.
    #[serde(default)]
    pub effects: Option<EffectsSection>,
}

fn default_friction_v_eps() -> f64 {
    0.05
}

/// `[sim.effects]`: what the CAN arm does that the simulator does not
/// (see `effects.rs`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectsSection {
    /// Ticks between issuing a command and the plant applying it.
    #[serde(default = "one_usize")]
    pub command_delay_ticks: usize,
    /// Extra ticks by which observations are older than the plant state.
    #[serde(default)]
    pub observation_delay_ticks: usize,
    /// Probability of one extra tick of command delay (missed bus cycle).
    #[serde(default)]
    pub jitter_probability: f64,
    /// Quantize feedback like the MIT status frame of the motors in `[hardware]`.
    #[serde(default = "yes")]
    pub quantize: bool,
    #[serde(default = "one_u64")]
    pub seed: u64,
}

fn one_usize() -> usize {
    1
}
fn one_u64() -> u64 {
    1
}

impl Default for SimSection {
    fn default() -> Self {
        toml::from_str("").unwrap()
    }
}

fn default_timestep() -> f64 {
    0.0005
}

/// One joint of leader neutral space -> one DOF of this robot.
///
/// `q = offset + scale * leader[from]`
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeleopMap {
    pub joint: String,
    pub from: String,
    #[serde(default = "one")]
    pub scale: f64,
    #[serde(default)]
    pub offset: f64,
}

impl RobotProfile {
    pub fn load(path: &Path) -> Result<(Self, PathBuf), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: RobotProfile = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Ok((p, dir))
    }
}

/// `[safety]`: keep monitored points inside a box (world = base frame) and
/// links apart.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetySection {
    pub box_min: [f64; 3],
    pub box_max: [f64; 3],
    /// Points checked against the box, in addition to the TCP.
    #[serde(default)]
    pub points: Vec<SafetyPoint>,
    #[serde(default = "yes")]
    pub self_collision: bool,
    /// Minimum distance kept between links [m].
    #[serde(default = "default_collision_margin")]
    pub collision_margin: f64,
    /// Joint tracking only: the guard checks the reference against a box
    /// shrunk by this much [m]. The arm follows the reference through a PD, so
    /// it can be off by the tracking error (14 mm at the TCP in the sine test
    /// with the default gains); the OSC needs no margin (it constrains the
    /// measured state).
    #[serde(default = "default_joint_margin")]
    pub joint_margin: f64,
    /// Pose (`[pose.*]`) at which link pairs already in contact are not
    /// checked (vendor meshes interpenetrate when folded).
    #[serde(default = "default_exclude_pose")]
    pub exclude_at_pose: String,
    /// Link pairs never checked, e.g. across the wrist, where the convex
    /// hulls overlap within the joint range though the real parts clear.
    #[serde(default)]
    pub exclude_pairs: Vec<[String; 2]>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyPoint {
    pub link: String,
    #[serde(default)]
    pub xyz: [f64; 3],
}

fn default_joint_margin() -> f64 {
    0.02
}
fn default_collision_margin() -> f64 {
    0.005
}
fn default_exclude_pose() -> String {
    "rest".into()
}
