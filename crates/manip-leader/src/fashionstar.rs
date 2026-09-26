//! Leader arm built from FashionStar bus servos (Star Arm 102 / reBot Arm 102).
//!
//! One cycle = one sync monitor (reads all servo angles with a single request, same
//! as the vendor's teleop script). Servos are **released** (torque off) at startup,
//! since this is the side a human moves by hand.
//!
//! # Servo angle -> neutral space
//!
//! ```text
//! θ   = unwrap(raw, window center)     absorbs the arbitrary multi-turn counter
//! out = clamp(sign*scale*(θ − offset), range)
//! ```
//!
//! - `offset` is the **servo angle at the zero pose**. 0 for units that went through
//!   Seeed's procedure (write `set_origin_point` to the servo's non-volatile memory at
//!   the zero pose). For units you don't want to write to, put the value read with
//!   `manip leader --zero` here.
//! - Default `sign` / `scale` are the `joint_directions` of upstream LeRobot's
//!   `rebot_102_leader` (gripper is −6, stretching the handle's 45° to fully open
//!   −270°). **Verify on hardware by moving one axis at a time** (some versions of
//!   Seeed's standalone plugin carry no signs; which one is right is unconfirmed).

use std::path::Path;
use std::time::Duration;

use fashionstar_driver::{FashionStarBus, FsCommands, StopMode};
use serde::Deserialize;

use crate::{Leader, LeaderError};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaderProfile {
    pub leader: LeaderSection,
    pub joint: Vec<LeaderJoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaderSection {
    pub name: String,
    pub port: String,
    #[serde(default = "default_baud")]
    pub baud: u32,
    /// Read rate [Hz] (rate-limited by the bus if that is slower).
    #[serde(default = "default_rate")]
    pub rate_hz: f64,
    /// Response timeout for one sync monitor [ms].
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Release the servos at startup (true for a hand-moved leader).
    #[serde(default = "yes")]
    pub release_on_start: bool,
    /// Reset the multi-turn counter to 0 at startup.
    #[serde(default = "yes")]
    pub reset_multi_turn: bool,
}

fn default_baud() -> u32 {
    1_000_000
}
fn default_rate() -> f64 {
    200.0
}
fn default_timeout() -> u64 {
    10
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaderJoint {
    /// Joint name in neutral space (referenced by the follower's `[[teleop]] from`).
    pub name: String,
    pub servo: u8,
    #[serde(default = "one")]
    pub sign: f64,
    #[serde(default = "one")]
    pub scale: f64,
    /// Servo angle at the zero pose [deg].
    #[serde(default)]
    pub offset_deg: f64,
    /// Range in neutral space [deg]. Used for the unwrap window and the final clamp.
    pub range_deg: [f64; 2],
}

fn one() -> f64 {
    1.0
}

impl LeaderProfile {
    /// Load, also returning the read rate [Hz].
    pub fn load(path: &Path) -> Result<(Self, f64), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: LeaderProfile = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        for j in &p.joint {
            if j.range_deg[0] > j.range_deg[1] {
                return Err(format!("{}: range_deg must be [min, max]", j.name));
            }
            if j.sign.abs() != 1.0 || j.scale == 0.0 {
                return Err(format!("{}: sign must be ±1 and scale non-zero", j.name));
            }
        }
        let rate = p.leader.rate_hz;
        Ok((p, rate))
    }
}

impl LeaderJoint {
    /// Servo angle [rad] (multi-turn) -> neutral space [rad].
    pub fn to_neutral(&self, raw_rad: f64) -> f64 {
        let (lo, hi) = (self.range_deg[0].to_radians(), self.range_deg[1].to_radians());
        let k = self.sign * self.scale;
        // Unwrap around the servo angle corresponding to the center of the neutral range.
        let center_raw = self.offset_deg.to_radians() + 0.5 * (lo + hi) / k;
        // The window is ±180° in neutral space, i.e. ±180°/|scale| in servo angle. With
        // |scale| > 1 (gripper) that window gets narrow, so unwrap by ±180° in servo angle.
        let theta = crate::unwrap_to_window(raw_rad, center_raw - std::f64::consts::PI, center_raw + std::f64::consts::PI);
        (k * (theta - self.offset_deg.to_radians())).clamp(lo, hi)
    }
}

pub struct FashionStarLeader {
    bus: FashionStarBus,
    joints: Vec<LeaderJoint>,
    ids: Vec<u8>,
    names: Vec<String>,
    last: Vec<f64>,
    missing: u64,
}

impl FashionStarLeader {
    pub fn open(p: LeaderProfile) -> Result<Self, LeaderError> {
        let mut bus = FashionStarBus::open(&p.leader.port, p.leader.baud, Duration::from_millis(p.leader.timeout_ms))
            .map_err(|e| LeaderError::Open(format!("{}: {e}", p.leader.port)))?;
        let ids: Vec<u8> = p.joint.iter().map(|j| j.servo).collect();
        for &id in &ids {
            if !bus.ping(id).map_err(|e| LeaderError::Open(e.to_string()))? {
                return Err(LeaderError::Open(format!("servo {id} not responding")));
            }
        }
        for &id in &ids {
            if p.leader.release_on_start {
                bus.stop(id, StopMode::Release).map_err(|e| LeaderError::Open(e.to_string()))?;
            }
            // Reset via broadcast (0xFF) doesn't work (per a note in Seeed's plugin).
            // One at a time.
            if p.leader.reset_multi_turn {
                bus.reset_multi_turn(id).map_err(|e| LeaderError::Open(e.to_string()))?;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
        let names = p.joint.iter().map(|j| j.name.clone()).collect();
        let n = p.joint.len();
        Ok(Self {
            bus,
            joints: p.joint,
            ids,
            names,
            last: vec![f64::NAN; n],
            missing: 0,
        })
    }

    /// Raw servo angles [rad] (multi-turn, before unwrapping). For `--zero` calibration.
    pub fn read_raw(&mut self) -> Result<Vec<Option<f64>>, LeaderError> {
        let m = self
            .bus
            .sync_monitor(&self.ids)
            .map_err(|e| LeaderError::Read(e.to_string()))?;
        Ok(m.into_iter().map(|x| x.map(|x| x.angle_rad as f64)).collect())
    }

    pub fn joints(&self) -> &[LeaderJoint] {
        &self.joints
    }
}

impl Leader for FashionStarLeader {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn read(&mut self) -> Result<Vec<f64>, LeaderError> {
        let raw = self.read_raw()?;
        let mut all = true;
        for (i, r) in raw.iter().enumerate() {
            match r {
                Some(a) => self.last[i] = self.joints[i].to_neutral(*a),
                None => all = false,
            }
        }
        if !all {
            self.missing += 1;
        }
        // Output nothing while any axis has never been read (don't produce NaN targets).
        if self.last.iter().any(|x| !x.is_finite()) {
            return Err(LeaderError::Read("angles not yet available for all servos".into()));
        }
        Ok(self.last.clone())
    }

    fn status_line(&self) -> String {
        format!("missing={}", self.missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(sign: f64, scale: f64, offset: f64, range: [f64; 2]) -> LeaderJoint {
        LeaderJoint {
            name: "j".into(),
            servo: 0,
            sign,
            scale,
            offset_deg: offset,
            range_deg: range,
        }
    }

    #[test]
    fn maps_sign_offset_and_wraps() {
        let d = |x: f64| x.to_radians();
        // shoulder_lift: sign −1, neutral [-200, 1]. Servo 90° -> neutral −90°.
        let j = joint(-1.0, 1.0, 0.0, [-200.0, 1.0]);
        assert!((j.to_neutral(d(90.0)) - d(-90.0)).abs() < 1e-9);
        // Same even if the multi-turn counter is off by one turn.
        assert!((j.to_neutral(d(90.0 + 360.0)) - d(-90.0)).abs() < 1e-9);
        // Offset: a unit whose servo reads 12° at the zero pose.
        let j = joint(1.0, 1.0, 12.0, [-150.0, 150.0]);
        assert!((j.to_neutral(d(42.0)) - d(30.0)).abs() < 1e-9);
    }

    #[test]
    fn gripper_scale_and_clamp() {
        let d = |x: f64| x.to_radians();
        // Gripper: ×−6, neutral [-270, 0]. Handle 30° -> −180°, 60° -> −270° (clamped).
        let j = joint(-1.0, 6.0, 0.0, [-270.0, 0.0]);
        assert!((j.to_neutral(d(30.0)) - d(-180.0)).abs() < 1e-9);
        assert!((j.to_neutral(d(60.0)) - d(-270.0)).abs() < 1e-9);
        assert!((j.to_neutral(d(-3.0)) - 0.0).abs() < 1e-9);
    }
}
