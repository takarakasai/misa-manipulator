//! Synthetic leader, for running the closed loop without a physical leader.
//!
//! Outputs `center + amp*sin(2π f t + phase)` per joint. Varying frequency and
//! amplitude lets you compare tracking bandwidth, the effect of inverse-dynamics FF,
//! and how the velocity limit kicks in, under identical conditions.

use std::time::Instant;

use crate::{Leader, LeaderError};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SineJoint {
    pub name: String,
    #[serde(default)]
    pub center: f64,
    #[serde(default)]
    pub amp: f64,
    #[serde(default = "default_freq")]
    pub freq_hz: f64,
    #[serde(default)]
    pub phase: f64,
}

fn default_freq() -> f64 {
    0.2
}

pub struct SineLeader {
    joints: Vec<SineJoint>,
    names: Vec<String>,
    t0: Instant,
}

impl SineLeader {
    pub fn new(joints: Vec<SineJoint>) -> Self {
        let names = joints.iter().map(|j| j.name.clone()).collect();
        Self {
            joints,
            names,
            t0: Instant::now(),
        }
    }

    /// Value at time `t` [s] (for deterministic reproduction in tests and sim).
    pub fn at(&self, t: f64) -> Vec<f64> {
        self.joints
            .iter()
            .map(|j| j.center + j.amp * (std::f64::consts::TAU * j.freq_hz * t + j.phase).sin())
            .collect()
    }
}

impl Leader for SineLeader {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn read(&mut self) -> Result<Vec<f64>, LeaderError> {
        Ok(self.at(self.t0.elapsed().as_secs_f64()))
    }

    fn status_line(&self) -> String {
        format!("sine t={:.1}s", self.t0.elapsed().as_secs_f64())
    }
}
