//! Binary run log and bit-exact replay.
//!
//! A log holds, per control cycle, everything [`Policy::step`] consumed and
//! produced: the observation, the mode requests, the target, and the gated
//! command (plus the SafetyGate verdict). Replaying feeds the recorded inputs
//! through the *current* code and compares commands **exactly**
//! (misa-core's `diff_commands`). If not a single bit changes, a refactor did
//! not change behaviour; if something changes, the first divergence says which
//! axis and field, and when.
//!
//! Format: `[u32 LE length][postcard]` records — one [`LogHeader`], then one
//! [`LogFrame`] per cycle (the same framing as misa-runner's recorder). The
//! per-cycle part embeds a misa-core [`Frame`] so the command comparison is the
//! shared one; `intent` is left at its default (it is the quadruped vocabulary).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use manip_model::ArmModel;
use misa_core::{Frame, Intent, Observation, SafetyVerdict, Time, diff_commands};
use nalgebra::{DVector, Isometry3, Quaternion, Translation3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::assemble;
use crate::config::RobotProfile;
use crate::policy::Policy;
use crate::supervisor::{Mode, Target};

/// Bump when [`LogHeader`] / [`LogFrame`] change shape (postcard is not
/// self-describing, so a reader must refuse other versions).
pub const LOG_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogHeader {
    pub format: u32,
    pub robot: String,
    /// Absolute path of the profile the run used, and its full text, so a
    /// replay can tell whether it is running against the same settings.
    pub profile_path: PathBuf,
    pub profile_text: String,
    pub axes: Vec<String>,
    pub rate_hz: f64,
    /// Measured pose the policy started from.
    pub q0: Vec<f64>,
}

/// Serializable form of [`Target`] (bit-exact round trip).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TargetRec {
    None,
    Joint(Vec<f64>),
    /// Position, quaternion `[i, j, k, w]`, posture.
    Tcp { pos: [f64; 3], quat: [f64; 4], posture: Vec<f64> },
}

impl From<&Target> for TargetRec {
    fn from(t: &Target) -> Self {
        match t {
            Target::None => TargetRec::None,
            Target::Joint(q) => TargetRec::Joint(q.as_slice().to_vec()),
            Target::Tcp { pose, posture } => {
                let p = pose.translation.vector;
                let c = pose.rotation.coords;
                TargetRec::Tcp {
                    pos: [p.x, p.y, p.z],
                    quat: [c.x, c.y, c.z, c.w],
                    posture: posture.as_slice().to_vec(),
                }
            }
        }
    }
}

impl From<&TargetRec> for Target {
    fn from(t: &TargetRec) -> Self {
        match t {
            TargetRec::None => Target::None,
            TargetRec::Joint(q) => Target::Joint(DVector::from_column_slice(q)),
            TargetRec::Tcp { pos, quat, posture } => Target::Tcp {
                pose: Isometry3::from_parts(
                    Translation3::new(pos[0], pos[1], pos[2]),
                    // Unchecked on purpose: renormalizing would change the last bits.
                    UnitQuaternion::new_unchecked(Quaternion::new(quat[3], quat[0], quat[1], quat[2])),
                ),
                posture: DVector::from_column_slice(posture),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFrame {
    /// Observation, gated command and verdict of this cycle (misa-core's shape).
    pub frame: Frame,
    pub requests: Vec<Mode>,
    pub target: TargetRec,
}

pub struct LogWriter {
    w: BufWriter<File>,
    seq: u64,
}

impl LogWriter {
    pub fn create(path: &Path, header: &LogHeader) -> Result<Self, String> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut s = Self { w: BufWriter::new(f), seq: 0 };
        s.put(header)?;
        Ok(s)
    }

    pub fn frame(
        &mut self,
        t: f64,
        obs: &Observation,
        requests: &[Mode],
        target: &Target,
        command: &misa_core::Command,
        verdict: &SafetyVerdict,
    ) -> Result<(), String> {
        let f = LogFrame {
            frame: Frame {
                seq: self.seq,
                time: Time::from_secs_f64(t),
                intent: Intent::default(),
                observation: obs.clone(),
                command: command.clone(),
                verdict: verdict.clone(),
            },
            requests: requests.to_vec(),
            target: target.into(),
        };
        self.seq += 1;
        self.put(&f)
    }

    fn put<T: Serialize>(&mut self, v: &T) -> Result<(), String> {
        let bytes = postcard::to_allocvec(v).map_err(|e| e.to_string())?;
        self.w.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        self.w.write_all(&bytes).map_err(|e| e.to_string())
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.w.flush().map_err(|e| e.to_string())
    }
}

pub fn read_log(path: &Path) -> Result<(LogHeader, Vec<LogFrame>), String> {
    let mut r = BufReader::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let header: LogHeader = next(&mut r)?.ok_or("empty log")?;
    if header.format != LOG_FORMAT {
        return Err(format!("log format {} (this build reads {LOG_FORMAT})", header.format));
    }
    let mut frames = Vec::new();
    while let Some(f) = next(&mut r)? {
        frames.push(f);
    }
    Ok((header, frames))
}

fn next<T: for<'de> Deserialize<'de>>(r: &mut impl Read) -> Result<Option<T>, String> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.to_string()),
    }
    let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
    // A truncated last record (process killed mid-write) ends the log.
    if r.read_exact(&mut buf).is_err() {
        return Ok(None);
    }
    postcard::from_bytes(&buf).map(Some).map_err(|e| e.to_string())
}

/// Result of a replay.
pub struct ReplayReport {
    pub frames: usize,
    pub divergences: Vec<misa_core::Divergence>,
    /// Whether the profile used now differs from the one recorded.
    pub profile_changed: bool,
}

/// Re-run the recorded inputs through the current code with `profile` and
/// compare commands exactly. `profile_path` is only for the changed-profile note.
pub fn replay(
    header: &LogHeader,
    frames: &[LogFrame],
    profile: &RobotProfile,
    profile_text: &str,
    arm: &ArmModel,
    limit: usize,
) -> Result<ReplayReport, String> {
    let names: Vec<String> = arm.dofs().iter().map(|d| d.name.clone()).collect();
    if names != header.axes {
        return Err(format!("axes differ: log {:?}, model {:?}", header.axes, names));
    }
    let mut policy = Policy::new(profile, arm, &DVector::from_column_slice(&header.q0))?;
    let mut replayed = Vec::with_capacity(frames.len());
    for lf in frames {
        let target: Target = (&lf.target).into();
        let out = policy.step(arm, &lf.frame.observation, &lf.requests, &target);
        replayed.push(Frame {
            seq: lf.frame.seq,
            time: lf.frame.time,
            intent: Intent::default(),
            observation: lf.frame.observation.clone(),
            command: policy.command().clone(),
            verdict: out.verdict,
        });
    }
    let recorded = frames.iter().map(|f| &f.frame);
    Ok(ReplayReport {
        frames: frames.len(),
        divergences: diff_commands(recorded, replayed.iter(), limit),
        profile_changed: header.profile_text != profile_text,
    })
}

/// Header for a live run.
pub fn header(profile: &RobotProfile, profile_path: &Path, arm: &ArmModel, q0: &DVector<f64>) -> Result<LogHeader, String> {
    let profile_path = std::fs::canonicalize(profile_path).map_err(|e| e.to_string())?;
    Ok(LogHeader {
        format: LOG_FORMAT,
        robot: profile.robot.name.clone(),
        profile_text: std::fs::read_to_string(&profile_path).map_err(|e| e.to_string())?,
        profile_path,
        axes: arm.dofs().iter().map(|d| d.name.clone()).collect(),
        rate_hz: profile.control.rate_hz,
        q0: q0.as_slice().to_vec(),
    })
}

/// Load the profile a log refers to (or `override_path`) and build its arm.
pub fn load_for_replay(
    header: &LogHeader,
    override_path: Option<&Path>,
) -> Result<(RobotProfile, String, ArmModel), String> {
    let path = override_path.unwrap_or(&header.profile_path);
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (p, dir) = RobotProfile::load(path)?;
    let arm = assemble::load_arm(&p, &dir)?;
    Ok((p, text, arm))
}
