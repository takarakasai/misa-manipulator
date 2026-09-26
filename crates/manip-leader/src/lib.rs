//! The target-producing side: leader arms, plus synthetic signals for running without
//! hardware.
//!
//! # Neutral joint space
//!
//! Leaders output in a **neutral joint space** (same convention as LeRobot actions:
//! an angle [rad] per joint name, gripper 0 when closed). Servo sign, zero point and
//! multi-turn unwrapping are absorbed on the leader side; the follower only holds a
//! "neutral space -> own model coordinates" mapping (sign, scale, offset). This lets
//! leader and follower be swapped independently (Star Arm 102 -> B601-DM or B601-RS,
//! the leader config stays the same).
//!
//! # Read on a separate thread
//!
//! [`LeaderThread`] keeps reading the leader free-running, and the control loop gets
//! the latest value **and its age** via [`LeaderThread::latest`]. Using it without
//! checking the age means that when a cable comes loose, a frozen value keeps being
//! used as the target.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(feature = "fashionstar")]
pub mod fashionstar;
pub mod synthetic;

#[derive(Debug, thiserror::Error)]
pub enum LeaderError {
    #[error("cannot open leader: {0}")]
    Open(String),
    #[error("cannot read from leader: {0}")]
    Read(String),
    #[error("invalid config: {0}")]
    Config(String),
}

/// One leader. **Blocking reads are fine** (it runs on its own thread).
pub trait Leader: Send {
    /// Names of the output joints (neutral space), in the same order as [`Leader::read`].
    fn names(&self) -> &[String];

    /// Read all joints once. Units are neutral-space rad (gripper is rad too).
    fn read(&mut self) -> Result<Vec<f64>, LeaderError>;

    /// One line appended to the status display.
    fn status_line(&self) -> String {
        String::new()
    }
}

/// Latest leader value.
#[derive(Debug, Clone)]
pub struct LeaderSample {
    pub q: Vec<f64>,
    /// Time the read completed.
    pub at: Instant,
    /// Sequence number (incremented on each successful read).
    pub seq: u64,
}

struct Shared {
    latest: Mutex<Option<LeaderSample>>,
    stop: AtomicBool,
    errors: AtomicU64,
    reads: AtomicU64,
    status: Mutex<String>,
}

/// Keeps reading a leader on a separate thread. Stops when dropped.
pub struct LeaderThread {
    names: Vec<String>,
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
    started: Instant,
}

impl LeaderThread {
    /// Read every `period` (rate-limited by the read itself if that is slower).
    pub fn spawn(mut leader: Box<dyn Leader>, period: Duration) -> Self {
        let names = leader.names().to_vec();
        let shared = Arc::new(Shared {
            latest: Mutex::new(None),
            stop: AtomicBool::new(false),
            errors: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            status: Mutex::new(String::new()),
        });
        let s = shared.clone();
        let handle = std::thread::Builder::new()
            .name("leader".into())
            .spawn(move || {
                let mut seq = 0u64;
                let mut next = Instant::now();
                let mut last_err_log = Instant::now() - Duration::from_secs(10);
                while !s.stop.load(Ordering::Relaxed) {
                    match leader.read() {
                        Ok(q) => {
                            seq += 1;
                            s.reads.fetch_add(1, Ordering::Relaxed);
                            *s.latest.lock().unwrap() = Some(LeaderSample {
                                q,
                                at: Instant::now(),
                                seq,
                            });
                        }
                        Err(e) => {
                            s.errors.fetch_add(1, Ordering::Relaxed);
                            // Don't log every cycle when a cable is unplugged.
                            if last_err_log.elapsed() > Duration::from_secs(1) {
                                log::warn!("leader: {e}");
                                last_err_log = Instant::now();
                            }
                        }
                    }
                    *s.status.lock().unwrap() = leader.status_line();
                    next += period;
                    let now = Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    } else {
                        next = now;
                    }
                }
            })
            .expect("leader thread");
        Self {
            names,
            shared,
            handle: Some(handle),
            started: Instant::now(),
        }
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Latest value. `None` if nothing has been read yet.
    pub fn latest(&self) -> Option<LeaderSample> {
        self.shared.latest.lock().unwrap().clone()
    }

    /// Effective read rate [Hz] and failure count.
    pub fn stats(&self) -> (f64, u64) {
        let reads = self.shared.reads.load(Ordering::Relaxed);
        let t = self.started.elapsed().as_secs_f64().max(1e-3);
        (reads as f64 / t, self.shared.errors.load(Ordering::Relaxed))
    }

    pub fn status_line(&self) -> String {
        self.shared.status.lock().unwrap().clone()
    }
}

impl Drop for LeaderThread {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Unwrap a multi-turn angle into the window of ±180° around the center of the range
/// of motion (same idea as `_round_to_valid_range` in Seeed's LeRobot plugin).
///
/// The servo's multi-turn counter can be arbitrary after each power-up, so take the
/// 360°-equivalent angle closest to the center of the range of motion.
pub fn unwrap_to_window(angle_rad: f64, lo: f64, hi: f64) -> f64 {
    use std::f64::consts::TAU;
    let center = 0.5 * (lo + hi);
    angle_rad - ((angle_rad - center) / TAU).round() * TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_picks_nearest_turn() {
        let d = |x: f64| x.to_radians();
        // Center 0: 350° becomes -10°.
        assert!((unwrap_to_window(d(350.0), d(-150.0), d(150.0)) - d(-10.0)).abs() < 1e-12);
        // Center 85° (-1..170): -200° becomes 160°.
        assert!((unwrap_to_window(d(-200.0), d(-1.0), d(170.0)) - d(160.0)).abs() < 1e-12);
        // Inside the window: unchanged.
        assert!((unwrap_to_window(d(42.0), d(-90.0), d(90.0)) - d(42.0)).abs() < 1e-12);
    }
}
