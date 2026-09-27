//! Exposes an arm on CAN as a [`Plant`]. **Not yet verified on real hardware**
//! (as of 2026-09-27 the motors have never been powered).
//!
//! # Coordinate conversion is confined here
//!
//! Upper layers only know the model's joint coordinates. The difference from the
//! motors comes down to three per-axis parameters:
//!
//! ```text
//! q_model = zero + sign * ratio * q_motor
//! ```
//!
//! - `sign` = ±1 (motor positive direction vs. model axis direction)
//! - `zero` = model angle when the motor is at its zero point
//! - `ratio` = model units / motor rad (1 for revolute axes; the pitch radius
//!   [m/rad] for the rack-and-pinion gripper)
//!
//! MIT gains and torque are converted so that power is conserved:
//! `τ_m = sign*ratio*τ`, `kp_m = ratio²*kp`, `kd_m = ratio²*kd`.
//!
//! # About going limp
//!
//! [`Plant::disarm`] disables the motors. **The arm will fall.** Folding the arm
//! before calling it is the upper layer's job (manip-runner's Park). DAMIAO may
//! re-energize on the next frame even after disable (misa-actuator handover §2),
//! so cut the power when you really need it stopped.
//!
//! # If the control loop stops
//!
//! The bus thread keeps sending the last target (a hold with gravity
//! compensation). Once the target is older than `stale_after`, position
//! stiffness is dropped, leaving only damping (the gravity FF is kept).
//! **It does not go limp** (the arm would fall).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use misa_actuator::{Actuator, RunMode};
use misa_core::{
    Axis, AxisHealth, AxisId, AxisRole, AxisState, AxisTable, Command, ControlMode, Observation,
    Plant, PlantCaps, Time,
};

/// Motor family. One family per bus (their frame formats differ).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    Damiao,
    Robstride,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MotorSpec {
    /// Joint name in the model (independent DOF).
    pub joint: String,
    pub id: u8,
    /// DAMIAO Master ID / RobStride host ID. Defaults to the family default when
    /// omitted (`0x10 + id` for DAMIAO, 0xFD for RobStride).
    #[serde(default)]
    pub host_id: Option<u16>,
    /// Model name (`dm4340p`, `dm4310`, `rs-06`, `rs-00`, etc.).
    pub model: String,
    #[serde(default = "one")]
    pub sign: f64,
    #[serde(default)]
    pub zero: f64,
    #[serde(default = "one")]
    pub ratio: f64,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct BusSpec {
    /// misa-can interface string (`can0`, `pcan:usb1`, `slcan:/dev/ttyACM0`).
    pub interface: String,
    pub vendor: Vendor,
    pub motor: Vec<MotorSpec>,
}

#[derive(Debug, Clone)]
pub struct CanOptions {
    pub buses: Vec<BusSpec>,
    /// Axis order (independent DOFs). Each axis must map to a motor on exactly one bus.
    pub joints: Vec<String>,
    /// Position stiffness is dropped once the target is older than this.
    pub stale_after: Duration,
}

/// Single-axis target in motor coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Setpoint {
    /// Powered but zero output (kp = kd = τ = 0).
    Limp,
    Mit {
        q: f32,
        v: f32,
        kp: f32,
        kd: f32,
        tau: f32,
    },
}

#[derive(Debug, Clone, Copy, Default)]
struct Feedback {
    q: f64,
    v: f64,
    tau: f64,
    temperature_c: f64,
    at: Option<Instant>,
    errors: u64,
}

struct Slot {
    setpoint: Mutex<(Setpoint, Instant)>,
    feedback: Mutex<Feedback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BusRequest {
    None,
    Enable,
    Disable,
}

struct BusShared {
    slots: Vec<Slot>,
    request: Mutex<BusRequest>,
    /// Result of the latest request (`None` = in progress).
    result: Mutex<Option<Result<(), String>>>,
    enabled: AtomicBool,
    stop: AtomicBool,
    cycles: AtomicU64,
}

struct Bus {
    spec: BusSpec,
    shared: Arc<BusShared>,
    handle: Option<JoinHandle<()>>,
    started: Instant,
}

pub struct CanArmPlant {
    table: AxisTable,
    caps: PlantCaps,
    buses: Vec<Bus>,
    /// Axis → (bus, motor index within the bus).
    map: Vec<(usize, usize)>,
    t0: Instant,
}

impl CanArmPlant {
    pub fn open(opts: CanOptions) -> Result<Self, String> {
        let mut map = vec![None; opts.joints.len()];
        for (bi, b) in opts.buses.iter().enumerate() {
            for (mi, m) in b.motor.iter().enumerate() {
                let ai = opts
                    .joints
                    .iter()
                    .position(|j| *j == m.joint)
                    .ok_or_else(|| format!("motor joint {} is not an independent DOF of the model", m.joint))?;
                if map[ai].is_some() {
                    return Err(format!("joint {} has two motors assigned", m.joint));
                }
                if m.ratio == 0.0 || m.sign.abs() != 1.0 {
                    return Err(format!("joint {}: sign must be ±1 and ratio must be non-zero", m.joint));
                }
                map[ai] = Some((bi, mi));
            }
        }
        let map: Vec<(usize, usize)> = map
            .into_iter()
            .enumerate()
            .map(|(i, m)| m.ok_or_else(|| format!("joint {} has no motor", opts.joints[i])))
            .collect::<Result<_, _>>()?;

        let mut buses = Vec::new();
        for spec in &opts.buses {
            let actuators = open_bus(spec)?;
            buses.push(Bus::spawn(spec.clone(), actuators, opts.stale_after));
        }
        let table = AxisTable::new(
            opts.joints
                .iter()
                .map(|j| Axis {
                    name: j.clone(),
                    role: AxisRole::Aux,
                })
                .collect(),
        )?;
        let caps = PlantCaps {
            modes: vec![ControlMode::Impedance, ControlMode::Torque],
            has_imu: false,
            has_contacts: false,
            driven: vec![true; opts.joints.len()],
        };
        Ok(Self {
            table,
            caps,
            buses,
            map,
            t0: Instant::now(),
        })
    }

    fn motor(&self, axis: usize) -> (&Bus, usize, &MotorSpec) {
        let (bi, mi) = self.map[axis];
        let b = &self.buses[bi];
        (b, mi, &b.spec.motor[mi])
    }

    fn request_all(&self, req: BusRequest, timeout: Duration) -> Result<(), String> {
        for b in &self.buses {
            *b.shared.result.lock().unwrap() = None;
            *b.shared.request.lock().unwrap() = req;
        }
        let t0 = Instant::now();
        for b in &self.buses {
            loop {
                if let Some(r) = b.shared.result.lock().unwrap().clone() {
                    r.map_err(|e| format!("{}: {e}", b.spec.interface))?;
                    break;
                }
                if t0.elapsed() > timeout {
                    return Err(format!("{}: {:?} did not complete within {:?}", b.spec.interface, req, timeout));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }
}

impl Plant for CanArmPlant {
    fn axes(&self) -> &AxisTable {
        &self.table
    }

    fn capabilities(&self) -> &PlantCaps {
        &self.caps
    }

    fn arm(&mut self) -> Result<(), String> {
        self.request_all(BusRequest::Enable, Duration::from_secs(2))
    }

    fn disarm(&mut self) -> Result<(), String> {
        self.request_all(BusRequest::Disable, Duration::from_secs(2))
    }

    fn status_line(&self) -> String {
        self.buses
            .iter()
            .map(|b| {
                let hz = b.shared.cycles.load(Ordering::Relaxed) as f64
                    / b.started.elapsed().as_secs_f64().max(1e-3);
                format!("{} {:.0}Hz", b.spec.interface, hz)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
        let now = Instant::now();
        for (i, a) in cmd.axes().iter().enumerate() {
            let (bus, mi, m) = self.motor(i);
            let sp = match a.mode {
                ControlMode::Idle => Setpoint::Limp,
                ControlMode::Impedance | ControlMode::Position => Setpoint::Mit {
                    q: (m.sign * (a.position_rad - m.zero) / m.ratio) as f32,
                    v: (m.sign * a.velocity_rad_s / m.ratio) as f32,
                    kp: (a.kp_nm_per_rad * m.ratio * m.ratio) as f32,
                    kd: (a.kd_nm_s_per_rad * m.ratio * m.ratio) as f32,
                    tau: (m.sign * a.torque_ff_nm * m.ratio) as f32,
                },
                ControlMode::Torque => Setpoint::Mit {
                    q: 0.0,
                    v: 0.0,
                    kp: 0.0,
                    kd: 0.0,
                    tau: (m.sign * a.torque_ff_nm * m.ratio) as f32,
                },
                ControlMode::Velocity => {
                    return Err("velocity control is not supported by this Plant".into());
                }
            };
            *bus.shared.slots[mi].setpoint.lock().unwrap() = (sp, now);
        }
        obs.time = Time::from_nanos(self.t0.elapsed().as_nanos() as u64);
        for i in 0..self.map.len() {
            let (bus, mi, m) = self.motor(i);
            let fb = *bus.shared.slots[mi].feedback.lock().unwrap();
            if let Some(s) = obs.get_mut(AxisId::new(i as u16)) {
                *s = AxisState {
                    position_rad: m.zero + m.sign * m.ratio * fb.q,
                    velocity_rad_s: m.sign * m.ratio * fb.v,
                    torque_nm: Some(m.sign * fb.tau / m.ratio),
                    health: AxisHealth {
                        valid: fb.at.is_some(),
                        age: fb.at.map(|t| now.saturating_duration_since(t)).unwrap_or(Duration::MAX),
                        fault_raw: 0,
                        temperature_c: fb.temperature_c.is_finite().then_some(fb.temperature_c),
                        voltage_v: None,
                    },
                };
            }
        }
        Ok(())
    }
}

/// Feedback resolution of one motor, in **model** units: `[position, velocity,
/// torque]` per LSB of the MIT status frame.
///
/// Derived from the vendor protocol tables, so a simulator can quantize its
/// observations exactly as the real bus would. DAMIAO packs position in 16 bits
/// and velocity / torque in 12 bits over `±{P,V,T}MAX`; RobStride packs all
/// three in 16 bits over its per-model MIT scales. The 12-bit DAMIAO velocity
/// is the coarse one: 0.015 rad/s per LSB on a DM4310 (±30 rad/s).
pub fn feedback_resolution(vendor: Vendor, m: &MotorSpec) -> Result<[f64; 3], String> {
    let [p, v, t, bits_vt] = match vendor {
        Vendor::Damiao => {
            let model = damiao_driver::MotorModel::from_name(&m.model)
                .ok_or_else(|| format!("unknown DAMIAO model: {}", m.model))?;
            let l = model.limits();
            [l.p_max as f64, l.v_max as f64, l.t_max as f64, 12.0]
        }
        Vendor::Robstride => {
            let model = robstride_driver::MotorModel::from_name(&m.model)
                .ok_or_else(|| format!("unknown RobStride model: {}", m.model))?;
            let s = robstride_driver::protocol::MitScales::for_model(model);
            [s.position as f64, s.velocity as f64, s.torque as f64, 16.0]
        }
    };
    let lsb = |half: f64, bits: f64| 2.0 * half / (2f64.powf(bits) - 1.0);
    // Motor units -> model units: q = zero + sign*ratio*q_m, tau = sign*tau_m/ratio.
    let r = m.ratio.abs();
    Ok([lsb(p, 16.0) * r, lsb(v, bits_vt) * r, lsb(t, bits_vt) / r])
}

type Motor = Box<dyn Actuator + Send>;

fn open_bus(spec: &BusSpec) -> Result<Vec<Motor>, String> {
    let mut out: Vec<Motor> = Vec::new();
    match spec.vendor {
        Vendor::Damiao => {
            use damiao_driver::{AnyCanBus, DamiaoMotor, MotorModel, Shared};
            let bus = Shared::new(AnyCanBus::open(&spec.interface).map_err(|e| e.to_string())?);
            for m in &spec.motor {
                let model = MotorModel::from_name(&m.model)
                    .ok_or_else(|| format!("unknown DAMIAO model name: {}", m.model))?;
                let master = m.host_id.unwrap_or(0x10 + m.id as u16);
                out.push(Box::new(DamiaoMotor::with_bus_and_master(bus.clone(), m.id, master, model)));
            }
        }
        Vendor::Robstride => {
            use robstride_driver::{AnyCanBus, Motor as RsMotor, MotorModel};
            let bus = misa_actuator::Shared::new(
                AnyCanBus::open(&spec.interface).map_err(|e| e.to_string())?,
            );
            for m in &spec.motor {
                let model = MotorModel::from_name(&m.model)
                    .ok_or_else(|| format!("unknown RobStride model name: {}", m.model))?;
                let host = m.host_id.unwrap_or(0xFD) as u8;
                out.push(Box::new(RsMotor::with_bus_and_host(bus.clone(), m.id, host, model)));
            }
        }
    }
    Ok(out)
}

impl Bus {
    fn spawn(spec: BusSpec, mut motors: Vec<Motor>, stale_after: Duration) -> Self {
        let now = Instant::now();
        let shared = Arc::new(BusShared {
            slots: (0..motors.len())
                .map(|_| Slot {
                    setpoint: Mutex::new((Setpoint::Limp, now)),
                    feedback: Mutex::new(Feedback {
                        temperature_c: f64::NAN,
                        ..Default::default()
                    }),
                })
                .collect(),
            request: Mutex::new(BusRequest::None),
            result: Mutex::new(None),
            enabled: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            cycles: AtomicU64::new(0),
        });
        let s = shared.clone();
        let name = format!("can-{}", spec.interface);
        let handle = std::thread::Builder::new()
            .name(name)
            .spawn(move || bus_loop(&s, &mut motors, stale_after))
            .expect("bus thread");
        Self {
            spec,
            shared,
            handle: Some(handle),
            started: Instant::now(),
        }
    }
}

fn bus_loop(s: &BusShared, motors: &mut [Motor], stale_after: Duration) {
    let mut last_err_log = Instant::now() - Duration::from_secs(10);
    while !s.stop.load(Ordering::Relaxed) {
        let req = std::mem::replace(&mut *s.request.lock().unwrap(), BusRequest::None);
        match req {
            BusRequest::None => {}
            BusRequest::Enable => {
                let r = motors.iter_mut().try_for_each(|m| {
                    m.set_run_mode(RunMode::Mit)?;
                    m.enable().map(|_| ())
                });
                s.enabled.store(r.is_ok(), Ordering::Relaxed);
                *s.result.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
            }
            BusRequest::Disable => {
                // Always disable the rest even if one motor fails.
                let mut first_err = None;
                for m in motors.iter_mut() {
                    if let Err(e) = m.disable() {
                        first_err.get_or_insert(e.to_string());
                    }
                }
                s.enabled.store(false, Ordering::Relaxed);
                *s.result.lock().unwrap() = Some(first_err.map_or(Ok(()), Err));
            }
        }

        let enabled = s.enabled.load(Ordering::Relaxed);
        for (i, m) in motors.iter_mut().enumerate() {
            let (sp, at) = *s.slots[i].setpoint.lock().unwrap();
            let r = if !enabled {
                m.measure()
            } else {
                match sp {
                    Setpoint::Limp => m.mit_control(0.0, 0.0, 0.0, 0.0, 0.0),
                    Setpoint::Mit { q, v, kp, kd, tau } => {
                        if at.elapsed() > stale_after {
                            // Control loop has stalled: drop stiffness, keep only damping + the last FF.
                            m.mit_control(q, 0.0, 0.0, kd.max(0.5), tau)
                        } else {
                            m.mit_control(q, v, kp, kd, tau)
                        }
                    }
                }
            };
            let mut fb = s.slots[i].feedback.lock().unwrap();
            match r {
                Ok(f) => {
                    fb.q = f.position_rad as f64;
                    fb.v = f.velocity_rad_per_s as f64;
                    fb.tau = f.torque_nm as f64;
                    fb.temperature_c = f.temperature_c as f64;
                    fb.at = Some(Instant::now());
                }
                Err(e) => {
                    fb.errors += 1;
                    if last_err_log.elapsed() > Duration::from_secs(1) {
                        log::warn!("motor {}: {e}", m.motor_id());
                        last_err_log = Instant::now();
                    }
                }
            }
        }
        s.cycles.fetch_add(1, Ordering::Relaxed);
        // All motors on the bus are serviced serially, so don't wait here (free-running).
        // Yield very briefly so we don't hog the CPU where round trips are very fast.
        std::thread::yield_now();
    }
}

impl Drop for CanArmPlant {
    fn drop(&mut self) {
        for b in &mut self.buses {
            b.shared.stop.store(true, Ordering::Relaxed);
            if let Some(h) = b.handle.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(model: &str, ratio: f64) -> MotorSpec {
        MotorSpec {
            joint: "j".into(),
            id: 1,
            host_id: None,
            model: model.into(),
            sign: 1.0,
            zero: 0.0,
            ratio,
        }
    }

    #[test]
    fn resolution_matches_protocol_tables() {
        // DM4310: ±12.5 rad / 16 bit, ±30 rad/s / 12 bit.
        let [p, v, _] = feedback_resolution(Vendor::Damiao, &spec("dm4310", 1.0)).unwrap();
        assert!((p - 25.0 / 65535.0).abs() < 1e-9);
        assert!((v - 60.0 / 4095.0).abs() < 1e-6);
        // Gripper through a 0.00605 m/rad rack: position step shrinks by the ratio,
        // force step grows by 1/ratio.
        let [pg, _, tg] = feedback_resolution(Vendor::Damiao, &spec("dm4310", 0.00605)).unwrap();
        assert!((pg - p * 0.00605).abs() < 1e-12);
        let [_, _, t] = feedback_resolution(Vendor::Damiao, &spec("dm4310", 1.0)).unwrap();
        assert!((tg - t / 0.00605).abs() < 1e-9);
        // RobStride: 16 bits on all three.
        let [p, v, t] = feedback_resolution(Vendor::Robstride, &spec("rs-00", 1.0)).unwrap();
        assert!(p > 0.0 && v < 0.01 && t < 0.01);
    }
}
