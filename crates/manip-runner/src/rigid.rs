//! Rigid-body Plant without MuJoCo. Integrates manip-model's forward dynamics
//! (`M⁻¹(τ − h)`).
//!
//! No contact, no friction, no mechanical joint stops. It exists **to exercise the
//! control laws, mode transitions and profile assembly where MuJoCo isn't available
//! (CI, SBC)**. Judge physical fidelity with MuJoCo (`--plant sim`).

use std::time::Duration;

use manip_model::ArmModel;
use misa_core::{
    Axis, AxisHealth, AxisId, AxisRole, AxisState, AxisTable, Command, ControlMode, Observation,
    Plant, PlantCaps, Time,
};
use nalgebra::DVector;

pub struct RigidPlant {
    arm: ArmModel,
    table: AxisTable,
    caps: PlantCaps,
    q: DVector<f64>,
    v: DVector<f64>,
    tau: DVector<f64>,
    substeps: usize,
    h: f64,
    armed: bool,
    time: Duration,
}

impl RigidPlant {
    pub fn new(arm: ArmModel, q0: DVector<f64>, control_period_s: f64, timestep_s: f64) -> Result<Self, String> {
        let n = arm.n();
        let table = AxisTable::new(
            arm.dofs()
                .iter()
                .map(|d| Axis {
                    name: d.name.clone(),
                    role: AxisRole::Aux,
                })
                .collect(),
        )?;
        let substeps = (control_period_s / timestep_s).round().max(1.0) as usize;
        Ok(Self {
            table,
            caps: PlantCaps {
                modes: vec![ControlMode::Impedance, ControlMode::Torque],
                has_imu: false,
                has_contacts: false,
                driven: vec![true; n],
            },
            q: q0,
            v: DVector::zeros(n),
            tau: DVector::zeros(n),
            substeps,
            h: control_period_s / substeps as f64,
            armed: false,
            time: Duration::ZERO,
            arm,
        })
    }
}

impl Plant for RigidPlant {
    fn axes(&self) -> &AxisTable {
        &self.table
    }

    fn capabilities(&self) -> &PlantCaps {
        &self.caps
    }

    fn arm(&mut self) -> Result<(), String> {
        self.armed = true;
        Ok(())
    }

    fn disarm(&mut self) -> Result<(), String> {
        self.armed = false;
        Ok(())
    }

    fn status_line(&self) -> String {
        format!("rigid ×{} t={:.2}s", self.substeps, self.time.as_secs_f64())
    }

    fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
        if self.armed {
            for _ in 0..self.substeps {
                let s = self.arm.evaluate(self.q.as_slice(), self.v.as_slice());
                for (i, a) in cmd.axes().iter().enumerate() {
                    let t = match a.mode {
                        ControlMode::Idle => 0.0,
                        ControlMode::Torque => a.torque_ff_nm,
                        _ => {
                            a.kp_nm_per_rad * (a.position_rad - self.q[i])
                                + a.kd_nm_s_per_rad * (a.velocity_rad_s - self.v[i])
                                + a.torque_ff_nm
                        }
                    };
                    let e = self.arm.dofs()[i].effort;
                    self.tau[i] = if e.is_finite() { t.clamp(-e, e) } else { t };
                }
                let qdd = s
                    .mass
                    .clone()
                    .cholesky()
                    .ok_or("mass matrix is not positive definite (check armature)")?
                    .solve(&(&self.tau - &s.nle));
                self.v += qdd * self.h;
                self.q += &self.v * self.h;
                // Range of motion: in place of a mechanical stop, stop the position and
                // zero the velocity.
                for (i, d) in self.arm.dofs().iter().enumerate() {
                    if self.q[i] < d.q_min || self.q[i] > d.q_max {
                        self.q[i] = d.clamp(self.q[i]);
                        self.v[i] = 0.0;
                    }
                }
                self.time += Duration::from_secs_f64(self.h);
            }
        }
        obs.time = Time::from_nanos(self.time.as_nanos() as u64);
        for i in 0..self.q.len() {
            if let Some(s) = obs.get_mut(AxisId::new(i as u16)) {
                *s = AxisState {
                    position_rad: self.q[i],
                    velocity_rad_s: self.v[i],
                    torque_nm: Some(self.tau[i]),
                    health: AxisHealth {
                        valid: true,
                        ..Default::default()
                    },
                };
            }
        }
        Ok(())
    }
}
