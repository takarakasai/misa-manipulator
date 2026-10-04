//! Rigid-body Plant without MuJoCo. Integrates manip-model's forward dynamics
//! (`M⁻¹(τ − h)`).
//!
//! No contact, no mechanical joint stops; friction smoothed (tanh) or, with
//! [`RigidPlant::with_stiction`], dry (sticks). It exists **to exercise the
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
    friction: Vec<(f64, f64)>,
    friction_v_eps: f64,
    stiction: bool,
}

impl RigidPlant {
    /// `friction` is per axis `(coulomb, viscous)`; `friction_v_eps` is the tanh
    /// smoothing velocity for the Coulomb part (see manip-plant-mujoco).
    pub fn new(
        arm: ArmModel,
        q0: DVector<f64>,
        control_period_s: f64,
        timestep_s: f64,
        friction: Vec<(f64, f64)>,
        friction_v_eps: f64,
    ) -> Result<Self, String> {
        if friction.len() != arm.n() {
            return Err(format!("friction table has {} entries, arm has {} DOFs", friction.len(), arm.n()));
        }
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
            friction,
            friction_v_eps: friction_v_eps.max(1e-4),
            stiction: false,
            arm,
        })
    }

    /// Dry Coulomb friction instead of the tanh ramp: per step, a joint whose
    /// velocity the friction impulse `h·fc` can cancel stops (and stays
    /// stopped while the other torques stay under `fc`); otherwise it slows
    /// by that impulse (time-stepping, like MuJoCo's `frictionloss`; the
    /// coupling through the off-diagonal inertia is ignored). The tanh ramp
    /// is a stiff damper around v = 0 and never sticks, so it can't show
    /// limit cycles or offsets the real gearboxes produce.
    pub fn with_stiction(mut self, on: bool) -> Self {
        self.stiction = on;
        self
    }
}

impl RigidPlant {
    /// Joint-side friction torque at the current velocity.
    fn joint_friction(&self) -> DVector<f64> {
        DVector::from_iterator(
            self.v.len(),
            self.v.iter().zip(&self.friction).map(|(&v, &(fc, fv))| {
                -fc * (v / self.friction_v_eps).tanh() - fv * v
            }),
        )
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
                let chol = s
                    .mass
                    .clone()
                    .cholesky()
                    .ok_or("mass matrix is not positive definite (check armature)")?;
                if self.stiction {
                    let viscous = DVector::from_iterator(self.v.len(), self.v.iter().zip(&self.friction).map(|(&v, &(_, fv))| -fv * v));
                    let qdd = chol.solve(&(&self.tau + viscous - &s.nle));
                    let minv = chol.inverse();
                    let v_free = &self.v + qdd * self.h;
                    for i in 0..self.v.len() {
                        let dv_max = self.h * self.friction[i].0 * minv[(i, i)];
                        self.v[i] = if v_free[i].abs() <= dv_max { 0.0 } else { v_free[i] - v_free[i].signum() * dv_max };
                    }
                } else {
                    let qdd = chol.solve(&(&self.tau + self.joint_friction() - &s.nle));
                    self.v += qdd * self.h;
                }
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
