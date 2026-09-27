//! Hardware non-idealities layered on top of a simulated plant.
//!
//! The MuJoCo and rigid-body plants are ideal: a command acts in the same tick,
//! and the observation is exact. The CAN arm is not:
//!
//! - **Latency.** The bus thread walks the motors one by one, so a command
//!   written by the control loop reaches a motor up to one bus cycle later, and
//!   the feedback it returns is correspondingly old.
//! - **Jitter.** The control loop and the bus thread are not synchronised; some
//!   ticks the new command misses the bus cycle and the old one is sent again.
//! - **Quantization.** MIT feedback is bit-packed: DAMIAO sends velocity in
//!   12 bits (0.015 rad/s per LSB on a DM4310), which is what the D term and
//!   the OSC see as "velocity".
//!
//! This wrapper adds those at tick granularity, around any [`Plant`]. Friction
//! is not here: it has to act on every physics substep, so it lives inside the
//! physics plants.
//!
//! Deterministic: jitter uses a seeded xorshift, so a run with effects is
//! reproducible (and can be recorded and replayed).

use std::collections::VecDeque;
use std::time::Duration;

use misa_core::{AxisId, AxisTable, Command, Observation, Plant, PlantCaps};

#[derive(Debug, Clone, PartialEq)]
pub struct Effects {
    /// Ticks between the controller issuing a command and the plant applying it.
    pub command_delay_ticks: usize,
    /// Extra ticks by which the observation is older than the plant state.
    pub observation_delay_ticks: usize,
    /// Probability that a tick's command is one tick later still (missed bus cycle).
    pub jitter_probability: f64,
    /// Per-axis `[position, velocity, torque]` LSB. `None` = no quantization.
    pub quantization: Option<Vec<[f64; 3]>>,
    pub seed: u64,
    /// Control period, for the `age` reported in the observation.
    pub period: Duration,
}

pub struct EffectsPlant {
    inner: Box<dyn Plant>,
    fx: Effects,
    /// Commands issued so far, newest at the back (bounded).
    commands: VecDeque<Command>,
    /// Observations produced by the inner plant, newest at the back (bounded).
    observations: VecDeque<Observation>,
    scratch: Observation,
    rng: u64,
}

impl EffectsPlant {
    pub fn new(inner: Box<dyn Plant>, fx: Effects) -> Self {
        let n = inner.axes().len();
        if let Some(q) = &fx.quantization {
            assert_eq!(q.len(), n, "quantization table length differs from axis count");
        }
        Self {
            rng: fx.seed.max(1),
            inner,
            fx,
            commands: VecDeque::new(),
            observations: VecDeque::new(),
            scratch: Observation::empty(n, 0),
        }
    }

    fn next_uniform(&mut self) -> f64 {
        // xorshift64*: tiny, deterministic, good enough for a coin flip.
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn quantize(x: f64, step: f64) -> f64 {
    if step > 0.0 { (x / step).round() * step } else { x }
}

impl Plant for EffectsPlant {
    fn axes(&self) -> &AxisTable {
        self.inner.axes()
    }

    fn capabilities(&self) -> &PlantCaps {
        self.inner.capabilities()
    }

    fn arm(&mut self) -> Result<(), String> {
        self.commands.clear();
        self.inner.arm()
    }

    fn disarm(&mut self) -> Result<(), String> {
        self.inner.disarm()
    }

    fn status_line(&self) -> String {
        format!(
            "{} +fx(delay {}+{} tick, jitter {:.0}%{})",
            self.inner.status_line(),
            self.fx.command_delay_ticks,
            self.fx.observation_delay_ticks,
            self.fx.jitter_probability * 100.0,
            if self.fx.quantization.is_some() { ", quantized" } else { "" }
        )
    }

    fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
        let keep = self.fx.command_delay_ticks + 2;
        self.commands.push_back(cmd.clone());
        while self.commands.len() > keep {
            self.commands.pop_front();
        }
        let mut delay = self.fx.command_delay_ticks;
        if self.fx.jitter_probability > 0.0 && self.next_uniform() < self.fx.jitter_probability {
            delay += 1;
        }
        // Until enough history exists, the oldest command stands in (the bus
        // would still be sending whatever it had).
        let idx = self.commands.len().saturating_sub(1 + delay);
        let applied = self.commands[idx].clone();
        self.inner.exchange(&applied, &mut self.scratch)?;

        self.observations.push_back(self.scratch.clone());
        while self.observations.len() > self.fx.observation_delay_ticks + 1 {
            self.observations.pop_front();
        }
        *obs = self.observations.front().cloned().expect("just pushed");
        let extra_age = self.fx.period * (self.observations.len() as u32 - 1);
        for i in 0..obs.len() {
            let id = AxisId::new(i as u16);
            let Some(a) = obs.get_mut(id) else { continue };
            a.health.age += extra_age;
            if let Some(q) = &self.fx.quantization {
                let [sp, sv, st] = q[i];
                a.position_rad = quantize(a.position_rad, sp);
                a.velocity_rad_s = quantize(a.velocity_rad_s, sv);
                a.torque_nm = a.torque_nm.map(|t| quantize(t, st));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use misa_core::{Axis, AxisCommand, AxisRole, AxisState, ControlMode};

    /// One-axis plant whose position is the last applied command's position.
    struct Echo {
        table: AxisTable,
        caps: PlantCaps,
    }

    impl Plant for Echo {
        fn axes(&self) -> &AxisTable {
            &self.table
        }
        fn capabilities(&self) -> &PlantCaps {
            &self.caps
        }
        fn arm(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn disarm(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn exchange(&mut self, cmd: &Command, obs: &mut Observation) -> Result<(), String> {
            let p = cmd.axes()[0].position_rad;
            *obs.get_mut(AxisId::new(0)).unwrap() = AxisState {
                position_rad: p,
                velocity_rad_s: p * 0.1,
                ..Default::default()
            };
            Ok(())
        }
    }

    fn echo() -> Box<dyn Plant> {
        Box::new(Echo {
            table: AxisTable::new(vec![Axis { name: "j".into(), role: AxisRole::Aux }]).unwrap(),
            caps: PlantCaps::default(),
        })
    }

    fn cmd(p: f64) -> Command {
        let mut c = Command::idle(1);
        *c.get_mut(AxisId::new(0)).unwrap() = AxisCommand {
            mode: ControlMode::Impedance,
            position_rad: p,
            ..AxisCommand::idle()
        };
        c
    }

    fn fx(cmd_delay: usize, obs_delay: usize) -> Effects {
        Effects {
            command_delay_ticks: cmd_delay,
            observation_delay_ticks: obs_delay,
            jitter_probability: 0.0,
            quantization: None,
            seed: 1,
            period: Duration::from_millis(2),
        }
    }

    #[test]
    fn delays_commands_and_observations() {
        let mut p = EffectsPlant::new(echo(), fx(1, 2));
        let mut obs = Observation::empty(1, 0);
        let mut seen = Vec::new();
        for k in 1..=6 {
            p.exchange(&cmd(k as f64), &mut obs).unwrap();
            seen.push(obs.axes()[0].position_rad);
        }
        // Command k is applied at tick k+1 (1 tick late) and seen 2 ticks after that.
        assert_eq!(seen, vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0]);
        assert_eq!(obs.axes()[0].health.age, Duration::from_millis(4));
    }

    #[test]
    fn quantizes_feedback() {
        let mut f = fx(0, 0);
        f.quantization = Some(vec![[0.25, 0.5, 0.0]]);
        let mut p = EffectsPlant::new(echo(), f);
        let mut obs = Observation::empty(1, 0);
        p.exchange(&cmd(1.1), &mut obs).unwrap();
        assert_eq!(obs.axes()[0].position_rad, 1.0);
        assert_eq!(obs.axes()[0].velocity_rad_s, 0.0);
    }

    #[test]
    fn jitter_is_deterministic() {
        let run = || {
            let mut f = fx(0, 0);
            f.jitter_probability = 0.3;
            let mut p = EffectsPlant::new(echo(), f);
            let mut obs = Observation::empty(1, 0);
            (1..=50)
                .map(|k| {
                    p.exchange(&cmd(k as f64), &mut obs).unwrap();
                    obs.axes()[0].position_rad
                })
                .collect::<Vec<_>>()
        };
        let a = run();
        assert_eq!(a, run());
        // Some ticks slipped by one, none by more.
        let slipped = a.iter().enumerate().filter(|(i, x)| **x != (*i + 1) as f64).count();
        assert!(slipped > 5 && slipped < 30, "slipped {slipped}");
        assert!(a.iter().enumerate().all(|(i, x)| (i + 1) as f64 - x <= 1.0));
    }
}
