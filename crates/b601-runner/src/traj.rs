//! Time-parametrized trajectories for motion commands.
//!
//! A [`Traj`] is a piecewise quintic in any number of coordinates, with
//! position, velocity and acceleration given at every knot, so a trajectory
//! can start from a moving reference (a motion replaced mid-way) and pass
//! through waypoints without stopping. Durations come from the limits: each
//! segment gets the time a rest-to-rest quintic needs, then the whole
//! trajectory is stretched until sampling shows every limit holds.

use nalgebra::DVector;

/// Peak speed and acceleration of a rest-to-rest quintic over distance `d`
/// in time `T`: `1.875·d/T` and `5.7735·d/T²`.
const QUINTIC_V: f64 = 1.875;
const QUINTIC_A: f64 = 5.773_502_691_896_258;

#[derive(Debug, Clone)]
pub struct Knot {
    pub p: DVector<f64>,
    pub v: DVector<f64>,
    pub a: DVector<f64>,
}

impl Knot {
    pub fn at_rest(p: DVector<f64>) -> Self {
        let n = p.len();
        Self { p, v: DVector::zeros(n), a: DVector::zeros(n) }
    }
}

#[derive(Debug, Clone)]
pub struct Traj {
    /// Knot times, starting at 0.
    t: Vec<f64>,
    knots: Vec<Knot>,
}

/// How far a trajectory sample is over the limits: the largest of
/// `|v| / v_max` and `√(|a| / a_max)` (both scale as 1/stretch, so this is the
/// factor to stretch time by). The caller decides what "|v|" means (per
/// joint, or the norm of a TCP block).
pub type Excess<'a> = &'a dyn Fn(&DVector<f64>, &DVector<f64>) -> f64;

/// Per-coordinate limits as an [`Excess`].
pub fn box_excess(v_max: &DVector<f64>, a_max: &DVector<f64>) -> impl Fn(&DVector<f64>, &DVector<f64>) -> f64 {
    let (vm, am) = (v_max.clone(), a_max.clone());
    move |v: &DVector<f64>, a: &DVector<f64>| {
        (0..v.len())
            .map(|i| (v[i].abs() / vm[i].max(1e-9)).max((a[i].abs() / am[i].max(1e-9)).sqrt()))
            .fold(0.0, f64::max)
    }
}

impl Traj {
    /// Trajectory from `start` through `points` (each reached at rest when it
    /// is the last, passed with a smooth velocity otherwise). `times` (from
    /// the start, one per point) are kept when they respect the limits and
    /// stretched otherwise; `min_duration` stretches a single move. Returns
    /// the trajectory and whether the timing was stretched beyond what was
    /// asked.
    pub fn through(
        start: Knot,
        points: &[DVector<f64>],
        times: Option<&[f64]>,
        min_duration: Option<f64>,
        v_max: &DVector<f64>,
        a_max: &DVector<f64>,
        excess: Excess,
    ) -> (Self, bool) {
        assert!(!points.is_empty());
        // Segment durations: as asked, or the time a rest-to-rest quintic
        // needs per coordinate.
        let mut prev = start.p.clone();
        let mut durations = Vec::with_capacity(points.len());
        for (k, p) in points.iter().enumerate() {
            let d = p - &prev;
            let need = (0..d.len())
                .map(|i| {
                    let x = d[i].abs();
                    (QUINTIC_V * x / v_max[i].max(1e-9)).max((QUINTIC_A * x / a_max[i].max(1e-9)).sqrt())
                })
                .fold(0.0, f64::max);
            let asked = match times {
                Some(ts) => ts[k] - if k == 0 { 0.0 } else { ts[k - 1] },
                None if points.len() == 1 => min_duration.unwrap_or(0.0),
                None => 0.0,
            };
            durations.push(asked.max(need).max(MIN_SEGMENT));
            prev = p.clone();
        }
        let asked_total: Option<f64> = times.map(|ts| *ts.last().unwrap()).or(min_duration);
        let mut stretched = false;
        let mut traj = Self::build(&start, points, &durations);
        for _ in 0..12 {
            let r = traj.peak_excess(excess);
            if r <= 1.0 + 1e-6 {
                break;
            }
            for d in &mut durations {
                *d *= r.min(4.0) * 1.01;
            }
            traj = Self::build(&start, points, &durations);
        }
        if let Some(asked) = asked_total
            && traj.duration() > asked + 1e-6
        {
            stretched = true;
        }
        (traj, stretched)
    }

    /// Knots with velocities at the vias: per coordinate, the mean of the
    /// neighbouring segment slopes, or 0 where the path turns back (no
    /// overshoot between waypoints).
    fn build(start: &Knot, points: &[DVector<f64>], durations: &[f64]) -> Self {
        let n = start.p.len();
        let mut knots = vec![start.clone()];
        for (k, p) in points.iter().enumerate() {
            let last = k + 1 == points.len();
            let v = if last {
                DVector::zeros(n)
            } else {
                let before = (p - &knots[k].p) / durations[k];
                let after = (&points[k + 1] - p) / durations[k + 1];
                DVector::from_iterator(
                    n,
                    (0..n).map(|i| if before[i] * after[i] <= 0.0 { 0.0 } else { 0.5 * (before[i] + after[i]) }),
                )
            };
            knots.push(Knot { p: p.clone(), v, a: DVector::zeros(n) });
        }
        let mut t = vec![0.0];
        for d in durations {
            t.push(t.last().unwrap() + d);
        }
        Self { t, knots }
    }

    pub fn duration(&self) -> f64 {
        *self.t.last().unwrap()
    }

    /// Position, velocity, acceleration at `t` (clamped to the ends).
    pub fn sample(&self, t: f64) -> Knot {
        let t = t.clamp(0.0, self.duration());
        let k = match self.t.iter().rposition(|&tk| tk <= t) {
            Some(k) if k + 1 < self.t.len() => k,
            _ => self.t.len() - 2,
        };
        let (k0, k1) = (&self.knots[k], &self.knots[k + 1]);
        let h = self.t[k + 1] - self.t[k];
        let s = t - self.t[k];
        let n = k0.p.len();
        let mut out = Knot::at_rest(DVector::zeros(n));
        for i in 0..n {
            let c = quintic(k0.p[i], k0.v[i], k0.a[i], k1.p[i], k1.v[i], k1.a[i], h);
            out.p[i] = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))));
            out.v[i] = c[1] + s * (2.0 * c[2] + s * (3.0 * c[3] + s * (4.0 * c[4] + s * 5.0 * c[5])));
            out.a[i] = 2.0 * c[2] + s * (6.0 * c[3] + s * (12.0 * c[4] + s * 20.0 * c[5]));
        }
        out
    }

    fn peak_excess(&self, excess: Excess) -> f64 {
        let mut worst = 0.0f64;
        for k in 0..self.t.len() - 1 {
            for j in 0..=SAMPLES {
                let t = self.t[k] + (self.t[k + 1] - self.t[k]) * j as f64 / SAMPLES as f64;
                let s = self.sample(t);
                worst = worst.max(excess(&s.v, &s.a));
            }
        }
        worst
    }
}

/// Samples per segment when checking limits.
const SAMPLES: usize = 40;
/// Shortest segment [s] (a zero-length move still takes a moment).
const MIN_SEGMENT: f64 = 0.02;

/// Coefficients of the quintic from `(p0, v0, a0)` to `(p1, v1, a1)` over `h`.
fn quintic(p0: f64, v0: f64, a0: f64, p1: f64, v1: f64, a1: f64, h: f64) -> [f64; 6] {
    let (h2, h3, h4, h5) = (h * h, h * h * h, h.powi(4), h.powi(5));
    let d = p1 - p0 - v0 * h - 0.5 * a0 * h2;
    let dv = v1 - v0 - a0 * h;
    let da = a1 - a0;
    [
        p0,
        v0,
        0.5 * a0,
        (10.0 * d - 4.0 * dv * h + 0.5 * da * h2) / h3,
        (-15.0 * d + 7.0 * dv * h - da * h2) / h4,
        (6.0 * d - 3.0 * dv * h + 0.5 * da * h2) / h5,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim(n: usize, v: f64, a: f64) -> (DVector<f64>, DVector<f64>) {
        (DVector::from_element(n, v), DVector::from_element(n, a))
    }

    #[test]
    fn point_to_point_respects_limits_and_ends_at_rest() {
        let (vm, am) = lim(2, 1.0, 4.0);
        let ex = box_excess(&vm, &am);
        let start = Knot::at_rest(DVector::from_vec(vec![0.0, 0.0]));
        let goal = DVector::from_vec(vec![1.2, -0.3]);
        let (t, stretched) = Traj::through(start, std::slice::from_ref(&goal), None, None, &vm, &am, &ex);
        assert!(!stretched);
        let end = t.sample(t.duration());
        assert!((end.p - &goal).norm() < 1e-12 && end.v.norm() < 1e-12 && end.a.norm() < 1e-9);
        for k in 0..=1000 {
            let s = t.sample(t.duration() * k as f64 / 1000.0);
            assert!(ex(&s.v, &s.a) <= 1.0 + 1e-6, "over the limits at {k}");
        }
        // Not needlessly slow: the speed limit is nearly reached.
        let peak = (0..=1000).map(|k| t.sample(t.duration() * k as f64 / 1000.0).v[0].abs()).fold(0.0, f64::max);
        assert!(peak > 0.9, "peak speed {peak}");
    }

    /// Starting from a moving reference: continuous, and the asked duration is
    /// kept when it is feasible.
    #[test]
    fn starts_from_motion_and_keeps_a_feasible_duration() {
        let (vm, am) = lim(1, 1.0, 4.0);
        let ex = box_excess(&vm, &am);
        let start = Knot { p: DVector::from_vec(vec![0.2]), v: DVector::from_vec(vec![0.5]), a: DVector::from_vec(vec![0.0]) };
        let (t, stretched) = Traj::through(start, &[DVector::from_vec(vec![0.6])], None, Some(2.0), &vm, &am, &ex);
        assert!(!stretched);
        assert!((t.duration() - 2.0).abs() < 1e-12);
        let s0 = t.sample(0.0);
        assert!((s0.p[0] - 0.2).abs() < 1e-12 && (s0.v[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn waypoints_pass_without_stopping_unless_the_path_turns() {
        let (vm, am) = lim(1, 1.0, 4.0);
        let ex = box_excess(&vm, &am);
        let start = Knot::at_rest(DVector::from_vec(vec![0.0]));
        let pts = [DVector::from_vec(vec![0.5]), DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.2])];
        let (t, _) = Traj::through(start, &pts, Some(&[1.0, 2.0, 4.0]), None, &vm, &am, &ex);
        assert!((t.duration() - 4.0).abs() < 1e-9, "duration {}", t.duration());
        assert!((t.sample(1.0).p[0] - 0.5).abs() < 1e-12);
        assert!(t.sample(1.0).v[0] > 0.3, "passes the first via moving");
        assert!(t.sample(2.0).v[0].abs() < 1e-12, "stops where the path turns");
        // Asked too fast: stretched, and reported.
        let (fast, stretched) = Traj::through(
            Knot::at_rest(DVector::from_vec(vec![0.0])),
            &pts,
            Some(&[0.1, 0.2, 0.3]),
            None,
            &vm,
            &am,
            &ex,
        );
        assert!(stretched && fast.duration() > 1.0);
        for k in 0..=2000 {
            let s = fast.sample(fast.duration() * k as f64 / 2000.0);
            assert!(ex(&s.v, &s.a) <= 1.0 + 1e-6);
        }
    }
}
