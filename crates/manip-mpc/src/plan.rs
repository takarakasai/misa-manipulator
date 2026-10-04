//! A planned joint trajectory: knots every `dt` with piecewise-constant
//! acceleration, sampled exactly (quadratic position) in between.

use manip_control::JointRef;
use nalgebra::DVector;

/// Joint trajectory of the TCP-chain DOFs over a horizon.
///
/// Knot `k` is at `t0 + k·dt` (`k = 0..=N`); the acceleration `a[k]` is held
/// from knot `k` to `k + 1` (`a[N] = 0`).
#[derive(Debug, Clone)]
pub struct JointPlan {
    /// Index of each planned DOF among all independent DOFs.
    pub idx: Vec<usize>,
    /// Time of knot 0 (the caller's clock).
    pub t0: f64,
    pub dt: f64,
    pub q: Vec<DVector<f64>>,
    pub v: Vec<DVector<f64>>,
    pub a: Vec<DVector<f64>>,
}

impl JointPlan {
    /// Horizon length `N` (number of intervals).
    pub fn horizon(&self) -> usize {
        self.q.len() - 1
    }

    /// End of the plan.
    pub fn t_end(&self) -> f64 {
        self.t0 + self.horizon() as f64 * self.dt
    }

    /// Reference of the planned DOFs at time `t` (held at the ends).
    pub fn sample(&self, t: f64) -> JointRef {
        let n_int = self.horizon();
        let s = ((t - self.t0) / self.dt).max(0.0);
        if s >= n_int as f64 {
            let q = self.q[n_int].clone();
            let n = q.len();
            return JointRef {
                q,
                v: self.v[n_int].clone(),
                a: DVector::zeros(n),
            };
        }
        let k = (s.floor() as usize).min(n_int - 1);
        let tau = (t - self.t0 - k as f64 * self.dt).max(0.0);
        let (q, v, a) = (&self.q[k], &self.v[k], &self.a[k]);
        JointRef {
            q: q + v * tau + a * (0.5 * tau * tau),
            v: v + a * tau,
            a: a.clone(),
        }
    }

    /// [`Self::sample`] for all independent DOFs: the planned DOFs from the
    /// plan, the others from `base`.
    pub fn sample_full(&self, t: f64, base: &JointRef) -> JointRef {
        let s = self.sample(t);
        let mut r = base.clone();
        for (k, &i) in self.idx.iter().enumerate() {
            r.q[i] = s.q[k];
            r.v[i] = s.v[k];
            r.a[i] = s.a[k];
        }
        r
    }
}
