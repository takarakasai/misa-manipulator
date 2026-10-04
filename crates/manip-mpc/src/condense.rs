//! Condensing the double integrator over the horizon.
//!
//! With the stage input `u_k = q̈` held over `h`:
//!
//! ```text
//! v_{k+1} = v_k + h·u_k
//! q_{k+1} = q_k + h·v_k + ½h²·u_k
//! ```
//!
//! so every knot is affine in the stacked inputs `U = [u_0; …; u_{N−1}]`:
//!
//! ```text
//! v_k = v_0 + h·Σ_{j<k} u_j
//! q_k = q_0 + k·h·v_0 + h²·Σ_{j<k} (k − j − ½)·u_j
//! ```

use nalgebra::{DMatrix, DVector};

/// `q_k = q_free[k] + Gq[k]·U`, `v_k = v_free[k] + Gv[k]·U` for `k = 0..=N`.
#[derive(Debug, Clone)]
pub struct Condensed {
    pub n: usize,
    pub horizon: usize,
    pub h: f64,
    pub q_free: Vec<DVector<f64>>,
    pub v_free: Vec<DVector<f64>>,
    /// `n × N·n` each.
    pub gq: Vec<DMatrix<f64>>,
    pub gv: Vec<DMatrix<f64>>,
}

impl Condensed {
    pub fn new(q0: &DVector<f64>, v0: &DVector<f64>, horizon: usize, h: f64) -> Self {
        let n = q0.len();
        let nu = horizon * n;
        let mut q_free = Vec::with_capacity(horizon + 1);
        let mut v_free = Vec::with_capacity(horizon + 1);
        let mut gq = Vec::with_capacity(horizon + 1);
        let mut gv = Vec::with_capacity(horizon + 1);
        for k in 0..=horizon {
            q_free.push(q0 + v0 * (k as f64 * h));
            v_free.push(v0.clone());
            let mut mq = DMatrix::zeros(n, nu);
            let mut mv = DMatrix::zeros(n, nu);
            for j in 0..k {
                let cq = h * h * (k as f64 - j as f64 - 0.5);
                for i in 0..n {
                    mq[(i, j * n + i)] = cq;
                    mv[(i, j * n + i)] = h;
                }
            }
            gq.push(mq);
            gv.push(mv);
        }
        Self {
            n,
            horizon,
            h,
            q_free,
            v_free,
            gq,
            gv,
        }
    }

    pub fn n_decision(&self) -> usize {
        self.horizon * self.n
    }

    /// `u_k = Gu[k]·U` (a selection).
    pub fn gu(&self, k: usize) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.n, self.n_decision());
        for i in 0..self.n {
            m[(i, k * self.n + i)] = 1.0;
        }
        m
    }

    /// Knot states for a given `U`.
    pub fn rollout(&self, u: &DVector<f64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let q = (0..=self.horizon).map(|k| &self.q_free[k] + &self.gq[k] * u).collect();
        let v = (0..=self.horizon).map(|k| &self.v_free[k] + &self.gv[k] * u).collect();
        (q, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The condensed knots equal stepping the double integrator.
    #[test]
    fn matches_forward_simulation() {
        let (n, horizon, h) = (3, 7, 0.04);
        let q0 = DVector::from_vec(vec![0.1, -0.2, 0.3]);
        let v0 = DVector::from_vec(vec![0.5, 0.0, -0.4]);
        let u = DVector::from_iterator(horizon * n, (0..horizon * n).map(|i| ((i * 37 % 11) as f64 - 5.0) * 0.7));
        let c = Condensed::new(&q0, &v0, horizon, h);
        let (qs, vs) = c.rollout(&u);
        let (mut q, mut v) = (q0.clone(), v0.clone());
        for k in 0..horizon {
            assert!((&qs[k] - &q).amax() < 1e-12 && (&vs[k] - &v).amax() < 1e-12, "knot {k}");
            let uk = u.rows(k * n, n).into_owned();
            q += &v * h + &uk * (0.5 * h * h);
            v += &uk * h;
        }
        assert!((&qs[horizon] - &q).amax() < 1e-12 && (&vs[horizon] - &v).amax() < 1e-12);
    }
}
