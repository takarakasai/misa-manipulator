//! Target shaping: turns a raw target (leader angles, TCP pose) into a
//! reference `(q, v, a)` whose velocity and acceleration stay within limits.
//!
//! # Why it's needed
//!
//! Using the leader angles as-is is dangerous in three ways.
//!
//! - **They jump.** Right after startup, when reception resumes, or when the
//!   leader is re-gripped, the target can shift by tens of degrees at once.
//!   The motor PD chases that at full force.
//! - **They carry no velocity or acceleration.** Inverse-dynamics feedforward
//!   needs `v_ref` and `a_ref`, and differencing the angles gives the second
//!   derivative of noise.
//! - **They are quantized.** The Star Arm's 12-bit encoder has 0.088° steps,
//!   and differentiating produces a spike at every step.
//!
//! The shaper produces a reference that "heads to the target with bounded
//! acceleration and approaches at a speed from which it can stop" (tracking
//! with acceleration and velocity limits). The reference always respects the
//! limits, and when the target stops it stops without overshoot.
//!
//! These limits are separate from SafetyGate's target rate limit. That one is
//! the last line of defense (it clips commands); this one **produces a
//! reference that never needs clipping in the first place**.

use nalgebra::{DVector, Isometry3, Translation3, UnitQuaternion, Vector3, Vector6};

/// Joint-space reference.
#[derive(Debug, Clone, PartialEq)]
pub struct JointRef {
    pub q: DVector<f64>,
    pub v: DVector<f64>,
    pub a: DVector<f64>,
}

impl JointRef {
    /// A reference at rest.
    pub fn at_rest(q: DVector<f64>) -> Self {
        let n = q.len();
        Self {
            q,
            v: DVector::zeros(n),
            a: DVector::zeros(n),
        }
    }
}

/// Per-joint shaping limits.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaperLimits {
    /// Reference velocity limit [rad/s or m/s].
    pub v_max: DVector<f64>,
    /// Reference acceleration limit [rad/s² or m/s²].
    pub a_max: DVector<f64>,
    /// Tracking time constant near the target [s]. Smaller is snappier, larger
    /// is smoother. This is also what smooths out the leader's quantization noise.
    pub time_constant_s: f64,
}

/// Joint-space shaper. Holds state (the current reference).
#[derive(Debug, Clone)]
pub struct JointShaper {
    limits: ShaperLimits,
    r: JointRef,
    /// The reference [`Self::follow`] was given last (cleared by anything else).
    followed: Option<JointRef>,
}

impl JointShaper {
    /// Starts from a reference at rest at `q`. **At startup, always build it
    /// from the measured angles.** Starting from 0 sends every axis toward 0
    /// in the first few cycles.
    pub fn new(limits: ShaperLimits, q: DVector<f64>) -> Self {
        assert_eq!(limits.v_max.len(), q.len());
        assert_eq!(limits.a_max.len(), q.len());
        Self {
            limits,
            r: JointRef::at_rest(q),
            followed: None,
        }
    }

    /// Re-anchors the reference in place (on mode switch or when reception resumes).
    pub fn reset(&mut self, q: DVector<f64>) {
        self.r = JointRef::at_rest(q);
        self.followed = None;
    }

    pub fn current(&self) -> &JointRef {
        &self.r
    }

    /// Replaces the reference including velocity and acceleration (to write
    /// back the result of stepping with temporarily changed limits).
    pub fn restore(&mut self, r: JointRef) {
        assert_eq!(r.q.len(), self.r.q.len());
        self.r = r;
    }

    pub fn limits(&self) -> &ShaperLimits {
        &self.limits
    }

    /// Advances toward `target` by `dt`.
    pub fn step(&mut self, target: &DVector<f64>, dt: f64) -> &JointRef {
        assert!(dt > 0.0);
        self.followed = None;
        for i in 0..target.len() {
            let (q, v, a) = pursue(
                self.r.q[i],
                self.r.v[i],
                target[i],
                self.limits.v_max[i],
                self.limits.a_max[i],
                self.limits.time_constant_s,
                dt,
            );
            self.r.q[i] = q;
            self.r.v[i] = v;
            self.r.a[i] = a;
        }
        &self.r
    }

    /// Follows a reference that already carries velocity and acceleration (a
    /// trajectory or velocity command generated within the limits). Once on
    /// it, the reference is taken as is; a gap (a trajectory that starts away
    /// from the current reference) is closed with this shaper's law in the
    /// frame moving with `target`, using the speed and acceleration it leaves
    /// free, so the result does not jump and stays near the limits.
    pub fn follow(&mut self, target: &JointRef, dt: f64) -> &JointRef {
        assert!(dt > 0.0);
        assert_eq!(target.q.len(), self.r.q.len());
        let prev = self.followed.take().unwrap_or_else(|| target.clone());
        for i in 0..target.q.len() {
            let (dq, dv) = (self.r.q[i] - prev.q[i], self.r.v[i] - prev.v[i]);
            if dq.abs() < 1e-9 && dv.abs() < 1e-9 {
                self.r.q[i] = target.q[i];
                self.r.v[i] = target.v[i];
                self.r.a[i] = target.a[i];
                continue;
            }
            let (v_max, a_max) = (self.limits.v_max[i], self.limits.a_max[i]);
            let v_free = (v_max - target.v[i].abs()).max(0.1 * v_max);
            let a_free = (a_max - target.a[i].abs()).max(0.1 * a_max);
            let (q, v, a) = pursue(dq, dv, 0.0, v_free, a_free, self.limits.time_constant_s, dt);
            self.r.q[i] = target.q[i] + q;
            self.r.v[i] = target.v[i] + v;
            self.r.a[i] = target.a[i] + a;
        }
        self.followed = Some(target.clone());
        &self.r
    }
}

/// 1-D tracking law. One step from `(q, v)` toward target `x` with
/// `|v| ≤ v_max` and `|a| ≤ a_max`. Returns the new `(q, v, a)`.
///
/// The desired speed is the minimum of the "stoppable speed"
/// `√(2·a_max·|e|)`, the "time-constant closing speed" `|e|/T`, and `v_max`.
/// With the former alone, the slope of √ becomes infinite near the target and
/// chatters, so the latter brings it down linearly.
fn pursue(q: f64, v: f64, x: f64, v_max: f64, a_max: f64, t: f64, dt: f64) -> (f64, f64, f64) {
    let e = x - q;
    let speed = (2.0 * a_max * e.abs()).sqrt().min(e.abs() / t.max(dt)).min(v_max);
    let v_des = speed.copysign(e);
    let a = ((v_des - v) / dt).clamp(-a_max, a_max);
    let v_new = v + a * dt;
    (q + 0.5 * (v + v_new) * dt, v_new, a)
}

/// TCP reference: pose, spatial velocity `[ω; v]`, spatial acceleration `[α; a]` (world coordinates).
#[derive(Debug, Clone, PartialEq)]
pub struct TcpRefState {
    pub pose: Isometry3<f64>,
    pub twist: Vector6<f64>,
    pub accel: Vector6<f64>,
}

/// TCP shaping limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcpShaperLimits {
    pub lin_v_max: f64,
    pub lin_a_max: f64,
    pub ang_v_max: f64,
    pub ang_a_max: f64,
    pub time_constant_s: f64,
}

/// TCP shaper. Applies the 1-D tracking law along the vector direction for
/// translation and about the rotation axis for rotation (applying it per axis
/// would turn diagonal motion into a polyline).
#[derive(Debug, Clone)]
pub struct TcpShaper {
    limits: TcpShaperLimits,
    r: TcpRefState,
    /// The reference [`Self::follow`] was given last (cleared by anything else).
    followed: Option<TcpRefState>,
}

impl TcpShaper {
    pub fn new(limits: TcpShaperLimits, pose: Isometry3<f64>) -> Self {
        Self {
            limits,
            r: TcpRefState {
                pose,
                twist: Vector6::zeros(),
                accel: Vector6::zeros(),
            },
            followed: None,
        }
    }

    pub fn reset(&mut self, pose: Isometry3<f64>) {
        self.r = TcpRefState {
            pose,
            twist: Vector6::zeros(),
            accel: Vector6::zeros(),
        };
        self.followed = None;
    }

    pub fn current(&self) -> &TcpRefState {
        &self.r
    }

    pub fn step(&mut self, target: &Isometry3<f64>, dt: f64) -> &TcpRefState {
        self.followed = None;
        let l = self.limits;
        // Translation: 1-D tracking along the error vector; velocity orthogonal to it is damped out.
        let p = self.r.pose.translation.vector;
        let v = self.r.twist.fixed_rows::<3>(3).into_owned();
        let (p_new, v_new, a_new) = pursue_vec(p, v, target.translation.vector, l.lin_v_max, l.lin_a_max, l.time_constant_s, dt);

        // Rotation: error rotation vector in world coordinates.
        let rot = self.r.pose.rotation;
        let w = self.r.twist.fixed_rows::<3>(0).into_owned();
        let err = (target.rotation * rot.inverse()).scaled_axis();
        // For rotation, treat the error vector as a position and track toward the origin 0.
        let (_, w_new, alpha) = pursue_vec(-err, w, Vector3::zeros(), l.ang_v_max, l.ang_a_max, l.time_constant_s, dt);
        let dq = UnitQuaternion::from_scaled_axis(0.5 * (w + w_new) * dt);

        self.r.pose = Isometry3::from_parts(Translation3::from(p_new), dq * rot);
        self.r.twist = stack6(&w_new, &v_new);
        self.r.accel = stack6(&alpha, &a_new);
        &self.r
    }

    /// [`JointShaper::follow`] for the TCP: a reference with twist and
    /// acceleration is taken as is once reached; a gap is closed in the frame
    /// moving with `target` (translation along the gap vector, rotation about
    /// the gap axis).
    pub fn follow(&mut self, target: &TcpRefState, dt: f64) -> &TcpRefState {
        let l = self.limits;
        let prev = self.followed.take().unwrap_or_else(|| target.clone());
        let lin = |x: &Vector6<f64>| x.fixed_rows::<3>(3).into_owned();
        let ang = |x: &Vector6<f64>| x.fixed_rows::<3>(0).into_owned();
        let dp = self.r.pose.translation.vector - prev.pose.translation.vector;
        let dv = lin(&self.r.twist) - lin(&prev.twist);
        let dr = (self.r.pose.rotation * prev.pose.rotation.inverse()).scaled_axis();
        let dw = ang(&self.r.twist) - ang(&prev.twist);
        if dp.norm() < 1e-9 && dv.norm() < 1e-9 && dr.norm() < 1e-9 && dw.norm() < 1e-9 {
            self.r = target.clone();
        } else {
            let free = |max: f64, used: f64| (max - used).max(0.1 * max);
            let (p, v, a) = pursue_vec(
                dp,
                dv,
                Vector3::zeros(),
                free(l.lin_v_max, lin(&target.twist).norm()),
                free(l.lin_a_max, lin(&target.accel).norm()),
                l.time_constant_s,
                dt,
            );
            let (r, w, alpha) = pursue_vec(
                dr,
                dw,
                Vector3::zeros(),
                free(l.ang_v_max, ang(&target.twist).norm()),
                free(l.ang_a_max, ang(&target.accel).norm()),
                l.time_constant_s,
                dt,
            );
            self.r = TcpRefState {
                pose: Isometry3::from_parts(
                    Translation3::from(target.pose.translation.vector + p),
                    UnitQuaternion::from_scaled_axis(r) * target.pose.rotation,
                ),
                twist: target.twist + stack6(&w, &v),
                accel: target.accel + stack6(&alpha, &a),
            };
        }
        self.followed = Some(target.clone());
        &self.r
    }
}

fn pursue_vec(
    p: Vector3<f64>,
    v: Vector3<f64>,
    x: Vector3<f64>,
    v_max: f64,
    a_max: f64,
    t: f64,
    dt: f64,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let e = x - p;
    let dist = e.norm();
    let v_des = if dist > 1e-12 {
        let speed = (2.0 * a_max * dist).sqrt().min(dist / t.max(dt)).min(v_max);
        e * (speed / dist)
    } else {
        Vector3::zeros()
    };
    let mut a = (v_des - v) / dt;
    let an = a.norm();
    if an > a_max {
        a *= a_max / an;
    }
    let v_new = v + a * dt;
    (p + 0.5 * (v + v_new) * dt, v_new, a)
}

pub(crate) fn stack6(ang: &Vector3<f64>, lin: &Vector3<f64>) -> Vector6<f64> {
    Vector6::new(ang.x, ang.y, ang.z, lin.x, lin.y, lin.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(n: usize) -> ShaperLimits {
        ShaperLimits {
            v_max: DVector::from_element(n, 1.0),
            a_max: DVector::from_element(n, 4.0),
            time_constant_s: 0.05,
        }
    }

    /// Even toward a jumped target, it respects the limits and stops without overshoot.
    #[test]
    fn joint_step_respects_limits_and_settles() {
        let mut s = JointShaper::new(limits(2), DVector::from_vec(vec![0.0, 0.0]));
        let target = DVector::from_vec(vec![1.0, -0.3]);
        let dt = 0.002;
        let mut max_q = f64::MIN;
        for _ in 0..3000 {
            let r = s.step(&target, dt).clone();
            for i in 0..2 {
                assert!(r.v[i].abs() <= 1.0 + 1e-9);
                assert!(r.a[i].abs() <= 4.0 + 1e-9);
            }
            max_q = max_q.max(r.q[0]);
        }
        let r = s.current();
        assert!((r.q[0] - 1.0).abs() < 1e-4 && (r.q[1] + 0.3).abs() < 1e-4);
        assert!(max_q < 1.0 + 2e-3, "overshoot {max_q}");
        assert!(r.v.norm() < 1e-3);
    }

    /// On a trajectory, `follow` returns it exactly; started away from it, it
    /// closes the gap within the limits and then rides on it.
    #[test]
    fn joint_follow_closes_the_gap_then_rides_the_trajectory() {
        let dt = 0.002;
        let traj = |t: f64| JointRef {
            q: DVector::from_vec(vec![0.5 * t, -0.2]),
            v: DVector::from_vec(vec![0.5, 0.0]),
            a: DVector::zeros(2),
        };
        let mut s = JointShaper::new(limits(2), DVector::from_vec(vec![0.0, 0.0]));
        let mut prev_q = s.current().q.clone();
        for k in 1..=1000 {
            let r = s.follow(&traj(k as f64 * dt), dt).clone();
            for i in 0..2 {
                assert!(r.v[i].abs() <= 1.0 + 1e-9, "v {}", r.v[i]);
                assert!(r.a[i].abs() <= 4.0 + 1e-9, "a {}", r.a[i]);
                assert!((r.q[i] - prev_q[i]).abs() <= 1.0 * dt + 1e-9);
            }
            prev_q = r.q.clone();
        }
        let r = s.current();
        assert_eq!(r, &traj(1000.0 * dt), "rides the trajectory exactly once on it");
    }

    #[test]
    fn tcp_follow_closes_the_gap_then_rides_the_trajectory() {
        let l = TcpShaperLimits {
            lin_v_max: 0.3,
            lin_a_max: 2.0,
            ang_v_max: 1.5,
            ang_a_max: 8.0,
            time_constant_s: 0.05,
        };
        let dt = 0.002;
        let traj = |t: f64| TcpRefState {
            pose: Isometry3::from_parts(
                Translation3::new(0.1 * t, 0.2, 0.3),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.2 * t),
            ),
            twist: Vector6::new(0.0, 0.0, 0.2, 0.1, 0.0, 0.0),
            accel: Vector6::zeros(),
        };
        let mut s = TcpShaper::new(l, Isometry3::translation(0.0, 0.15, 0.3));
        for k in 1..=1500 {
            let r = s.follow(&traj(k as f64 * dt), dt).clone();
            assert!(r.twist.fixed_rows::<3>(3).norm() <= 0.3 + 1e-6);
            assert!(r.twist.fixed_rows::<3>(0).norm() <= 1.5 + 1e-6);
        }
        let (r, want) = (s.current(), traj(1500.0 * dt));
        assert!((r.pose.translation.vector - want.pose.translation.vector).norm() < 1e-6);
        assert!(r.pose.rotation.angle_to(&want.pose.rotation) < 1e-6);
        assert!((r.twist - want.twist).norm() < 1e-6);
    }

    #[test]
    fn tcp_step_reaches_target() {
        let l = TcpShaperLimits {
            lin_v_max: 0.3,
            lin_a_max: 2.0,
            ang_v_max: 1.5,
            ang_a_max: 8.0,
            time_constant_s: 0.05,
        };
        let mut s = TcpShaper::new(l, Isometry3::identity());
        let target = Isometry3::from_parts(
            Translation3::new(0.1, -0.05, 0.2),
            UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5),
        );
        for _ in 0..3000 {
            let r = s.step(&target, 0.002).clone();
            assert!(r.twist.fixed_rows::<3>(3).norm() <= 0.3 + 1e-9);
            assert!(r.twist.fixed_rows::<3>(0).norm() <= 1.5 + 1e-9);
        }
        let r = s.current();
        assert!((r.pose.translation.vector - target.translation.vector).norm() < 1e-4);
        assert!(r.pose.rotation.angle_to(&target.rotation) < 1e-3);
    }
}
