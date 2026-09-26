//! CSV recording. One row = one control cycle.
//!
//! The minimum needed to compare tracking quality afterwards: measured, reference,
//! commanded torque, TCP. Column names include joint names, so the columns stay
//! readable when the robot changes.

use std::io::Write;

use manip_control::JointCommand;
use manip_model::{ArmModel, ArmState};

use crate::supervisor::{Mode, TickInfo};

pub struct Recorder {
    w: std::io::BufWriter<std::fs::File>,
}

impl Recorder {
    pub fn create(path: &std::path::Path, arm: &ArmModel) -> std::io::Result<Self> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut cols = vec!["t".to_string(), "mode".into(), "tick_us".into()];
        for d in arm.dofs() {
            for k in ["q", "v", "qref", "vref", "tau"] {
                cols.push(format!("{k}_{}", d.name));
            }
        }
        for k in ["tcp_x", "tcp_y", "tcp_z", "ref_x", "ref_y", "ref_z", "sigma_min"] {
            cols.push(k.into());
        }
        writeln!(w, "{}", cols.join(","))?;
        Ok(Self { w })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn row(
        &mut self,
        t: f64,
        mode: Mode,
        tick_us: f64,
        s: &ArmState,
        cmd: &JointCommand,
        info: &TickInfo,
    ) -> std::io::Result<()> {
        let mut out = format!("{t:.4},{mode:?},{tick_us:.0}");
        let tau = cmd.torque_at(&s.q, &s.v);
        for i in 0..s.q.len() {
            let (qr, vr) = info
                .reference
                .as_ref()
                .map(|r| (r.q[i], r.v[i]))
                .unwrap_or((f64::NAN, f64::NAN));
            out += &format!(",{:.6},{:.5},{:.6},{:.5},{:.4}", s.q[i], s.v[i], qr, vr, tau[i]);
        }
        let p = s.tcp_pose.translation.vector;
        let r = info
            .tcp_reference
            .map(|x| x.translation.vector)
            .unwrap_or(nalgebra::Vector3::from_element(f64::NAN));
        let sigma = info.osc.as_ref().map(|o| o.sigma_min).unwrap_or(f64::NAN);
        out += &format!(",{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.4}", p.x, p.y, p.z, r.x, r.y, r.z, sigma);
        writeln!(self.w, "{out}")
    }
}
