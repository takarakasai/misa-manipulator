//! Live 3D view of the arm in MuJoCo's passive viewer.
//!
//! The viewer owns its **own** display-only MuJoCo model (built from the same
//! `.misa`) and only poses it: joint angles come from the control loop through
//! a shared slot, are written to `qpos`, and `mj_forward` places the bodies.
//! No physics runs here. That makes the same view work for every plant (sim,
//! rigid, virtual CAN, real CAN) and keeps rendering out of the control
//! thread: winit wants its event loop on the main thread, so the control loop
//! runs on a worker thread and this runs on the main one.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use articara::mjcf::{GroundPlaneCfg, MjcfExportOptions};
use articara::mujoco_sim::MujocoSim;
use articara::robot::RobotModel;
use mujoco::viewer::MjViewer;

/// Latest joint values, in the order of the `names` given to [`run_viewer`].
pub type PoseSlot = Arc<Mutex<Option<Vec<f64>>>>;

/// Show the model at `misa_path` and follow `slot` until `done` is set.
/// Closing the window stops drawing but keeps waiting for `done`.
pub fn run_viewer(misa_path: &str, names: &[String], slot: PoseSlot, done: &AtomicBool, title: &str) -> Result<(), String> {
    let robot = RobotModel::from_misa(std::path::Path::new(misa_path))?;
    let mut sim = MujocoSim::new(
        &robot,
        MjcfExportOptions {
            base_pos: Some([0.0; 3]),
            base_locked_axes: [true; 6],
            ground_plane: Some(GroundPlaneCfg { z: 0.0, half_size: 0.4, roll: 0.0, pitch: 0.0 }),
            ..MjcfExportOptions::default()
        },
    )?;
    let adr: Vec<Option<usize>> = names.iter().map(|n| sim.joint_dof_adr(n)).collect();
    for (n, a) in names.iter().zip(&adr) {
        if a.is_none() {
            log::warn!("viewer: joint {n} not in the display model");
        }
    }
    let mut viewer = MjViewer::builder()
        .window_name(title.to_string())
        .build_passive(sim.mj_model())
        .map_err(|e| format!("viewer: {e}"))?;
    while !done.load(Ordering::Relaxed) {
        if let Some(q) = slot.lock().unwrap().take() {
            let d = sim.mj_data_mut();
            for (a, v) in adr.iter().zip(q) {
                if let Some(a) = a {
                    d.qpos_mut()[*a] = v;
                }
            }
            d.forward();
        }
        if viewer.running() {
            viewer.sync_data(sim.mj_data_mut());
            viewer.render().map_err(|e| format!("viewer: {e}"))?;
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    Ok(())
}
