//! Imports a URDF into `.misa`, decimating meshes to produce a self-contained model.
//!
//! ```sh
//! manip-import <in.urdf> <out_dir> --name rebot_b601_dm \
//!     [--visual-tris 6000] [--collision-tris 800] [--finger-travel 0.0285]
//! ```
//!
//! Output is `<out_dir>/<name>.misa` and `<out_dir>/meshes/{visual,collision}/*.stl`.
//! Mesh references are rewritten as paths relative to the `.misa`.
//!
//! # Why decimate
//!
//! The vendor URDF meshes are 6 MB per part (120k faces), over 100 MB for two
//! models. That can't go into the repository, and MuJoCo turns collision
//! meshes into convex hulls anyway, so fine detail is meaningless. The quality
//! bar is "looks like the same part" for visual meshes and "the convex hull
//! doesn't change" for collision meshes.
//!
//! # What it does not do
//!
//! Mass, inertia and joint definitions are **never changed**. They determine
//! how good the control law is, so if import silently changed them the root
//! cause would be untraceable. The exception is `--finger-travel` (finger
//! range of motion only): its value differs between upstream URDF versions
//! (0.05 / 0.0285), so the user must choose it explicitly.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use misarta::mesh::MeshData;
use misarta::native::schema::{Geom, MisaFile};
use nalgebra as na;

#[derive(Parser)]
#[command(about = "Import a URDF into .misa (with mesh decimation)")]
struct Args {
    urdf: PathBuf,
    out_dir: PathBuf,
    /// Output file name (without extension) and robot name.
    #[arg(long)]
    name: String,
    /// Target triangle count per visual mesh.
    #[arg(long, default_value_t = 6000)]
    visual_tris: usize,
    /// Target triangle count per collision mesh.
    #[arg(long, default_value_t = 800)]
    collision_tris: usize,
    /// Travel [m] of prismatic fingers (names containing `finger`). The original sign is kept.
    #[arg(long)]
    finger_travel: Option<f64>,
    /// Adds a mimic relation: `slave=master[:multiplier]` (repeatable).
    ///
    /// Use when one motor drives both fingers but the URDF describes them as
    /// two independent prismatic joints (as the reBot B601-RS URDF does). The
    /// follower's range of motion is rewritten to the leader's times
    /// `multiplier`.
    #[arg(long = "mimic")]
    mimics: Vec<String>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let urdf_dir = args.urdf.parent().unwrap_or(Path::new(".")).to_path_buf();
    let import = misarta_formats::urdf::import(&args.urdf)?;
    for w in &import.warnings {
        eprintln!("warning: {w}");
    }
    let mut file = import.file;
    file.robot.name = args.name.clone();

    std::fs::create_dir_all(args.out_dir.join("meshes/visual")).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(args.out_dir.join("meshes/collision")).map_err(|e| e.to_string())?;

    // Convert each mesh only once even if several links reference it.
    let mut done: BTreeMap<(String, &'static str), String> = BTreeMap::new();
    let (mut bytes_in, mut bytes_out) = (0u64, 0u64);

    let mut convert = |src: &str, kind: &'static str, tris: usize| -> Result<String, String> {
        if let Some(rel) = done.get(&(src.to_string(), kind)) {
            return Ok(rel.clone());
        }
        let src_path = resolve(&urdf_dir, src);
        bytes_in += std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
        let mesh = load_mesh(&src_path)?;
        let n0 = mesh.indices.len();
        let ratio = (tris as f64 / n0.max(1) as f64).min(1.0);
        let out = if ratio < 1.0 {
            misarta::decimate::decimate(&mesh, ratio)
        } else {
            mesh
        };
        let stem = src_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("cannot read mesh name: {src}"))?
            .to_lowercase();
        let rel = format!("meshes/{kind}/{stem}.stl");
        let dst = args.out_dir.join(&rel);
        write_binary_stl(&dst, &out).map_err(|e| format!("{}: {e}", dst.display()))?;
        bytes_out += std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0);
        eprintln!("  {kind:9} {stem:48} {n0:>7} → {:>6} tris", out.indices.len());
        done.insert((src.to_string(), kind), rel.clone());
        Ok(rel)
    };

    for link in &mut file.link {
        for v in &mut link.visual {
            if let Geom::Mesh { file: f, .. } = &mut v.geom {
                *f = convert(f, "visual", args.visual_tris)?;
            }
        }
        for c in &mut link.collision {
            if let Geom::Mesh { file: f, .. } = &mut c.geom {
                *f = convert(f, "collision", args.collision_tris)?;
            }
        }
    }

    if let Some(travel) = args.finger_travel {
        set_finger_travel(&mut file, travel);
    }
    for m in &args.mimics {
        add_mimic(&mut file, m)?;
    }
    give_massless_links_mass(&mut file);

    let out = args.out_dir.join(format!("{}.misa", args.name));
    misarta::native::save(&out, &file).map_err(|e| e.to_string())?;
    // Read it back to confirm it builds through the same path the control side uses.
    let reloaded = misarta::native::load(&out).map_err(|e| e.to_string())?;
    let (model, _, _) = misarta::native::build_model(&reloaded.file).map_err(|e| e.to_string())?;
    eprintln!(
        "wrote {} (nq={}, {} links); meshes {:.1} MB → {:.1} MB",
        out.display(),
        model.nq,
        model.link_names.len(),
        bytes_in as f64 / 1e6,
        bytes_out as f64 / 1e6
    );
    Ok(())
}

/// Resolves a URDF `filename` to a real file. `package://` can't be located
/// from the URDF's location, so the package name is dropped and the path is
/// resolved relative to the URDF's parent directory.
fn resolve(urdf_dir: &Path, name: &str) -> PathBuf {
    if let Some(rest) = name.strip_prefix("package://") {
        let rest = rest.split_once('/').map(|(_, r)| r).unwrap_or(rest);
        return urdf_dir.join("..").join(rest);
    }
    let name = name.strip_prefix("file://").unwrap_or(name);
    let p = Path::new(name);
    if p.is_absolute() { p.to_path_buf() } else { urdf_dir.join(p) }
}

fn load_mesh(path: &Path) -> Result<MeshData, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("stl") => MeshData::from_stl(path),
        Some("obj") => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            MeshData::from_obj_bytes(&bytes)
        }
        _ => Err(format!("unsupported mesh format: {}", path.display())),
    }
}

fn set_finger_travel(file: &mut MisaFile, travel: f64) {
    for j in &mut file.joint {
        if !j.name.contains("finger") {
            continue;
        }
        let l = &mut j.limit;
        if l.upper > 0.0 && l.lower >= 0.0 {
            l.upper = travel;
        } else if l.lower < 0.0 && l.upper <= 0.0 {
            l.lower = -travel;
        }
        eprintln!("  finger {} → [{}, {}]", j.name, l.lower, l.upper);
    }
}

/// Gives massless links beyond a moving joint an explicit tiny mass (1 g).
///
/// Without `<inertial>` in the MJCF, MuJoCo derives mass from geometry volume
/// × water density. The control side (misarta) keeps it at 0, so **only the
/// sim gets heavier** (0.18 kg on the B601-DM fingers, shifting shoulder
/// gravity torque by 15 %). Set it explicitly here so both see the same value.
/// Overwrite it once the real mass is known.
fn give_massless_links_mass(file: &mut MisaFile) {
    use misarta::native::schema::JointKind;
    let moving: Vec<String> = file
        .joint
        .iter()
        .filter(|j| j.kind != JointKind::Fixed)
        .map(|j| j.child.clone())
        .collect();
    for link in &mut file.link {
        if link.inertial.mass <= 0.0 && moving.contains(&link.name) {
            eprintln!("  warning: link {} has zero mass; setting 1 g placeholder", link.name);
            link.inertial.mass = 1e-3;
            link.inertial.ixx = 1e-7;
            link.inertial.iyy = 1e-7;
            link.inertial.izz = 1e-7;
        }
    }
}

fn add_mimic(file: &mut MisaFile, spec: &str) -> Result<(), String> {
    let (slave, rest) = spec
        .split_once('=')
        .ok_or_else(|| format!("--mimic must be of the form slave=master[:mult]: {spec}"))?;
    let (master, mult) = match rest.split_once(':') {
        Some((m, k)) => (m, k.parse::<f64>().map_err(|e| format!("{spec}: {e}"))?),
        None => (rest, 1.0),
    };
    let master_limit = file
        .joint
        .iter()
        .find(|j| j.name == master)
        .ok_or_else(|| format!("joint {master} not found"))?
        .limit
        .clone();
    let sj = file
        .joint
        .iter_mut()
        .find(|j| j.name == slave)
        .ok_or_else(|| format!("joint {slave} not found"))?;
    let (a, b) = (master_limit.lower * mult, master_limit.upper * mult);
    sj.limit.lower = a.min(b);
    sj.limit.upper = a.max(b);
    file.mimic.retain(|m| m.joint != slave);
    file.mimic.push(misarta::native::schema::Mimic {
        joint: slave.to_string(),
        source: master.to_string(),
        multiplier: mult,
        offset: 0.0,
    });
    eprintln!("  mimic {slave} = {mult} * {master}  → [{}, {}]", sj.limit.lower, sj.limit.upper);
    Ok(())
}

/// Binary STL (same format as articara's decimate_stl). Normals are recomputed
/// from the decimated vertices; reusing old normals breaks the shading.
fn write_binary_stl(path: &Path, m: &MeshData) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = [0u8; 80];
    let tag = b"binary STL, decimated by misa-manipulator/manip-import";
    header[..tag.len()].copy_from_slice(tag);
    f.write_all(&header)?;
    f.write_all(&(m.indices.len() as u32).to_le_bytes())?;
    for tri in &m.indices {
        let p: Vec<_> = tri.iter().map(|&i| m.vertices[i as usize]).collect();
        let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
        let n = if n.norm() > 1e-20 { n / n.norm() } else { na::Vector3::zeros() };
        for v in [n.x, n.y, n.z] {
            f.write_all(&(v as f32).to_le_bytes())?;
        }
        for q in &p {
            for v in [q.x, q.y, q.z] {
                f.write_all(&(v as f32).to_le_bytes())?;
            }
        }
        f.write_all(&0u16.to_le_bytes())?;
    }
    f.flush()
}
