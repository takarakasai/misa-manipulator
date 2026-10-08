#!/usr/bin/env python3
"""Compare control configurations on real teleop sessions.

Each real run log (`logs/hw_lead_*.mlog`) holds the leader targets the arm
got, cycle by cycle. This script

1. scores the real run itself (`manip replay --record`: the measured states),
2. plays the same targets against the rigid plant with sticking friction
   (`--source log --plant rigid`, `[sim] stiction = true`) under each
   configuration, and scores those the same way.

Metrics, over the cycles in the controlled mode:
- err_rms / err_p95 [mm]: TCP vs. the TCP the target asks for (FK of the
  leader's joint target). Includes the teleop lag, the same for every config.
- still_err [mm]: median of that error while the target is still (offset at
  rest; MPC keeps q_margin from the joint limits, ~5 mm at the folded pose).
- still_v [mm/s]: rms TCP speed (over 50 ms) while the target is still
  (swinging at rest).
- chatter [N·m]: rms of the torque minus its 50 ms moving average, summed over
  the arm joints (the 3-8 Hz shaking seen on the real arm).
- holds: falls to Hold.

Usage: scripts/teleop_study.py [--logs a.mlog b.mlog] [--configs ltv ilqr ...] [--real]
"""

import argparse
import concurrent.futures as cf
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIP = ROOT / "target/release/manip"
PROFILE = ROOT / "robots/rebot_b601_dm.toml"
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# name -> (mode, edits): edits are [mpc] keys to set (None removes the section
# override), applied on top of the profile with [sim] stiction = true.
CONFIGS = {
    "joint": ("joint", {}),
    "osc": ("osc", {}),
    "ltv": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[]"}),
    "ltv_ref_ki": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[0.05, 0.5]", "track_ki": "100.0"}),
    "ilqr": ("mpc", {"planner": '"ilqr"', "replan_from_reference": "[]"}),
    "ilqr_ref": ("mpc", {"planner": '"ilqr"', "replan_from_reference": "[0.05, 0.5]"}),
    "ilqr_ref_look": ("mpc", {"planner": '"ilqr"', "replan_from_reference": "[0.05, 0.5]", "target_lookahead_s": "0.15"}),
    "ilqr_ref_ki": ("mpc", {"planner": '"ilqr"', "replan_from_reference": "[0.05, 0.5]", "track_ki": "100.0"}),
    "ltv_ki": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[]", "track_ki": "100.0"}),
    "ltv_ref": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[0.05, 0.5]"}),
    "ltv_look": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[0.05, 0.5]", "track_ki": "100.0", "target_lookahead_s": "0.15"}),
    "ltv_ref_look": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[0.05, 0.5]", "target_lookahead_s": "0.15"}),
    "ltv_look10": ("mpc", {"planner": '"ltv"', "replan_from_reference": "[0.05, 0.5]", "track_ki": "100.0", "target_lookahead_s": "0.10"}),
}


# Plant variants. On the real hw_lead_ltv6 run (which replays bit-exact on
# the current code) the tanh friction sim matched: TCP 10.3 vs 9.9 mm rms,
# lag 80 vs 80 ms, per-joint tracking alike. Sticking friction at the
# identified Coulomb values was far too pessimistic (23.9 mm, 130 ms); at half
# of them it was close (12.3 mm, 90 ms) and is kept as a robustness check.
SIMS = {"tanh": (False, 1.0), "stick": (True, 1.0), "stick_half": (True, 0.5)}


def profile_text(edits: dict, sim: str = "tanh") -> str:
    t = PROFILE.read_text().replace("../models", str((ROOT / "models").resolve()))
    stiction, k = SIMS[sim]
    if stiction:
        t = t.replace("[sim]\n", "[sim]\nstiction = true\n", 1)
    if k != 1.0:
        t = re.sub(r"^sim_friction = ([0-9.]+)", lambda m: f"sim_friction = {float(m.group(1)) * k:.4f}", t, flags=re.M)
    # The configuration replaces the profile's [mpc] keys entirely (so each
    # config is the defaults plus exactly its edits).
    m = re.search(r"^\[mpc\]\n(?:(?!\[)[^\n]*\n)*", t, re.M)
    if m:
        t = t[: m.start()] + t[m.end():]
    body = "".join(f"{k} = {v}\n" for k, v in edits.items())
    t += "\n[mpc]\n" + body
    return t


def load(csv_path: Path):
    with open(csv_path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = [x for x in r]
    cols = {h: i for i, h in enumerate(head)}
    mode = np.array([x[cols["mode"]] for x in rows])

    def col(name):
        return np.array([float(x[cols[name]]) for x in rows])

    return cols, mode, col


def score(csv_path: Path, mode_name: str, dt: float = 0.002) -> dict:
    cols, mode, col = load(csv_path)
    sel = mode == mode_name.capitalize()
    if sel.sum() < 100:
        return {"n": int(sel.sum())}
    tcp = np.stack([col("tcp_x"), col("tcp_y"), col("tcp_z")], 1)
    ref = np.stack([col("goal_x"), col("goal_y"), col("goal_z")], 1)
    ok = sel & np.isfinite(ref).all(1)
    if ok.sum() < 100:
        return {"n": int(sel.sum()), "no_ref": True}
    err = np.linalg.norm(tcp - ref, axis=1)
    # Target still: its TCP moved < 2 mm over the last 0.5 s (a hand-held
    # leader never stops completely; its encoder steps alone are ~0.6 mm at
    # the TCP), and the arm has had 0.3 s to arrive (it lags ~80 ms).
    k = int(0.5 / dt)
    span = np.full(len(ref), np.inf)
    span[k:] = np.linalg.norm(ref[k:] - ref[:-k], axis=1)
    calm = ok & (span < 0.002)
    a = int(0.3 / dt)
    run = np.zeros(len(calm), dtype=int)
    for i in range(1, len(calm)):
        run[i] = run[i - 1] + 1 if calm[i] else 0
    still = calm & (run > a)
    # TCP speed over 50 ms: tick-to-tick differences are dominated by the
    # quantized position feedback (~0.1 mm per step at the TCP).
    w = int(0.05 / dt)
    tv = np.zeros(len(tcp))
    tv[w:] = np.linalg.norm(tcp[w:] - tcp[:-w], axis=1) / (w * dt)
    chat = 0.0
    m = int(0.05 / dt)
    for j in ARM:
        name = f"taum_{j}" if f"taum_{j}" in cols and np.isfinite(col(f"taum_{j}")[sel]).all() else f"tau_{j}"
        x = col(name)
        hp = x - np.convolve(x, np.ones(m) / m, mode="same")
        chat += np.sqrt(np.mean(hp[sel] ** 2))
    holds = int(np.sum((mode[1:] == "Hold") & (mode[:-1] == mode_name.capitalize())))
    return {
        "n": int(sel.sum()),
        "err_rms": 1e3 * np.sqrt(np.mean(err[ok] ** 2)),
        "err_p95": 1e3 * np.percentile(err[ok], 95),
        "still_s": still.sum() * dt,
        "still_err": 1e3 * np.median(err[still]) if still.any() else float("nan"),
        "still_v": 1e3 * np.sqrt(np.mean(tv[still] ** 2)) if still.any() else float("nan"),
        "chatter": chat,
        "holds": holds,
    }


def run_real(mlog: Path, out: Path) -> Path:
    csv_path = out / f"real_{mlog.stem}.csv"
    if not csv_path.exists():
        subprocess.run([str(MANIP), "replay", str(mlog), "--record", str(csv_path)], capture_output=True)
    return csv_path


def run_sim(mlog: Path, cfg: str, out: Path, duration: float, sim: str) -> Path:
    mode, edits = CONFIGS[cfg]
    prof = out / f"p_{sim}_{cfg}.toml"
    if not prof.exists():
        prof.write_text(profile_text(edits, sim))
    csv_path = out / f"sim_{sim}_{mlog.stem}_{cfg}.csv"
    if not csv_path.exists():
        cmd = [str(MANIP), "run", "--robot", str(prof), "--plant", "rigid", "--source", "log", "--leader", str(mlog),
               "--mode", mode, "--fast", "--duration", f"{duration:.2f}", "--record", str(csv_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {mlog.stem} {cfg}: exit {r.returncode}: {r.stderr[-300:]}", file=sys.stderr)
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--real", action="store_true", help="also score the real runs")
    ap.add_argument("--sims", nargs="*", default=["tanh"], choices=list(SIMS))
    a = ap.parse_args()
    logs = [Path(x) for x in a.logs] if a.logs else sorted(ROOT.glob("logs/hw_lead_*.mlog"))
    out = Path(a.out or tempfile.mkdtemp(prefix="teleop_study_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"# work dir {out}", file=sys.stderr)

    # Real runs first (their length sets the sim duration).
    with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        reals = dict(zip(logs, ex.map(lambda m: run_real(m, out), logs)))
    durations = {}
    for m, p in reals.items():
        _, mode, col = load(p)
        durations[m] = len(mode) * 0.002
    jobs = [(m, c, sim) for m in logs for c in a.configs for sim in a.sims]
    with cf.ThreadPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        sims = dict(zip(jobs, ex.map(lambda j: run_sim(j[0], j[1], out, durations[j[0]], j[2]), jobs)))
        scores = dict(zip(jobs, ex.map(lambda j: score(sims[j], CONFIGS[j[1]][0]), jobs)))

    keys = ["err_rms", "err_p95", "still_err", "still_v", "chatter", "holds"]
    fmt = lambda s: " ".join(f"{s.get(k, float('nan')):9.2f}" for k in keys)
    print(f"{'log':16s} {'sim':10s} {'config':12s} " + " ".join(f"{k:>9s}" for k in keys))
    for m in logs:
        if a.real:
            print(f"{m.stem:16s} {'REAL':10s} {'(as run)':12s} " + fmt(score(reals[m], "mpc")))
        for sim in a.sims:
            for c in a.configs:
                print(f"{m.stem:16s} {sim:10s} {c:12s} " + fmt(scores[(m, c, sim)]))
    # Mean over the logs, per sim and config.
    print()
    print(f"{'MEAN':16s} {'sim':10s} {'config':12s} " + " ".join(f"{k:>9s}" for k in keys))
    for sim in a.sims:
        for c in a.configs:
            ss = [scores[(m, c, sim)] for m in logs]
            mean = {k: float(np.nanmean([x.get(k, np.nan) for x in ss])) for k in keys}
            print(f"{'':16s} {sim:10s} {c:12s} " + fmt(mean))


if __name__ == "__main__":
    main()
