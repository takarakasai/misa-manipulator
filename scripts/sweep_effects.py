#!/usr/bin/env python3
"""Sweep hardware effects (latency, jitter, quantization) in the MuJoCo sim.

For each robot x scenario x effect setting, runs `manip run --fast --record`
and reports tracking error and a chatter metric:

- joint scenario: rms |q - qref| over the six arm joints [mrad].
- osc scenario:   rms TCP position error [mm].
- chatter:        rms of the tick-to-tick change of commanded torque, summed
                  over the arm joints [N*m per tick]. A value that rises with
                  latency is the first sign of a loop losing phase margin.
- fallback:       ticks where the OSC could not solve and dropped to Hold.

Usage (from the repo root, after
`cargo build --release -p manip-runner --features sim`):

    scripts/sweep_effects.py [--robots dm,rs] [--delays 0,1,2,3,4]
"""

import argparse
import csv
import math
import os
import subprocess
import sys

MANIP = "./target/release/manip"


def run(robot, source, mode, extra, out):
    cmd = [MANIP, "run", "--robot", f"robots/rebot_b601_{robot}.toml", "--plant", "sim",
           "--source", source, "--mode", mode, "--start-pose", "ready",
           "--duration", "12", "--fast", "--record", out] + extra
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "warn")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


def metrics(path, mode_name):
    rows = list(csv.DictReader(open(path)))
    fallbacks = sum(1 for a, b in zip(rows, rows[1:]) if a["mode"] == "Osc" and b["mode"] == "Hold")
    active = [r for r in rows if r["mode"] == mode_name]
    if len(active) < 100:
        return None, None, fallbacks
    active = active[len(active) // 5:]  # skip the entry transient
    joints = [k[2:] for k in rows[0] if k.startswith("q_")][:6]
    if mode_name == "Osc":
        e = [math.dist([float(r["tcp_x"]), float(r["tcp_y"]), float(r["tcp_z"])],
                       [float(r["ref_x"]), float(r["ref_y"]), float(r["ref_z"])]) for r in active]
    else:
        e = [float(r["q_" + j]) - float(r["qref_" + j]) for r in active for j in joints]
    err = 1e3 * math.sqrt(sum(x * x for x in e) / len(e))
    d = [sum((float(b["tau_" + j]) - float(a["tau_" + j])) ** 2 for j in joints)
         for a, b in zip(active, active[1:])]
    chatter = math.sqrt(sum(d) / len(d))
    return err, chatter, fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", default="dm,rs")
    ap.add_argument("--delays", default="0,1,2,3,4")
    ap.add_argument("--outdir", default="logs/sweep")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    delays = [int(x) for x in a.delays.split(",")]
    settings = [("ideal", ["--ideal"])]
    settings += [(f"delay{d}", ["--delay-ticks", str(d), "--jitter", "0"]) for d in delays]
    settings += [(f"delay{d}+jit10", ["--delay-ticks", str(d), "--jitter", "0.1"]) for d in delays]
    print(f"{'robot':5} {'scenario':8} {'setting':14} {'error':>10} {'chatter':>9} {'fallback':>8}")
    for robot in a.robots.split(","):
        for scen, source, mode, mode_name, unit in [
            ("joint", "sine", "joint", "Joint", "mrad"),
            ("osc", "circle", "osc", "Osc", "mm"),
        ]:
            for name, extra in settings:
                out = f"{a.outdir}/{robot}_{scen}_{name}.csv"
                rc, err_txt = run(robot, source, mode, extra, out)
                if rc != 0:
                    tail = err_txt.strip().splitlines()[-1:] or [""]
                    print(f"{robot:5} {scen:8} {name:14} FAILED: {tail[0]}")
                    continue
                err, chatter, fb = metrics(out, mode_name)
                if err is None:
                    print(f"{robot:5} {scen:8} {name:14} {'(left mode)':>14} {fb:>8}")
                else:
                    print(f"{robot:5} {scen:8} {name:14} {err:7.2f}{unit:>4} {chatter:9.3f} {fb:>8}")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
