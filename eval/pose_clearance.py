"""Is the map itself in the way of a start pose? Scans the bare map (no obstacles) with true depth, HPC only.

    python -m eval.pose_clearance --config configs/sweeps/airsim_replay_gates.yaml --out results/pose_clearance

The start poses were scanned at one altitude, but the controller climbs. For every pose in the config, this puts the
drone at each altitude and distance along the path, reads the corridor window of the true depth exactly as the flight
loop's scenery guard does, and repeats the whole scan (``--passes``). A point below the drone radius in every pass is
map scenery; a point below it in one pass only is a bad capture. Ground truth is fine here: this checks the simulator,
no policy runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from perception.depth_utils import corridor_window
from sim.drone import DroneParams, KinematicState

ALTS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)


def probe(sim: Any, x: float, alt: float, y: float = 0.0) -> dict[str, float]:
    obs = sim.observe(KinematicState(t=0.0, x=x, y=y, z=-alt, yaw=0.0, speed=0.0))
    w = obs.depth[corridor_window(obs.depth.shape)]
    return {"depth_min": round(float(w.min()), 3), "depth_median": round(float(np.median(w)), 2), "frac_zero": round(float(np.mean(w <= 0.0)), 4)}


def scan(sim: Any, starts: Sequence[Sequence[float]], xs: Sequence[float], alts: Sequence[float], passes: int, sink: Any,
         ys: Sequence[float] = (0.0,)) -> list[dict[str, Any]]:
    rows = []
    for n in range(passes):
        for i, start in enumerate(starts):
            sim.empty_map(start)
            for alt in alts:
                for y in ys:
                    for x in xs:
                        row = {"pass": n, "pose": i, "alt": float(alt), "x": float(x), "y": float(y), **probe(sim, float(x), float(alt), float(y))}
                        sink.write(json.dumps(row) + "\n")
                        rows.append(row)
            sink.flush()
            print(f"pass {n} pose {i} done", flush=True)
    return rows


def report(rows: Sequence[dict[str, Any]], radius: float) -> list[str]:
    points: dict[tuple[int, float, float, float], list[dict[str, Any]]] = {}
    for r in rows:
        points.setdefault((r["pose"], r["alt"], r.get("y", 0.0), r["x"]), []).append(r)
    poses, alts, ys = sorted({k[0] for k in points}), sorted({k[1] for k in points}), sorted({k[2] for k in points})
    many_y = len(ys) > 1  # a lateral scan: one row per (pose, altitude, lateral offset)
    out = [f"corridor depth below {radius} m (hit in every pass = map; hit in some passes only = bad capture); no obstacles spawned", ""]
    out += ["| pose | alt (m) |" + (" y (m) |" if many_y else "") + " points | map hits | x of map hits | bad captures | x of bad captures | points with depth under 2 m in every pass |",
            "|---|---|" + ("---|" if many_y else "") + "---|---|---|---|---|---|"]
    for p in poses:
        for y in ys:
            for a in alts:
                here = {k[3]: v for k, v in points.items() if k[:3] == (p, a, y)}
                hits = {x: [r["depth_min"] < radius for r in v] for x, v in here.items()}
                solid = sorted(x for x, h in hits.items() if all(h))
                glitch = sorted(x for x, h in hits.items() if any(h) and not all(h))
                near = sum(all(r["depth_min"] < 2.0 for r in v) for v in here.values())
                show = lambda xs: " ".join(f"{x:g}" for x in xs) or "none"  # noqa: E731
                lat = f" {y:g} |" if many_y else ""
                out.append(f"| {p} | {a:g} |{lat} {len(here)} | {len(solid)} | {show(solid)} | {len(glitch)} | {show(glitch)} | {near} |")
    zero = [r for r in rows if r["frac_zero"] > 0]
    diff = [max(r["depth_min"] for r in v) - min(r["depth_min"] for r in v) for v in points.values() if len(v) > 1]
    out += ["", f"captures: {len(rows)}; with zero depth in the window: {len(zero)}"]
    if diff:
        out.append(f"corridor depth change between passes: median {np.median(diff):.3f} m, 95th percentile {np.percentile(diff, 95):.3f} m, max {max(diff):.2f} m")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="a sweep config with backend_kw.starts")
    p.add_argument("--out", default="results/pose_clearance")
    p.add_argument("--x-max", type=float, default=40.0)
    p.add_argument("--x-step", type=float, default=1.0)
    p.add_argument("--alts", type=float, nargs="+", default=list(ALTS))
    p.add_argument("--ys", type=float, nargs="+", default=[0.0], help="lateral offsets (m) from the path, e.g. -12 -6 0 6 12")
    p.add_argument("--passes", type=int, default=2)
    a = p.parse_args(argv)
    import os

    from eval.factory import make_sim

    cfg = yaml.safe_load(Path(a.config).read_text())
    kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg["backend_kw"].items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = make_sim("airsim", **kw)
    try:
        with open(out / "scan.jsonl", "w") as f:
            rows = scan(sim, kw["starts"], np.arange(0.0, a.x_max + 1e-6, a.x_step), a.alts, a.passes, f, a.ys)
    finally:
        sim.close()
    text = "\n".join(report(rows, DroneParams().radius)) + "\n"
    (out / "report.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
