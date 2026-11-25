"""Job A world checks before the scenario freeze (plan v7, PLAN_SCENARIO_FREEZE.md; DS 2026-11-25).

Three parts, all with the drone pose set directly (no flights). Ground truth is used freely: this checks the
simulator and the course design, not a policy (CLAUDE.md rule 5 concerns run time only).

- ``region``: the bare map (no obstacles) around each start pose, over the whole area a cone course can reach:
  parallel lines across ``ys`` (path frame, metres right of the start heading) at every height in ``alts``,
  marching forward along x. Each capture reads the corridor window of the true depth. A point whose corridor
  minimum is below the drone radius in every pass is a map object; in some passes only, a bad capture. To save
  captures, a line skips ahead by what the window shows as free (the window is wider than the drone beyond about
  4 m, so a free reading of d metres clears the tube up to about d - 4 m). Map objects are part of the world
  (DS 2026-11-25: a crash into one is logged as "scenery"), so only two things reject a pose: an object at the
  start, or at the landing area (the camera cannot see below the drone while it descends). Everything else is
  reported as how much map is in the way, per pose.
- ``probe``: each obstacle family alone, straight ahead at several distances and yaws, on one clean pose:
  true depth and segmentation of the obstacle, whether it renders vertical or horizontal, our geometric corridor
  distance, the fast stack's corridor estimate (``--stack da2``), and the true depth 1 m in front of it.
- ``ground``: hovering low at each pose's start: whether the scenery guard would fire and what the fast stack reads,
  so landing at 1 m is not counted as a crash.

    python -m eval.world_check --config configs/sweeps/airsim_world_check.yaml --out results/world_check --stack da2

Resumable: every part appends to its JSONL and skips rows already written. ``report.md`` is rebuilt from the JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from perception.depth_utils import corridor_window
from perception.frame_bank import OBSTACLE_SEG_RGB
from sim.drone import DroneParams, KinematicState
from sim.scenario import DEFAULT_FAMILIES, Obstacle, Scenario

RADIUS = DroneParams().radius
YS = tuple(float(y) for y in np.arange(-25.0, 25.0 + 1e-6, 2.5))
ALTS = (1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 24.0)
X_MAX = 66.0
WINDOW_FREE_FROM_M = 4.0  # beyond this the corridor window is wider than the drone (half-angle about 4.5 deg)
LANDING = {"x": (48.0, 62.0), "y": 12.0, "alts": (1.0, 2.0)}  # where a cone course can end (goal 50 to 60 m, up to 10 m aside)
START = {"x": (0.0, 4.0), "y": 2.5}
NEAR_M = 2.0
PROBE_FAMILIES = (*DEFAULT_FAMILIES, "pole_10cm_axis_y")
PROBE_DISTS = (10.0, 15.0, 25.0)
PROBE_YAWS_DEG = (0.0, 11.0)
GROUND_ALTS = (1.0, 1.5, 2.0)


def window_stats(depth: np.ndarray) -> dict[str, float]:
    w = np.asarray(depth, dtype=np.float32)[corridor_window(depth.shape)]
    return {"depth_min": round(float(w.min()), 3), "depth_median": round(float(np.median(w)), 2),
            "frac_below_radius": round(float(np.mean(w < RADIUS)), 4), "frac_zero": round(float(np.mean(w <= 0.0)), 4)}


def bare(sim: Any, start: Sequence[float]) -> None:
    """The bare map at ``start`` ([x, y, yaw_deg, ...]): ``empty_map`` on Project AirSim; an empty scenario otherwise."""
    if hasattr(sim, "empty_map"):
        sim.empty_map(start)
    else:
        sim.reset(Scenario(seed=0, split="probe", alt_m=2.0, speed_mps=5.0, goal_m=40.0, condition="clear"))


def _done(path: Path, key: Sequence[str]) -> set[tuple]:
    if not path.exists():
        return set()
    return {tuple(json.loads(line)[k] for k in key) for line in path.read_text().splitlines() if line.strip()}


# ---- region ------------------------------------------------------------------------------------


def scan_line(sim: Any, y: float, alt: float, x_max: float = X_MAX, min_step: float = 2.0, max_step: float = 20.0) -> list[dict[str, Any]]:
    """Captures along one line; skips ahead by what the window shows free, but never more than ``max_step``."""
    rows, x = [], 0.0
    while x <= x_max + 1e-6:
        obs = sim.observe(KinematicState(t=0.0, x=x, y=y, z=-alt, yaw=0.0, speed=0.0))
        st = window_stats(obs.depth)
        rows.append({"x": round(x, 2), **st})
        x += float(np.clip(st["depth_min"] - WINDOW_FREE_FROM_M, min_step, max_step))
    return rows


def run_region(sim: Any, starts: Sequence[Sequence[float]], out: Path, ys: Iterable[float] = YS, alts: Iterable[float] = ALTS,
               passes: int = 2, x_max: float = X_MAX) -> None:
    path = out / "region.jsonl"
    done = _done(path, ("pass", "pose", "y", "alt"))
    with open(path, "a") as f:
        for n in range(passes):
            for i, start in enumerate(starts):
                todo = [(y, a) for y in ys for a in alts if (n, i, float(y), float(a)) not in done]
                if not todo:
                    continue
                bare(sim, start)
                for y, a in todo:
                    pts = scan_line(sim, float(y), float(a), x_max)
                    f.write(json.dumps({"pass": n, "pose": i, "y": float(y), "alt": float(a), "points": pts}) + "\n")
                    f.flush()
                print(f"region: pass {n} pose {i} done", flush=True)


def _cells(lines: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    """Every capture, keyed by (pose, y, alt, x), across passes. Lines march differently per pass, so a point is
    compared across passes only where both captured it."""
    out: dict[tuple, list] = defaultdict(list)
    for ln in lines:
        for p in ln["points"]:
            out[(ln["pose"], ln["y"], ln["alt"], p["x"])].append(p)
    return out


def region_report(lines: list[dict[str, Any]], passes: int) -> tuple[list[str], dict[int, dict[str, Any]]]:
    cells = _cells(lines)
    per: dict[int, dict[str, Any]] = defaultdict(lambda: {"points": 0, "map": [], "bad": 0, "near": 0, "start_hits": [], "landing_hits": []})
    for (pose, y, alt, x), ps in cells.items():
        r = per[pose]
        r["points"] += 1
        hit = [p["depth_min"] < RADIUS for p in ps]
        is_map = len(ps) >= passes and all(hit)
        if any(hit) and not is_map:
            r["bad"] += 1
        if len(ps) >= passes and all(p["depth_min"] < NEAR_M for p in ps):
            r["near"] += 1
        if is_map:
            r["map"].append((x, y, alt))
            if START["x"][0] <= x <= START["x"][1] and abs(y) <= START["y"]:
                r["start_hits"].append((x, y, alt))
            if LANDING["x"][0] <= x <= LANDING["x"][1] and abs(y) <= LANDING["y"] and alt in LANDING["alts"]:
                r["landing_hits"].append((x, y, alt))
    L = ["## Region scan (bare map, no obstacles spawned)", "",
         f"Corridor-window depth below {RADIUS} m in every pass = map object; in some passes only = bad capture. "
         f"Reject a pose only for map objects at the start (x {START['x'][0]:g} to {START['x'][1]:g}, |y| <= {START['y']:g}) "
         f"or at the landing area (x {LANDING['x'][0]:g} to {LANDING['x'][1]:g}, |y| <= {LANDING['y']:g}, heights {LANDING['alts']}). "
         "Other map objects are part of the world (crashes into them are logged as scenery).", "",
         "| pose | points | map objects | share | under 2 m (all passes) | bad captures | start | landing area | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for pose in sorted(per):
        r = per[pose]
        ok = not r["start_hits"] and not r["landing_hits"]
        verdicts[pose] = {"keep": ok, **{k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()}}
        L.append(f"| {pose} | {r['points']} | {len(r['map'])} | {len(r['map']) / max(r['points'], 1):.3f} | {r['near']} | {r['bad']} | "
                 f"{len(r['start_hits'])} | {len(r['landing_hits'])} | {'keep' if ok else 'REJECT'} |")
    L += ["", "Map objects by height (pose: height -> count; x and y of the first five):"]
    for pose in sorted(per):
        by_alt: dict[float, list] = defaultdict(list)
        for x, y, a in per[pose]["map"]:
            by_alt[a].append((x, y))
        parts = [f"{a:g} m: {len(v)} {sorted(v)[:5]}" for a, v in sorted(by_alt.items())]
        L.append(f"- pose {pose}: " + ("; ".join(parts) if parts else "none"))
    return L, verdicts


# ---- obstacle probe ----------------------------------------------------------------------------


def seg_extent(seg: np.ndarray) -> dict[str, Any]:
    mask = (np.asarray(seg)[..., :3] == np.array(OBSTACLE_SEG_RGB, dtype=np.uint8)).all(axis=-1)
    if not mask.any():
        return {"seg_px": 0, "seg_h": 0, "seg_w": 0, "renders": "invisible"}
    rows, cols = np.where(mask)
    h, w = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
    return {"seg_px": int(mask.sum()), "seg_h": h, "seg_w": w, "renders": "vertical" if h > 2 * w else "horizontal" if w > 2 * h else "compact"}


def run_probe(sim: Any, starts: Sequence[Sequence[float]], out: Path, stack: Any = None, pose: int = 0,
              families: Iterable[str] = PROBE_FAMILIES, dists: Iterable[float] = PROBE_DISTS, yaws_deg: Iterable[float] = PROBE_YAWS_DEG,
              alt: float = 2.0) -> None:
    from sim.raycast import corridor_truth

    path = out / "probe.jsonl"
    done = _done(path, ("family", "dist", "yaw_deg"))
    n = len(starts) if starts else 1
    with open(path, "a") as f:
        for fam in families:
            for d in dists:
                for yd in yaws_deg:
                    if (fam, float(d), float(yd)) in done:
                        continue
                    ob = Obstacle(fam, float(d), 0.0, alt, float(np.radians(yd)))
                    sc = Scenario(seed=pose, split="probe", alt_m=alt, speed_mps=5.0, goal_m=40.0, condition="clear", obstacles=[ob])
                    sim.reset(sc)
                    assert not starts or int(sc.seed) % n == pose
                    st = KinematicState(t=0.0, x=0.0, y=0.0, z=-alt, yaw=0.0, speed=0.0)
                    obs = sim.observe(st)
                    gt_geo, fam_geo = corridor_truth(st, sim.boxes, sim.cam)
                    row = {"family": fam, "dist": float(d), "yaw_deg": float(yd), **window_stats(obs.depth), **seg_extent(obs.seg),
                           "corridor_geometry": round(float(gt_geo), 3), "geometry_family": fam_geo}
                    if stack is not None:
                        row["est_corridor_min"] = round(float(stack.perceive_obs(obs, st, sim).features["corridor_min"]), 3)
                    front = ob.obb().half[0] + 1.0  # 1 m in front of its near face
                    close = sim.observe(KinematicState(t=0.0, x=float(d) - front, y=0.0, z=-alt, yaw=0.0, speed=0.0))
                    row["depth_min_1m_in_front"] = window_stats(close.depth)["depth_min"]
                    f.write(json.dumps(row) + "\n")
                    f.flush()
        print("probe done", flush=True)


def probe_report(rows: list[dict[str, Any]], has_stack: bool) -> list[str]:
    L = ["## Obstacle probe (one obstacle straight ahead, drone at 2 m)", "",
         "`true` = corridor minimum of the true depth image; `geometry` = our box geometry (labels use it); `est` = fast stack. "
         "`renders` from the segmentation extent. `1 m` = true depth with the drone 1 m in front of the near face "
         "(about 0.7 m expected: the camera sits about 0.3 m ahead).", "",
         "| family | dist | yaw | true | geometry | est | seg px | renders (h x w) | 1 m |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["family"], r["dist"], r["yaw_deg"])):
        est = f"{r['est_corridor_min']:.2f}" if has_stack and "est_corridor_min" in r else "n/a"
        L.append(f"| {r['family']} | {r['dist']:g} | {r['yaw_deg']:g} | {r['depth_min']:.2f} | {r['corridor_geometry']:.2f} | {est} | "
                 f"{r['seg_px']} | {r['renders']} ({r['seg_h']} x {r['seg_w']}) | {r['depth_min_1m_in_front']:.2f} |")
    return L


# ---- ground ------------------------------------------------------------------------------------


def run_ground(sim: Any, starts: Sequence[Sequence[float]], out: Path, stack: Any = None, alts: Iterable[float] = GROUND_ALTS) -> None:
    path = out / "ground.jsonl"
    done = _done(path, ("pose", "alt"))
    with open(path, "a") as f:
        for i, start in enumerate(starts):
            todo = [a for a in alts if (i, float(a)) not in done]
            if not todo:
                continue
            bare(sim, start)
            for a in todo:
                st = KinematicState(t=0.0, x=0.0, y=0.0, z=-float(a), yaw=0.0, speed=0.0)
                obs = sim.observe(st)
                row = {"pose": i, "alt": float(a), **window_stats(obs.depth)}
                row["guard_fires"] = row["depth_min"] < RADIUS
                if stack is not None:
                    row["est_corridor_min"] = round(float(stack.perceive_obs(obs, st, sim).features["corridor_min"]), 3)
                f.write(json.dumps(row) + "\n")
                f.flush()
        print("ground done", flush=True)


def ground_report(rows: list[dict[str, Any]], d_safe: float = 8.0) -> list[str]:
    L = ["## Ground check (hovering at the start, bare map)", "",
         f"Landing is safe to count only if the guard does not fire (true depth min >= {RADIUS} m) and the fast stack does not read "
         f"the ground as blocked (est >= {d_safe:g} m) at 1 m.", "",
         "| pose | height | true min | below radius | guard fires | est |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["pose"], r["alt"])):
        est = f"{r['est_corridor_min']:.2f}" if "est_corridor_min" in r else "n/a"
        L.append(f"| {r['pose']} | {r['alt']:g} | {r['depth_min']:.2f} | {r['frac_below_radius']:.3f} | {r['guard_fires']} | {est} |")
    return L


# ---- report and CLI ----------------------------------------------------------------------------


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_report(out: Path, starts: Sequence[Sequence[float]], passes: int) -> str:
    L = ["# World checks before the scenario freeze", "", "Poses (index: x, y, yaw_deg, clear_m):"]
    L += [f"- {i}: {list(s)}" for i, s in enumerate(starts)] + [""]
    region = _read(out / "region.jsonl")
    if region:
        lines, verdicts = region_report(region, passes)
        L += lines + [""]
        (out / "region_verdicts.json").write_text(json.dumps(verdicts, indent=1))
    probe = _read(out / "probe.jsonl")
    if probe:
        L += probe_report(probe, any("est_corridor_min" in r for r in probe)) + [""]
    ground = _read(out / "ground.jsonl")
    if ground:
        L += ground_report(ground) + [""]
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    return text


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="a config with backend and backend_kw.starts")
    p.add_argument("--out", default="results/world_check")
    p.add_argument("--parts", nargs="+", default=["region", "probe", "ground"], choices=["region", "probe", "ground"])
    p.add_argument("--stack", default=None, help="fast stack for probe and ground (e.g. da2); omit for true depth only")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--probe-pose", type=int, default=0, help="index of the clean pose used for the obstacle probe")
    a = p.parse_args(argv)

    from eval.factory import make_sim, make_stack

    cfg = yaml.safe_load(Path(a.config).read_text())
    kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg["backend_kw"].items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = make_sim(cfg.get("backend", "airsim"), **kw)
    stack = make_stack(a.stack) if a.stack else None
    starts = kw.get("starts") or [[0.0, 0.0, 0.0, 1000.0]]
    try:
        if "region" in a.parts:
            run_region(sim, starts, out, passes=a.passes)
        if "probe" in a.parts:
            run_probe(sim, starts, out, stack, pose=a.probe_pose)
        if "ground" in a.parts:
            run_ground(sim, starts, out, stack)
    finally:
        if hasattr(sim, "close"):
            sim.close()
    print(write_report(out, starts, a.passes), flush=True)


if __name__ == "__main__":
    main()
