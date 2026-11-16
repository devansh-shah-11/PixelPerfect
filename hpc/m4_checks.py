"""Chunks M4.1 and M4.2 on a live Project AirSim: do wires render, which way do the glTF axes point, and does pause/step work?

Run by ``hpc/m4_checks.sbatch`` (the simulator must already be running). Writes
``m4_checks.json`` and a contact sheet ``m4_obstacles.png`` into ``--out``.

M4.2, pause and step (D10 depends on it):
  - is the world paused after ``pause()``, and does sim time stay put while it is (2 s wall wait)?
  - does ``continue_for_sim_time(100 ms)`` advance sim time by exactly 100 ms, every time?
  - does a ``set_pose`` stay put over 10 paused steps (non-physics drone, no drift)?
  - wall seconds per flight-loop step (set_pose + 100 ms step + get_images), median and p95 over 50 steps.

M4.1, thin obstacles and axes (the wire story, PLAN_DETAILED.md §11 blocker 4):
  for each obstacle in ``CASES`` at 3, 6 and 10 m straight ahead at camera height: segmentation
  pixels, mask bounding box (wider than tall means the long axis is horizontal), median ground-truth
  depth on the mask against the expected distance, and RGB pixels changed against an empty frame.
  Verdict per obstacle: "visible to segmentation and depth", "segmentation only", or "invisible".

Uses the helpers from hpc/capability/capability_test.py that the frame bank ran on.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim.airsim_backend import load_capability_module  # noqa: E402

CASES = ["wire_1cm_axis_z", "wire_3cm_axis_z", "wire_1cm_axis_y", "pole_10cm_axis_z", "pole_10cm_axis_y", "glass_pane_alpha20", "box_1m_grey"]
DISTANCES = (3.0, 6.0, 10.0)
SEG_RGB_BGR = (231, 243, 185)  # OBSTACLE_SEG_RGB (185, 243, 231) as the BGR array the client returns [UNVERIFIED order]


def pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if x else None


def check_pause_step(ct, world, drone, start) -> dict:
    out: dict = {}
    world.pause()
    out["paused_after_pause"] = bool(world.is_paused())
    t0 = world.get_sim_time()
    time.sleep(2.0)
    out["sim_time_change_while_paused_ns"] = int(world.get_sim_time() - t0)
    deltas = []
    for _ in range(20):
        a = world.get_sim_time()
        world.continue_for_sim_time(ct.STEP_NS)
        deltas.append(int(world.get_sim_time() - a))
    out["step_ns_requested"] = ct.STEP_NS
    out["step_ns_observed"] = {"min": min(deltas), "max": max(deltas), "all_equal": len(set(deltas)) == 1}
    out["paused_after_steps"] = bool(world.is_paused())
    x, y, z, yaw = start
    drone.set_pose(ct.make_pose(x, y, z, yaw=yaw))
    world.continue_for_sim_time(ct.STEP_NS)
    p0 = drone.get_ground_truth_pose()
    for _ in range(10):
        world.continue_for_sim_time(ct.STEP_NS)
    p1 = drone.get_ground_truth_pose()
    out["pose_before_after_10_steps"] = [str(p0)[:160], str(p1)[:160]]
    walls, walls_no_img = [], []
    for k in range(50):
        t = time.time()
        drone.set_pose(ct.make_pose(x + 0.5 * k * math.cos(yaw), y + 0.5 * k * math.sin(yaw), z, yaw=yaw))
        world.continue_for_sim_time(ct.STEP_NS)
        walls_no_img.append(time.time() - t)
        ct.grab(drone)
        walls.append(time.time() - t)
    out["wall_s_per_step"] = {"median": pct(walls, 50), "p95": pct(walls, 95), "median_without_images": pct(walls_no_img, 50)}
    out["sim_seconds_per_wall_second"] = 0.1 / pct(walls, 50) if walls else None
    return out


def check_obstacles(ct, world, drone, start, out_dir: Path) -> dict:
    import cv2

    x, y, z, yaw = start
    base_pose = ct.make_pose(x, y, z, yaw=yaw)
    empty = ct.settle_grab(world, drone, base_pose)
    results, tiles = {}, []
    for name in CASES:
        spec = ct.CATALOG.get(name) or {"glb": (0.01, 6.0, 0.01, (0.05, 0.05, 0.05, 1.0), False)}  # wire_1cm_axis_y
        dx, dy, dz, rgba, blend = spec["glb"]
        rows = []
        for d in DISTANCES:
            ox, oy = x + d * math.cos(yaw), y + d * math.sin(yaw)
            obj = f"m4_{name}_{int(d)}"
            row: dict = {"distance_m": d}
            try:
                res = world.spawn_object_from_file(obj, "gltf", ct.box_gltf(dx, dy, dz, rgba, blend, binary=True), True,
                                                   ct.make_pose(ox, oy, z, yaw=yaw), [1.0, 1.0, 1.0], False)
                spawned = str(res) if res not in (None, False, "") else obj
                try:
                    world.set_segmentation_id_by_name(spawned, 100, False, True)
                except Exception as err:  # noqa: BLE001
                    row["seg_id_error"] = str(err)[:120]
                rgb, depth, seg = ct.resettle(world, drone)
                mask = np.all(seg[..., :3] == np.array(SEG_RGB_BGR), axis=-1) if seg is not None else np.zeros((1, 1), bool)
                if not mask.any() and seg is not None:  # try the other channel order before concluding anything
                    mask = np.all(seg[..., :3] == np.array(SEG_RGB_BGR[::-1]), axis=-1)
                    row["seg_colour_order"] = "rgb" if mask.any() else None
                row["seg_px"] = int(mask.sum())
                if mask.any():
                    ys, xs = np.where(mask)
                    row["bbox_wh_px"] = [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
                    row["long_axis"] = "horizontal" if row["bbox_wh_px"][0] > row["bbox_wh_px"][1] else "vertical"
                    row["depth_on_mask_median_m"] = float(np.median(depth[mask])) if depth is not None else None
                h, w = depth.shape
                centre = depth[int(0.4 * h): int(0.6 * h), int(0.45 * w): int(0.55 * w)]
                row["corridor_min_m"] = float(np.nanmin(centre))
                row["depth_sees_it"] = bool(row["corridor_min_m"] < d + 1.0)
                row["rgb_px_changed_vs_empty"] = ct.n_changed(empty[0], rgb)
                tiles.append((f"{name} {d:g} m", rgb, seg, depth))
            except Exception as err:  # noqa: BLE001
                row["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            finally:
                try:
                    world.destroy_object(obj)
                except Exception:  # noqa: BLE001
                    pass
                world.continue_for_sim_time(2 * ct.STEP_NS)
            rows.append(row)
        seg_ok = [r.get("seg_px", 0) > 0 for r in rows]
        depth_ok = [r.get("depth_sees_it", False) for r in rows]
        verdict = ("visible to segmentation and depth" if all(seg_ok) and all(depth_ok) else
                   "segmentation only" if any(seg_ok) and not any(depth_ok) else
                   "invisible" if not any(seg_ok) and not any(depth_ok) else "partly visible (see rows)")
        results[name] = {"verdict": verdict, "rows": rows}
        print(f"[{name}] {verdict}: " + "; ".join(f"{r['distance_m']:g} m seg {r.get('seg_px')} px depth {r.get('corridor_min_m', float('nan')):.1f} m {r.get('long_axis', '')}" for r in rows), flush=True)
    if tiles:
        strips = []
        for title, rgb, seg, depth in tiles:
            dv = cv2.applyColorMap(np.clip(depth / 30.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            strip = np.concatenate([rgb[..., :3], seg[..., :3], dv], axis=1)
            cv2.putText(strip, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            strips.append(strip)
        cv2.imwrite(str(out_dir / "m4_obstacles.png"), np.concatenate(strips, axis=0))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", required=True, help="directory with scene_cap_400.jsonc (written by make_configs.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--start", default="0,0,-2.0,0", help="x,y,z,yaw_deg in world NED; pick an open spot (see the capability scene scan)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y, z, yaw_deg = (float(v) for v in a.start.split(","))
    start = (x, y, z, math.radians(yaw_deg))

    ct = load_capability_module()
    from projectairsim import Drone, ProjectAirSimClient, World

    client = ProjectAirSimClient()
    ct.connect_with_retry(client)
    report: dict = {"start": a.start}
    try:
        world = World(client, "scene_cap_400.jsonc", delay_after_load_sec=2, sim_config_path=a.configs)
        drone = Drone(client, world, "Drone1")
        for key, fn in (("pause_step_M4.2", lambda: check_pause_step(ct, world, drone, start)),
                        ("obstacles_M4.1", lambda: check_obstacles(ct, world, drone, start, out))):
            t = time.time()
            try:
                report[key] = {"ok": True, **fn(), "seconds": round(time.time() - t, 1)}
            except Exception as err:  # noqa: BLE001
                import traceback

                report[key] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
            (out / "m4_checks.json").write_text(json.dumps(report, indent=1, default=str))
            print(f"== {key}: {'ok' if report[key]['ok'] else 'FAILED ' + report[key]['error']}", flush=True)
    finally:
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps({k: v for k, v in report.get("pause_step_M4.2", {}).items() if k != "pose_before_after_10_steps"}, indent=1, default=str))
    print(f"written: {out}/m4_checks.json, {out}/m4_obstacles.png")


if __name__ == "__main__":
    main()
