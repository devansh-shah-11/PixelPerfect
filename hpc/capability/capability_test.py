"""Project AirSim capability test (answers: what can this simulator do for the study?).

Per resolution it checks: clock/pause, get_images while paused, seconds per kinematic step
(set_pose + continue_for_sim_time + get_images), world.step(), render determinism, and the
asset list. At the higher resolution it also spawns a pole, thin wires and a box and records
whether they appear in depth, segmentation and RGB at several distances.

Failure handling (the goal is to learn as much as possible from ONE job):
- every section runs in a worker thread with its own timeout, so a blocked request cannot
  hang the run; a timeout triggers a reconnect + scene reload before the next section
- exceptions are recorded (type, message, trace) and the run continues
- results.json is rewritten after every section, and on SIGTERM
- a global time budget skips the remaining sections instead of being killed mid-way
- missing image channels, failed spawns (other candidate assets are tried) and failed
  loop iterations are recorded as partial results, not crashes

Based on the v1.0.1 client source; API usage is NOT yet verified against a running sim.

Usage: python capability_test.py --configs /scratch/$USER/airsim/pa_configs/sim_config --out DIR
"""

import argparse
import json
import math
import os
import re
import signal
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import ImageType, Pose, Quaternion, Vector3
from projectairsim.utils import rpy_to_quaternion, unpack_image

CAM = "FrontCam"
TYPES = [ImageType.SCENE, ImageType.DEPTH_PERSPECTIVE, ImageType.SEGMENTATION]
START_Z = -1.5  # NED: 1.5 m above the origin plane
RESULTS: dict = {}
STATE = {"out_dir": ".", "deadline": None}
POOL = ThreadPoolExecutor(max_workers=64)  # a timed-out call keeps its thread; allow many


# ---------------------------------------------------------------- helpers
def make_pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
    w, qx, qy, qz = rpy_to_quaternion(roll, pitch, yaw)
    return Pose(
        {
            "translation": Vector3({"x": x, "y": y, "z": z}),
            "rotation": Quaternion({"w": w, "x": qx, "y": qy, "z": qz}),
            "frame_id": "DEFAULT_ID",
        }
    )


def grab(drone):
    """Fetch (rgb, depth_m float32, seg) from FrontCam. Any missing channel comes back None."""
    imgs = drone.get_images(CAM, [int(t) for t in TYPES])
    out = []
    for t in TYPES:
        msg = imgs.get(int(t)) if isinstance(imgs, dict) else None
        out.append(None if msg is None else unpack_image(msg))
    rgb, depth, seg = out
    if depth is not None:
        depth = depth.astype(np.float32)
    return rgb, depth, seg


def need(**chans):
    missing = [k for k, v in chans.items() if v is None]
    if missing:
        raise RuntimeError(f"image channel(s) not returned: {missing}")


def corridor_min(depth):
    if depth is None:
        return None
    h, w = depth.shape[:2]
    c = depth[int(0.4 * h) : int(0.6 * h), int(0.45 * w) : int(0.55 * w)]
    f = c[np.isfinite(c)]
    return float(f.min()) if f.size else None


def color_ids(seg):
    s = seg.astype(np.int64)
    return (s[..., 0] << 16) | (s[..., 1] << 8) | s[..., 2]


def dump():
    try:
        with open(os.path.join(STATE["out_dir"], "results.json"), "w") as f:
            json.dump(RESULTS, f, indent=2, default=str)
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: could not write results.json: {err}", flush=True)


def run_section(store, name, fn, timeout=120.0):
    """Run fn in a thread with a timeout. Returns (ok, timed_out). Never raises."""
    if STATE["deadline"] is not None and time.time() > STATE["deadline"]:
        store[name] = {"ok": False, "skipped": "time budget exhausted"}
        print(f"[{name}] skipped: time budget exhausted", flush=True)
        dump()
        return False, False
    t0, timed_out = time.time(), False
    fut = POOL.submit(fn)
    try:
        out = fut.result(timeout=timeout) or {}
        store[name] = {"ok": True, "seconds": round(time.time() - t0, 1), **out}
    except FutureTimeout:
        timed_out = True
        store[name] = {"ok": False, "error": f"timeout after {timeout}s (a call blocked)"}
    except Exception as err:  # noqa: BLE001
        store[name] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
    short = {k: v for k, v in store[name].items() if k != "trace"}
    print(f"[{name}] {json.dumps(short, default=str)[:700]}", flush=True)
    dump()
    return store[name]["ok"], timed_out


def connect_with_retry(client, tries=12, wait=10.0):
    last = None
    for i in range(tries):
        try:
            client.connect()
            return
        except Exception as err:  # noqa: BLE001
            last = err
            print(f"connect attempt {i + 1}/{tries} failed: {err}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"could not connect to the sim: {last}")


def load_scene(client, configs, width, shared):
    shared["world"] = World(client, f"scene_cap_{width}.jsonc", delay_after_load_sec=2, sim_config_path=configs)
    shared["drone"] = Drone(client, shared["world"], "Drone1")
    w = shared["world"]
    return {"clock_type": w.get_sim_clock_type(), "paused": w.is_paused(), "sim_time_ns": w.get_sim_time()}


def recover(client, configs, width, shared, store, after):
    """After a blocked call: reconnect and reload the scene. Records the outcome."""
    try:
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        connect_with_retry(client, tries=3, wait=5.0)
        load_scene(client, configs, width, shared)
        store[f"recovery_after_{after}"] = {"ok": True}
        return True
    except Exception as err:  # noqa: BLE001
        store[f"recovery_after_{after}"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
        print(f"[recovery_after_{after}] FAILED: {err}", flush=True)
        dump()
        return False


# ---------------------------------------------------------------- per-resolution tests
def run_resolution(client, configs, out_dir, width):
    key = f"{width}px"
    store: dict = {}
    RESULTS[key] = store
    os.makedirs(os.path.join(out_dir, key), exist_ok=True)
    shared: dict = {"usable": True}
    W, D = (lambda: shared["world"]), (lambda: shared["drone"])  # noqa: E731  (re-read after recovery)

    def sec(name, fn, timeout=120.0):
        if not shared["usable"]:
            store[name] = {"ok": False, "skipped": "session unusable after earlier failure"}
            print(f"[{name}] skipped: session unusable", flush=True)
            dump()
            return False
        ok, timed_out = run_section(store, name, fn, timeout)
        if timed_out and name != "load_scene":
            shared["usable"] = recover(client, configs, width, shared, store, name)
        return ok

    if not sec("load_scene", lambda: load_scene(client, configs, width, shared), timeout=180.0):
        return
    try:
        W().pause()
    except Exception as err:  # noqa: BLE001
        store["initial_pause"] = {"ok": False, "error": str(err)}

    def t_set_pose():
        try:
            D().enable_api_control()
        except Exception as err:  # noqa: BLE001  (may be unsupported for non-physics)
            store["enable_api_control_note"] = {"ok": True, "note": f"{type(err).__name__}: {err}"}
        ret = D().set_pose(make_pose(0.0, 0.0, START_Z))
        W().continue_for_sim_time(50_000_000)
        return {"set_pose_returned": str(ret), "pose_after": str(D().get_ground_truth_pose())[:200]}

    sec("set_pose", t_set_pose)

    def t_paused_images():
        W().pause()
        t0 = time.perf_counter()
        fut = POOL.submit(lambda: grab(D()))
        blocked = False
        try:
            rgb, depth, seg = fut.result(timeout=20)
        except FutureTimeout:
            blocked = True
            W().continue_for_single_step()
            rgb, depth, seg = fut.result(timeout=20)
        info = {"blocked_until_step": blocked, "seconds": round(time.perf_counter() - t0, 3)}
        if rgb is not None:
            cv2.imwrite(os.path.join(out_dir, key, "paused_rgb.png"), rgb)
            info.update(rgb_shape=list(rgb.shape), rgb_std=round(float(rgb.std()), 1))
        if depth is not None:
            f = depth[np.isfinite(depth)]
            info["depth_min_max_m"] = [float(f.min()), float(f.max())] if f.size else None
        if seg is not None:
            cv2.imwrite(os.path.join(out_dir, key, "paused_seg.png"), seg)
            info["seg_unique_colors"] = int(len(np.unique(color_ids(seg))))
        info["channels_missing"] = [n for n, v in (("rgb", rgb), ("depth", depth), ("seg", seg)) if v is None]
        return info

    sec("get_images_while_paused", t_paused_images, timeout=90.0)

    def t_loop():
        n, dt_ns = 60, 100_000_000  # 100 ms sim time per decision step
        t_pose, t_step, t_img, errors = [], [], [], []
        consecutive = 0
        for i in range(n):
            try:
                t0 = time.perf_counter()
                D().set_pose(make_pose(0.05 * i, 0.0, START_Z))
                t1 = time.perf_counter()
                W().continue_for_sim_time(dt_ns)
                t2 = time.perf_counter()
                grab(D())
                t3 = time.perf_counter()
                t_pose.append(t1 - t0); t_step.append(t2 - t1); t_img.append(t3 - t2)
                consecutive = 0
            except Exception as err:  # noqa: BLE001
                errors.append(f"iter {i}: {type(err).__name__}: {err}")
                consecutive += 1
                if consecutive >= 3:
                    break
        if not t_pose:
            raise RuntimeError(f"no loop iteration succeeded: {errors[:3]}")
        med = lambda a: round(statistics.median(a) * 1000, 1)  # noqa: E731
        total = med([a + b + c for a, b, c in zip(t_pose, t_step, t_img)])
        return {"iterations_ok": len(t_pose), "iterations_failed": len(errors), "errors": errors[:3], "sim_dt_ms": dt_ns / 1e6,
                "median_ms": {"set_pose": med(t_pose), "continue_for_sim_time": med(t_step), "get_images_rgb_depth_seg": med(t_img), "total": total},
                "sim_speed_vs_realtime": round(100.0 / total, 2) if total else None}

    sec("loop_timing", t_loop, timeout=300.0)

    def t_step_api():
        t0 = W().get_sim_time()
        res = W().step(5_000_000)
        return {"sim_time_advanced_ns": W().get_sim_time() - t0, "result_keys": list(res.keys()) if isinstance(res, dict) else str(type(res))}

    sec("world_step_api", t_step_api)

    def t_determinism():
        p = make_pose(1.0, 0.0, START_Z)
        D().set_pose(p); W().continue_for_sim_time(50_000_000); a = grab(D())
        D().set_pose(make_pose(5.0, 2.0, START_Z, yaw=1.0)); W().continue_for_sim_time(50_000_000); grab(D())
        D().set_pose(p); W().continue_for_sim_time(50_000_000); b = grab(D())
        out = {}
        if a[0] is not None and b[0] is not None:
            diff = np.abs(a[0].astype(int) - b[0].astype(int))
            out.update(rgb_max_abs_diff=int(diff.max()), rgb_frac_pixels_differ=round(float((diff.sum(-1) > 0).mean()), 4))
        if a[1] is not None and b[1] is not None:
            ok = np.isfinite(a[1]) & np.isfinite(b[1])
            out["depth_max_abs_diff_m"] = float(np.abs(a[1][ok] - b[1][ok]).max()) if ok.any() else None
        if not out:
            raise RuntimeError("no channel available for comparison")
        return out

    sec("render_determinism", t_determinism)

    def t_assets():
        assets = W().list_assets(".*")
        assets = list(assets) if not isinstance(assets, str) else [assets]
        shared["assets"] = assets
        return {"count": len(assets), "first_80": [str(a) for a in assets[:80]]}

    sec("assets", t_assets)

    if width == 1280:
        run_obstacles(sec, store, shared, W, D, out_dir, key)


def pick_all(assets, pattern):
    rx = re.compile(pattern, re.I)
    return [str(a) for a in assets if rx.search(str(a))]


def run_obstacles(sec, store, shared, W, D, out_dir, key):
    """Spawn objects one at a time and measure visibility at several distances."""
    assets = shared.get("assets")
    if not assets:
        store["obstacles"] = {"ok": False, "skipped": "no asset list available"}
        return
    cyls = pick_all(assets, r"cylinder|pole|pipe|rod|post|cable|wire")[:3]
    boxes = pick_all(assets, r"cube|box")[:2]
    store["asset_candidates"] = {"ok": True, "cylinder_like": cyls, "box_like": boxes}
    print(f"[asset_candidates] cylinder_like={cyls} box_like={boxes}", flush=True)
    specs = []
    if cyls:
        specs += [("pole_10cm", cyls, [0.1, 0.1, 5.0], (0, 0, 0)),
                  ("wire_3cm", cyls, [0.03, 0.03, 6.0], (math.pi / 2, 0, 0)),
                  ("wire_1cm", cyls, [0.01, 0.01, 6.0], (math.pi / 2, 0, 0))]
    if boxes:
        specs.append(("box_1m", boxes, [1.0, 1.0, 1.0], (0, 0, 0)))
    if not specs:
        store["obstacles"] = {"ok": False, "skipped": "no cylinder-like or box-like asset names matched"}
        return

    obj_x, distances = 20.0, [10.0, 6.0, 3.0]
    pose_at = lambda d: make_pose(obj_x - d - 0.3, 0.0, START_Z)  # noqa: E731

    def t_baseline():
        base = {}
        for d in distances:
            D().set_pose(pose_at(d)); W().continue_for_sim_time(50_000_000); base[d] = grab(D())
        shared["base"] = base
        return {"distances_m": distances}

    if not sec("obstacle_baselines", t_baseline, timeout=90.0):
        return

    for name, candidates, scale, (r, p, y) in specs:
        def t_obstacle(name=name, candidates=candidates, scale=scale, r=r, p=p, y=y):
            attempts, spawned, used = [], None, None
            for asset in candidates:
                try:
                    res = W().spawn_object(name, asset, make_pose(obj_x, 0.0, START_Z, r, p, y), scale, False)
                    if res is None or res is False:
                        attempts.append(f"{asset}: spawn returned {res!r}")
                        continue
                    spawned, used = res, asset
                    break
                except Exception as err:  # noqa: BLE001
                    attempts.append(f"{asset}: {type(err).__name__}: {err}")
            if used is None:
                raise RuntimeError(f"spawn failed for all candidates: {attempts}")
            rows = []
            try:
                for d in distances:
                    row = {"true_distance_m": d}
                    try:
                        D().set_pose(pose_at(d)); W().continue_for_sim_time(50_000_000)
                        rgb, depth, seg = grab(D())
                        b_rgb, b_depth, b_seg = shared["base"][d]
                        row["corridor_min_depth_with"], row["corridor_min_depth_without"] = corridor_min(depth), corridor_min(b_depth)
                        if rgb is not None and b_rgb is not None:
                            row["rgb_changed_pixels"] = int((np.abs(rgb.astype(int) - b_rgb.astype(int)).sum(-1) > 30).sum())
                            cv2.imwrite(os.path.join(out_dir, key, f"{name}_d{int(d)}_rgb.png"), rgb)
                        if seg is not None and b_seg is not None:
                            row["seg_new_color_pixels"] = int((~np.isin(color_ids(seg), np.unique(color_ids(b_seg)))).sum())
                            cv2.imwrite(os.path.join(out_dir, key, f"{name}_d{int(d)}_seg.png"), seg)
                    except Exception as err:  # noqa: BLE001
                        row["error"] = f"{type(err).__name__}: {err}"
                    rows.append(row)
            finally:
                for target in {str(spawned) if isinstance(spawned, str) else name, name}:
                    try:
                        W().destroy_object(target)
                        break
                    except Exception:  # noqa: BLE001
                        continue
            return {"asset": used, "spawn_returned": str(spawned), "scale": scale, "failed_attempts": attempts, "rows": rows}

        sec(f"obstacle_{name}", t_obstacle, timeout=120.0)


# ---------------------------------------------------------------- main
def run(args, client=None):
    STATE["out_dir"] = args.out
    STATE["deadline"] = time.time() + args.budget_sec
    os.makedirs(args.out, exist_ok=True)
    client = client or ProjectAirSimClient()

    def on_term(signum, _frame):
        RESULTS["terminated"] = {"ok": False, "error": f"received signal {signum}; partial results"}
        dump()
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    try:
        connect_with_retry(client)
        for width in (400, 1280):
            try:
                run_resolution(client, args.configs, args.out, width)
            except Exception as err:  # noqa: BLE001  (belt and braces: never lose the other resolution)
                RESULTS.setdefault(f"{width}px", {})["resolution_crash"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
                dump()
    except Exception as err:  # noqa: BLE001
        RESULTS["fatal"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
        dump()
    finally:
        dump()
        try:
            client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    print("\n=== SUMMARY ===")
    for res, secs in RESULTS.items():
        if isinstance(secs, dict) and all(isinstance(v, dict) for v in secs.values()):
            print(res, {k: ("ok" if v.get("ok") else ("SKIP" if "skipped" in v else "FAIL: " + str(v.get("error", "")))[:90]) for k, v in secs.items()})
        else:
            print(res, secs)
    return RESULTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="sim_config dir written by make_configs.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-sec", type=float, default=840.0, help="skip remaining sections after this many seconds")
    run(parser.parse_args())
    os._exit(0)  # a timed-out request thread may still be blocked
