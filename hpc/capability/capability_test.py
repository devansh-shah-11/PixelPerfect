"""Project AirSim capability test v2: what can this simulator (and this environment) do for the study?

Runs per environment (Blocks / Neighborhood / CityEnviron / LandscapeMountains; the binary that is
running decides the map, the scene config only places the robot). Sections, in priority order:

  A. 400x225 session (the resolution episodes will use)
     basics        load, set_pose, concurrent get_images while paused, loop timing, per-channel timing,
                   world.step(), render repeatability, asset + scene object lists
     depth_range   spawned 1 m box at 2..40 m: true depth vs measured (max range, units check)
     scene_scan    grid of poses x 4 yaws at two altitudes -> contact sheets + suggested start poses
     obstacles     catalog: poles, wires (3 axes), boxes (grey/black/white), glass pane, white wall,
                   building block, ball, textured wall; each at 10/6/3 m against ONE empty baseline,
                   with depth error, RGB/seg change, bbox of change, and "after destroy" cleanup check
     dynamic       box moving toward the drone: depth tracks it?
     weather       rain/snow/fog/dust/leaves/wetness/road-snow, rain+fog, + time of day + sun intensity
     video         60-step flight with RGB | depth | seg panel -> mp4 (clear and, if it worked, rain+fog)
  B. 400x225 with depth noise enabled (does noise-settings change depth?)
  C. 1280x720 session: loop timing + thin obstacles / glass only (visibility of thin things)

Failure handling: every section runs in a worker thread with its own timeout; a timeout triggers a
reconnect + scene reload; results.json is rewritten after every section and on SIGTERM; a global time
budget skips remaining sections; failed spawns/channels are recorded as partial results.

Fixes from the first run (job 15689): enable_api_control is unsupported for non-physics robots (removed);
get_images while paused only works if a step happens concurrently (tested that way now); first capture
after a pose change/spawn can be stale (settle_grab: step, discard one frame, keep the next); spawned
objects get a unique segmentation ID; one empty baseline per session instead of per distance.

API usage is based on the v1.0.1 client source; sections not yet run on a live sim may still hit API
mistakes, which show up as FAIL with a trace in results.json.
"""

import argparse
import base64
import json
import math
import os
import re
import signal
import statistics
import struct
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import ImageType, Pose, Quaternion, Vector3
from projectairsim.utils import rpy_to_quaternion, unpack_image

CAM = "FrontCam"
TYPES = [ImageType.SCENE, ImageType.DEPTH_PERSPECTIVE, ImageType.SEGMENTATION]
STEP_NS = 100_000_000  # one decision step of sim time (100 ms)
DEFAULT_START = (0.0, 0.0, -1.5, 0.0)  # x, y, z (NED, up is negative), yaw radians
RESULTS: dict = {}
STATE = {"out_dir": ".", "deadline": None, "args": None}
POOL = ThreadPoolExecutor(max_workers=64)  # a timed-out call keeps its thread; allow many
AUX: dict = {}  # cross-session data (clean depth for the noisy-depth comparison)


# ---------------------------------------------------------------- geometry / image helpers
def make_pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> Pose:
    w, qx, qy, qz = rpy_to_quaternion(roll, pitch, yaw)
    return Pose(
        {
            "translation": Vector3({"x": x, "y": y, "z": z}),
            "rotation": Quaternion({"w": w, "x": qx, "y": qy, "z": qz}),
            "frame_id": "DEFAULT_ID",
        }
    )


def ahead(start, d):
    """Point d metres ahead of start (x, y, z, yaw) along its heading, same altitude."""
    x, y, z, yaw = start
    return x + d * math.cos(yaw), y + d * math.sin(yaw), z


def grab(drone, types=TYPES):
    """Fetch (rgb, depth_m float32, seg) from FrontCam. Any missing channel comes back None."""
    imgs = drone.get_images(CAM, [int(t) for t in types])
    out = {}
    for t in types:
        msg = imgs.get(int(t)) if isinstance(imgs, dict) else None
        out[t] = None if msg is None else unpack_image(msg)
    depth = out.get(ImageType.DEPTH_PERSPECTIVE)
    if depth is not None:
        depth = depth.astype(np.float32)
    return out.get(ImageType.SCENE), depth, out.get(ImageType.SEGMENTATION)


def settle_grab(world, drone, pose, warm_steps=3, types=TYPES):
    """set_pose, let the renderer catch up, discard one frame, then keep a fresh one.

    The first run returned stale frames right after a pose change or spawn (the d10 box frame was
    identical to the previous d3 frame), so never trust the first capture after a change.
    """
    drone.set_pose(pose)
    return resettle(world, drone, warm_steps, types)


def resettle(world, drone, warm_steps=3, types=TYPES):
    """Same as settle_grab but without moving the drone (after spawning/moving/weather changes)."""
    world.continue_for_sim_time(warm_steps * STEP_NS)
    grab(drone, types)  # discarded
    world.continue_for_sim_time(STEP_NS)
    return grab(drone, types)


def n_changed(a, b, thr=30):
    """Pixels whose max channel difference exceeds thr (None if either frame is missing)."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return int((np.abs(a.astype(int) - b.astype(int)).max(-1) > thr).sum())


def change_bbox(a, b, thr=30):
    if a is None or b is None or a.shape != b.shape:
        return None
    m = np.abs(a.astype(int) - b.astype(int)).max(-1) > thr
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


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


def depth_vis(depth, max_m=30.0):
    d = np.where(np.isfinite(depth), depth, max_m)
    d8 = (255 * (1 - np.clip(d / max_m, 0, 1))).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def depth_stats(depth):
    if depth is None:
        return None
    f = depth[np.isfinite(depth)]
    return {"finite_frac": round(float(np.isfinite(depth).mean()), 3),
            "min_m": round(float(f.min()), 2) if f.size else None,
            "max_finite_m": round(float(f.max()), 2) if f.size else None}


# ---------------------------------------------------------------- glTF (GLB) box builder
def box_gltf(dx, dy, dz, rgba=(0.5, 0.5, 0.5, 1.0), blend=False, roughness=0.8, binary=True) -> bytes:
    """Axis-aligned box centred at the origin, sizes in metres along glTF x, y, z.

    Which glTF axis ends up forward/sideways/up in the sim is not documented, so obstacles that care
    about orientation are spawned once per axis and the bbox of the change reports the mapping.
    """
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    faces = [
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
        ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
    ]
    pos, nrm, idx = [], [], []
    for i, (n, vs) in enumerate(faces):
        for v in vs:
            pos += v
            nrm += n
        b = 4 * i
        idx += [b, b + 1, b + 2, b, b + 2, b + 3]
    pos_b = struct.pack(f"<{len(pos)}f", *pos)
    nrm_b = struct.pack(f"<{len(nrm)}f", *nrm)
    idx_b = struct.pack(f"<{len(idx)}H", *idx)
    idx_b += b"\x00" * (-len(idx_b) % 4)
    buf = pos_b + nrm_b + idx_b
    gltf = {
        "asset": {"version": "2.0", "generator": "capability_test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0, "roughnessFactor": roughness},
                       "alphaMode": "BLEND" if blend else "OPAQUE", "doubleSided": bool(blend)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 24, "type": "VEC3", "min": [-hx, -hy, -hz], "max": [hx, hy, hz]},
            {"bufferView": 1, "componentType": 5126, "count": 24, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(nrm_b), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_b) + len(nrm_b), "byteLength": len(idx) * 2, "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }
    if not binary:  # self-contained .gltf with the buffer as a data URI
        gltf["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(buf).decode()
        return json.dumps(gltf).encode()
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(buf)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(buf), b"BIN\x00") + buf)


# ---------------------------------------------------------------- runner infrastructure
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
        store[name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1800:]}
    short = {k: v for k, v in store[name].items() if k not in ("trace", "rows", "all")}
    print(f"[{name}] {json.dumps(short, default=str)[:600]}", flush=True)
    dump()
    return store[name]["ok"], timed_out


def connect_with_retry(client, tries=30, wait=10.0):
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


class Session:
    """One loaded scene config (one resolution/noise variant) and its results store."""

    def __init__(self, client, configs, out_dir, scene_file, key):
        self.client, self.configs, self.out_dir, self.scene_file, self.key = client, configs, out_dir, scene_file, key
        self.store: dict = {}
        RESULTS[key] = self.store
        self.world = self.drone = None
        self.usable = True
        self.assets: list = []
        self.start = STATE["args"].start_pose or DEFAULT_START
        self.base = None  # empty-scene (rgb, depth, seg) at self.start
        self.flags: dict = {}
        self.epoch = 0  # bumped on every reconnect; work started in an older epoch must not write results
        os.makedirs(os.path.join(out_dir, key), exist_ok=True)

    W = property(lambda self: self.world)
    D = property(lambda self: self.drone)

    def path(self, name):
        return os.path.join(self.out_dir, self.key, name)

    def load(self):
        self.world = World(self.client, self.scene_file, delay_after_load_sec=2, sim_config_path=self.configs)
        self.drone = Drone(self.client, self.world, "Drone1")
        return {"clock_type": self.world.get_sim_clock_type(), "paused": self.world.is_paused(), "sim_time_ns": self.world.get_sim_time()}

    def recover(self, after):
        self.epoch += 1
        try:
            try:
                self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            connect_with_retry(self.client, tries=3, wait=5.0)
            self.load()
            self.W.pause()
            self.base = None
            self.store[f"recovery_after_{after}"] = {"ok": True}
            return True
        except Exception as err:  # noqa: BLE001
            self.store[f"recovery_after_{after}"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
            print(f"[recovery_after_{after}] FAILED: {err}", flush=True)
            dump()
            return False

    def sec(self, name, fn, timeout=120.0, force=False):
        only, skip = STATE["args"].only, STATE["args"].skip
        if not force and name != "load_scene" and ((only and name not in only) or name in skip):
            self.store[name] = {"ok": False, "skipped": "excluded by --only/--skip"}
            return False
        if not self.usable:
            self.store[name] = {"ok": False, "skipped": "session unusable after earlier failure"}
            print(f"[{name}] skipped: session unusable", flush=True)
            dump()
            return False
        ok, timed_out = run_section(self.store, name, fn, timeout)
        if timed_out and name != "load_scene":
            self.usable = self.recover(name)
        return ok

    def start_pose(self, **kw):
        x, y, z, yaw = self.start
        return make_pose(x, y, z, yaw=yaw, **kw)

    def baseline(self):
        """Empty-scene frame at the start pose (captured once per session)."""
        if self.base is None:
            self.base = settle_grab(self.W, self.D, self.start_pose())
            for n, i in (("rgb", 0), ("seg", 2)):
                if self.base[i] is not None:
                    cv2.imwrite(self.path(f"baseline_{n}.png"), self.base[i])
        return self.base


# ---------------------------------------------------------------- spawning
def pick_all(assets, pattern):
    rx = re.compile(pattern, re.I)
    return [str(a) for a in assets if rx.search(str(a))]


def spawn_spec(s: Session, name, spec, pose):
    """Spawn per spec: {"glb": (dx,dy,dz,rgba,blend)} first, then {"asset": regex, "scale": [...]}.

    Returns (spawned_name, how, attempts). Raises RuntimeError if every route failed.
    """
    attempts = []
    if "glb" in spec:
        dx, dy, dz, rgba, blend = spec["glb"]
        for binary in (True, False):
            try:
                data = box_gltf(dx, dy, dz, rgba, blend, binary=binary)
                res = s.W.spawn_object_from_file(name, "gltf", data, binary, pose, [1.0, 1.0, 1.0], False)
                if res not in (None, False, ""):
                    return str(res), f"gltf_{'glb' if binary else 'json'}", attempts
                attempts.append(f"gltf binary={binary}: returned {res!r}")
            except Exception as err:  # noqa: BLE001
                attempts.append(f"gltf binary={binary}: {type(err).__name__}: {err}")
    if "asset" in spec:
        for asset in pick_all(s.assets, spec["asset"])[:3]:
            try:
                res = s.W.spawn_object(name, asset, pose, spec.get("scale", [1.0, 1.0, 1.0]), False)
                if res not in (None, False, ""):
                    return str(res), f"asset:{asset}", attempts
                attempts.append(f"{asset}: returned {res!r}")
            except Exception as err:  # noqa: BLE001
                attempts.append(f"{asset}: {type(err).__name__}: {err}")
    raise RuntimeError(f"spawn failed: {attempts}")


def destroy(s: Session, spawned, name):
    for target in dict.fromkeys([spawned, name]):
        try:
            s.W.destroy_object(target)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


# ---------------------------------------------------------------- A. basics
def sec_basics(s: Session, n_loop=60):
    def t_set_pose():
        ret = s.D.set_pose(s.start_pose())
        s.W.continue_for_sim_time(5 * STEP_NS)
        return {"set_pose_returned": str(ret), "pose_after": str(s.D.get_ground_truth_pose())[:200]}

    s.sec("set_pose", t_set_pose)

    def t_paused_images():
        s.W.pause()
        t0 = time.perf_counter()
        fut = POOL.submit(lambda: grab(s.D))
        time.sleep(0.5)
        s.W.continue_for_single_step()  # the capture only completes when a step happens
        rgb, depth, seg = fut.result(timeout=40)
        info = {"concurrent_step_seconds": round(time.perf_counter() - t0, 3)}
        info["channels_missing"] = [n for n, v in (("rgb", rgb), ("depth", depth), ("seg", seg)) if v is None]
        if rgb is not None:
            info["rgb_shape"] = list(rgb.shape)
            cv2.imwrite(s.path("paused_rgb.png"), rgb)
        if seg is not None:
            info["seg_unique_colors"] = int(len(np.unique(color_ids(seg))))
        info["depth"] = depth_stats(depth)
        return info

    s.sec("images_while_paused_concurrent_step", t_paused_images, timeout=90.0)

    def t_loop():
        t_pose, t_step, t_img, errors, consecutive = [], [], [], [], 0
        x0, y0, z0, yaw = s.start
        for i in range(n_loop):
            try:
                t0 = time.perf_counter()
                s.D.set_pose(make_pose(x0 + 0.05 * i * math.cos(yaw), y0 + 0.05 * i * math.sin(yaw), z0, yaw=yaw))
                t1 = time.perf_counter()
                s.W.continue_for_sim_time(STEP_NS)
                t2 = time.perf_counter()
                grab(s.D)
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
        return {"iterations_ok": len(t_pose), "iterations_failed": len(errors), "errors": errors[:3], "sim_dt_ms": STEP_NS / 1e6,
                "median_ms": {"set_pose": med(t_pose), "continue_for_sim_time": med(t_step), "get_images_rgb_depth_seg": med(t_img), "total": total},
                "sim_speed_vs_realtime": round(STEP_NS / 1e6 / total, 2) if total else None}

    s.sec("loop_timing", t_loop, timeout=420.0)

    def t_channels():
        out = {}
        for name, types in (("rgb", [ImageType.SCENE]), ("depth", [ImageType.DEPTH_PERSPECTIVE]), ("seg", [ImageType.SEGMENTATION]),
                            ("rgb+depth", TYPES[:2]), ("all3", TYPES)):
            ts = []
            for _ in range(8):
                s.W.continue_for_sim_time(STEP_NS)
                t0 = time.perf_counter()
                grab(s.D, types)
                ts.append(time.perf_counter() - t0)
            out[name] = round(statistics.median(ts) * 1000, 1)
        return {"median_get_images_ms": out}

    s.sec("channel_timing", t_channels, timeout=240.0)

    def t_step_api():
        t0 = s.W.get_sim_time()
        res = s.W.step(5_000_000)
        return {"sim_time_advanced_ns": s.W.get_sim_time() - t0, "result_keys": list(res.keys()) if isinstance(res, dict) else str(type(res))}

    s.sec("world_step_api", t_step_api)

    def t_repeatability():
        p = s.start_pose()
        s.D.set_pose(p); s.W.continue_for_sim_time(3 * STEP_NS); grab(s.D)
        frames = []
        for _ in range(3):  # same pose, successive steps
            s.W.continue_for_sim_time(STEP_NS)
            frames.append(grab(s.D))
        out = {"successive_same_pose": []}
        for a, b in zip(frames, frames[1:]):
            row = {"rgb_changed_gt0": n_changed(a[0], b[0], 0), "rgb_changed_gt30": n_changed(a[0], b[0], 30)}
            if a[0] is not None and b[0] is not None:
                row["rgb_max_abs_diff"] = int(np.abs(a[0].astype(int) - b[0].astype(int)).max())
            if a[1] is not None and b[1] is not None:
                ok = np.isfinite(a[1]) & np.isfinite(b[1])
                row["depth_max_abs_diff_m"] = float(np.abs(a[1][ok] - b[1][ok]).max()) if ok.any() else None
            out["successive_same_pose"].append(row)
        x, y, z, yaw = s.start
        s.D.set_pose(make_pose(x + 5.0, y + 2.0, z, yaw=yaw + 1.0)); s.W.continue_for_sim_time(3 * STEP_NS); grab(s.D)
        back = settle_grab(s.W, s.D, p)
        out["after_moving_away_and_back"] = {"rgb_changed_gt0": n_changed(frames[-1][0], back[0], 0), "rgb_changed_gt30": n_changed(frames[-1][0], back[0], 30)}
        return out

    s.sec("render_repeatability", t_repeatability)

    def t_lists():
        assets = s.W.list_assets(".*")
        assets = list(assets) if not isinstance(assets, str) else [assets]
        s.assets = [str(a) for a in assets]
        objs = s.W.list_objects(".*")
        objs = list(objs) if not isinstance(objs, str) else [objs]
        material_like = pick_all(s.assets, r"glass|transluc|mirror|water|snow|material|mat_")
        building_like = pick_all(s.assets, r"build|house|tower|wall|roof|window|door|tree|car|truck|bus")
        return {"asset_count": len(s.assets), "assets": s.assets, "scene_object_count": len(objs), "scene_objects_first_150": [str(o) for o in objs[:150]],
                "material_or_glass_like_assets": material_like, "building_or_vehicle_like_assets": building_like}

    s.sec("assets_and_scene_objects", t_lists)


# ---------------------------------------------------------------- A. depth range probe
def sec_depth_range(s: Session):
    def t():
        base = s.baseline()
        spec = {"glb": (1.0, 1.0, 1.0, (0.55, 0.55, 0.55, 1.0), False), "asset": r"1M_Cube|^cube$", "scale": [1.0, 1.0, 1.0]}
        ax, ay, az = ahead(s.start, 10.0)
        spawned, how, _ = spawn_spec(s, "range_box", spec, make_pose(ax, ay, az, yaw=s.start[3]))
        rows = []
        try:
            for d in (2, 4, 5, 6, 8, 12, 20, 40):
                px, py, pz = ahead(s.start, d + 0.5)  # front face at distance d
                s.W.set_object_pose(spawned, make_pose(px, py, pz, yaw=s.start[3]), True)
                rgb, depth, seg = resettle(s.W, s.D)
                m = corridor_min(depth)
                rows.append({"true_front_m": d, "corridor_min_depth_m": m, "err_m": None if m is None else round(m - d, 2),
                             "rgb_changed_vs_empty": n_changed(rgb, base[0]), "bbox": change_bbox(rgb, base[0])})
        finally:
            destroy(s, spawned, "range_box")
        return {"how": how, "baseline_depth": depth_stats(base[1]), "rows": rows}

    s.sec("depth_range_probe", t, timeout=180.0)


# ---------------------------------------------------------------- A. scene scan
def sec_scene_scan(s: Session):
    def t():
        a = STATE["args"]
        n, ext = 4, a.scan_extent
        grid = [(x, y) for x in np.linspace(-ext, ext, n) for y in np.linspace(-ext, ext, n)]
        rows, sheets = [], {}
        for alt in (-2.0, -25.0):
            tiles = []
            for (x, y) in grid:
                panel, ps = [], []
                for k in range(4):
                    yaw = k * math.pi / 2
                    rgb, depth, seg = settle_grab(s.W, s.D, make_pose(float(x), float(y), alt, yaw=yaw), warm_steps=2)
                    st = depth_stats(depth) or {}
                    st.update(x=round(float(x), 1), y=round(float(y), 1), z=alt, yaw_deg=90 * k, corridor_min_m=corridor_min(depth),
                              seg_colors=int(len(np.unique(color_ids(seg)))) if seg is not None else None)
                    ps.append(st)
                    if rgb is not None:
                        panel.append(label(rgb, f"x{x:.0f} y{y:.0f} z{-alt:.0f} yaw{90 * k}"))
                rows += ps
                if len(panel) == 4:
                    tiles.append(np.hstack(panel))
            if tiles:
                sheets[alt] = np.vstack(tiles)
                cv2.imwrite(s.path(f"scan_alt{int(-alt)}m.png"), sheets[alt])
        # suggest start poses at street level: all four directions open (min depth large / sky)
        by_pos = {}
        for r in rows:
            if r["z"] == -2.0:
                by_pos.setdefault((r["x"], r["y"]), []).append(r)
        scored = []
        for (x, y), rs in by_pos.items():
            clear = [(r["corridor_min_m"] if r["corridor_min_m"] is not None else 100.0) for r in rs]
            scored.append((min(clear), x, y, [r["yaw_deg"] for r, c in zip(rs, clear) if c > 15]))
        scored.sort(reverse=True)
        suggested = [{"x": x, "y": y, "z": -2.0, "min_clear_m": round(c, 1), "open_yaws_deg": yaws} for c, x, y, yaws in scored[:5]]
        s.flags["suggested"] = suggested
        if suggested and a.start_pose is None and a.auto_start:
            b = suggested[0]
            yaw = math.radians(b["open_yaws_deg"][0]) if b["open_yaws_deg"] else 0.0
            s.start = (b["x"], b["y"], -2.0, yaw)
            s.base = None
        return {"extent_m": ext, "n_poses": len(rows), "suggested_start_poses": suggested, "start_used_for_later_sections": list(s.start), "rows": rows}

    s.sec("scene_scan", t, timeout=420.0)


# ---------------------------------------------------------------- A. obstacle catalog
GREY, BLACK, WHITE, RED = (0.55, 0.55, 0.55, 1.0), (0.02, 0.02, 0.02, 1.0), (0.95, 0.95, 0.95, 1.0), (0.8, 0.1, 0.1, 1.0)
CATALOG = {
    # name: (spec, expected front-face offset from centre along view axis is ignored: glb axis mapping unknown)
    "pole_10cm_axis_z": {"glb": (0.1, 0.1, 5.0, GREY, False)},
    "pole_10cm_axis_y": {"glb": (0.1, 5.0, 0.1, GREY, False)},
    "pole_10cm_asset": {"asset": r"^cylinder$", "scale": [0.1, 0.1, 5.0]},
    "wire_3cm_axis_z": {"glb": (0.03, 0.03, 6.0, BLACK, False)},
    "wire_3cm_axis_y": {"glb": (0.03, 6.0, 0.03, BLACK, False)},
    "wire_1cm_axis_z": {"glb": (0.01, 0.01, 6.0, BLACK, False), "thin": True},
    "wire_1cm_axis_y": {"glb": (0.01, 6.0, 0.01, BLACK, False), "thin": True},
    "wire_3mm_axis_y": {"glb": (0.003, 6.0, 0.003, BLACK, False), "thin": True},
    "box_1m_grey": {"glb": (1.0, 1.0, 1.0, GREY, False), "asset": r"1M_Cube", "scale": [1, 1, 1]},
    "box_1m_black": {"glb": (1.0, 1.0, 1.0, BLACK, False)},
    "box_1m_white": {"glb": (1.0, 1.0, 1.0, WHITE, False)},
    "wall_white_featureless": {"glb": (0.2, 8.0, 3.0, WHITE, False), "thin": True},
    "glass_pane_alpha20": {"glb": (0.05, 3.0, 2.5, (0.7, 0.85, 0.95, 0.2), True), "thin": True},
    "glass_pane_alpha50": {"glb": (0.05, 3.0, 2.5, (0.7, 0.85, 0.95, 0.5), True)},
    "building_block": {"glb": (12.0, 20.0, 25.0, (0.6, 0.45, 0.4, 1.0), False)},
    "ball_20cm_red": {"glb": (0.2, 0.2, 0.2, RED, False), "asset": r"^sphere$", "scale": [0.2, 0.2, 0.2]},
    "textured_cube_checker": {"asset": r"1M_Cube|^cube$", "scale": [2.0, 2.0, 2.0], "texture": "checker"},
}


def sec_obstacles(s: Session, only_thin=False, distances=(10.0, 6.0, 3.0)):
    names = [n for n, sp in CATALOG.items() if sp.get("thin") or not only_thin]
    ck = s.path("checker.png")
    chk = (np.indices((64, 64)).sum(0) // 8 % 2 * 255).astype(np.uint8)
    cv2.imwrite(ck, np.dstack([chk, 255 - chk, chk]))

    def t_baseline():
        b = s.baseline()
        return {"start_pose": list(s.start), "baseline_depth": depth_stats(b[1])}

    if not s.sec("obstacle_baseline", t_baseline, timeout=90.0):
        return
    for i, name in enumerate(names):
        spec = CATALOG[name]

        def t_obstacle(name=name, spec=spec, i=i):
            base_rgb, base_depth, base_seg = s.baseline()
            x, y, z = ahead(s.start, distances[0])
            spawned, how, attempts = spawn_spec(s, name, spec, make_pose(x, y, z, yaw=s.start[3]))
            info = {"how": how, "spawn_attempts_failed": attempts, "spawned_name": spawned}
            try:
                info["seg_id_set"] = bool(s.W.set_segmentation_id_by_name(spawned, 100 + i, False, True))
            except Exception as err:  # noqa: BLE001
                info["seg_id_set"] = f"{type(err).__name__}: {err}"
            if spec.get("texture"):
                try:
                    info["texture_ok"] = bool(s.W.set_object_texture_from_file(spawned, ck))
                except Exception as err:  # noqa: BLE001
                    info["texture_ok"] = f"{type(err).__name__}: {err}"
            rows, prev = [], None
            try:
                s.W.continue_for_sim_time(5 * STEP_NS)
                for d in distances:
                    row = {"true_distance_m": d}
                    try:
                        px, py, pz = ahead(s.start, d)
                        s.W.set_object_pose(spawned, make_pose(px, py, pz, yaw=s.start[3]), True)
                        rgb, depth, seg = resettle(s.W, s.D)
                        row["sim_time_ns"] = s.W.get_sim_time()
                        m = corridor_min(depth)
                        row.update(corridor_min_depth_with=m, corridor_min_depth_empty=corridor_min(base_depth),
                                   rgb_changed_vs_empty=n_changed(rgb, base_rgb), rgb_change_bbox=change_bbox(rgb, base_rgb),
                                   rgb_changed_vs_prev_distance=n_changed(rgb, prev),
                                   seg_changed_vs_empty=n_changed(seg, base_seg, 0),
                                   depth_changed_px=int((np.abs(np.nan_to_num(depth, posinf=1e3) - np.nan_to_num(base_depth, posinf=1e3)) > 0.3).sum()) if depth is not None and base_depth is not None else None)
                        prev = rgb
                        if rgb is not None:
                            cv2.imwrite(s.path(f"{name}_d{int(d)}_rgb.png"), rgb)
                        if seg is not None:
                            cv2.imwrite(s.path(f"{name}_d{int(d)}_seg.png"), seg)
                    except Exception as err:  # noqa: BLE001
                        row["error"] = f"{type(err).__name__}: {err}"
                    rows.append(row)
            finally:
                info["destroyed"] = destroy(s, spawned, name)
                try:
                    rgb, _, _ = resettle(s.W, s.D)
                    info["after_destroy_rgb_changed_vs_empty"] = n_changed(rgb, base_rgb)
                except Exception as err:  # noqa: BLE001
                    info["after_destroy_rgb_changed_vs_empty"] = f"{type(err).__name__}: {err}"
            info["rows"] = rows
            return info

        s.sec(f"obstacle_{name}", t_obstacle, timeout=180.0)


# ---------------------------------------------------------------- A. dynamic obstacle
def sec_dynamic(s: Session):
    def t():
        base = s.baseline()
        spec = CATALOG["box_1m_grey"]
        x, y, z = ahead(s.start, 20.0)
        spawned, how, _ = spawn_spec(s, "mover", spec, make_pose(x, y, z, yaw=s.start[3]))
        rows = []
        try:
            for k in range(0, 31, 3):  # 5 m/s toward the drone for 3 s of sim time
                d = 20.0 - 5.0 * (k * 0.1)
                px, py, pz = ahead(s.start, d + 0.5)
                s.W.set_object_pose(spawned, make_pose(px, py, pz, yaw=s.start[3]), True)
                rgb, depth, _ = resettle(s.W, s.D, warm_steps=1)
                m = corridor_min(depth)
                rows.append({"true_front_m": round(d, 1), "corridor_min_depth_m": m, "err_m": None if m is None else round(m - d, 2)})
        finally:
            destroy(s, spawned, "mover")
        return {"how": how, "rows": rows}

    s.sec("dynamic_obstacle", t, timeout=180.0)


# ---------------------------------------------------------------- A. weather, time of day
def sec_weather(s: Session):
    from projectairsim.types import WeatherParameter as WP

    def t_weather():
        clear = resettle(s.W, s.D)
        cv2.imwrite(s.path("weather_clear_rgb.png"), clear[0])
        rows = {}

        def measure(tag, params):
            for p, v in params.items():
                s.W.set_weather_visual_effects_param(int(p), v)
            rgb, depth, seg = resettle(s.W, s.D, warm_steps=6)
            row = {"rgb_changed_vs_clear": n_changed(rgb, clear[0]), "rgb_mean_abs_diff": round(float(np.abs(rgb.astype(int) - clear[0].astype(int)).mean()), 2)}
            if depth is not None and clear[1] is not None:
                both = np.isfinite(depth) & np.isfinite(clear[1])
                row["depth_mean_abs_diff_m"] = round(float(np.abs(depth[both] - clear[1][both]).mean()), 3) if both.any() else None
                row["depth_finite_frac"] = round(float(np.isfinite(depth).mean()), 3)
            cv2.imwrite(s.path(f"weather_{tag}_rgb.png"), rgb)
            rows[tag] = row
            for p in params:
                s.W.set_weather_visual_effects_param(int(p), 0.0)

        try:
            s.W.enable_weather_visual_effects()
            for p in (WP.RAIN, WP.SNOW, WP.FOG, WP.DUST, WP.MAPLE_LEAF, WP.ROAD_WETNESS, WP.ROAD_SNOW):
                try:
                    measure(p.name.lower(), {p: 0.8})
                except Exception as err:  # noqa: BLE001
                    rows[p.name.lower()] = {"error": f"{type(err).__name__}: {err}"}
            try:
                measure("rain_fog", {WP.RAIN: 0.8, WP.FOG: 0.6})
            except Exception as err:  # noqa: BLE001
                rows["rain_fog"] = {"error": f"{type(err).__name__}: {err}"}
        finally:
            try:
                s.W.reset_weather_effects()
                s.W.disable_weather_visual_effects()
            except Exception:  # noqa: BLE001
                pass
        s.flags["weather_ok"] = any(isinstance(r, dict) and (r.get("rgb_changed_vs_clear") or 0) > 500 for r in rows.values())
        return {"rows": rows, "weather_visibly_changes_rgb": s.flags["weather_ok"]}

    s.sec("weather_effects", t_weather, timeout=300.0)

    def t_tod():
        rows = {}
        try:
            for hh in (6, 9, 12, 17, 19, 22):
                try:
                    ok = s.W.set_sun_position_from_date_time(datetime(2026, 6, 21, hh, 0, 0))
                    rgb, depth, _ = resettle(s.W, s.D, warm_steps=4)
                    rows[f"sun_{hh:02d}h"] = {"api_returned": str(ok), "rgb_mean": round(float(rgb.mean()), 1)}
                    cv2.imwrite(s.path(f"tod_{hh:02d}h_rgb.png"), rgb)
                except Exception as err:  # noqa: BLE001
                    rows[f"sun_{hh:02d}h"] = {"error": f"{type(err).__name__}: {err}"}
            for lux in (75000, 20000, 2000, 200):
                try:
                    ok = s.W.set_sunlight_intensity(float(lux))
                    rgb, _, _ = resettle(s.W, s.D, warm_steps=4)
                    rows[f"lux_{lux}"] = {"api_returned": str(ok), "rgb_mean": round(float(rgb.mean()), 1)}
                    cv2.imwrite(s.path(f"lux_{lux}_rgb.png"), rgb)
                except Exception as err:  # noqa: BLE001
                    rows[f"lux_{lux}"] = {"error": f"{type(err).__name__}: {err}"}
        finally:
            for fn in (lambda: s.W.set_sunlight_intensity(75000.0), lambda: s.W.set_sun_position_from_date_time(datetime(2026, 6, 21, 12, 0, 0))):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
        return {"rows": rows}

    s.sec("time_of_day_and_sun", t_tod, timeout=240.0)


# ---------------------------------------------------------------- A. video
def open_writer(path_base, fps, size):
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        p = path_base + ext
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, p, fourcc
        w.release()
    return None, None, None


def sec_video(s: Session):
    def fly(tag, n=60):
        x0, y0, z0, yaw0 = s.start
        writer, path, codec, t0, errs = None, None, None, time.perf_counter(), 0
        for i in range(n):
            yaw = yaw0 + math.radians(1.5) * i
            step = 0.4  # 4 m/s at 10 Hz
            x0 += step * math.cos(yaw); y0 += step * math.sin(yaw)
            s.D.set_pose(make_pose(x0, y0, z0, yaw=yaw))
            s.W.continue_for_sim_time(STEP_NS)
            rgb, depth, seg = grab(s.D)
            if rgb is None or depth is None or seg is None:
                errs += 1
                continue
            panel = np.hstack([rgb, depth_vis(depth), seg])
            if writer is None:
                writer, path, codec = open_writer(s.path(f"video_{tag}"), 10, (panel.shape[1], panel.shape[0]))
                if writer is None:
                    cv2.imwrite(s.path(f"video_{tag}_frame0.png"), panel)
                    return {"error": "no video codec opened (mp4v/MJPG); wrote frame0.png instead"}
            writer.write(panel)
        if writer is not None:
            writer.release()
        wall = time.perf_counter() - t0
        return {"file": os.path.basename(path), "codec": codec, "bytes": os.path.getsize(path), "frames": n - errs, "missing_frames": errs,
                "wall_seconds": round(wall, 1), "wall_s_per_step": round(wall / n, 3)}

    def t_clear():
        out = fly("clear")
        if "wall_s_per_step" in out:  # the number the compute budget needs
            out["est_wall_minutes_for_300_step_episode"] = round(out["wall_s_per_step"] * 300 / 60, 1)
        return out

    s.sec("video_clear", t_clear, timeout=300.0)

    def t_weather():
        from projectairsim.types import WeatherParameter as WP
        if not s.flags.get("weather_ok"):
            return {"skipped_reason": "weather did not visibly change RGB"}
        try:
            s.W.enable_weather_visual_effects()
            s.W.set_weather_visual_effects_param(int(WP.RAIN), 0.8)
            s.W.set_weather_visual_effects_param(int(WP.FOG), 0.6)
            return fly("rain_fog")
        finally:
            try:
                s.W.reset_weather_effects(); s.W.disable_weather_visual_effects()
            except Exception:  # noqa: BLE001
                pass

    s.sec("video_rain_fog", t_weather, timeout=300.0)


# ---------------------------------------------------------------- B. noisy depth
def sec_clean_reference(s: Session):
    """Clean depth at the final start pose, for the noisy-depth comparison in the next session."""
    def t():
        _, depth, _ = settle_grab(s.W, s.D, s.start_pose())
        AUX["clean_depth_400"] = depth
        return {"depth": depth_stats(depth), "start_pose": list(s.start)}

    s.sec("clean_depth_reference", t)


def sec_noisy_depth(s: Session):
    def t():
        a = settle_grab(s.W, s.D, s.start_pose())
        b = resettle(s.W, s.D)
        out = {"depth": depth_stats(a[1])}
        if a[1] is not None and b[1] is not None:
            ok = np.isfinite(a[1]) & np.isfinite(b[1])
            out["temporal_depth_abs_diff_mean_m"] = round(float(np.abs(a[1][ok] - b[1][ok]).mean()), 4) if ok.any() else None
        clean = AUX.get("clean_depth_400")
        if clean is not None and a[1] is not None and clean.shape == a[1].shape:
            ok = np.isfinite(a[1]) & np.isfinite(clean)
            out["vs_clean_session_abs_diff_mean_m"] = round(float(np.abs(a[1][ok] - clean[ok]).mean()), 4) if ok.any() else None
        if a[1] is not None:
            cv2.imwrite(s.path("noisy_depth_vis.png"), depth_vis(a[1]))
        return out

    s.sec("noisy_depth_effect", t, timeout=120.0)


# ---------------------------------------------------------------- A2. frame bank (long run)
BANK_CONDS = ["clear", "clear", "clear", "rain", "fog", "snow", "dusk"]
BANK_OBSTACLES = ["none", "none", "pole_10cm_axis_z", "wire_3cm_axis_z", "wire_1cm_axis_z", "box_1m_grey", "box_1m_black",
                  "wall_white_featureless", "glass_pane_alpha20", "building_block", "ball_20cm_red"]


def sec_frame_bank(s: Session):
    """Many short straight flights under varied conditions, every frame saved with ground truth for later analysis.

    Exploratory data for understanding failure modes. Episodes use seeds 10000+ep, which are NOT the seeds in
    configs/seeds.yaml, and must not be used as train/test data without a split by episode. Ground truth
    (distance to the obstacle) is stored only as labels. Resumable: finished episodes (episodes.jsonl) are skipped.
    """
    from projectairsim.types import WeatherParameter as WP

    a = STATE["args"]
    bank = a.bank_dir or s.path("bank")
    os.makedirs(os.path.join(bank, "frames"), exist_ok=True)
    ep_file, fr_file = os.path.join(bank, "episodes.jsonl"), os.path.join(bank, "frames.jsonl")
    done = set()
    if os.path.exists(ep_file):
        with open(ep_file) as f:
            done = {json.loads(line)["episode"] for line in f if line.strip()}

    def set_condition(cond):
        if cond == "rain":
            s.W.enable_weather_visual_effects(); s.W.set_weather_visual_effects_param(int(WP.RAIN), 0.8)
        elif cond == "fog":
            s.W.enable_weather_visual_effects(); s.W.set_weather_visual_effects_param(int(WP.FOG), 0.7)
        elif cond == "snow":
            s.W.enable_weather_visual_effects(); s.W.set_weather_visual_effects_param(int(WP.SNOW), 0.8)
        elif cond == "dusk":
            s.W.set_sun_position_from_date_time(datetime(2026, 6, 21, 19, 30, 0))

    def reset_condition():
        for fn in (s.W.reset_weather_effects, s.W.disable_weather_visual_effects,
                   lambda: s.W.set_sun_position_from_date_time(datetime(2026, 6, 21, 12, 0, 0))):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def run_episode(ep):
        rng = np.random.default_rng(10_000 + ep)
        cond = BANK_CONDS[int(rng.integers(len(BANK_CONDS)))]
        obs_name = BANK_OBSTACLES[int(rng.integers(len(BANK_OBSTACLES)))]
        sug = s.flags.get("suggested") or []
        if sug:
            b = sug[int(rng.integers(len(sug)))]
            yaws = b["open_yaws_deg"] or [0]
            x0, y0, yaw = b["x"], b["y"], math.radians(float(yaws[int(rng.integers(len(yaws)))]))
        else:
            x0, y0, _, yaw = s.start
            x0 += float(rng.uniform(-3, 3)); y0 += float(rng.uniform(-3, 3))
        z0 = float(rng.uniform(-4.0, -1.5))
        info = {"episode": ep, "condition": cond, "obstacle": obs_name, "start": [round(x0, 2), round(y0, 2), round(z0, 2), round(yaw, 3)]}
        spawned, rows, t0, epoch = None, [], time.time(), s.epoch
        try:
            set_condition(cond)
            lateral = 0.0
            if obs_name != "none":
                dist0 = float(rng.uniform(8, 25))
                lateral = float(rng.choice([0.0, 0.0, 1.2, -1.2, 3.0, -3.0]))
                ox = x0 + dist0 * math.cos(yaw) - lateral * math.sin(yaw)
                oy = y0 + dist0 * math.sin(yaw) + lateral * math.cos(yaw)
                spawned, how, _ = spawn_spec(s, f"bank_{ep}", CATALOG[obs_name], make_pose(ox, oy, z0, yaw=yaw))
                info.update(obstacle_how=how, obstacle_xy=[round(ox, 2), round(oy, 2)], obstacle_lateral_m=lateral, obstacle_start_dist_m=round(dist0, 1))
                try:
                    s.W.set_segmentation_id_by_name(spawned, 100, False, True)
                except Exception:  # noqa: BLE001
                    pass
            x, y, hd = x0, y0, yaw
            s.D.set_pose(make_pose(x, y, z0, yaw=hd))
            s.W.continue_for_sim_time(6 * STEP_NS)
            grab(s.D)
            for k in range(a.bank_steps):
                hd += math.radians(float(rng.normal(0, 0.5)))
                x += 0.4 * math.cos(hd); y += 0.4 * math.sin(hd)  # 4 m/s at 10 Hz
                s.D.set_pose(make_pose(x, y, z0, yaw=hd))
                s.W.continue_for_sim_time(STEP_NS)
                rgb, depth, seg = grab(s.D)
                if rgb is None or depth is None or seg is None:
                    rows.append({"episode": ep, "step": k, "error": "missing channel"})
                    continue
                stem = f"{ep:04d}_{k:03d}"
                cv2.imwrite(os.path.join(bank, "frames", stem + "_rgb.png"), rgb)
                cv2.imwrite(os.path.join(bank, "frames", stem + "_seg.png"), seg)
                np.save(os.path.join(bank, "frames", stem + "_depth.npy"), depth.astype(np.float16))
                row = {"episode": ep, "step": k, "stem": stem, "pose": [round(x, 3), round(y, 3), round(z0, 3), round(hd, 4)],
                       "condition": cond, "obstacle": obs_name, "corridor_min_depth_m": corridor_min(depth), "sim_time_ns": s.W.get_sim_time()}
                if spawned:  # ground truth, for labels and analysis only
                    rx, ry = info["obstacle_xy"][0] - x, info["obstacle_xy"][1] - y
                    along = rx * math.cos(hd) + ry * math.sin(hd)
                    lat = -rx * math.sin(hd) + ry * math.cos(hd)
                    row.update(gt_along_m=round(along, 2), gt_lateral_m=round(lat, 2))
                    if along < 0.5:
                        rows.append(row)
                        info["reached_obstacle_at_step"] = k
                        break
                rows.append(row)
        finally:
            if spawned:
                destroy(s, spawned, f"bank_{ep}")
            reset_condition()
            try:
                s.W.continue_for_sim_time(3 * STEP_NS)
            except Exception:  # noqa: BLE001
                pass
        if s.epoch != epoch:  # this episode timed out and the session was rebuilt meanwhile: discard
            raise RuntimeError("episode abandoned after a timeout/reconnect")
        info.update(frames=sum(1 for r in rows if "stem" in r), errors=sum(1 for r in rows if "error" in r), wall_s=round(time.time() - t0, 1))
        with open(fr_file, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(ep_file, "a") as f:  # written last: a crash mid-episode means it is redone
            f.write(json.dumps(info) + "\n")
        return info

    def t():
        if not s.assets:
            try:
                s.assets = [str(x) for x in s.W.list_assets(".*")]
            except Exception:  # noqa: BLE001
                pass
        new, frames, failed, t_start = 0, 0, [], time.time()
        for ep in range(a.bank_episodes):
            if ep in done:
                continue
            if time.time() > STATE["deadline"] - 90:
                break
            # each episode is its own guarded section: a hang costs one episode (timeout + reconnect), not the run
            ok = s.sec("bank_episode", lambda ep=ep: run_episode(ep), timeout=300.0, force=True)
            if ok:
                new += 1
                frames += s.store["bank_episode"].get("frames", 0)
                if new % 5 == 0:
                    print(f"[frame_bank] {new} episodes, {frames} frames, {round(time.time() - t_start)} s", flush=True)
            else:
                failed.append(f"episode {ep}: {s.store['bank_episode'].get('error') or s.store['bank_episode'].get('skipped')}")
                if not s.usable or (len(failed) >= 5 and new == 0):
                    break
        return {"dir": bank, "episodes_requested": a.bank_episodes, "already_done": len(done), "new_episodes": new, "new_frames": frames,
                "failed": failed[:10], "stopped_early": (not s.usable) or (len(failed) >= 5 and new == 0), "wall_seconds": round(time.time() - t_start, 1)}

    try:
        out = t()
        s.store["frame_bank"] = {"ok": out["new_episodes"] > 0 or out["already_done"] > 0, **out}
    except Exception as err:  # noqa: BLE001
        s.store["frame_bank"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
    print(f"[frame_bank] {json.dumps({k: v for k, v in s.store['frame_bank'].items() if k != 'trace'}, default=str)[:500]}", flush=True)
    dump()


# ---------------------------------------------------------------- orchestration
def run_session(client, configs, scene_file, key, plan, start=None):
    s = Session(client, configs, STATE["out_dir"], scene_file, key)
    if start:
        s.start = start
    if not s.sec("load_scene", s.load, timeout=240.0):
        return s
    try:
        s.W.pause()
    except Exception as err:  # noqa: BLE001
        s.store["initial_pause"] = {"ok": False, "error": str(err)}
    for fn in plan:
        try:
            fn(s)
        except Exception as err:  # noqa: BLE001  (a bug in the test code must not lose later sections)
            s.store[f"harness_error_{getattr(fn, '__name__', 'fn')}"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
            dump()
    return s


def run(args, client=None):
    STATE.update(out_dir=args.out, deadline=time.time() + args.budget_sec, args=args)
    args.start_pose = tuple(float(v) for v in args.start.split(",")) if args.start else None
    if args.start_pose:
        args.start_pose = (*args.start_pose[:3], math.radians(args.start_pose[3]))
    args.only = set(filter(None, (args.only or "").split(",")))
    args.skip = set(filter(None, (args.skip or "").split(",")))
    os.makedirs(args.out, exist_ok=True)
    RESULTS["meta"] = {"env": args.env, "started": datetime.now().isoformat(timespec="seconds"), "budget_sec": args.budget_sec}
    client = client or ProjectAirSimClient()

    def on_term(signum, _frame):
        RESULTS["terminated"] = {"ok": False, "error": f"received signal {signum}; partial results"}
        dump()
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    try:
        connect_with_retry(client)
        plans = [
            ("scene_cap_400.jsonc", "400px", [sec_basics, sec_depth_range, sec_scene_scan, sec_obstacles, sec_dynamic, sec_weather, sec_video, sec_clean_reference]),
            ("scene_cap_400n.jsonc", "400px_noisy_depth", [sec_noisy_depth]),
            ("scene_cap_1280.jsonc", "1280px", [lambda s: sec_basics(s, n_loop=20), lambda s: sec_obstacles(s, only_thin=True)]),
            ("scene_cap_400.jsonc", "400px_frame_bank", [sec_frame_bank]),  # last: long, uses the remaining time budget
        ]
        carry = None
        for scene_file, key, plan in plans:
            try:
                s = run_session(client, args.configs, scene_file, key, plan, start=carry)
                if key == "400px":
                    carry = s.start  # later sessions reuse the start pose chosen by the scan
            except Exception as err:  # noqa: BLE001
                RESULTS.setdefault(key, {})["session_crash"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-1500:]}
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
        if isinstance(secs, dict) and secs and all(isinstance(v, dict) for v in secs.values()):
            print(res, {k: ("ok" if v.get("ok") else ("SKIP" if "skipped" in v else "FAIL: " + str(v.get("error", ""))[:90])) for k, v in secs.items()})
        else:
            print(res, secs)
    return RESULTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="sim_config dir written by make_configs.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="Blocks", help="label only; the running binary decides the map")
    parser.add_argument("--budget-sec", type=float, default=5400.0, help="stop starting new work after this many seconds; keep >= 10 min below the sbatch --time minus boot/load")
    parser.add_argument("--start", default=None, help="x,y,z,yaw_deg start pose (NED); default: scan picks one (z=-2)")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false", help="keep the default start pose instead of the scan's best")
    parser.add_argument("--bank-episodes", type=int, default=100000, help="frame_bank: max number of flights; in practice it runs until the time budget ends (resumable)")
    parser.add_argument("--bank-steps", type=int, default=80, help="frame_bank: max steps per flight (10 Hz sim time)")
    parser.add_argument("--bank-dir", default=None, help="frame_bank output dir; use a persistent path to resume across jobs")
    parser.add_argument("--scan-extent", type=float, default=60.0, help="half-width in metres of the scene scan grid")
    parser.add_argument("--only", default="", help="comma list of section names to run")
    parser.add_argument("--skip", default="", help="comma list of section names to skip")
    run(parser.parse_args())
    os._exit(0)  # a timed-out request thread may still be blocked
