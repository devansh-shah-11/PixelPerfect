"""Plan B simulator: a raycast renderer over the scenario geometry (PLAN_DETAILED.md §11, chunk A-07 / M4 fallback).

Loses photorealism, gains speed, exact geometry and laptop portability. It
renders what the flight loop needs, with the same conventions as the Project
AirSim frame bank, so code written against one runs on the other:

- ``depth``: distance along each pixel's ray (DEPTH_PERSPECTIVE), float32 metres, sky saturated at 65504;
- ``seg``: spawned obstacles in ``perception.frame_bank.OBSTACLE_SEG_RGB``, everything else black;
- glass (``OBB.see_through``) is invisible to depth and segmentation, as it is in Project AirSim, but
  tints the RGB image and is solid for collisions;
- ``rgb``: flat-shaded sky gradient, checkered ground, obstacles in their catalog colours; "dusk" darkens.

Any result produced on this backend says so ("raycast" in the run config); it
is a fallback and a test bed, not the realistic simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from perception.depth_utils import CAMERA_HFOV_DEG
from perception.frame_bank import OBSTACLE_SEG_RGB
from sim.drone import KinematicState
from sim.geometry import OBB, ray_ground, ray_obb
from sim.scenario import CATALOG, Scenario

SKY_DEPTH = 65504.0


@dataclass
class Observation:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, ray distance, glass not visible
    seg: np.ndarray  # (H, W, 3) uint8


class Camera:
    """Level pinhole camera at the drone, looking along its heading. Square pixels, horizontal FOV ``hfov``."""

    def __init__(self, width: int = 400, height: int = 225, hfov_deg: float = CAMERA_HFOV_DEG) -> None:
        self.w, self.h = width, height
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        u = (np.arange(width) + 0.5 - width / 2) / f
        v = (np.arange(height) + 0.5 - height / 2) / f
        uu, vv = np.meshgrid(u, v)
        d = np.stack([np.ones_like(uu), uu, vv], axis=-1)  # camera frame: forward, right, down
        self._dirs_cam = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)

    def rays(self, yaw: float, rows: slice | None = None, cols: slice | None = None) -> np.ndarray:
        """Unit ray directions in NED world coordinates, (N, 3), optionally for a pixel window only."""
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # camera (fwd, right, down) -> NED
        d = self._dirs_cam.reshape(self.h, self.w, 3)
        if rows is not None or cols is not None:
            d = d[rows or slice(None), cols or slice(None)]
        return d.reshape(-1, 3) @ R.T


def first_hits(origin: np.ndarray, dirs: np.ndarray, boxes: list[OBB]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest distance per ray over ``boxes`` and the index of the box hit (-1 for none)."""
    best = np.full(len(dirs), np.inf)
    idx = np.full(len(dirs), -1)
    for i, b in enumerate(boxes):
        t = ray_obb(origin, dirs, b)
        closer = t < best
        best[closer], idx[closer] = t[closer], i
    return best, idx


class RaycastSim:
    """``reset(scenario)`` then ``observe(state)``. Same interface as the Project AirSim backend."""

    name = "raycast"

    def __init__(self, camera: Camera | None = None) -> None:
        self.cam = camera or Camera()
        self.scenario: Scenario | None = None
        self.boxes: list[OBB] = []

    def reset(self, scenario: Scenario) -> None:
        self.scenario, self.boxes = scenario, scenario.obbs()

    def render_depth(self, state: KinematicState, keep: Callable[[OBB], bool] = lambda b: not b.see_through) -> np.ndarray:
        """Ray-distance depth with only the boxes ``keep`` accepts (default: everything the depth camera sees)."""
        dirs = self.cam.rays(state.yaw)
        o = state.position
        t_obj, _ = first_hits(o, dirs, [b for b in self.boxes if keep(b)])
        t = np.minimum(t_obj, ray_ground(o, dirs))
        return np.where(np.isfinite(t), t, SKY_DEPTH).astype(np.float32).reshape(self.cam.h, self.cam.w)

    def observe(self, state: KinematicState) -> Observation:
        dirs = self.cam.rays(state.yaw)
        o = state.position
        opaque = [b for b in self.boxes if not b.see_through]
        glass = [b for b in self.boxes if b.see_through]
        t_obj, idx = first_hits(o, dirs, opaque)
        t_gnd = ray_ground(o, dirs)
        t = np.minimum(t_obj, t_gnd)
        depth = np.where(np.isfinite(t), t, SKY_DEPTH).astype(np.float32)

        rgb = np.empty((len(dirs), 3))
        sky = ~np.isfinite(t)
        up = np.clip(-dirs[:, 2], 0, 1)[:, None]
        rgb[sky] = (np.array([0.75, 0.82, 0.92]) * (1 - up[sky]) + np.array([0.35, 0.5, 0.85]) * up[sky])
        gnd = np.isfinite(t_gnd) & (t_gnd <= t_obj)
        p = o + dirs[gnd] * t_gnd[gnd, None]
        checker = ((np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2)[:, None]
        rgb[gnd] = np.array([0.42, 0.45, 0.40]) * (0.85 + 0.15 * checker)
        obj = np.isfinite(t_obj) & (t_obj < t_gnd)
        colours = np.array([CATALOG[b.family][1] for b in opaque]) if opaque else np.zeros((0, 3))
        shade = 0.75 + 0.25 * np.clip(-dirs[obj, 2] + 0.5, 0, 1)[:, None]
        rgb[obj] = colours[idx[obj]] * shade
        if glass:  # tint where a pane lies in front of whatever the ray hits
            t_glass, _ = first_hits(o, dirs, glass)
            tint = t_glass < t
            rgb[tint] = 0.7 * rgb[tint] + 0.3 * np.array([0.7, 0.85, 0.95])
        fog = np.clip(np.where(np.isfinite(t), t, 200.0) / 200.0, 0, 0.6)[:, None]
        rgb = rgb * (1 - fog) + np.array([0.8, 0.82, 0.85]) * fog
        if self.scenario is not None and self.scenario.condition == "dusk":
            rgb *= 0.4
        seg = np.zeros((len(dirs), 3), np.uint8)
        seg[obj] = OBSTACLE_SEG_RGB
        H, W = self.cam.h, self.cam.w
        return Observation(rgb=(np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(H, W, 3),
                           depth=depth.reshape(H, W), seg=seg.reshape(H, W, 3))


def corridor_truth(state: KinematicState, boxes: list[OBB], cam: Camera) -> tuple[float, str | None]:
    """Exact nearest solid surface inside the corridor window, glass included, and the family it belongs to.

    Ground truth from the scenario geometry, so it is the same on every backend. Offline / labels only.
    """
    from perception.depth_utils import MAX_RANGE_M, corridor_window

    rows, cols = corridor_window((cam.h, cam.w))
    dirs = cam.rays(state.yaw, rows, cols)
    t_obj, idx = first_hits(state.position, dirs, boxes)
    t = np.minimum(t_obj, ray_ground(state.position, dirs))
    k = int(np.argmin(t))
    d = float(t[k])
    if not np.isfinite(d) or d > MAX_RANGE_M:
        return float(MAX_RANGE_M), None
    return d, (boxes[idx[k]].family if t_obj[k] <= t[k] and idx[k] >= 0 else "ground")


def scenery_truth(depth_gt: np.ndarray, state: KinematicState, boxes: list[OBB], cam: Camera,
                  tube_m: float = 0.8, placed_tol_m: float = 0.3) -> tuple[float, str | None]:
    """Nearest map surface (tree, post, building, rising ground) in the drone's path, from the simulator's true depth.

    ``corridor_truth`` only knows the placed obstacles and a flat ground plane, so the surrogate VLM and the oracle
    could not see map scenery. Here each corridor-window pixel of the true depth image (distance along the ray, as
    both backends give it) becomes a 3D point; it counts when it lies within ``tube_m`` of the drone's height and of
    its heading line (the path the drone would sweep) and in front of any placed obstacle on that ray (else the pixel
    is that obstacle). Flat ground below a drone flying higher than ``tube_m`` is left out by the height test; ground
    rising into the path counts, as it should. Returns (distance, side to pass it: "left", "right" or "over"), or
    (inf, None). Offline / evaluation only, like ``corridor_truth`` (rule 5).
    """
    from perception.depth_utils import MAX_RANGE_M, corridor_window

    rows, cols = corridor_window((cam.h, cam.w))
    d = np.asarray(depth_gt, dtype=np.float64)[rows, cols].reshape(-1)
    dirs = cam.rays(state.yaw, rows, cols)
    ok = np.isfinite(d) & (d > 0) & (d < MAX_RANGE_M)
    if not ok.any():
        return float("inf"), None
    t_box, _ = first_hits(state.position, dirs, boxes)
    ok &= d < t_box - placed_tol_m
    p = state.position + dirs * d[:, None]
    rel = p - state.position
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    lateral = -s * rel[:, 0] + c * rel[:, 1]  # right of the heading is positive (NED)
    ok &= (np.abs(rel[:, 2]) <= tube_m) & (np.abs(lateral) <= tube_m)
    if not ok.any():
        return float("inf"), None
    near = ok & (d <= float(d[ok].min()) + 1.0)  # the surface nearest the drone, 1 m deep
    side = "left" if float(np.mean(lateral[near])) > 0 else "right"  # pass on the side it leaves free
    return float(d[ok].min()), side
