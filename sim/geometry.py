"""Obstacle geometry: oriented boxes, ray casting and swept collision (PLAN_DETAILED.md D5, chunk M4.4).

World frame is NED, as in Project AirSim: x north, y east, z down, so altitude is
``-z`` and the ground is the plane ``z = 0``. Every obstacle is an oriented box
(``OBB``) with a yaw; pitch and roll are never used.

Collision is our own check against known geometry (D5: thin wires may not
trigger engine colliders). The drone is a sphere of radius ``r``; a step is a
segment from the old to the new position, and it collides if the segment meets
the box inflated by ``r``. Inflating the box (rather than taking the exact
Minkowski sum with rounded corners) is slightly conservative at the corners,
which is the safe side for a crash count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OBB:
    """Oriented box: centre (NED, metres), yaw (rad), half extents along its own forward, lateral and down axes."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float = 0.0
    name: str = ""
    family: str = ""
    see_through: bool = False  # invisible to depth and segmentation (glass), but solid for collisions

    def rotation(self) -> np.ndarray:
        """Columns are the box's forward, lateral and down axes in world coordinates."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_local(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, dtype=np.float64) - np.asarray(self.center)) @ self.rotation()

    def inflated(self, r: float) -> OBB:
        return OBB(self.center, tuple(h + r for h in self.half), self.yaw, self.name, self.family, self.see_through)


def ray_box_local(o: np.ndarray, d: np.ndarray, half: np.ndarray, t_max: np.ndarray | float = np.inf) -> np.ndarray:
    """Slab test in box coordinates. ``o`` (3,) or (N,3), ``d`` (N,3). Returns entry distance per ray, ``inf`` if missed.

    A ray starting inside the box returns 0.
    """
    o = np.broadcast_to(np.asarray(o, dtype=np.float64), d.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (-half - o) * inv
        t2 = (half - o) * inv
    lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
    # A ray parallel to a slab (d = 0) is inside it for all t if the origin is between the planes, else never:
    # the slab's interval is then (-inf, inf) or empty (+inf, -inf). Set after min/max, or "empty" becomes "everything".
    par = d == 0
    inside = np.abs(o) <= half
    lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
    hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
    t_near = np.max(lo, axis=-1)
    t_far = np.min(hi, axis=-1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (np.maximum(t_near, 0.0) <= t_max)
    return np.where(hit, np.maximum(t_near, 0.0), np.inf)


def ray_obb(origin: np.ndarray, dirs: np.ndarray, box: OBB) -> np.ndarray:
    """Distance along each unit ray in ``dirs`` (N,3) from ``origin`` (3,) to ``box``; ``inf`` if missed."""
    R = box.rotation()
    return ray_box_local(box.to_local(origin), np.asarray(dirs) @ R, np.asarray(box.half, dtype=np.float64))


def ray_ground(origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Distance to the ground plane z = 0 (NED) along each ray; ``inf`` for rays that point up or level."""
    oz = float(origin[2])
    dz = np.asarray(dirs)[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -oz / dz
    return np.where((dz > 0) & (t > 0), t, np.inf)


def segment_hits(p0: np.ndarray, p1: np.ndarray, box: OBB, radius: float) -> bool:
    """Does a sphere of ``radius`` moving from ``p0`` to ``p1`` touch ``box``? Exact for the inflated box."""
    b = box.inflated(radius)
    a, c = b.to_local(p0), b.to_local(p1)
    seg = c - a
    length = float(np.linalg.norm(seg))
    half = np.asarray(b.half, dtype=np.float64)
    if length < 1e-12:
        return bool(np.all(np.abs(a) <= half))
    t = ray_box_local(a, (seg / length)[None, :], half, t_max=length)
    return bool(np.isfinite(t[0]))


def hits_ground(p0: np.ndarray, p1: np.ndarray, radius: float) -> bool:
    """Sphere below the ground at either end of the step (z is down, ground at 0)."""
    return bool(max(float(p0[2]), float(p1[2])) > -radius)


def best_side(position: np.ndarray, yaw: float, box: OBB, r: float) -> str:
    """Which way past ``box`` needs the smallest move: "left", "right" or "over" (ground truth, for the surrogate only).

    In the drone's frame (forward along ``yaw``, lateral positive to the right, NED), the box's lateral extent is
    projected onto the drone's lateral axis; passing left means moving past its left edge, right past its right edge,
    over means rising above its top; each by the drone radius ``r``. Evaluation-side helper (CLAUDE.md rule 5).
    """
    p = np.asarray(position, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(box.center, dtype=np.float64) - p
    lat = -s * d[0] + c * d[1]  # box centre to the right of the drone
    rel = box.yaw - yaw
    half_lat = abs(box.half[0] * np.sin(rel)) + abs(box.half[1] * np.cos(rel))
    top_alt = -(box.center[2] - box.half[2])  # NED: the box top is centre minus half-height in z
    # to pass on the left the drone must end up at lateral <= lat - half_lat - r, on the right at >= lat + half_lat + r
    need = {"left": half_lat + r - lat, "right": lat + half_lat + r, "over": top_alt - (-p[2]) + r}
    return min(need, key=lambda k: (need[k], ("over", "left", "right").index(k)))
