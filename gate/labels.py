"""Gate labels from saved frames (PLAN_DETAILED.md §4.3, task W-B3). L1 only for now.

L1, "missed obstacle": the ground truth says something is inside ``d_safe`` in
the forward corridor, while the fast stack's estimate says the corridor is
clear. The controller would continue when it should avoid.

Labels use ground truth, so they are computed offline from the flight log and
never at run time (CLAUDE.md rule 5).
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import numpy as np

from perception.depth_utils import CAMERA_HFOV_DEG, CORRIDOR_COLS, MAX_RANGE_M, clip_depth, corridor_window

# ---- ground truth for obstacles the cameras cannot see -------------------------------------
# Glass is see-through to both the depth and the segmentation cameras in the frame bank (SB,
# job 33462: no mask pixels in any of 50 glass frames, depth reads the background). For these
# types the obstacle's distance comes from the episode metadata (gt_along_m, gt_lateral_m)
# instead of the images.
METADATA_GT_TYPES = ("glass_pane_alpha20", "glass_pane_alpha50")

# Half of the smaller in-plane size of each spawned obstacle, metres (CATALOG in
# hpc/capability/capability_test.py). Poles and wires use half their length: both "axis_z"
# types render as horizontal bars at camera height, not vertical posts (job 33462 sheets).
HALF_WIDTH_M = {
    "ball_20cm_red": 0.1, "box_1m_black": 0.5, "box_1m_grey": 0.5, "box_1m_white": 0.5,
    "building_block": 6.0, "glass_pane_alpha20": 1.25, "glass_pane_alpha50": 1.25,
    "wall_white_featureless": 1.5, "pole_10cm_axis_z": 2.5, "wire_1cm_axis_z": 3.0, "wire_3cm_axis_z": 3.0,
}


def window_half_width(along_m: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Half-width in metres of the corridor window at ``along_m`` ahead of the camera."""
    half_frac = (CORRIDOR_COLS[1] - CORRIDOR_COLS[0]) / 2  # of the full image width
    return half_frac * 2 * along_m * math.tan(math.radians(hfov_deg) / 2)


def metadata_in_window(obstacle: str, along_m: float | None, lateral_m: float | None, hfov_deg: float = CAMERA_HFOV_DEG) -> bool | None:
    """Whether the episode metadata puts the obstacle inside the corridor window; None without metadata."""
    if along_m is None or lateral_m is None or obstacle not in HALF_WIDTH_M:
        return None
    return along_m > 0 and abs(lateral_m) < HALF_WIDTH_M[obstacle] + window_half_width(along_m, hfov_deg)


def with_metadata_fallback(gt: dict[str, Any], frame: dict[str, Any], hfov_deg: float = CAMERA_HFOV_DEG) -> dict[str, Any]:
    """Add ``meta_in_window`` and ``gt_source`` to a ``clean_corridor_min`` result; for see-through types, take the
    obstacle distance from the metadata. ``frame`` is a frames.jsonl row."""
    out = dict(gt)
    obstacle = str(frame.get("obstacle"))
    along, lateral = frame.get("gt_along_m"), frame.get("gt_lateral_m")
    meta_in = metadata_in_window(obstacle, along, lateral, hfov_deg)
    out["meta_in_window"] = meta_in
    out["gt_source"] = "seg"
    if obstacle in METADATA_GT_TYPES and meta_in is not None:
        out["gt_source"] = "metadata"
        out["corridor_min_gt_obstacle"] = float(along) if meta_in else float(MAX_RANGE_M)
        out["corridor_min_gt_clean"] = min(out["corridor_min_gt_obstacle"], float(out["corridor_min_gt_scene"]))
    return out


class Outcome(IntEnum):
    """How the fast stack's corridor reading compares with ground truth."""

    AGREE_CLEAR = 0  # both say clear
    MISSED = 1  # GT says obstacle, estimate says clear: the L1 positive
    FALSE_STOP = 2  # GT says clear, estimate says obstacle: logged, not a positive
    AGREE_AVOID = 3  # both say obstacle


def outcomes(
    corridor_min_gt: np.ndarray | float,
    corridor_min_est: np.ndarray | float,
    d_safe: float,
) -> np.ndarray:
    """``Outcome`` code per frame. Inputs are corridor minimum depths in metres.

    Both inputs must come from ``perception.depth_utils.corridor_min`` so that
    they share the same clipping and the same corridor.
    """
    gt = np.asarray(corridor_min_gt, dtype=np.float64)
    est = np.asarray(corridor_min_est, dtype=np.float64)
    if np.isnan(gt).any() or np.isnan(est).any():
        raise ValueError("corridor minimum depth contains NaN; clip depth before labelling")
    gt_obstacle = gt < d_safe
    est_obstacle = est < d_safe
    out = np.full(gt.shape, Outcome.AGREE_CLEAR, dtype=np.int8)
    out[gt_obstacle & ~est_obstacle] = Outcome.MISSED
    out[~gt_obstacle & est_obstacle] = Outcome.FALSE_STOP
    out[gt_obstacle & est_obstacle] = Outcome.AGREE_AVOID
    return out


def l1_labels(
    corridor_min_gt: np.ndarray | float,
    corridor_min_est: np.ndarray | float,
    d_safe: float,
) -> np.ndarray:
    """L1 gate label per frame: 1 for a missed obstacle, else 0."""
    return (outcomes(corridor_min_gt, corridor_min_est, d_safe) == Outcome.MISSED).astype(np.int8)


def clean_corridor_min(
    depth_gt: np.ndarray, obstacle_mask: np.ndarray, max_range: float = MAX_RANGE_M, scene_percentile: float = 2.0
) -> dict[str, float | int]:
    """Ground-truth corridor distance that ignores weather particles.

    Rain particles are in the simulator's depth image but not in its
    segmentation image, so a plain minimum reports "obstacle at 2 m" in an empty
    scene. Here the spawned obstacle (``obstacle_mask``) contributes its true
    minimum, so a one-pixel wire still counts, while the rest of the scene
    contributes a low percentile, which a few particle pixels cannot move.
    """
    rows, cols = corridor_window(depth_gt.shape)
    corridor = clip_depth(depth_gt, max_range)[rows, cols]
    mask = obstacle_mask[rows, cols]
    obstacle = float(corridor[mask].min()) if mask.any() else float(max_range)
    rest = corridor[~mask]
    scene = float(np.percentile(rest, scene_percentile)) if rest.size else float(max_range)
    return {
        "corridor_min_gt_obstacle": obstacle,
        "corridor_min_gt_scene": scene,
        "corridor_min_gt_clean": min(obstacle, scene),
        "obstacle_px": int(mask.sum()),
    }
