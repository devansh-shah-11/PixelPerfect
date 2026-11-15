"""Gate labels from saved frames (PLAN_DETAILED.md §4.3, task W-B3). L1 only for now.

L1, "missed obstacle": the ground truth says something is inside ``d_safe`` in
the forward corridor, while the fast stack's estimate says the corridor is
clear. The controller would continue when it should avoid.

Labels use ground truth, so they are computed offline from the flight log and
never at run time (CLAUDE.md rule 5).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from perception.depth_utils import MAX_RANGE_M, clip_depth, corridor_window


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
