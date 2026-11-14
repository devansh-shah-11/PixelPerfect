"""Depth clipping and forward-corridor statistics (tasks W-S2 and B-04).

Used on both ground-truth depth (labels and metrics) and estimated depth (run
time), so the two are always compared under the same rules.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Simulator depth saturates on sky (65504 m in CityEnviron, 16416 m in
# Neighborhood; hpc/instructions.md §3.5). Nothing past this range matters to
# the controller, so everything farther counts as "far".
MAX_RANGE_M = 100.0

# Forward corridor as a fixed central window, as fractions of image height and
# width. Same window as corridor_min() in hpc/capability/capability_test.py, so
# values match `corridor_min_depth_m` in the frame bank's frames.jsonl.
# PLAN.md §2 defines the corridor as the projected flight tube; this window is
# the MVP stand-in until the camera intrinsics and T_h are fixed.
CORRIDOR_ROWS = (0.4, 0.6)
CORRIDOR_COLS = (0.45, 0.55)


def valid_depth_mask(depth: np.ndarray, max_range: float = MAX_RANGE_M) -> np.ndarray:
    """True where depth is a real surface: finite, positive and within ``max_range``."""
    return np.isfinite(depth) & (depth > 0) & (depth <= max_range)


def clip_depth(depth: np.ndarray, max_range: float = MAX_RANGE_M) -> np.ndarray:
    """Depth in metres as float32, with sky, saturated and invalid pixels set to ``max_range``.

    Invalid pixels (NaN, inf, zero or negative) are treated as far, not near, so
    a missing reading never looks like an obstacle.
    """
    out = np.asarray(depth, dtype=np.float32).copy()
    out[~valid_depth_mask(out, max_range)] = max_range
    return out


def corridor_window(shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Row and column slices of the forward corridor for an image of ``shape``."""
    h, w = shape[:2]
    rows = slice(int(CORRIDOR_ROWS[0] * h), int(CORRIDOR_ROWS[1] * h))
    cols = slice(int(CORRIDOR_COLS[0] * w), int(CORRIDOR_COLS[1] * w))
    return rows, cols


def corridor_min(depth: np.ndarray, max_range: float = MAX_RANGE_M) -> float:
    """Nearest depth in the forward corridor, in metres; ``max_range`` if nothing is in range."""
    return float(clip_depth(depth, max_range)[corridor_window(depth.shape)].min())


def corridor_features(depth: np.ndarray, max_range: float = MAX_RANGE_M) -> dict[str, Any]:
    """Scalar corridor statistics for the gate and the threshold baseline (PLAN.md §4.2).

    ``depth`` must be estimated depth at run time (CLAUDE.md rule 5).
    """
    corridor = clip_depth(depth, max_range)[corridor_window(depth.shape)]
    grad_rows, grad_cols = np.gradient(corridor)
    return {
        "corridor_min": float(corridor.min()),
        "corridor_p05": float(np.percentile(corridor, 5)),
        "corridor_p25": float(np.percentile(corridor, 25)),
        "corridor_median": float(np.median(corridor)),
        "corridor_var": float(corridor.var()),
        "corridor_grad_mean": float(np.hypot(grad_rows, grad_cols).mean()),
        "corridor_far_frac": float((corridor >= max_range).mean()),
    }
