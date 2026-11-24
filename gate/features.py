"""Gate input features (PLAN_DETAILED.md §4.2, chunks M3.3 and M3.4).

All features are computed from run-time information only (CLAUDE.md rule 5):

- corridor statistics of the *estimated* depth (``perception.depth_utils.corridor_features``);
- temporal changes of those statistics against the previous frame of the same flight, **per metre
  flown** (not per frame), so the gate sees the same values at any cruise speed: the frame bank moves
  0.40 m per frame, the flight loop 0.5 m (5 m/s at 10 Hz);
- optional thin-structure image features from the RGB corridor crop (``gate.image_features``).

The same functions are used offline (training on the frame bank) and online
(``TemporalFeatures`` inside the flight loop), so the gate sees identical
inputs in both places.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

DEPTH_KEYS = ("corridor_min", "corridor_p05", "corridor_p25", "corridor_median", "corridor_var", "corridor_grad_mean", "corridor_far_frac")
TEMPORAL_SOURCE = ("corridor_min", "corridor_p05", "corridor_median", "corridor_var")
TEMPORAL_KEYS = tuple(f"dpm_{k}" for k in TEMPORAL_SOURCE)  # change per metre flown
IMAGE_KEYS = ("img_grad_mean", "img_edge_frac", "img_hline", "img_vline", "img_contrast", "img_brightness")

# Distance the frame bank moves per frame (4 m/s at 10 Hz). Measured over the bank:
# median 0.40 m, p10 0.38, p90 0.42 (AGENT_LOG.md, 2026-11-20 17:50).
BANK_STEP_M = 0.40
# Below this displacement (braking, holding for a VLM answer) the change per metre is
# undefined; the previous rate is carried over instead.
MIN_MOVE_M = 0.05


def temporal(prev: dict[str, Any] | None, cur: dict[str, Any], dist_m: float) -> dict[str, float]:
    """Change of selected depth features since the previous frame, divided by the metres flown.

    Zeros on a flight's first frame (``prev`` is None). ``dist_m`` must be at least ``MIN_MOVE_M``.
    """
    if prev is None:
        return {f"dpm_{k}": 0.0 for k in TEMPORAL_SOURCE}
    if dist_m < MIN_MOVE_M:
        raise ValueError(f"distance flown {dist_m} m is below MIN_MOVE_M; carry the previous rate instead")
    return {f"dpm_{k}": (float(cur[k]) - float(prev[k])) / dist_m for k in TEMPORAL_SOURCE}


class TemporalFeatures:
    """Online version for the flight loop: call ``reset()`` per flight, then ``update(features, position)`` every frame.

    ``position`` is the drone's (x, y, z) when the frame was taken (run-time state, not ground truth).
    When the drone has barely moved since the last frame, the previous rates are returned unchanged
    and the reference frame is kept, so a hold followed by a move is measured over the whole move.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev: dict[str, Any] | None = None
        self._prev_pos: np.ndarray | None = None
        self._rates: dict[str, float] | None = None

    def update(self, features: dict[str, Any], position: Sequence[float]) -> dict[str, Any]:
        pos = np.asarray(position, dtype=np.float64)
        if self._prev is None:
            self._rates = temporal(None, features, 0.0)
        else:
            dist = float(np.linalg.norm(pos - self._prev_pos))
            if dist < MIN_MOVE_M:
                return {**features, **self._rates}
            self._rates = temporal(self._prev, features, dist)
        self._prev, self._prev_pos = dict(features), pos
        return {**features, **self._rates}


def add_temporal_offline(rows: Iterable[dict[str, Any]], step_m: float = BANK_STEP_M) -> list[dict[str, Any]]:
    """Attach temporal features to bank-run rows, using the previous step of the same episode.

    Bank rows carry no pose, so every step counts as ``step_m`` metres (the bank's nominal 0.40 m;
    the real step varies by about 5%).

    A missing step (e.g. frames dropped by a filter) is treated as the start of the flight, so a feature
    never spans a gap.
    """
    rows = sorted(rows, key=lambda r: (int(r["episode"]), int(r["step"])))
    out, prev = [], None
    for r in rows:
        f = r["features"]
        if prev is None or int(prev["episode"]) != int(r["episode"]) or int(prev["step"]) != int(r["step"]) - 1:
            p = None
        else:
            p = prev["features"]
        out.append({**r, "features": {**f, **temporal(p, f, step_m)}})
        prev = r
    return out


def feature_names(use_temporal: bool = True, use_image: bool = False) -> list[str]:
    return list(DEPTH_KEYS) + (list(TEMPORAL_KEYS) if use_temporal else []) + (list(IMAGE_KEYS) if use_image else [])


def matrix(feature_dicts: Sequence[dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    """(N, F) float matrix in the order of ``names``. Raises on a missing feature rather than guessing."""
    try:
        return np.array([[float(d[k]) for k in names] for d in feature_dicts], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"feature {e.args[0]!r} missing; available: {sorted(feature_dicts[0])}") from None
