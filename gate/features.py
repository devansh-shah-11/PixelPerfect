"""Gate input features (PLAN_DETAILED.md §4.2, chunks M3.3 and M3.4).

All features are computed from run-time information only (CLAUDE.md rule 5):

- corridor statistics of the *estimated* depth (``perception.depth_utils.corridor_features``);
- temporal changes of those statistics against the previous frame of the same flight;
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
TEMPORAL_KEYS = tuple(f"d_{k}" for k in TEMPORAL_SOURCE)
IMAGE_KEYS = ("img_grad_mean", "img_edge_frac", "img_hline", "img_vline", "img_contrast", "img_brightness")


def temporal(prev: dict[str, Any] | None, cur: dict[str, Any]) -> dict[str, float]:
    """Change of selected depth features since the previous frame; zeros on a flight's first frame."""
    return {f"d_{k}": (float(cur[k]) - float(prev[k])) if prev is not None else 0.0 for k in TEMPORAL_SOURCE}


class TemporalFeatures:
    """Online version for the flight loop: call ``reset()`` per flight, then ``update(features)`` every frame."""

    def __init__(self) -> None:
        self._prev: dict[str, Any] | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, features: dict[str, Any]) -> dict[str, Any]:
        out = {**features, **temporal(self._prev, features)}
        self._prev = dict(features)
        return out


def add_temporal_offline(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach temporal features to bank-run rows, using the previous step of the same episode.

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
        out.append({**r, "features": {**f, **temporal(p, f)}})
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
