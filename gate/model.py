"""A trained gate behind ``interfaces.Gate`` (chunk M3.4; used by ``policies.base.GatePolicy`` in M5).

Loads a pickle written by ``gate.train``. ``score`` takes the same feature dict
the fast stack produces at run time, with the temporal features added by
``gate.features.TemporalFeatures`` (and the image features from
``gate.image_features.image_features`` if the model was trained with them).

Pickles are not portable across scikit-learn versions; retrain rather than
copying one between environments.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from gate.features import matrix
from interfaces import DroneState


class GateModel:
    def __init__(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.model, self.names = blob["model"], list(blob["names"])
        self.kind, self.feature_set = blob.get("kind"), blob.get("feature_set")

    def score(self, features: dict[str, Any], drone_state: DroneState) -> float:
        return float(self.model.predict_proba(matrix([features], self.names))[0, 1])
