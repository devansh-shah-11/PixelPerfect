"""The learned gate inside the flight loop (M5.1 / M5.3): built from a sweep config, fed run-time features only."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from control.rules import RuleController
from eval.factory import make_judge, make_sim, make_stack
from eval.run_episode import LoopConfig, run_episode
from gate.features import DEPTH_KEYS, IMAGE_KEYS, TEMPORAL_KEYS
from policies.base import GatePolicy, make_policy
from sim.scenario import generate

sklearn = pytest.importorskip("sklearn")


def _pickle(tmp_path, names, always_high: bool = False):
    """A real sklearn model on random data, saved the way gate.train saves it."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, len(names))), rng.integers(0, 2, 200)
    model = DummyClassifier(strategy="constant", constant=1).fit(x, np.r_[y[:-1], 1]) if always_high else LogisticRegression().fit(x, y)
    path = tmp_path / "gate.pkl"
    path.write_bytes(pickle.dumps({"model": model, "names": list(names), "kind": "test", "feature_set": "test"}))
    return path


def _fly(policy, seed: int = 101_000):
    sim = make_sim("raycast", width=120, height=68)
    judge, name = make_judge("surrogate", default_tpr=0.8, default_fpr=0.05, seed=0)
    return run_episode(generate(seed, "val"), sim, make_stack("blind"), RuleController(), policy, judge,
                       LoopConfig(t_max=6.0), judge_name=name)


def test_make_policy_loads_the_gate_from_a_path(tmp_path, monkeypatch):
    names = list(DEPTH_KEYS + TEMPORAL_KEYS + IMAGE_KEYS)
    path = _pickle(tmp_path, names)
    monkeypatch.setenv("GATE_DIR", str(tmp_path))
    p = make_policy({"policy": "gate", "model_path": "$GATE_DIR/gate.pkl", "tau": 0.4, "cooldown_s": 0.5})
    assert isinstance(p, GatePolicy) and p.tau == 0.4 and p.feature_names == names


def test_loop_computes_image_features_only_when_the_gate_uses_them(tmp_path):
    with_img = make_policy({"policy": "gate", "model_path": str(_pickle(tmp_path, DEPTH_KEYS + TEMPORAL_KEYS + IMAGE_KEYS)), "tau": 0.5})
    records, result = _fly(with_img)
    assert all(r.gate_score is not None for r in records)  # scored every frame, so every image feature was present
    ms = result.extra["image_features_ms"]
    assert 0 < ms["median"] <= ms["p95"]

    (tmp_path / "d").mkdir()
    depth_only = make_policy({"policy": "gate", "model_path": str(_pickle(tmp_path / "d", DEPTH_KEYS + TEMPORAL_KEYS)), "tau": 0.5})
    records, result = _fly(depth_only)
    assert all(r.gate_score is not None for r in records)
    assert "image_features_ms" not in result.extra


def test_a_gate_that_always_fires_asks_and_pays_the_delay(tmp_path):
    p = make_policy({"policy": "gate", "model_path": str(_pickle(tmp_path, DEPTH_KEYS, always_high=True)), "tau": 0.5, "cooldown_s": 1.0})
    records, result = _fly(p)
    assert result.n_calls >= 2 and result.delay_charged_s == pytest.approx(result.n_calls * 1.0)
    assert result.judge == "surrogate VLM"
