"""Tests for the gate: features, image features, metadata ground truth, training, and the Gate wrapper (M3.3 to M3.5)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import yaml

from gate.features import DEPTH_KEYS, TemporalFeatures, add_temporal_offline, matrix
from gate.image_features import image_features
from gate.labels import METADATA_GT_TYPES, metadata_in_window, window_half_width, with_metadata_fallback
from gate.model import GateModel
from gate.train import cut_at_fpr, ece, main as train_main
from interfaces import DroneState, Gate
from perception.bank_split import make_split

S = DroneState(t=0.0, position=(0.0, 0.0, -2.0), yaw=0.0, speed=5.0)


def feats(cm, var=1.0, grad=0.1):
    return {"corridor_min": cm, "corridor_p05": cm + 1, "corridor_p25": cm + 2, "corridor_median": cm + 3,
            "corridor_var": var, "corridor_grad_mean": grad, "corridor_far_frac": 0.0}


# ---- features ------------------------------------------------------------------------------


def test_temporal_offline_matches_online_and_respects_gaps():
    rows = [{"episode": 1, "step": s, "features": feats(30 - s)} for s in (0, 1, 2, 4)]  # step 3 missing
    off = add_temporal_offline(rows)
    assert off[0]["features"]["d_corridor_min"] == 0.0
    assert off[1]["features"]["d_corridor_min"] == -1.0
    assert off[3]["features"]["d_corridor_min"] == 0.0  # after the gap: treated as a fresh start
    online = TemporalFeatures()
    got = [online.update(r["features"]) for r in rows[:3]]
    assert [g["d_corridor_min"] for g in got] == [0.0, -1.0, -1.0]
    online.reset()
    assert online.update(feats(5.0))["d_corridor_min"] == 0.0


def test_matrix_refuses_missing_features():
    with pytest.raises(KeyError):
        matrix([{"a": 1.0}], ["a", "b"])


def test_image_features_see_thin_lines():
    blank = np.full((225, 400, 3), 150, np.uint8)
    hline = blank.copy()
    hline[112, :] = 20  # one-pixel horizontal wire through the corridor
    vline = blank.copy()
    vline[:, 200] = 20
    b, h, v = image_features(blank), image_features(hline), image_features(vline)
    assert b["img_grad_mean"] == pytest.approx(0.0) and b["img_edge_frac"] == 0.0
    assert h["img_hline"] > 10 * b["img_hline"] and h["img_hline"] > h["img_vline"]
    assert v["img_vline"] > 10 * b["img_vline"] and v["img_vline"] > v["img_hline"]


# ---- ground truth for see-through obstacles ----------------------------------------------


def test_metadata_fallback_only_for_see_through_types():
    assert window_half_width(10.0) == pytest.approx(1.0)
    seg_gt = {"corridor_min_gt_obstacle": 100.0, "corridor_min_gt_scene": 40.0, "corridor_min_gt_clean": 40.0, "obstacle_px": 0}
    glass_in = with_metadata_fallback(seg_gt, {"obstacle": METADATA_GT_TYPES[0], "gt_along_m": 6.0, "gt_lateral_m": 0.5})
    assert glass_in["gt_source"] == "metadata" and glass_in["corridor_min_gt_clean"] == 6.0
    glass_out = with_metadata_fallback(seg_gt, {"obstacle": METADATA_GT_TYPES[0], "gt_along_m": 6.0, "gt_lateral_m": 9.0})
    assert glass_out["corridor_min_gt_clean"] == 40.0 and glass_out["meta_in_window"] is False
    box = with_metadata_fallback(seg_gt, {"obstacle": "box_1m_grey", "gt_along_m": 6.0, "gt_lateral_m": 0.0})
    assert box["gt_source"] == "seg" and box["corridor_min_gt_clean"] == 40.0 and box["meta_in_window"] is True
    assert metadata_in_window("none", None, None) is None


def test_bank_gt_labels_glass_from_metadata(tmp_path):
    sys.path.insert(0, "tests")
    import test_eval_bank as t

    from gate.bank_gt import write_bank_gt

    bank = t.make_bank(tmp_path, {0: {"obstacle": "glass_pane_alpha20", "condition": "clear", "in_corr": True, "dist": 6.0, "seg": False},
                                  1: {"obstacle": "box_1m_grey", "condition": "clear", "in_corr": True, "dist": 7.0}})
    write_bank_gt(bank, tmp_path / "gt.jsonl")
    rows = {r["stem"]: r for r in map(json.loads, (tmp_path / "gt.jsonl").read_text().splitlines())}
    assert rows["0000_000"]["gt_source"] == "metadata" and rows["0000_000"]["corridor_min_gt_clean"] == pytest.approx(6.0)
    assert rows["0001_000"]["gt_source"] == "seg" and rows["0001_000"]["corridor_min_gt_clean"] == pytest.approx(7.0)


# ---- training ------------------------------------------------------------------------------


def synthetic(tmp_path, leak=False):
    """Flights where a missed obstacle shows up as a gradient bump and a horizontal image line, not as variance."""
    rng = np.random.default_rng(0)
    run, gt, img, eps = [], [], [], []
    for ep in range(80):
        obstacle = ["pole_10cm_axis_z", "wire_1cm_axis_z", "box_1m_grey", "none"][ep % 4]
        eps.append({"episode": ep, "condition": "clear", "obstacle": obstacle})
        for step in range(12):
            missed = obstacle != "none" and step >= 6 and rng.random() < 0.8
            f = feats(cm=40.0 + rng.normal(0, 3), var=rng.gamma(2.0), grad=(0.32 if missed else 0.2) + rng.normal(0, 0.1))
            stem = f"{ep:04d}_{step:03d}"
            run.append({"stem": stem, "episode": ep, "step": step, "obstacle": obstacle, "condition": "clear",
                        "gt_along_m": 20.0 - step, "corridor_min_gt": 5.0 if missed else 40.0, "corridor_min_est": f["corridor_min"], "features": f})
            gt.append({"stem": stem, "corridor_min_gt_clean": 5.0 if missed else 40.0, "gt_source": "seg"})
            hl = (3.0 if missed else 1.5) + rng.normal(0, 1.0)
            img.append({"stem": stem, "img_grad_mean": 5.0, "img_edge_frac": 0.1, "img_hline": (1.0 if missed else 0.0) if leak else hl,
                        "img_vline": 1.0, "img_contrast": 30.0, "img_brightness": 120.0})
    split = make_split(eps, seed=0)
    (tmp_path / "split.yaml").write_text(yaml.safe_dump(split))
    for name, rows in (("run", run), ("gt", gt), ("img", img)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_train_end_to_end_and_gate_model(tmp_path, capsys):
    d = synthetic(tmp_path)
    train_main(["--run", str(d / "run.jsonl"), "--gt", str(d / "gt.jsonl"), "--split", str(d / "split.yaml"),
                "--image-features", str(d / "img.jsonl"), "--d-safe", "8", "--models", "logreg", "gbt", "--out", str(d / "out")])
    res = json.loads((d / "out" / "gate_results.json").read_text())["results"]
    assert res["baseline_var"]["auroc"] < 0.65  # variance carries no signal here
    assert 0.75 < res["gbt/depth+temporal+image"]["auroc"] < 0.99
    assert res["gbt/depth+temporal+image"]["auroc"] > res["gbt/depth"]["auroc"]  # the image line adds signal
    assert res["best_single"]["features"][0] in ("corridor_grad_mean", "img_hline")
    text = (d / "out" / "gate_report.md").read_text()
    assert "beats" in text and "WARNING" not in text
    g = GateModel(d / "out" / "gate_gbt_depth+temporal+image.pkl")
    assert isinstance(g, Gate)
    hi = g.score({**feats(40.0, grad=0.5), "d_corridor_min": 0, "d_corridor_p05": 0, "d_corridor_median": 0, "d_corridor_var": 0,
                  "img_grad_mean": 5.0, "img_edge_frac": 0.1, "img_hline": 5.0, "img_vline": 1.0, "img_contrast": 30.0, "img_brightness": 120.0}, S)
    lo = g.score({**feats(40.0, grad=0.1), "d_corridor_min": 0, "d_corridor_p05": 0, "d_corridor_median": 0, "d_corridor_var": 0,
                  "img_grad_mean": 5.0, "img_edge_frac": 0.1, "img_hline": 0.5, "img_vline": 1.0, "img_contrast": 30.0, "img_brightness": 120.0}, S)
    assert hi > 0.5 > lo


def test_train_flags_leakage_and_holdout(tmp_path):
    d = synthetic(tmp_path, leak=True)
    train_main(["--run", str(d / "run.jsonl"), "--gt", str(d / "gt.jsonl"), "--split", str(d / "split.yaml"),
                "--image-features", str(d / "img.jsonl"), "--models", "gbt", "--holdout-family", "wire_1cm_axis_z", "--out", str(d / "out")])
    text = (d / "out" / "gate_report.md").read_text()
    assert "WARNING" in text and "Held out of training: wire_1cm_axis_z" in text


def test_ece_and_cut():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert ece(y, np.array([1.0, 1.0, 0.0, 0.0])) == 1.0
    s = np.arange(20, dtype=float)
    assert ((s[np.zeros(20, int) == 0] > cut_at_fpr(np.zeros(20, int), s, 0.1)).mean()) <= 0.1
