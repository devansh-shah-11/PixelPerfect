"""Tests for perception/diagnose_depth.py (M1.9). No model: a fake fast stack stands in."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from interfaces import DepthEstimate
from perception.bank_split import make_split
from perception.diagnose_depth import decompose, describe, load_frames, main, pixel_check, pixel_scores, ray_to_planar, rel_errors


def test_ray_to_planar_geometry():
    ray = np.full((225, 400), 10.0)
    planar = ray_to_planar(ray, hfov_deg=90.0)
    assert planar[112, 199] == pytest.approx(10.0, rel=1e-3)  # centre: ray and planar agree
    assert planar[112, 0] == pytest.approx(10.0 / math.sqrt(2), rel=0.01)  # left edge of a 90 degree camera
    assert (planar <= ray + 1e-9).all()


def test_mean_is_dominated_by_misses():
    gt = np.array([10.0] * 95 + [2.0] * 5)
    est = np.array([11.0] * 95 + [35.0] * 5)  # good everywhere except 5 missed obstacles
    d = describe(rel_errors(gt, est))
    assert d["median"] == pytest.approx(0.1)
    assert d["mean"] > 0.8  # 5 misses at 16.5 each drag the mean up
    assert d["share_of_error_from_worst_5pct"] > 0.85
    assert d["trimmed_mean_5pct"] == pytest.approx(0.1)


def _row(stem, ep, gt, est, obstacle="pole", obstacle_px=0, obs_m=100.0, scene_m=None):
    r = {"stem": stem, "episode": ep, "step": 0, "obstacle": obstacle, "condition": "clear", "corridor_min_gt": gt, "corridor_min_est": est}
    c = {"stem": stem, "obstacle_px": obstacle_px, "corridor_min_gt_obstacle": obs_m, "corridor_min_gt_scene": scene_m if scene_m is not None else gt, "corridor_min_gt_clean": gt}
    return r, c


def test_decompose_separates_obstacle_from_scene(tmp_path):
    pairs = [_row(f"{i:04d}_000", i, 10.0, 11.0) for i in range(20)]  # scene frames, 10% error
    pairs += [_row(f"{100 + i:04d}_000", 100 + i, 2.0, 35.0, obstacle_px=10, obs_m=2.0, scene_m=30.0) for i in range(5)]  # missed obstacles
    rows = [{**r, "gt_clean": c} for r, c in pairs]
    a = decompose(rows)
    assert a["groups"]["nearest = scene"]["median"] == pytest.approx(0.1)
    assert a["groups"]["nearest = obstacle"]["median"] == pytest.approx(16.5)
    assert a["groups"]["all"]["mean"] > 3
    # end to end through the CLI, with a split and files
    eps = [{"episode": int(r["episode"]), "condition": "clear", "obstacle": r["obstacle"]} for r, _ in pairs]
    split = make_split(eps, seed=0)
    split["splits"]["train"] = sorted(e["episode"] for e in eps)  # everything train for this test
    split["splits"]["val"], split["splits"]["test"] = [], []
    from perception.bank_split import split_checksum

    split["checksum"] = split_checksum(split["splits"])
    (tmp_path / "split.yaml").write_text(yaml.safe_dump(split))
    (tmp_path / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r, _ in pairs))
    (tmp_path / "gt.jsonl").write_text("".join(json.dumps(c) + "\n" for _, c in pairs))
    assert len(load_frames(tmp_path / "run.jsonl", tmp_path / "gt.jsonl", tmp_path / "split.yaml")) == 25
    main(["--run", str(tmp_path / "run.jsonl"), "--gt", str(tmp_path / "gt.jsonl"), "--split", str(tmp_path / "split.yaml"), "--out", str(tmp_path / "out")])
    text = (tmp_path / "out" / "depth_diag.md").read_text()
    assert "nearest = scene" in text and "nearest = obstacle" in text


def test_pixel_scores_split_scene_and_obstacle():
    gt = np.full((20, 40), 10.0)
    est = np.full((20, 40), 10.0)
    mask = np.zeros((20, 40), bool)
    mask[8:12, 18:22] = True
    gt[mask] = 3.0
    est[mask] = 30.0  # sees through the obstacle
    s = pixel_scores(gt, est, mask)
    assert s["px_scene"]["abs_rel_median"] == 0.0 and s["px_scene"]["delta_1.25"] == 1.0
    assert s["px_obstacle"]["median_est_over_gt"] == pytest.approx(10.0)


class SeeThroughModel:
    """Returns ground-truth-like planar depth of the background everywhere: blind to every obstacle."""

    def perceive(self, rgb):
        return DepthEstimate(depth=np.full(rgb.shape[:2], 80.0, np.float32), features={})


def test_pixel_check_runs_on_a_fake_bank(tmp_path):
    import sys

    sys.path.insert(0, "tests")
    import test_eval_bank as t

    eps = {k: {"obstacle": "pole_10cm_axis_z", "condition": "clear", "in_corr": True, "dist": 5.0, "steps": 2} for k in range(3)}
    bank = t.make_bank(tmp_path, eps)
    rows = [json.loads(line) for line in (bank / "frames.jsonl").read_text().splitlines()]
    out = pixel_check(bank, rows, per_type=2, out_dir=tmp_path / "px", model=SeeThroughModel())
    assert out["frames"] == 2
    assert out["by_obstacle"]["pole_10cm_axis_z"]["obstacle_median_est_over_gt"] > 10  # blind to the pole
    assert (tmp_path / "px" / "depth_sheet.png").exists()
