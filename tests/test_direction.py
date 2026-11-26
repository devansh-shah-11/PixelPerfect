"""Tests for the L2 direction benchmark: answer key, goal inputs, depth baselines, scoring, and the driver with a fake judge."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sim.geometry import OBB, best_side
from sim.scenario import half_extents
from vlm import direction as D
from vlm import eval_direction as E
from vlm.judge import QwenVLJudge

H, W = 90, 160


def frame(along, lateral, family="box_1m_grey", episode=0, step=0):
    return {"episode": episode, "step": step, "stem": f"{episode:04d}_{step:03d}", "split": "val", "obstacle": family, "condition": "clear",
            "gt_along_m": along, "gt_lateral_m": lateral, "pose": [0, 0, -3.0, 0.0]}


def test_needs_agree_with_best_side():
    for fam in ("box_1m_grey", "wire_3cm_axis_z", "pole_10cm_axis_y", "wall_white_featureless"):
        for lat in (-0.4, 0.0, 0.3):
            need = D.moves_needed(10.0, lat, fam)
            box = OBB(center=(10.0, lat, -3.0), half=half_extents(fam), yaw=0.0, family=fam)
            assert best_side(np.array([0.0, 0.0, -3.0]), 0.0, box, D.DRONE_RADIUS_M) == min(need, key=lambda k: (need[k], ("over", "left", "right").index(k)))


def test_labels_straight_avoid_and_skip():
    assert D.direction_label(frame(10.0, 3.0), 0.0)["kind"] == "straight"  # box 3 m to the side: not on the line
    assert D.direction_label(frame(40.0, 0.0), 0.0)["kind"] == "straight"  # far away
    assert D.direction_label(frame(20.0, 0.0), 0.0)["kind"] == "skip"  # between d_pos and d_neg
    lab = D.direction_label(frame(10.0, 0.0), 0.0)
    assert lab["kind"] == "avoid" and set(lab["valid"]) == {"left", "right", "over"}
    wire = D.direction_label(frame(10.0, 0.0, "wire_3cm_axis_z"), 0.0)
    assert wire["best"] == "over" and wire["need"]["over"] < 0.5 < wire["need"]["left"]  # a 6 m bar: over is far cheaper than round it
    pole = D.direction_label(frame(10.0, 0.0, "pole_10cm_axis_y"), 0.0)
    assert pole["best"] in ("left", "right") and "over" in pole["valid"]  # a vertical post: round it, climbing 2.8 m is dearer
    assert D.direction_label(frame(None, 0.0), 0.0)["kind"] == "skip"


def test_goal_side_decides_between_equal_sides_and_the_cheaper_over_still_wins_when_far_cheaper():
    box = D.direction_label(frame(10.0, 0.0), 25.0)
    assert box["best"] == "right"  # sides cost the same; the goal is on the right
    assert D.direction_label(frame(10.0, 0.0), -25.0)["best"] == "left"
    wall = D.direction_label(frame(10.0, 1.0, "wall_white_featureless"), 25.0)
    assert wall["kind"] == "avoid" and wall["best"] in wall["valid"]


def test_goal_bearing_is_deterministic_per_episode_and_uses_all_values():
    assert D.goal_bearing_deg(7, 0) == D.goal_bearing_deg(7, 0)
    assert {D.goal_bearing_deg(e, 0) for e in range(60)} == set(D.GOAL_BEARINGS_DEG)


def test_prompts_and_inputs_carry_no_ground_truth_and_follow_the_goal_input():
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for g in D.GOAL_INPUTS:
        for im in D.IMAGES:
            imgs, prompt = D.render_inputs(rgb, g, 25.0, im)
            assert len(imgs) == (2 if im == "both" else 1)
            assert "{" not in prompt and "obstacle at" not in prompt.lower()
            assert ("GOAL" in prompt) == (g in ("arrow", "arrow_text"))
            assert ("Your position" in prompt) == (g in ("text", "arrow_text"))
    plain, _ = D.render_inputs(rgb, "none", 25.0, "raw")
    arrow, _ = D.render_inputs(rgb, "arrow", 25.0, "raw")
    assert not plain[0].any() and arrow[0].any()
    with pytest.raises(ValueError):
        D.direction_prompt("coords", True)


def test_goal_marker_is_on_the_goal_side_of_the_image():
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    cols = lambda b: np.nonzero(D.draw_goal_marker(rgb, b).any(axis=(0, 2)))[0].mean()  # noqa: E731
    assert cols(25.0) > W / 2 > cols(-25.0)


def test_option_logprobs_are_a_distribution_over_the_four_options():
    ids = {"straight": [0], "left": [1, 2], "right": [3], "over": [4]}
    logits = np.zeros(6)
    logits[3] = 5.0
    lp = D.option_logprobs(logits, ids)
    assert max(lp, key=lp.get) == "right" and sum(math.exp(v) for v in lp.values()) == pytest.approx(1.0)


def test_controller_choice_and_depth_veto():
    clear = np.full((40, 80), 50.0, np.float32)
    assert D.controller_choice(clear, 0.0) == "straight"
    blocked = np.full((40, 80), 50.0, np.float32)
    blocked[16:24, 36:44] = 3.0
    blocked[:, :20] = 12.0
    assert D.controller_choice(blocked, 0.0, "choose") == "right"  # more room on the right
    assert D.controller_choice(blocked, 0.0, "climb_first") == "over"
    bands = {"left": 20.0, "right": 4.0, "over": 30.0}
    assert D.depth_veto("right", bands, "left") == "left"  # depth positively reads the VLM's side as blocked
    assert D.depth_veto("left", bands, "right") == "left"
    assert D.depth_veto("straight", bands, "left") == "straight"
    assert D.depth_veto("right", {"left": 99, "right": 99, "over": 99}, "left") == "right"  # clear never vetoes


def test_summary_counts_safe_false_steer_and_regret():
    base = {"model": "m", "condition": "c", "episode": 0, "obstacle": "box_1m_grey"}
    rows = [
        {**base, "kind": "avoid", "valid": ["left", "right"], "best": "left", "need": {"left": 1.0, "right": 2.0, "over": 9.0}, "choice": "right"},
        {**base, "episode": 1, "kind": "avoid", "valid": ["left", "right"], "best": "left", "need": {"left": 1.0, "right": 2.0, "over": 9.0}, "choice": "over"},
        {**base, "episode": 2, "kind": "straight", "valid": ["straight"], "best": "straight", "need": {}, "choice": "left"},
    ]
    s = D.summarize(rows)["m|c"]
    assert s["n_avoid"] == 2 and s["safe"] == pytest.approx(0.5) and s["exact"] == 0.0
    assert s["regret_m"] == pytest.approx((1.0 + 8.0) / 2) and s["false_steer"] == 1.0
    assert "| m | c |" in D.table({"m|c": s})


class FakeDirJudge(QwenVLJudge):
    """Always picks the side the goal marker is on, by the prompt text; skips the model."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._dir4_ids = {"straight": [0], "left": [1], "right": [2], "over": [3]}

    def _logits(self, images, prompt=None):
        logits = np.zeros(6)
        logits[2 if prompt and "GOAL" in prompt else 1] = 4.0
        return logits


def test_ask_direction_reads_four_way_logits():
    j = FakeDirJudge()
    lp, latency = j.ask_direction([np.zeros((H, W, 3), np.uint8)], "plain")
    assert max(lp, key=lp.get) == "left" and latency >= 0
    lp, _ = j.ask_direction([np.zeros((H, W, 3), np.uint8)], "GOAL marker")
    assert max(lp, key=lp.get) == "right"


def test_driver_is_resumable_labels_frames_and_builds_baselines(tmp_path, monkeypatch):
    import perception.frame_bank as fb

    monkeypatch.setattr(fb, "load_rgb", lambda bank, stem: np.full((H, W, 3), 90, np.uint8))
    deep = np.full((H, W), 40.0, np.float32)
    deep[36:54, 70:90] = 3.0
    monkeypatch.setattr(fb, "load_depth_gt", lambda bank, stem: deep)
    frames = [{**frame(10.0, 0.0, episode=e, step=0), "split": "val"} for e in range(6)] + [{**frame(40.0, 0.0, episode=9), "split": "val"}]
    out = tmp_path / "rows.jsonl"
    judge = FakeDirJudge()
    rows = E.evaluate({"fake": judge}, tmp_path, frames, out, ["none", "arrow"], ["raw"])
    n = len(rows)
    assert n == 7 * (2 + 2) and {r["condition"] for r in rows if r["model"] == "fake"} == {"none_raw", "arrow_raw"}
    assert len(E.evaluate({"fake": judge}, tmp_path, frames, out, ["none", "arrow"], ["raw"])) == n  # nothing redone
    full = E.with_baselines(rows)
    models = {r["model"] for r in full}
    assert {"oracle", "random", "goal_side", "fake+veto_depth_gt"} <= models
    s = D.summarize(full)
    assert s["oracle|-"]["safe"] == 1.0 and s["oracle|-"]["false_steer"] == 0.0
    json.dumps(s)
