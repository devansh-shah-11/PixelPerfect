"""Checks for the shared interfaces (task S-04)."""
from __future__ import annotations

import dataclasses

import numpy as np

from interfaces import (
    Action,
    Controller,
    DepthEstimate,
    DroneState,
    FastStack,
    FrameRecord,
    Gate,
    Policy,
    RuntimeStep,
    VLMAnswer,
    VLMJudge,
)

STATE = DroneState(t=1.0, position=(0.0, 0.0, 5.0), yaw=0.0, speed=5.0)


def make_record(**overrides) -> FrameRecord:
    fields = dict(
        episode_id="val-0007",
        split="val",
        seed=7,
        t=1.2,
        rgb_path="rgb/000012.png",
        depth_gt_path="depth_gt/000012.npy",
        seg_gt_path="seg/000012.png",
        depth_est_path="depth_est/000012.npy",
        pose=(1.0, 2.0, 5.0, 0.1),
        velocity=(5.0, 0.0, 0.0),
        corridor_min_gt=3.5,
        corridor_min_est=12.0,
        action_proposed=Action.CONTINUE,
        asked=True,
        vlm_answer=True,
        vlm_conf=2.3,
        vlm_latency_real_s=0.41,
        delay_charged_s=1.0,
        action_taken=Action.STOP,
        collision=False,
        obstacle_family="wire",
        gate_score=0.87,
    )
    fields.update(overrides)
    return FrameRecord(**fields)


def test_actions_match_plan():
    assert {a.value for a in Action} == {"continue", "stop", "yaw_left", "yaw_right", "climb"}


def test_frame_record_round_trip():
    rec = make_record()
    assert FrameRecord.from_json(rec.to_json()) == rec


def test_frame_record_round_trip_when_not_asked():
    rec = make_record(asked=False, vlm_answer=None, vlm_conf=None, vlm_latency_real_s=None,
                      delay_charged_s=0.0, obstacle_family=None, gate_score=None)
    assert FrameRecord.from_json(rec.to_json()) == rec


def test_frame_record_has_every_plan_field():
    plan_fields = {
        "episode_id", "seed", "t", "rgb_path", "depth_gt_path", "seg_gt_path", "depth_est_path",
        "pose", "velocity", "corridor_min_gt", "corridor_min_est", "action_proposed", "asked",
        "vlm_answer", "vlm_conf", "vlm_latency_real_s", "delay_charged_s", "action_taken",
        "collision", "obstacle_family",
    }
    assert plan_fields <= {f.name for f in dataclasses.fields(FrameRecord)}


def test_runtime_types_carry_no_ground_truth():
    """CLAUDE.md rule 5: types passed to controller, gate and policies hold no GT."""
    forbidden = ("gt", "seg", "collision", "obstacle_family")
    for cls in (DroneState, DepthEstimate, RuntimeStep):
        for f in dataclasses.fields(cls):
            assert not any(word in f.name for word in forbidden), f"{cls.__name__}.{f.name}"


def test_vlm_answer_unpacks_like_plan_tuple():
    answer, confidence, latency_s = VLMAnswer(True, 1.5, 0.3)
    assert (answer, confidence, latency_s) == (True, 1.5, 0.3)


def test_dummy_components_satisfy_protocols():
    class Fast:
        def perceive(self, rgb):
            return DepthEstimate(depth=np.ones(rgb.shape[:2], np.float32), features={"corridor_min": 1.0})

    class Ctrl:
        def act(self, depth, drone_state):
            return Action.STOP if depth.features["corridor_min"] < 2.0 else Action.CONTINUE

    class AlwaysHalf:
        def score(self, features, drone_state):
            return 0.5

    class Never:
        name = "never"

        def should_ask(self, t, features, drone_state, history):
            return False

    class Judge:
        def ask(self, image, path_overlay, crop):
            return VLMAnswer(False, -1.0, 0.0)

    assert isinstance(Fast(), FastStack)
    assert isinstance(Ctrl(), Controller)
    assert isinstance(AlwaysHalf(), Gate)
    assert isinstance(Never(), Policy)
    assert isinstance(Judge(), VLMJudge)

    est = Fast().perceive(np.zeros((4, 6, 3), np.uint8))
    assert est.depth.shape == (4, 6)
    assert Ctrl().act(est, STATE) is Action.STOP
