"""Tests for eval/crash_causes.py (M5.4)."""
from __future__ import annotations

from pathlib import Path

from eval.crash_causes import analyse, classify, reference_crashes
from eval.metrics import EpisodeResult
from interfaces import Action, FrameRecord


def rec(t: float, asked: bool = False, answer: bool | None = None, collision: bool = False) -> FrameRecord:
    return FrameRecord(
        episode_id="val-1-p", split="val", seed=1, t=t, rgb_path="", depth_gt_path="", seg_gt_path="", depth_est_path="",
        pose=(0.0, 0.0, -2.0, 0.0), velocity=(5.0, 0.0, 0.0), corridor_min_gt=5.0, corridor_min_est=20.0,
        action_proposed=Action.CONTINUE, asked=asked, vlm_answer=answer, vlm_conf=None, vlm_latency_real_s=None,
        delay_charged_s=1.0 if asked else 0.0, action_taken=Action.CONTINUE, collision=collision, obstacle_family="wire")


def flight(asks: dict[float, bool | None], t_end: float = 10.0) -> list[FrameRecord]:
    ts = [round(0.1 * k, 1) for k in range(int(t_end * 10) + 1)]
    return [rec(t, t in asks, asks.get(t), collision=(t == ts[-1])) for t in ts]


def test_controller_when_true_depth_also_crashes():
    assert classify(flight({}), reference_crashed=True)[0] == "controller"


def test_gate_did_not_ask_when_no_ask_near_the_crash():
    # an ask 8 s before the crash is outside the 3 s window
    assert classify(flight({2.0: False}), reference_crashed=False)[0] == "gate_did_not_ask"


def test_vlm_wrong_when_asked_and_said_no():
    assert classify(flight({8.0: False}), reference_crashed=False)[0] == "vlm_wrong"


def test_delay_when_vlm_said_yes_but_crashed():
    assert classify(flight({8.0: True}), reference_crashed=False)[0] == "delay"


def test_unknown_without_log_and_unchecked_reference_is_flagged():
    assert classify(None, None)[0] == "unknown"
    cause, detail = classify(flight({}), None)
    assert cause == "gate_did_not_ask" and detail["reference_checked"] is False


def test_analyse_reads_a_sweep_dir(tmp_path: Path):
    def res(seed: int, crashed: bool) -> EpisodeResult:
        return EpisodeResult(episode_id=f"val-{seed}-gate", seed=seed, split="val", policy="gate", judge="surrogate VLM", crashed=crashed,
                             reached_goal=not crashed, distance_m=30.0, mission_time_s=6.0, n_frames=60, n_calls=1, delay_charged_s=1.0)

    (tmp_path / "flights").mkdir()
    (tmp_path / "results.jsonl").write_text(res(1, True).to_json() + "\n" + res(2, False).to_json() + "\n" + res(3, True).to_json() + "\n")
    (tmp_path / "flights" / "val-1-gate.jsonl").write_text("".join(r.to_json() + "\n" for r in flight({9.0: False})))
    ref_path = tmp_path / "ref.jsonl"
    ref = EpisodeResult(episode_id="val-3-never", seed=3, split="val", policy="never", judge="none", crashed=True, reached_goal=False,
                        distance_m=1.0, mission_time_s=1.0, n_frames=1, n_calls=0, delay_charged_s=0.0)
    ref_path.write_text(ref.to_json() + "\n")
    rows = analyse(tmp_path, reference_crashes(ref_path))
    assert [(r["seed"], r["cause"]) for r in rows] == [(1, "vlm_wrong"), (3, "controller")]


def test_scenery_when_the_drone_hit_a_map_object():
    # even when true depth also crashes and the flight log is missing: a map object is its own cause
    assert classify(None, reference_crashed=True, collision_source="scenery")[0] == "scenery"
    assert classify(flight({8.0: False}), reference_crashed=False, collision_source="scenery")[0] == "scenery"
    assert classify(flight({}), reference_crashed=False, collision_source="obstacle")[0] == "gate_did_not_ask"
