"""Flight forking and replay labels (M6.1, M6.3)."""
from __future__ import annotations

import numpy as np

from control.rules import RuleController
from eval.factory import make_judge, make_sim, make_stack
from eval.replay import AskOnce, RecordingNever, fork_judge, replay_flight
from eval.run_episode import LoopConfig, run_episode
from sim.scenario import generate
from vlm.surrogate import SurrogateJudge

SEED = 101_000  # first val seed; the blind stand-in crashes into thin obstacles


def _parts():
    return make_sim("raycast", width=80, height=45), make_stack("blind")


def test_resuming_from_a_fork_point_reproduces_the_flight():
    sim, stack = _parts()
    scenario = generate(SEED, "val")
    snaps = []
    records, base = run_episode(scenario, sim, stack, RuleController(), RecordingNever(), None, LoopConfig(), snapshots=snaps)
    assert len(snaps) == len(records)
    k = len(records) // 2
    rest, again = run_episode(scenario, sim, stack, None, RecordingNever(), None, LoopConfig(), resume=snaps[k])
    assert [r.pose for r in rest] == [r.pose for r in records[k:]]
    assert again.crashed == base.crashed and again.mission_time_s == base.mission_time_s
    # the fork's copies were used, not the originals: replaying again gives the same answer
    rest2, _ = run_episode(scenario, sim, stack, None, RecordingNever(), None, LoopConfig(), resume=snaps[k])
    assert [r.pose for r in rest2] == [r.pose for r in rest]


def test_t_stop_ends_a_fork_at_the_horizon():
    sim, stack = _parts()
    scenario = generate(SEED, "val")
    snaps = []
    run_episode(scenario, sim, stack, RuleController(), RecordingNever(), None, LoopConfig(), snapshots=snaps)
    recs, res = run_episode(scenario, sim, stack, None, RecordingNever(), None, LoopConfig(), resume=snaps[0], t_stop=1.0)
    assert res.mission_time_s <= 1.0 + 1e-9 and len(recs) <= 11


def test_ask_once_asks_only_at_the_fork_point():
    p = AskOnce()
    assert [p.should_ask(0.1 * i, {}, None, []) for i in range(4)] == [True, False, False, False]


def test_fork_judge_gets_its_own_seed_per_frame():
    j = SurrogateJudge(seed=0)
    a, b = fork_judge(j, 101_000, 3), fork_judge(j, 101_000, 4)
    assert a.seed != b.seed and j.seed == 0
    assert fork_judge("not a dataclass", 1, 2) == "not a dataclass"


def test_replay_rows_are_consistent():
    sim, stack = _parts()
    judge, name = make_judge("surrogate", default_tpr=1.0, default_fpr=0.0)
    rows = replay_flight(generate(SEED, "val"), sim, stack, judge, LoopConfig(), horizon_s=3.0, every=2, judge_name=name)
    assert rows and [r["k"] for r in rows] == list(range(0, 2 * len(rows), 2))
    for r in rows:
        assert r["l3"] == int(r["crash_without"] and not r["crash_with"])
        assert (r["crash_with"] is None) == (r["outcome"] == "SAFE_NOT_FORKED") and (r["crash_with"] is None) == (not r["crash_without"])
        assert {"corridor_min", "dpm_corridor_min", "img_grad_mean"} <= set(r["features"])
    if rows[0]["base_crash_t"] is not None:  # a perfect judge saves at least one frame before a crash
        assert sum(r["l3"] for r in rows) > 0


def test_fork_all_matches_the_shortcut_on_l3():
    sim, stack = _parts()
    judge, name = make_judge("surrogate", default_tpr=0.8, default_fpr=0.05)
    scen = generate(SEED, "val")
    fast = replay_flight(scen, sim, stack, judge, LoopConfig(), horizon_s=2.0, every=3, judge_name=name)
    full = replay_flight(scen, sim, stack, judge, LoopConfig(), horizon_s=2.0, every=3, judge_name=name, fork_all=True)
    assert [r["l3"] for r in fast] == [r["l3"] for r in full]
    assert all(r["crash_with"] is not None for r in full)
