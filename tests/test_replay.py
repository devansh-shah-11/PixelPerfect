"""Flight forking and replay labels (M6.1, M6.3)."""
from __future__ import annotations

import numpy as np
import pytest

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


def test_train_replay_compares_gates_at_equal_query_rates(tmp_path):
    import json

    from gate.train_replay import ask_mask, at_rate, main as train_replay_main

    s = np.array([0.9, 0.1, 0.8, 0.2])
    assert ask_mask(s, 0.5).tolist() == [True, False, True, False]
    rows = [{"seed": 1, "l3": 1}, {"seed": 1, "l3": 0}, {"seed": 2, "l3": 1}, {"seed": 2, "l3": 0}]
    r = at_rate(rows, s, 0.5)
    assert r["l3_recall"] == 1.0 and r["flights_saved"] == 1.0 and r["wasted"] == 0.0

    sim, stack = _parts()
    judge, name = make_judge("surrogate", default_tpr=0.9, default_fpr=0.05)
    paths = {}
    for split, seeds in (("train", (100_000, 100_001, 100_002)), ("val", (101_000, 101_001))):
        p = tmp_path / f"{split}.jsonl"
        with open(p, "w") as f:
            for sd in seeds:
                for row in replay_flight(generate(sd, split), sim, stack, judge, LoopConfig(), horizon_s=2.0, every=2, judge_name=name):
                    f.write(json.dumps(row) + "\n")
        paths[split] = p
    train_replay_main(["--train", str(paths["train"]), "--val", str(paths["val"]), "--model", "logreg", "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "replay_gate_results.json").read_text())
    assert {"random"} <= set(res["at_rate"]) and all(len(v) == 4 for v in res["at_rate"].values())
    if "l3_minus_l1" in res:
        assert len(res["l3_minus_l1"]) == 4
    with pytest.raises(SystemExit):  # val flights passed as train are refused
        train_replay_main(["--train", str(paths["val"]), "--val", str(paths["val"]), "--out", str(tmp_path / "o2")])


def test_paired_diff_is_zero_for_identical_gates_and_positive_for_a_better_one():
    from gate.train_replay import paired_diff

    rows = [{"seed": sd, "l3": int(i == 0)} for sd in range(10) for i in range(5)]
    good = np.array([1.0 if r["l3"] else 0.0 for r in rows])
    d = paired_diff(rows, good, good, 0.2, n_boot=200)
    assert d["d_flights_saved"] == 0.0 and d["d_flights_saved_ci"] == [0.0, 0.0] and d["p_a_better"] == 0.5
    bad = 1.0 - good
    d = paired_diff(rows, good, bad, 0.2, n_boot=200)
    assert d["d_flights_saved"] == 1.0 and d["d_flights_saved_ci"][0] == 1.0 and d["p_a_better"] == 1.0
