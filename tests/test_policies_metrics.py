"""Tests for policies/base.py (M3.2) and eval/metrics.py (M3.1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import (
    EpisodeResult, markdown, paired_bootstrap_safety, percentiles, policy_summary, results_table, safety_retained, wilson_ci,
)
from interfaces import Action, DroneState, Policy, RuntimeStep
from policies.base import Always, FixedInterval, GatePolicy, Never, Oracle, Random, Threshold, make_policy

S = DroneState(t=0.0, position=(0.0, 0.0, -2.0), yaw=0.0, speed=5.0)


def step(t: float, asked: bool) -> RuntimeStep:
    return RuntimeStep(t=t, asked=asked, vlm_answer=None, action_taken=Action.CONTINUE, gate_score=None)


def run(policy, n=50, dt=0.1, features=None):
    hist, asks = [], []
    for k in range(n):
        t = k * dt
        a = policy.should_ask(t, features or {"corridor_min": 5.0}, S, hist)
        asks.append(a)
        hist.append(step(t, a))
    return asks


# ---- policies ------------------------------------------------------------------------


def test_all_policies_implement_the_interface():
    for p in (Never(), Always(), Random(0.1), FixedInterval(1.0), Threshold(), Oracle()):
        assert isinstance(p, Policy)


def test_never_and_always():
    assert not any(run(Never()))
    assert all(run(Always()))  # cooldown 0: every frame
    assert sum(run(Always(cooldown_s=1.0), n=50, dt=0.1)) == 5  # 5 s of flight, one per second


def test_cooldown_blocks_query_storms():
    asks = run(Threshold("corridor_min", tau=10.0, cooldown_s=0.5), n=20, dt=0.1)
    assert sum(asks) == 4 and asks[0]  # t = 0.0, 0.5, 1.0, 1.5


def test_random_is_reproducible_per_seed_and_matches_rate():
    a, b = Random(0.3, cooldown_s=0.0), Random(0.3, cooldown_s=0.0)
    a.reset(7), b.reset(7)
    assert run(a, n=200) == run(b, n=200)
    b.reset(8)
    assert run(a.__class__(0.3, cooldown_s=0.0), n=10) is not None  # fresh instance works without reset
    c = Random(0.3, cooldown_s=0.0)
    c.reset(1)
    assert abs(np.mean(run(c, n=4000)) - 0.3) < 0.03


def test_fixed_interval():
    assert sum(run(FixedInterval(every_s=2.0), n=50, dt=0.1)) == 3  # t = 0, 2, 4


def test_threshold_directions():
    low = Threshold("corridor_min", tau=6.0, below=True, cooldown_s=0)
    assert low.should_ask(0, {"corridor_min": 5.0}, S, []) and not low.should_ask(0, {"corridor_min": 7.0}, S, [])
    high = Threshold("corridor_var", tau=2.0, below=False, cooldown_s=0)
    assert high.should_ask(0, {"corridor_var": 3.0}, S, []) and high.last_score == 3.0


class FakeGate:
    def score(self, features, drone_state):
        return features["corridor_min"] / 10.0


def test_gate_policy_logs_score_even_in_cooldown():
    g = GatePolicy(FakeGate(), tau=0.4, cooldown_s=1.0)
    assert g.should_ask(0.0, {"corridor_min": 5.0}, S, [])
    assert not g.should_ask(0.1, {"corridor_min": 9.0}, S, [step(0.0, True)])  # in cooldown
    assert g.last_score == pytest.approx(0.9)  # still scored for the log
    assert not g.should_ask(2.0, {"corridor_min": 3.0}, S, [step(0.0, True)])  # 0.3 < tau


def test_oracle_needs_truth_and_spends_it():
    o = Oracle()
    with pytest.raises(RuntimeError):
        o.should_ask(0, {}, S, [])
    o.set_truth(True)
    assert o.should_ask(0, {}, S, [])
    with pytest.raises(RuntimeError):  # the label is not reused on the next frame
        o.should_ask(0.1, {}, S, [])


def test_make_policy():
    assert make_policy({"policy": "random", "p": 0.2}).p == 0.2
    assert make_policy({"policy": "threshold", "feature": "corridor_var", "tau": 1.0, "below": False}).feature == "corridor_var"
    with pytest.raises(ValueError):
        make_policy({"policy": "psychic"})


# ---- metrics ------------------------------------------------------------------------


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(20, 200)  # 10%
    assert lo == pytest.approx(0.0657, abs=1e-3) and hi == pytest.approx(0.1496, abs=1e-3)
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0) and wilson_ci(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_safety_retained():
    assert safety_retained(0.3, 0.3, 0.1) == 0.0
    assert safety_retained(0.3, 0.1, 0.1) == 1.0
    assert safety_retained(0.3, 0.2, 0.1) == pytest.approx(0.5)
    assert math.isnan(safety_retained(0.1, 0.1, 0.2))  # always-ask not safer: undefined


def ep(policy, seed, crashed, calls=0, judge="surrogate VLM", dist=60.0, time_s=12.0, lat=()):
    return EpisodeResult(episode_id=f"{policy}-{seed}", seed=seed, split="val", policy=policy, judge=judge if calls else "none",
                         crashed=crashed, reached_goal=not crashed, distance_m=dist, mission_time_s=time_s, n_frames=120,
                         n_calls=calls, delay_charged_s=float(calls), latencies_s=list(lat))


def test_policy_summary_and_paired_safety():
    rng = np.random.default_rng(0)
    seeds = range(300)
    never = [ep("never", s, bool(rng.random() < 0.3)) for s in seeds]
    always = [ep("always", s, False, calls=120) for s in seeds]
    half = [ep("gate", s, never[s].crashed and s % 2 == 0, calls=6, lat=(0.08, 0.09)) for s in seeds]  # fixes every other crash
    t = results_table(never + always + half)
    assert t["gate"]["calls_per_flight"] == 6 and t["gate"]["query_rate"] == pytest.approx(0.05)
    sr = t["gate"]["safety_retained"]
    assert 0.35 < sr["safety_retained"] < 0.65 and sr["ci_low"] < sr["safety_retained"] < sr["ci_high"]
    assert t["never"]["safety_retained"]["safety_retained"] == 0.0 and t["always"]["safety_retained"]["safety_retained"] == 1.0
    assert t["gate"]["latency_s"]["p50"] == pytest.approx(0.085)
    s = policy_summary(never)
    assert s["crashes_per_km"] == pytest.approx(sum(r.crashed for r in never) / 18.0)  # 300 x 60 m = 18 km
    md = markdown(t)
    assert "surrogate VLM" in md and "| gate |" in md


def test_tables_refuse_to_mix_judges():
    with pytest.raises(ValueError):
        results_table([ep("a", 0, False, calls=3, judge="surrogate VLM"), ep("b", 0, False, calls=3, judge="qwen3-vl-2b")])


def test_paired_bootstrap_uses_only_common_seeds():
    out = paired_bootstrap_safety([ep("g", 1, False)], [ep("n", 2, True)], [ep("a", 3, False)])
    assert out["seeds"] == 0 and math.isnan(out["safety_retained"])


def test_percentiles_empty():
    assert math.isnan(percentiles([])["p95"])
