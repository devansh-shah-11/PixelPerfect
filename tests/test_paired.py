import pytest

from eval.metrics import EpisodeResult
from eval.paired import compare, mcnemar_exact, repeatability


def _r(seed: int, crashed: bool, calls: int = 0, t: float = 5.0) -> EpisodeResult:
    return EpisodeResult(episode_id=f"val-{seed}", seed=seed, split="val", policy="p", judge="none", crashed=crashed, reached_goal=not crashed,
                         distance_m=1.0, mission_time_s=t, n_frames=10, n_calls=calls, delay_charged_s=0.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)


def test_compare_pairs_by_seed_and_ignores_unshared_seeds():
    a = [_r(s, s < 5, calls=1) for s in range(20)]
    b = [_r(s, False, calls=3) for s in reversed(range(25))]
    d = compare(a, b, n_boot=500)
    assert d["seeds"] == 20 and d["only_a_crashed"] == 5 and d["only_b_crashed"] == 0
    assert d["d_crash"] == 0.25 and d["d_crash_ci"][0] > 0 and d["d_calls"] == -2.0 and d["d_calls_ci"] == [-2.0, -2.0]


def test_repeatability_reports_flips_and_time_shifts():
    rows = [{"seed": 1, "base_crash_t": None}, {"seed": 2, "base_crash_t": 5.0}, {"seed": 2, "base_crash_t": 5.0}, {"seed": 3, "base_crash_t": 5.0}]
    r = repeatability([_r(1, True, t=4.0), _r(2, True, t=5.0), _r(3, True, t=5.4)], rows)
    assert r["flipped"] == [(1, None, 4.0)] and [s for s, *_ in r["time_shifted"]] == [3]
