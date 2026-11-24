"""Noise and harm split for crashes on seeds that never-ask survives."""
from eval.ask_harm import main, split_flights
from eval.metrics import EpisodeResult


def _r(seed: int, policy: str, crashed: bool, calls: int) -> EpisodeResult:
    return EpisodeResult(episode_id=f"val-{seed}-{policy}", seed=seed, split="val", policy=policy, judge="j", crashed=crashed, reached_goal=not crashed,
                         distance_m=1.0, mission_time_s=5.0, n_frames=10, n_calls=calls, delay_charged_s=0.0)


RESULTS = [_r(1, "never", False, 0), _r(2, "never", True, 0), _r(1, "g", True, 2), _r(2, "g", False, 0), _r(1, "h", True, 0), _r(2, "h", True, 0),
           _r(1, "always", True, 9), _r(1, "k", False, 1)]


def test_split_separates_noise_from_candidates():
    d = split_flights(RESULTS)
    assert {(r.seed, r.policy) for r in d["flips"]} == {(2, "g"), (1, "h")}
    assert {(r.seed, r.policy) for r in d["candidates"]} == {(1, "g"), (1, "always")}
    assert (d["asked_safe_n"], d["asked_safe_crashed"]) == (2, 1)


def test_cli_runs_on_real_flight_logs(tmp_path, capsys):
    from eval.sweep import run

    cfg = {"backend": "raycast", "stack": "blind", "judge": "surrogate", "judge_kw": {"default_tpr": 1.0, "default_fpr": 1.0}, "split": "val",
           "seeds": [101_000, 101_001], "loop": {"t_max": 10.0}, "flight_logs": True, "out": str(tmp_path / "s"),
           "policies": [{"name": "never", "policy": "never"}, {"name": "r", "policy": "random", "p": 0.3}]}
    run(cfg)
    # make the asked flights candidates: mark never as safe and r as crashed, keeping the real logs
    import json

    p = tmp_path / "s" / "results.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    for r in rows:
        r["crashed"] = r["policy"] == "r"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["--sweep", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert "ask t=" in out and "end t=" in out and "flights per policy" in out
