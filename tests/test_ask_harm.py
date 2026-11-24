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


def test_guard_split_uses_the_last_logged_frame(tmp_path):
    from eval.ask_harm import guard_split
    from interfaces import Action, FrameRecord

    def frame(gt, fam):
        return FrameRecord(episode_id="e", split="val", seed=1, t=1.0, rgb_path="", depth_gt_path="", seg_gt_path="", depth_est_path="",
                           pose=(1.0, 0.0, -2.0, 0.0), velocity=(0.0, 0.0, 0.0), corridor_min_gt=gt, corridor_min_est=20.0, action_proposed=Action.CONTINUE,
                           asked=False, vlm_answer=None, vlm_conf=None, vlm_latency_real_s=None, delay_charged_s=0.0, action_taken=Action.CONTINUE,
                           collision=False, obstacle_family=fam, gate_score=None)

    results, where = [], {}
    for seed, recs in ((1, [frame(0.8, "pole_10cm_axis_z")]), (2, [frame(40.0, None)]), (3, [])):
        r = _r(seed, "g", True, 1)
        r.extra["collision_source"] = "scenery"
        results.append(r)
        where[("g", seed)] = tmp_path / f"{seed}.jsonl"
        where[("g", seed)].write_text("".join(x.to_json() + "\n" for x in recs))
    lines = guard_split(results, where)
    assert lines[0].endswith("1 {'g': 1}") and "elsewhere (map or bad capture): 2" in lines[1] and "no log" in "".join(lines)
