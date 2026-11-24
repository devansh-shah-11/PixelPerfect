"""Replay labels by flight forking (PLAN_DETAILED.md §4.3 L3, D16 headline 1; chunks M6.1 and M6.3).

For every seed: fly once with the fast stack only (never ask), keeping a ``ForkPoint`` per step.
Then, at every ``every``-th frame ``t``, fork the flight and ask the judge exactly once at ``t``
(pause, delay ``d``, brake and hold, answer applied), and nothing after that.

- Fork (a), fast stack only, needs no extra flight: it *is* the never-ask flight from ``t`` on, so
  "crashes within the horizon" is the L2 label with ``k = horizon_s``.
- Fork (b) runs for ``horizon_s + d`` simulated seconds, so both forks get the same flying time
  after ``t``; otherwise the hold alone would push a crash past the horizon and look like a save.

By default only frames whose never-ask flight crashes within the horizon are forked: everywhere else
``crash_without`` is false, so L3 is 0 whatever fork (b) does, and the row's outcome is
``SAFE_NOT_FORKED``. ``fork_all: true`` forks every frame, which also counts where asking *causes*
a crash (``ReplayOutcome.CAUSED``); it costs about ``(horizon_s + d) / dt`` steps per frame.

Each row carries the gate's run-time features at ``t`` (depth, temporal, image), the L1, L2 and L3
labels and the ``ReplayOutcome``, so gate-L1 and gate-L3 can be trained on the same flight frames.

    python -m eval.replay --config configs/replay/raycast_smoke.yaml

Config keys: as in ``eval.sweep`` (``backend``, ``backend_kw``, ``stack``, ``judge``, ``judge_kw``,
``delay_s``, ``split``, ``seeds``, ``scenario``, ``loop``, ``out``) plus ``horizon_s``, ``every`` and ``fork_all``.
Resumable per seed (CLAUDE.md rule 10). Test seeds are refused: labels are training data.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from control.rules import RuleController
from eval.factory import fork_judge, make_judge, make_sim, make_stack
from eval.run_episode import ForkPoint, LoopConfig, run_episode
from eval.sweep import _log_run, apply_overrides, resolve_scenario, resolve_seeds
from gate.features import IMAGE_KEYS
from gate.labels import ReplayOutcome, l2_labels, replay_outcomes
from sim.scenario import Scenario, generate


class RecordingNever:
    """Never asks; keeps the features the loop hands to ``should_ask``, image features included."""

    name = "never"
    feature_names = tuple(IMAGE_KEYS)  # makes the loop compute image features, as for the image gate

    def __init__(self) -> None:
        self.features: list[dict[str, float]] = []

    def reset(self, seed: int) -> None:
        self.features = []

    def should_ask(self, t: float, features: dict[str, Any], drone_state: Any, history: Any) -> bool:
        self.features.append({k: float(v) for k, v in features.items()})
        return False


class AskOnce:
    """Asks at the first step it sees (the fork point) and never again."""

    name = "ask_once"
    feature_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._asked = False

    def reset(self, seed: int) -> None:
        self._asked = False

    def should_ask(self, t: float, features: dict[str, Any], drone_state: Any, history: Any) -> bool:
        first, self._asked = not self._asked, True
        return first


def replay_flight(scenario: Scenario, sim: Any, stack: Any, judge: Any, cfg: LoopConfig, horizon_s: float,
                  every: int = 1, judge_name: str = "none", fork_all: bool = False) -> list[dict[str, Any]]:
    """Replay rows for every ``every``-th frame of one never-ask flight."""
    if horizon_s <= 0 or every < 1:
        raise ValueError("horizon_s must be > 0 and every >= 1")
    snaps: list[ForkPoint] = []
    base_policy = RecordingNever()
    records, base = run_episode(scenario, sim, stack, RuleController(d_safe=cfg.d_safe), base_policy, None, cfg,
                                judge_name=judge_name, snapshots=snaps)
    times = [r.t for r in records]
    crash_time = base.mission_time_s if base.crashed else None
    l2 = l2_labels(times, crash_time, horizon_s)
    rows = []
    for k in range(0, len(records), every):
        snap, rec = snaps[k], records[k]
        crash_without = bool(l2[k])
        if crash_without or fork_all:
            _, fork = run_episode(scenario, sim, stack, None, AskOnce(), fork_judge(judge, scenario.seed, k), cfg,
                                  judge_name=judge_name, resume=snap, t_stop=snap.state.t + horizon_s + cfg.delay_s)
            crash_with, fork_crash_t = bool(fork.crashed), (fork.mission_time_s if fork.crashed else None)
            outcome_name = ReplayOutcome(int(replay_outcomes([crash_without], [crash_with])[0])).name
        else:
            crash_with, fork_crash_t, outcome_name = None, None, "SAFE_NOT_FORKED"
        rows.append({
            "episode_id": base.episode_id, "split": scenario.split, "seed": scenario.seed, "k": k, "t": rec.t,
            "obstacle_family": rec.obstacle_family, "corridor_min_gt": rec.corridor_min_gt, "corridor_min_est": rec.corridor_min_est,
            "features": base_policy.features[k],
            "l1": int(rec.corridor_min_gt < cfg.d_safe <= rec.corridor_min_est), "l2": int(l2[k]),
            "l3": int(outcome_name == ReplayOutcome.SAVED.name), "outcome": outcome_name,
            "crash_without": crash_without, "crash_with": crash_with, "fork_crash_t": fork_crash_t,
            "base_crash_t": crash_time, "horizon_s": horizon_s, "delay_s": cfg.delay_s, "judge": judge_name,
        })
    return rows


def run(cfg: dict[str, Any]) -> int:
    if cfg.get("split", "val") == "test":
        raise SystemExit("replay labels are training data: test seeds are refused (CLAUDE.md rule 3)")
    scen_cfg, frozen_starts = resolve_scenario(cfg)
    seeds = resolve_seeds(cfg)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    rows_path, done_path = out / "replay_labels.jsonl", out / "done_seeds.txt"
    done = {int(s) for s in done_path.read_text().split()} if done_path.exists() else set()
    # Strings may name environment variables, e.g. "${PP_SIM_CONFIGS}" set by hpc/airsim_job.sbatch (as in eval.sweep).
    backend_kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg.get("backend_kw", {}).items()}
    if frozen_starts and cfg.get("backend") == "airsim":
        backend_kw["starts"] = frozen_starts
    sim = make_sim(cfg.get("backend", "raycast"), **backend_kw)
    stack = make_stack(cfg.get("stack", "true-depth"), **cfg.get("stack_kw", {}))
    judge, judge_name = make_judge(cfg.get("judge", "surrogate"), **cfg.get("judge_kw", {}))
    if judge is None:
        raise SystemExit("replay labels need a judge")
    loop = LoopConfig(delay_s=cfg.get("delay_s", 1.0), **cfg.get("loop", {}))
    todo = [s for s in seeds if s not in done]
    print(f"[{time.strftime('%H:%M:%S')}] {len(todo)} flights to replay, {len(seeds) - len(todo)} done", flush=True)
    n = 0
    try:
        for seed in todo:
            scenario = generate(seed, cfg.get("split", "val"), scen_cfg)
            rows = replay_flight(scenario, sim, stack, judge, loop, float(cfg.get("horizon_s", 3.0)), int(cfg.get("every", 1)), judge_name,
                                 bool(cfg.get("fork_all", False)))
            with open(rows_path, "a") as f:  # one write per flight, then mark it done
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            with open(done_path, "a") as f:
                f.write(f"{seed}\n")
            n += len(rows)
            print(f"  seed {seed}: {len(rows)} frames, {sum(r['l3'] for r in rows)} L3 positive", flush=True)
    finally:
        # Job 39930 finished all 20 flights, then hung until its 2 h limit without printing the summary:
        # the Project AirSim client was never disconnected [UNVERIFIED as the cause]. Close it explicitly.
        if hasattr(sim, "close"):
            sim.close()
    return n


def summarize(rows_path: Path) -> str:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by = {name: sum(r["outcome"] == name for r in rows) for name in [o.name for o in ReplayOutcome] + ["SAFE_NOT_FORKED"]}
    agree = sum(r["l1"] == r["l3"] for r in rows)
    return (f"{len(rows)} frames from {len({r['seed'] for r in rows})} flights | outcomes {by} | "
            f"L1 positive {sum(r['l1'] for r in rows)}, L2 {sum(r['l2'] for r in rows)}, L3 {sum(r['l3'] for r in rows)} | "
            f"L1 and L3 agree on {agree}/{len(rows)} frames")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides")
    p.add_argument("--no-log", action="store_true", help="do not append to logs/runs.jsonl (tests)")
    a = p.parse_args(argv)
    cfg = apply_overrides(yaml.safe_load(Path(a.config).read_text()), a.overrides)
    start, outcome, n = time.time(), "failed", 0
    try:
        n = run(cfg)
        outcome = "success"
    finally:
        if not a.no_log:
            _log_run(a.config, cfg, n, time.time() - start, outcome)
    print(summarize(Path(cfg["out"]) / "replay_labels.jsonl"), flush=True)


if __name__ == "__main__":
    main()
