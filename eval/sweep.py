"""Resumable sweep: every policy over the same seeds (PLAN_DETAILED.md D15, §7 rules; chunks M4.7 and M5.3).

    python -m eval.sweep --config configs/sweeps/smoke_raycast.yaml

One line per finished flight goes to ``<out>/results.jsonl``; on restart,
finished ``(policy, seed)`` pairs are skipped (CLAUDE.md rule 10, the 20-minute
idle shutdown). At the end the metrics table is printed and written to
``<out>/table.md``. Each run appends one line to ``logs/runs.jsonl``.

Config keys (YAML): ``backend``, ``stack``, ``judge`` (+ ``judge_kw``), ``delay_s``, ``split``,
``seeds`` (a list, or ``{file: configs/seeds.yaml, n: 100}``), ``scenario`` (ScenarioConfig fields),
``policies`` (list of policy specs, each with a unique ``name``), ``out``, ``flight_logs`` (bool).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from control.rules import RuleController
from eval.factory import make_judge, make_sim, make_stack
from eval.metrics import EpisodeResult, markdown, read_results, results_table
from eval.run_episode import LoopConfig, run_episode
from policies.base import make_policy
from sim.scenario import ScenarioConfig, generate, load_seed_lists

REPO = Path(__file__).resolve().parent.parent


def resolve_seeds(cfg: dict[str, Any]) -> list[int]:
    s = cfg["seeds"]
    if isinstance(s, list):
        return [int(x) for x in s]
    lists = load_seed_lists(REPO / s["file"] if not Path(s["file"]).is_absolute() else s["file"])
    split = cfg.get("split", "val")
    if split == "test" and not cfg.get("final_test_run", False):
        raise SystemExit("refusing to fly test seeds: set final_test_run: true only for the one final run (PLAN_DETAILED.md D14)")
    return lists[split][: s.get("n")]


def run(cfg: dict[str, Any]) -> list[EpisodeResult]:
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = {(r.policy, r.seed) for r in read_results(results_path)} if results_path.exists() else set()
    # Strings in backend_kw may name environment variables, e.g. configs: ${PP_SIM_CONFIGS} set by hpc/airsim_job.sbatch.
    backend_kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg.get("backend_kw", {}).items()}
    sim = make_sim(cfg.get("backend", "raycast"), **backend_kw)
    stack = make_stack(cfg.get("stack", "true-depth"), **cfg.get("stack_kw", {}))
    judge, judge_name = make_judge(cfg.get("judge", "none"), **cfg.get("judge_kw", {}))
    scen_cfg = ScenarioConfig(**cfg.get("scenario", {}))
    loop = LoopConfig(delay_s=cfg.get("delay_s", 1.0), **cfg.get("loop", {}))
    seeds = resolve_seeds(cfg)
    names = [p["name"] for p in cfg["policies"]]
    if len(set(names)) != len(names):
        raise SystemExit("policy names must be unique")
    with open(results_path, "a") as f:
        for spec in cfg["policies"]:
            spec = dict(spec)
            name = spec.pop("name")
            policy = make_policy(spec)
            policy.name = name
            todo = [s for s in seeds if (name, s) not in done]
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(todo)} flights to fly, {len(seeds) - len(todo)} done", flush=True)
            for seed in todo:
                scenario = generate(seed, cfg.get("split", "val"), scen_cfg)
                records, result = run_episode(scenario, sim, stack, RuleController(d_safe=loop.d_safe), policy, judge, loop, judge_name=judge_name)
                if cfg.get("flight_logs"):
                    (out / "flights").mkdir(exist_ok=True)
                    (out / "flights" / f"{result.episode_id}.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
                f.write(result.to_json() + "\n")
                f.flush()
    return read_results(results_path)


def _log_run(cfg_path: str, cfg: dict[str, Any], n: int, wall: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    (REPO / "logs").mkdir(exist_ok=True)
    with open(REPO / "logs" / "runs.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(),
                            "command": " ".join(sys.argv), "config": cfg_path, "seeds": cfg.get("seeds"), "n_episodes": n,
                            "wall_seconds": round(wall, 1), "outcome": outcome}) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--no-log", action="store_true", help="do not append to logs/runs.jsonl (tests)")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    start, outcome, results = time.time(), "failed", []
    try:
        results = run(cfg)
        outcome = "success"
    finally:
        if not a.no_log:
            _log_run(a.config, cfg, len(results), time.time() - start, outcome)
    table = results_table(results)
    text = markdown(table)
    extra = f"backend {cfg.get('backend', 'raycast')}, fast stack {cfg.get('stack', 'true-depth')}, delay {cfg.get('delay_s', 1.0)} s, split {cfg.get('split', 'val')}"
    (Path(cfg["out"]) / "table.md").write_text(extra + "\n\n" + text + "\n")
    (Path(cfg["out"]) / "table.json").write_text(json.dumps(table, indent=1, default=float))
    print("\n" + extra + "\n\n" + text)


if __name__ == "__main__":
    main()
