"""Resumable sweep: every policy over the same seeds (PLAN_DETAILED.md D15, §7 rules; chunks M4.7 and M5.3).

    python -m eval.sweep --config configs/sweeps/raycast_smoke.yaml
    python -m eval.sweep --config configs/sweeps/airsim_bench.yaml --set stack=clear \
        --set judge_kw.model_id=Qwen/Qwen3-VL-8B-Instruct@4bit --set out=results/sweeps/bench_vlm_8b-4bit

One line per finished flight goes to ``<out>/results.jsonl``; on restart,
finished ``(policy, seed)`` pairs are skipped (CLAUDE.md rule 10, the 20-minute
idle shutdown). At the end the metrics table is printed and written to
``<out>/table.md``. Each run appends one line to ``logs/runs.jsonl``.

Config keys (YAML): ``backend``, ``stack``, ``judge`` (+ ``judge_kw``), ``delay_s``, ``split``,
``seeds`` (a list, or ``{file: configs/seeds.yaml, n: 100}``: always the first n, D24), ``scenario``
(ScenarioConfig fields, or ``{file: configs/scenario.yaml}`` for the frozen settings, which also carry
the start poses), ``policies`` (list of policy specs, each with a unique ``name``), ``out``,
``flight_logs`` (bool). ``--set a.b=value`` overrides a key, so one config serves every model variant.

Test seeds fly only from the frozen scenario file and only with ``final_test_run: true`` (D14, D24).
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
from eval.factory import fork_judge, make_judge, make_sim, make_stack
from eval.metrics import EpisodeResult, markdown, read_results, results_table
from eval.run_episode import LoopConfig, run_episode
from policies.base import make_policy
from sim.scenario import ScenarioConfig, generate, load_scenario_config, load_seed_lists

REPO = Path(__file__).resolve().parent.parent


def resolve_seeds(cfg: dict[str, Any]) -> list[int]:
    s = cfg["seeds"]
    if isinstance(s, list):
        return [int(x) for x in s]
    lists = load_seed_lists(_repo_path(s["file"]))
    split = cfg.get("split", "val")
    if split == "test" and not cfg.get("final_test_run", False):
        raise SystemExit("refusing to fly test seeds: set final_test_run: true only for the one final run (PLAN_DETAILED.md D14)")
    return lists[split][: s.get("n")]


def _repo_path(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else REPO / p


def resolve_scenario(cfg: dict[str, Any]) -> tuple[ScenarioConfig, list | None]:
    """(scenario settings, frozen start poses or None). Test seeds are refused unless the settings are frozen."""
    s = cfg.get("scenario", {}) or {}
    if "file" in s:
        frozen = load_scenario_config(_repo_path(s["file"]))
        return ScenarioConfig(**frozen["scenario"]), frozen["starts"]
    if cfg.get("split", "val") == "test":
        raise SystemExit("refusing to fly test seeds with unfrozen scenario settings: use scenario: {file: configs/scenario.yaml} (PLAN_DETAILED.md D24)")
    return ScenarioConfig(**s), None


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides in place (value parsed as YAML). Returns ``cfg``."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--set {key}: {part!r} is not a mapping")
        node[leaf] = yaml.safe_load(value)
    return cfg


def run(cfg: dict[str, Any]) -> list[EpisodeResult]:
    scen_cfg, frozen_starts = resolve_scenario(cfg)  # before anything is created: a refused test run leaves no trace
    seeds = resolve_seeds(cfg)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = {(r.policy, r.seed) for r in read_results(results_path)} if results_path.exists() else set()
    # Strings in backend_kw may name environment variables, e.g. configs: ${PP_SIM_CONFIGS} set by hpc/airsim_job.sbatch.
    backend_kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg.get("backend_kw", {}).items()}
    if frozen_starts and cfg.get("backend") == "airsim":
        if backend_kw.get("starts") not in (None, frozen_starts):
            raise SystemExit("backend_kw.starts differs from the frozen start poses in the scenario file (PLAN_DETAILED.md D24)")
        backend_kw["starts"] = frozen_starts
    sim = make_sim(cfg.get("backend", "raycast"), **backend_kw)
    stack = make_stack(cfg.get("stack", "true-depth"), **cfg.get("stack_kw", {}))
    judge, judge_name = make_judge(cfg.get("judge", "none"), **cfg.get("judge_kw", {}))
    loop = LoopConfig(delay_s=cfg.get("delay_s", 1.0), **cfg.get("loop", {}))
    names = [p["name"] for p in cfg["policies"]]
    if len(set(names)) != len(names):
        raise SystemExit("policy names must be unique")
    try:
        _fly(cfg, seeds, scen_cfg, sim, stack, judge, judge_name, loop, results_path, done, out)
    finally:
        if hasattr(sim, "close"):  # an open Project AirSim client can keep the job alive past the last flight (job 39930)
            sim.close()
    return read_results(results_path)


def _fly(cfg: dict[str, Any], seeds: list[int], scen_cfg: ScenarioConfig, sim: Any, stack: Any, judge: Any, judge_name: str,
         loop: LoopConfig, results_path: Path, done: set, out: Path) -> None:
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
                records, result = run_episode(scenario, sim, stack, RuleController(d_safe=loop.d_safe), policy,
                                              fork_judge(judge, seed, 0), loop, judge_name=judge_name)  # own seed per flight: resuming cannot change the draws
                if cfg.get("flight_logs"):
                    (out / "flights").mkdir(exist_ok=True)
                    (out / "flights" / f"{result.episode_id}.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
                f.write(result.to_json() + "\n")
                f.flush()


def _log_run(cfg_path: str, cfg: dict[str, Any], n: int, wall: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    (REPO / "logs").mkdir(exist_ok=True)
    with open(REPO / "logs" / "runs.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(),
                            "command": " ".join(sys.argv), "config": cfg_path, "seeds": cfg.get("seeds"),
                            "stack": cfg.get("stack"), "judge": cfg.get("judge"), "judge_kw": cfg.get("judge_kw"), "n_episodes": n,
                            "wall_seconds": round(wall, 1), "outcome": outcome}) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a config key, dotted for nested ones, e.g. --set judge_kw.model_id=Qwen/Qwen3-VL-8B-Instruct@4bit")
    p.add_argument("--no-log", action="store_true", help="do not append to logs/runs.jsonl (tests)")
    a = p.parse_args(argv)
    cfg = apply_overrides(yaml.safe_load(Path(a.config).read_text()), a.overrides)
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
    print("\n" + extra + "\n\n" + text, flush=True)


if __name__ == "__main__":
    main()
