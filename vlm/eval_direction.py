"""L2 direction benchmark on the frame bank: depth only, VLM only and depth + VLM choose a move (working plan v3).

    python -m vlm.eval_direction --bank /scratch/dns5508/frame_bank/CityEnviron --models Qwen/Qwen3-VL-8B-Instruct \
        --splits val --stride 4 --goal-inputs none arrow text arrow_text --images both --out results/direction_pilot.jsonl
    python -m vlm.eval_direction --summarize-only --out results/direction_pilot.jsonl        # re-score, no model, no bank

Per frame (train and val only; test stays closed, CLAUDE.md rule 3): the label (``vlm.direction.direction_label``,
ground truth, offline only), the rule controller's choice on the true depth (CPU, an upper bound) and, with
``--depth-model``, on the fast stack's estimate; then every model under every (goal input, image) condition. Rows are
keyed (model, condition, stem) and the file is appended to, so a killed job resumes (CLAUDE.md rule 10). Each VLM row
keeps the four log-probabilities, so other cuts and combinations are re-scored with no GPU. Combinations (depth veto)
and the oracle, random and always-goal-side baselines are computed at scoring time.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vlm import direction as D
from vlm.eval_bank import REPO, RUNS_LOG, read_rows, select_frames

BASE_KEYS = ("episode", "step", "stem", "split", "obstacle", "weather", "kind", "valid", "best", "need", "goal_deg", "along", "lateral")


def base_row(fr: dict[str, Any], goal_deg: float, label: dict[str, Any]) -> dict[str, Any]:
    return {"episode": int(fr["episode"]), "step": int(fr["step"]), "stem": fr["stem"], "split": fr["split"], "obstacle": fr["obstacle"],
            "weather": fr["condition"], "kind": label["kind"], "valid": label["valid"], "best": label["best"],
            "need": {k: round(float(v), 3) for k, v in label["need"].items()}, "goal_deg": goal_deg,
            "along": fr.get("gt_along_m"), "lateral": fr.get("gt_lateral_m")}


def evaluate(judges: dict[str, Any], bank: str | Path, frames: list[dict[str, Any]], out_path: str | Path,
             goal_inputs: Iterable[str], images: Iterable[str], depth_stack: Any | None = None, goal_seed: int = 0,
             hfov_deg: float = 90.0) -> list[dict[str, Any]]:
    """Fill ``out_path`` with every row not yet in it (frames the label skips are not asked); return all rows."""
    from perception.frame_bank import load_depth_gt, load_rgb

    goal_inputs, images = tuple(goal_inputs), tuple(images)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    done = {(r["model"], r["condition"], r["stem"]) for r in rows}
    conds = [f"{g}_{im}" for g in goal_inputs for im in images]
    base_models = ["depth_gt", "depth_est"] if depth_stack is not None else ["depth_gt"]
    need_work = lambda fr: any((m, c, fr["stem"]) not in done for m in judges for c in conds) or any(  # noqa: E731
        (f"{b}_{mode}", "-", fr["stem"]) not in done for b in base_models for mode in ("choose", "climb_first"))
    todo = [fr for fr in frames if need_work(fr)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(frames)} frames, {len(todo)} with work left; models {list(judges)}; conditions {conds}", flush=True)
    with open(out_path, "a") as f:
        def emit(row: dict[str, Any]) -> None:
            f.write(json.dumps(row) + "\n")
            rows.append(row)

        for k, fr in enumerate(todo, 1):
            stem = fr["stem"]
            goal = D.goal_bearing_deg(int(fr["episode"]), goal_seed)
            label = D.direction_label(fr, goal)
            if label["kind"] == "skip":
                continue
            base = base_row(fr, goal, label)
            rgb = load_rgb(bank, stem)
            alt = -float(fr["pose"][2]) if fr.get("pose") else 3.0
            gt_depth = load_depth_gt(bank, stem)
            maps = {"depth_gt": gt_depth}
            if depth_stack is not None:
                maps["depth_est"] = depth_stack.perceive(rgb).depth
            for name, depth in maps.items():
                bands = D.depth_bands(depth)
                for mode in ("choose", "climb_first"):
                    key = (f"{name}_{mode}", "-", stem)
                    if key not in done:
                        emit({"model": key[0], "condition": "-", **base, "choice": D.controller_choice(depth, goal, mode, alt),
                              "bands": {a: round(b, 2) for a, b in bands.items()}})
            for model_name, judge in judges.items():
                for g in goal_inputs:
                    for im in images:
                        cond = f"{g}_{im}"
                        if (model_name, cond, stem) in done:
                            continue
                        imgs, prompt = D.render_inputs(rgb, g, goal, im, hfov_deg)
                        logp, latency = judge.ask_direction(imgs, prompt)
                        emit({"model": model_name, "condition": cond, **base, "choice": max(logp, key=logp.get),
                              "logp": {a: round(b, 4) for a, b in logp.items()}, "latency_s": round(latency, 4)})
            f.flush()
            if k % 100 == 0:
                print(f"  {k}/{len(todo)} frames", flush=True)
    return rows


def with_baselines(rows: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Add oracle, random, always-goal-side and depth-veto rows (computed from stored rows; no model)."""
    rng = np.random.default_rng(seed)
    out = list(rows)
    seen: set[str] = set()
    est = {r["stem"]: r for r in rows if r["model"] == "depth_est_choose"}
    gt = {r["stem"]: r for r in rows if r["model"] == "depth_gt_choose"}
    for r in rows:
        if r["stem"] in seen or not r["model"].startswith("depth_gt_"):
            continue
        seen.add(r["stem"])
        core = {k: r[k] for k in BASE_KEYS if k in r}
        side = D.goal_side(r["goal_deg"])
        out.append({"model": "oracle", "condition": "-", **core, "choice": r["best"]})
        out.append({"model": "random", "condition": "-", **core, "choice": D.OPTIONS[int(rng.integers(len(D.OPTIONS)))]})
        out.append({"model": "goal_side", "condition": "-", **core, "choice": "straight" if r["kind"] == "straight" else (side or "over")})
    for r in rows:
        if "logp" not in r:
            continue
        for name, ref in (("depth_est", est), ("depth_gt", gt)):
            d = ref.get(r["stem"])
            if d is not None:
                core = {k: r[k] for k in BASE_KEYS if k in r}
                out.append({"model": f"{r['model']}+veto_{name}", "condition": r["condition"], **core,
                            "choice": D.depth_veto(r["choice"], d["bands"], d["choice"])})
    return out


def _log_run(args: argparse.Namespace, n_rows: int, wall_seconds: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(), "command": " ".join(sys.argv),
                            "config": {"models": args.models, "goal_inputs": args.goal_inputs, "images": args.images, "splits": args.splits,
                                       "stride": args.stride, "per_type": args.per_type, "depth_model": args.depth_model},
                            "seeds": f"bank split {args.split}; goal seed {args.goal_seed}", "n_episodes": n_rows,
                            "wall_seconds": round(wall_seconds, 1), "outcome": outcome}) + "\n")


def main(argv: list[str] | None = None) -> None:
    from vlm.judge import REFERENCE_MODEL, QwenVLJudge

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank")
    p.add_argument("--split", default=str(REPO / "configs" / "bank_split.yaml"))
    p.add_argument("--splits", nargs="+", default=["val"], help="never 'test' before the end (CLAUDE.md rule 3)")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--per-type", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=[REFERENCE_MODEL])
    p.add_argument("--one-at-a-time", action="store_true")
    p.add_argument("--goal-inputs", nargs="+", default=["none", "arrow", "text"], choices=list(D.GOAL_INPUTS))
    p.add_argument("--images", nargs="+", default=["both"], choices=list(D.IMAGES))
    p.add_argument("--depth-model", action="store_true", help="also run Depth Anything V2 (GPU) for the depth-estimate baseline")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=str(REPO / "results" / "direction_bench.jsonl"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if "test" in args.splits:
        raise SystemExit("refusing to evaluate on the test split before the end of the project (CLAUDE.md rule 3)")
    if not args.summarize_only:
        if not args.bank:
            raise SystemExit("--bank is required unless --summarize-only")
        frames = select_frames(args.bank, args.split, args.splits, args.stride, args.per_type, args.seed)
        stack = None
        if args.depth_model:
            from perception.depth_anything import DepthAnythingV2

            stack = DepthAnythingV2()
        start, outcome, rows = time.time(), "failed", []
        try:
            if args.one_at_a_time:
                for m in args.models:
                    judge = QwenVLJudge(model_id=m, dtype=args.dtype)
                    rows = evaluate({m: judge}, args.bank, frames, args.out, args.goal_inputs, args.images, stack, args.goal_seed)
                    from vlm.eval_bank import release

                    release(judge)
            else:
                rows = evaluate({m: QwenVLJudge(model_id=m, dtype=args.dtype) for m in args.models}, args.bank, frames, args.out,
                                args.goal_inputs, args.images, stack, args.goal_seed)
            outcome = "success"
        finally:
            _log_run(args, len(rows), time.time() - start, outcome)
    rows = read_rows(args.out)
    if not rows:
        raise SystemExit(f"no rows in {args.out}")
    summary = D.summarize(with_baselines(rows))
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    text = D.table(summary)
    Path(args.out).with_suffix(".report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
