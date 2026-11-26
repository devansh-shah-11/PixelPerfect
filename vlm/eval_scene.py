"""Scene-read benchmark on the frame bank: what is ahead and where? (plan v7, step A.)

    python -m vlm.eval_scene --bank /scratch/dns5508/frame_bank/CityEnviron --models Qwen/Qwen3-VL-8B-Instruct \
        --splits val --stride 4 --conditions json_raw grid_raw yesno_both --depth-model --out results/scene_val.jsonl
    python -m vlm.eval_scene --summarize-only --out results/scene_val.jsonl --yesno-cut -4.02 --grid-cut 0   # re-score, no GPU

Conditions: ``json_raw`` (one generated list of objects: box, kind, range), ``grid_raw`` (six yes/no questions, one per cell of a
3 x 2 grid, read from logits, the cheap fallback), ``yesno_both`` (today's single yes/no with path overlay and crop; one box over the
corridor when it says yes). Truth is the true depth of every surface, placed or scenery (``vlm.scene.truth_objects``), offline only.
Rows hold the raw model output, so cuts and parsing are re-scored without the GPU. The file is appended to and keyed
(model, condition, stem): a killed job resumes (CLAUDE.md rule 10). The test split is refused (rule 3).
Also scores a steering arm: the boxes turned into a move (``vlm.scene.boxes_to_move``), against the direction answer key.
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
from vlm import scene as S
from vlm.eval_bank import REPO, RUNS_LOG, ground_truth, read_rows, select_frames
from vlm.eval_direction import base_row, with_baselines

CONDITIONS = ("json_raw", "grid_raw", "yesno_both")
CORRIDOR_BOX = [450.0, 400.0, 550.0, 600.0]  # the highlighted window of vlm.overlay, in 0-1000 units
D_SAFE_M = 8.0  # the fast stack's L1 threshold: depth "misses" an object when its estimate there is not nearer than this


def depth_misses(depth_est: np.ndarray, box: list[float], alt_m: float, hfov_deg: float = 90.0) -> bool:
    """True when the fast stack's depth estimate does not show the object as nearer than ``D_SAFE_M`` inside its box."""
    h, w = depth_est.shape[:2]
    c0, c1 = int(box[0] / S.BOX_SCALE * w), max(int(np.ceil(box[2] / S.BOX_SCALE * w)), int(box[0] / S.BOX_SCALE * w) + 1)
    r0, r1 = int(box[1] / S.BOX_SCALE * h), max(int(np.ceil(box[3] / S.BOX_SCALE * h)), int(box[1] / S.BOX_SCALE * h) + 1)
    r1 = max(min(r1, S.ground_row(alt_m, D_SAFE_M, (h, w), hfov_deg)), r0 + 1)  # rows that may be flat ground do not count
    return bool(float(np.min(depth_est[r0:r1, c0:c1])) >= D_SAFE_M)


def build_pred(row: dict[str, Any], yesno_cut: float, grid_cut: float) -> tuple[list[dict[str, Any]], bool]:
    """The objects a stored model output means at the given cuts. Returns (objects, parsed ok)."""
    out, cond = row["out"], row["condition"]
    if cond.startswith("json"):
        return S.parse_objects(out["text"])
    if cond.startswith("grid"):
        return S.grid_objects(out["scores"], grid_cut), True
    return ([{"box": CORRIDOR_BOX, "kind": "other", "range": "near"}] if out["confidence"] > yesno_cut else []), True


def evaluate(judge: Any, model: str, bank: str | Path, frames: list[dict[str, Any]], out_path: str | Path, conditions: Iterable[str],
             depth_stack: Any | None = None, goal_seed: int = 0, hfov_deg: float = 90.0, max_new_tokens: int = 200,
             load_rgb=None, load_depth=None, load_mask=None) -> list[dict[str, Any]]:
    """Fill ``out_path`` with every (model, condition, frame) not yet in it; return all rows."""
    from vlm.judge import prompt_text
    from vlm.overlay import render

    if load_rgb is None:
        from perception.frame_bank import load_depth_gt, load_obstacle_mask, load_rgb
        load_depth, load_mask = load_depth_gt, load_obstacle_mask
    conditions = tuple(conditions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    done = {(r["model"], r["condition"], r["stem"]) for r in rows}
    todo = [fr for fr in frames if any((model, c, fr["stem"]) not in done for c in conditions)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(frames)} frames, {len(todo)} with work left; conditions {list(conditions)}", flush=True)
    with open(out_path, "a") as f:
        for k, fr in enumerate(todo, 1):
            stem = fr["stem"]
            rgb, depth, mask = load_rgb(bank, stem), load_depth(bank, stem), load_mask(bank, stem)
            alt = -float(fr["pose"][2]) if fr.get("pose") else 3.0
            gt = ground_truth(bank, fr, hfov_deg)
            placed = float(gt["corridor_min_gt_obstacle"])
            truth = S.truth_objects(depth, mask, alt, placed if placed < 99.0 else None, fr.get("obstacle"), hfov_deg)
            if depth_stack is not None:
                est = depth_stack.perceive(rgb).depth
                for t in truth:
                    t["depth_misses"] = depth_misses(est, t["box"], alt, hfov_deg)
            goal = D.goal_bearing_deg(int(fr["episode"]), goal_seed)
            label = D.direction_label(fr, goal)
            base = {**base_row(fr, goal, label), "truth": [{**t, "box": [round(v, 1) for v in t["box"]], "dist": round(t["dist"], 2)} for t in truth],
                    "alt_m": round(alt, 2), "depth_gt_choice": D.controller_choice(depth, goal, "choose", alt),
                    "depth_gt_bands": {a: round(b, 2) for a, b in D.depth_bands(depth).items()}}
            for cond in conditions:
                if (model, cond, stem) in done:
                    continue
                if cond == "json_raw":
                    text, lat = judge.generate([rgb], S.scene_prompt(), max_new_tokens)
                    out = {"text": text}
                elif cond == "grid_raw":
                    scores, lat = {}, 0.0
                    for r in range(len(S.GRID_ROWS)):
                        for c in range(len(S.GRID_COLS)):
                            s, dt = judge.score_yes_no([rgb], S.grid_prompt(r, c))
                            scores[S.cell_name(r, c)], lat = round(float(s), 3), lat + dt
                    out = {"scores": scores}
                else:
                    judge.use_overlay, judge.use_crop, judge.prompt, judge.invert = True, True, prompt_text("cross", True), False
                    overlay, crop = render(rgb)
                    ans = judge.ask(rgb, overlay, crop)
                    out, lat = {"confidence": round(float(ans.confidence), 4)}, float(ans.latency_s)
                row = {"model": model, "condition": cond, **base, "out": out, "latency_s": round(float(lat), 4)}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)} frames", flush=True)
    return rows


def score_rows(rows: list[dict[str, Any]], yesno_cut: float, grid_cut: float) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        pred, ok = build_pred(r, yesno_cut, grid_cut)
        out.append({**r, "pred": pred, "ok": ok, "score": S.score_frame(pred, r["truth"])})
    return out


def steering_rows(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows in the ``vlm.direction`` format: the move the boxes mean, plus the true-depth rule controller (an upper bound)."""
    out, seen = [], set()
    for r in scored:
        core = {k: r[k] for k in ("episode", "step", "stem", "split", "obstacle", "weather", "kind", "valid", "best", "need", "goal_deg", "along", "lateral")}
        if r["stem"] not in seen:
            seen.add(r["stem"])
            out.append({"model": "depth_gt_choose", "condition": "-", **core, "choice": r["depth_gt_choice"], "bands": r["depth_gt_bands"]})
        if r["condition"] != "yesno_both":
            out.append({"model": f"scene_{r['condition']}", "condition": "boxes", **core, "choice": S.boxes_to_move(r["pred"], r["goal_deg"])})
    return out


def summarize(rows: list[dict[str, Any]], yesno_cut: float = -4.02, grid_cut: float = 0.0) -> dict[str, Any]:
    scored = score_rows(rows, yesno_cut, grid_cut)
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in scored:
        by_cond.setdefault(f"{r['model']}|{r['condition']}", []).append(r)
    scene = {k: S.summarize_condition(v) for k, v in by_cond.items()}
    base_src = next(iter(by_cond.values()))
    for name, rs in S.baseline_rows(base_src).items():
        scene[f"{name}|-"] = S.summarize_condition(rs)
    steer = D.summarize(with_baselines(steering_rows(scored)))
    return {"cuts": {"yesno": yesno_cut, "grid": grid_cut}, "scene": scene, "steering": {k: v for k, v in steer.items()},
            "scene_table": S.table(scene), "steering_table": D.table(steer)}


def _log_run(args: argparse.Namespace, n_rows: int, wall_seconds: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(), "command": " ".join(sys.argv),
                            "config": {"models": args.models, "conditions": args.conditions, "splits": args.splits, "stride": args.stride,
                                       "depth_model": args.depth_model},
                            "seeds": f"bank split {args.splits}; goal seed {args.goal_seed}", "n_episodes": n_rows,
                            "wall_seconds": round(wall_seconds, 1), "outcome": outcome}) + "\n")


def main(argv: list[str] | None = None) -> None:
    from vlm.judge import REFERENCE_MODEL

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank")
    p.add_argument("--split", default=str(REPO / "configs" / "bank_split.yaml"))
    p.add_argument("--splits", nargs="+", default=["val"], help="never 'test' before the end (CLAUDE.md rule 3)")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--per-type", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-seed", type=int, default=0)
    p.add_argument("--models", nargs="+", default=[REFERENCE_MODEL])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--depth-model", action="store_true", help="also run Depth Anything V2 (GPU) to mark the objects the fast stack misses")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--yesno-cut", type=float, default=-4.02, help="cut on the yes/no confidence (the flights' cut)")
    p.add_argument("--grid-cut", type=float, default=0.0)
    p.add_argument("--out", default=str(REPO / "results" / "scene_bench.jsonl"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if "test" in args.splits:
        raise SystemExit("refusing to evaluate on the test split before the end of the project (CLAUDE.md rule 3)")
    if not args.summarize_only:
        if not args.bank:
            raise SystemExit("--bank is required unless --summarize-only")
        from vlm.judge import QwenVLJudge

        frames = select_frames(args.bank, args.split, args.splits, args.stride, args.per_type, args.seed)
        stack = None
        if args.depth_model:
            from perception.depth_anything import DepthAnythingV2

            stack = DepthAnythingV2()
        start, outcome, rows = time.time(), "failed", []
        try:
            for m in args.models:
                judge = QwenVLJudge(model_id=m, dtype=args.dtype)
                rows = evaluate(judge, m, args.bank, frames, args.out, args.conditions, stack, args.goal_seed, max_new_tokens=args.max_new_tokens)
                from vlm.eval_bank import release

                release(judge)
            outcome = "success"
        finally:
            _log_run(args, len(rows), time.time() - start, outcome)
    rows = read_rows(args.out)
    if not rows:
        raise SystemExit(f"no rows in {args.out}")
    summary = summarize(rows, args.yesno_cut, args.grid_cut)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, default=float))
    text = ("Scene read: frame and object level (truth = true depth of every surface, placed or scenery)\n\n" + summary["scene_table"]
            + "\nSteering arm (boxes turned into a move; same answer key as the direction pilot)\n\n" + summary["steering_table"])
    Path(args.out).with_suffix(".report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
