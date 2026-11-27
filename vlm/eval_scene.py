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

OLD_CONDITIONS = ("json_raw", "grid_raw", "yesno_both")  # the first run (job 44259): prompt v1
V2_CONDITIONS = ("v2_both", "v2near_both", "v2thin_both", "v2_raw", "jsonv2_both", "jsonv2_raw")  # plan v8 step 1: prompt v2
CONDITIONS = OLD_CONDITIONS + V2_CONDITIONS
CORRIDOR_BOX = S.CORRIDOR_BOX
D_SAFE_M = S.DEPTH_SAFE_M  # the fast stack acts below this: a depth "miss" is an object nearer than this that the estimate shows farther
TUNE_MAX_FPR = 0.10  # the cut of a yes/no question: lowest one with at most this false-alarm rate on TRAIN frames (as in the benchmark)


def is_yesno(cond: str) -> bool:
    return not (cond.startswith("json") or cond.startswith("grid"))


def has_boxes(cond: str) -> bool:
    return cond.startswith("json") or cond.startswith("grid")


def depth_misses(depth_est: np.ndarray, box: list[float], alt_m: float, hfov_deg: float = 90.0) -> bool:
    """True when the fast stack's depth estimate does not show the object as nearer than ``D_SAFE_M`` inside its box.

    Stored per object; the scorer counts it as a depth miss only when the object is truly nearer than ``D_SAFE_M`` (``vlm.scene.score_frame``)."""
    h, w = depth_est.shape[:2]
    c0, c1 = int(box[0] / S.BOX_SCALE * w), max(int(np.ceil(box[2] / S.BOX_SCALE * w)), int(box[0] / S.BOX_SCALE * w) + 1)
    r0, r1 = int(box[1] / S.BOX_SCALE * h), max(int(np.ceil(box[3] / S.BOX_SCALE * h)), int(box[1] / S.BOX_SCALE * h) + 1)
    r1 = max(min(r1, S.ground_row(alt_m, D_SAFE_M, (h, w), hfov_deg)), r0 + 1)  # rows that may be flat ground do not count
    return bool(float(np.min(depth_est[r0:r1, c0:c1])) >= D_SAFE_M)


def build_pred(row: dict[str, Any], cut: float, grid_cut: float) -> tuple[list[dict[str, Any]], bool]:
    """The objects a stored model output means at the given cuts (``cut`` for a yes/no question). Returns (objects, parsed ok)."""
    out, cond = row["out"], row["condition"]
    if cond.startswith("json"):
        return S.parse_objects(out["text"])
    if cond.startswith("grid"):
        return S.grid_objects(out["scores"], grid_cut), True
    return ([{"box": CORRIDOR_BOX, "kind": "other", "range": "near"}] if out["confidence"] > cut else []), True


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
                elif cond in V2_CONDITIONS:
                    both = cond.endswith("_both")
                    overlay, crop = render(rgb)
                    imgs = [overlay, crop] if both else [rgb]
                    key = cond.rsplit("_", 1)[0]
                    if key.startswith("json"):
                        text, lat = judge.generate(imgs, S.scene_prompt_v2(), max_new_tokens)
                        out = {"text": text}
                    else:
                        sc, lat = judge.score_yes_no(imgs, S.path_prompt(key, overlay=both))
                        out = {"confidence": round(float(sc), 4)}
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


def has_near_truth(r: dict[str, Any]) -> bool:
    return any(t["dist"] < S.NEAR_M for t in r["truth"])


def tune_cuts(rows: list[dict[str, Any]], default: float) -> dict[str, float]:
    """Per yes/no condition: the lowest cut with at most ``TUNE_MAX_FPR`` false alarms on the TRAIN rows (frames with nothing within 15 m).

    Conditions without train rows keep ``default``. Chosen on train, reported on val (the benchmark's own rule)."""
    from vlm.eval_bank import tune_tau

    cuts: dict[str, float] = {}
    for cond in sorted({r["condition"] for r in rows if is_yesno(r["condition"])}):
        tr = [{"confidence": r["out"]["confidence"], "label": "neg"} for r in rows
              if r["condition"] == cond and r["split"] == "train" and not has_near_truth(r)]
        cut = tune_tau(tr, TUNE_MAX_FPR) if tr else float("nan")
        cuts[cond] = default if cut != cut else cut
    return cuts


def score_rows(rows: list[dict[str, Any]], cuts: dict[str, float] | float, grid_cut: float) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        cut = cuts.get(r["condition"], -4.02) if isinstance(cuts, dict) else float(cuts)
        pred, ok = build_pred(r, cut, grid_cut)
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
        if has_boxes(r["condition"]):
            out.append({"model": f"scene_{r['condition']}", "condition": "boxes", **core, "choice": S.boxes_to_move(r["pred"], r["goal_deg"])})
    return out


def family_auroc(scored: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per yes/no condition and obstacle type: AUROC of the confidence on frames with an object within 15 m vs frames with none."""
    from vlm.eval_bank import auroc

    out: dict[str, dict[str, float]] = {}
    for cond in sorted({r["condition"] for r in scored if is_yesno(r["condition"])}):
        rs = [r for r in scored if r["condition"] == cond]
        neg = np.array([r["out"]["confidence"] for r in rs if not has_near_truth(r)], dtype=float)
        row = {}
        for fam in sorted({r["obstacle"] for r in rs if has_near_truth(r)}):
            pos = np.array([r["out"]["confidence"] for r in rs if r["obstacle"] == fam and has_near_truth(r)], dtype=float)
            if len(pos) >= 8 and len(neg):
                row[fam] = round(auroc(pos, neg), 3)
        out[f"{rs[0]['model']}|{cond}"] = row
    return out


def summarize(rows: list[dict[str, Any]], yesno_cut: float = -4.02, grid_cut: float = 0.0, report_split: str = "val") -> dict[str, Any]:
    """Cuts are tuned on the train rows (when there are any) and every number is reported on ``report_split`` rows (all rows if none)."""
    cuts = tune_cuts(rows, yesno_cut)
    shown = [r for r in rows if r["split"] == report_split] or rows
    scored = score_rows(shown, cuts, grid_cut)
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in scored:
        by_cond.setdefault(f"{r['model']}|{r['condition']}", []).append(r)
    scene = {k: S.summarize_condition(v) for k, v in by_cond.items()}
    base_src = next(iter(by_cond.values()))
    for name, rs in S.baseline_rows(base_src).items():
        scene[f"{name}|-"] = S.summarize_condition(rs)
    steer = D.summarize(with_baselines(steering_rows(scored)))
    fam = family_auroc(scored)
    fam_text = "\n".join(f"- {k}: " + ", ".join(f"{f} {v}" for f, v in d.items()) for k, d in fam.items())
    return {"cuts": {"yes_no": cuts, "grid": grid_cut}, "report_split": report_split if shown is not rows else "all", "scene": scene,
            "steering": dict(steer), "family_auroc": fam, "scene_table": S.table(scene), "steering_table": D.table(steer),
            "family_text": fam_text}


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
    p.add_argument("--report-split", default="val", help="the rows every number is reported on; yes/no cuts are tuned on train rows")
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
    summary = summarize(rows, args.yesno_cut, args.grid_cut, args.report_split)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, default=float))
    text = ("Scene read: frame and object level (truth = true depth of every surface, placed or scenery). "
            f"Yes/no cuts tuned on train (FPR <= {TUNE_MAX_FPR}): " + json.dumps({k: round(v, 2) for k, v in summary["cuts"]["yes_no"].items()}) + "\n\n"
            + summary["scene_table"] + "\nAUROC by obstacle type, yes/no questions (object within 15 m vs none)\n\n" + summary["family_text"]
            + "\n\nSteering arm (boxes turned into a move; same answer key as the direction pilot)\n\n" + summary["steering_table"])
    Path(args.out).with_suffix(".report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
