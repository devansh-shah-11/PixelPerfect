"""Chunk M2.4: standalone VLM evaluation on labelled frame-bank frames.

    python -m vlm.eval_bank --bank /scratch/dns5508/frame_bank/CityEnviron \
        --out results/vlm_eval_val.jsonl --examples results/vlm_examples

Asks the judge the fixed question about every val-split frame under four
conditions (overlay and crop each on or off), for each model, and reports
precision and recall by obstacle type and by distance. Resumable: rows already
in ``--out`` are skipped, keyed by (model, condition, stem).

Ground truth (offline only, CLAUDE.md rule 5), from ``gate.labels.clean_corridor_min``:
the nearest true surface inside the corridor window, spawned obstacle or scene.
A frame is **positive** if that distance is below ``--d-pos`` (something is in
the highlighted region at a range the controller cares about), **negative** if
it is above ``--d-neg`` (nothing close), and **ambiguous** in between, which is
left out of precision and recall but counted. Both thresholds are reported with
the table, since the answer to "does anything cross the region" depends on
them. Distance buckets use the same distance.

``--examples DIR`` writes an eyeball sheet per model and condition: overlay and
crop side by side with the judge's answer and confidence written on it, for
up to ``--n-examples`` positives and negatives per obstacle type.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gate.labels import clean_corridor_min
from interfaces import VLMJudge
from perception.bank_split import load_split, split_of_episode
from perception.depth_utils import MAX_RANGE_M
from perception.frame_bank import frame_rows, load_depth_gt, load_obstacle_mask, load_rgb
from vlm.judge import DEFAULT_MODEL, REFERENCE_MODEL, QwenVLJudge
from vlm.overlay import render

REPO = Path(__file__).resolve().parent.parent
RUNS_LOG = REPO / "logs" / "runs.jsonl"

CONDITIONS = {  # name -> (use_overlay, use_crop)
    "both": (True, True),
    "overlay": (True, False),
    "crop": (False, True),
    "raw": (False, False),
}
BUCKETS = ((0, 3), (3, 6), (6, 10), (10, 20), (20, float("inf")))
MIN_ALONG_M = 1.0  # as in perception.analyze_bank: closer and the camera is inside the obstacle


def bucket(distance_m: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= distance_m < hi:
            return f"{lo}-{hi:g} m" if hi != float("inf") else f">{lo} m"
    return "n/a"


def frame_label(gt: dict[str, float | int], d_pos: float, d_neg: float) -> str:
    """'pos', 'neg' or 'ambiguous' from the clean corridor distance."""
    d = float(gt["corridor_min_gt_clean"])
    if d < d_pos:
        return "pos"
    if d > d_neg:
        return "neg"
    return "ambiguous"


def select_frames(bank: str | Path, split_path: str | Path, splits: tuple[str, ...], per_type: int | None, seed: int) -> list[dict[str, Any]]:
    """Frames of the chosen splits, dropping frames where the camera is past the obstacle; optionally capped per obstacle type."""
    of_episode = split_of_episode(load_split(split_path))
    rows = []
    for r in frame_rows(bank):
        if of_episode.get(int(r["episode"])) not in splits:
            continue
        along = r.get("gt_along_m")
        if along is not None and along < MIN_ALONG_M:
            continue
        rows.append(r)
    if per_type is None:
        return rows
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[str(r["obstacle"])].append(r)
    out = []
    for name in sorted(by_type):
        group = by_type[name]
        idx = sorted(rng.choice(len(group), size=min(per_type, len(group)), replace=False))
        out.extend(group[i] for i in idx)
    return out


def ground_truth(bank: str | Path, stem: str) -> dict[str, float | int]:
    return clean_corridor_min(load_depth_gt(bank, stem), load_obstacle_mask(bank, stem), MAX_RANGE_M)


def evaluate(
    judges: dict[str, VLMJudge],
    bank: str | Path,
    frames: list[dict[str, Any]],
    out_path: str | Path,
    d_pos: float,
    d_neg: float,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
    examples_dir: str | Path | None = None,
    n_examples: int = 6,
) -> list[dict[str, Any]]:
    """Run every (model, condition, frame) not yet in ``out_path``; return all rows (old and new)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if out_path.exists():
        with open(out_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    done = {(r["model"], r["condition"], r["stem"]) for r in rows}
    gt_cache: dict[str, dict[str, float | int]] = {}
    sheets: dict[tuple[str, str], ExampleSheet] = {}

    with open(out_path, "a") as f:
        for model_name, judge in judges.items():
            for cond in conditions:
                use_overlay, use_crop = CONDITIONS[cond]
                judge.use_overlay, judge.use_crop = use_overlay, use_crop  # type: ignore[attr-defined]
                todo = [r for r in frames if (model_name, cond, r["stem"]) not in done]
                print(f"[{time.strftime('%H:%M:%S')}] {model_name} / {cond}: {len(todo)} frames to do, {len(frames) - len(todo)} done", flush=True)
                sheet = ExampleSheet(Path(examples_dir) / f"{_slug(model_name)}_{cond}", n_examples) if examples_dir else None
                for k, r in enumerate(todo, 1):
                    stem = r["stem"]
                    if stem not in gt_cache:
                        gt_cache[stem] = ground_truth(bank, stem)
                    gt = gt_cache[stem]
                    rgb = load_rgb(bank, stem)
                    overlay, crop = render(rgb)
                    ans = judge.ask(rgb, overlay, crop)
                    label = frame_label(gt, d_pos, d_neg)
                    row = {
                        "model": model_name, "condition": cond, "stem": stem,
                        "episode": r["episode"], "step": r["step"], "obstacle": r["obstacle"], "cond_weather": r["condition"],
                        "gt_clean_m": round(float(gt["corridor_min_gt_clean"]), 3),
                        "gt_obstacle_m": round(float(gt["corridor_min_gt_obstacle"]), 3),
                        "obstacle_px": int(gt["obstacle_px"]),
                        "label": label, "bucket": bucket(float(gt["corridor_min_gt_clean"])),
                        "answer": bool(ans.answer), "confidence": round(float(ans.confidence), 4), "latency_s": round(float(ans.latency_s), 4),
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows.append(row)
                    if sheet is not None:
                        sheet.maybe_add(row, overlay, crop)
                    if k % 200 == 0:
                        print(f"  {k}/{len(todo)}", flush=True)
                if sheet is not None:
                    sheet.save()
    return rows


# ---- summary -----------------------------------------------------------------------


def prf(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    pos = [r for r in rows if r["label"] == "pos"]
    neg = [r for r in rows if r["label"] == "neg"]
    tp = sum(r["answer"] for r in pos)
    fp = sum(r["answer"] for r in neg)
    recall = tp / len(pos) if pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / len(neg) if neg else float("nan")
    return {"n_pos": len(pos), "n_neg": len(neg), "n_ambiguous": sum(r["label"] == "ambiguous" for r in rows),
            "tp": int(tp), "fp": int(fp), "recall": recall, "precision": precision, "fpr": fpr}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per model and condition: overall PRF, recall by obstacle type, recall and FPR by distance bucket."""
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    for (model, cond), g in sorted(groups.items()):
        by_type = defaultdict(list)
        by_bucket = defaultdict(list)
        for r in g:
            by_type[str(r["obstacle"])].append(r)
            by_bucket[r["bucket"]].append(r)
        out[f"{model} / {cond}"] = {
            "overall": prf(g),
            "by_obstacle": {t: prf(v) for t, v in sorted(by_type.items())},
            "by_bucket": {b: prf(v) for b, v in sorted(by_bucket.items(), key=lambda kv: _bucket_key(kv[0]))},
        }
    return out


def _bucket_key(name: str) -> float:
    return float(name.lstrip(">").split("-")[0].split()[0])


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def markdown_tables(summary: dict[str, Any], d_pos: float, d_neg: float) -> str:
    lines = [f"Positive: clean corridor distance < {d_pos:g} m. Negative: > {d_neg:g} m. In between: ambiguous, excluded.", "",
             "| Model / condition | n_pos | n_neg | recall | precision | FPR |", "|---|---|---|---|---|---|"]
    for key, s in summary.items():
        o = s["overall"]
        lines.append(f"| {key} | {o['n_pos']} | {o['n_neg']} | {_fmt(o['recall'])} | {_fmt(o['precision'])} | {_fmt(o['fpr'])} |")
    for key, s in summary.items():
        if not key.endswith("/ both"):
            continue
        lines += ["", f"**{key}**, recall by obstacle type (n_pos):", "", "| Obstacle | n_pos | recall | n_neg | FPR |", "|---|---|---|---|---|"]
        for t, p in s["by_obstacle"].items():
            lines.append(f"| {t} | {p['n_pos']} | {_fmt(p['recall'])} | {p['n_neg']} | {_fmt(p['fpr'])} |")
        lines += ["", f"**{key}**, by distance of the nearest true surface in the region:", "", "| Distance | n_pos | recall | n_neg | FPR |", "|---|---|---|---|---|"]
        for b, p in s["by_bucket"].items():
            lines.append(f"| {b} | {p['n_pos']} | {_fmt(p['recall'])} | {p['n_neg']} | {_fmt(p['fpr'])} |")
    return "\n".join(lines)


# ---- eyeball sheet -------------------------------------------------------------------


class ExampleSheet:
    """Collects up to ``n`` positive and ``n`` negative frames per obstacle type and saves one PNG per type."""

    def __init__(self, out_dir: Path, n: int) -> None:
        self.out_dir, self.n = out_dir, n
        self.tiles: dict[str, list[Image.Image]] = defaultdict(list)
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def maybe_add(self, row: dict[str, Any], overlay: np.ndarray, crop: np.ndarray) -> None:
        key = (str(row["obstacle"]), row["label"])
        if row["label"] == "ambiguous" or self.counts[key] >= self.n:
            return
        self.counts[key] += 1
        self.tiles[str(row["obstacle"])].append(tile(row, overlay, crop))

    def save(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for obstacle, tiles in self.tiles.items():
            w, h = tiles[0].size
            sheet = Image.new("RGB", (w, h * len(tiles)), "white")
            for i, t in enumerate(tiles):
                sheet.paste(t, (0, i * h))
            p = self.out_dir / f"{obstacle}.png"
            sheet.save(p)
            paths.append(p)
        return paths


def tile(row: dict[str, Any], overlay: np.ndarray, crop: np.ndarray, text_h: int = 18) -> Image.Image:
    """Overlay and crop side by side with the label and the judge's answer written above."""
    ov, cr = Image.fromarray(overlay), Image.fromarray(crop)
    cr = cr.resize((int(cr.width * ov.height / cr.height), ov.height))
    im = Image.new("RGB", (ov.width + cr.width + 8, ov.height + text_h), "white")
    im.paste(ov, (0, text_h))
    im.paste(cr, (ov.width + 8, text_h))
    verdict = "correct" if (row["answer"] == (row["label"] == "pos")) else "WRONG"
    msg = (f"{row['stem']}  {row['obstacle']}  truth={row['label']} ({row['gt_clean_m']} m)  "
           f"judge={'yes' if row['answer'] else 'no'} ({row['confidence']:+.2f})  {verdict}")
    ImageDraw.Draw(im).text((4, 2), msg, fill="red" if verdict == "WRONG" else "black")
    return im


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-Instruct", "").lower()


def _log_run(args: argparse.Namespace, n_rows: int, wall_seconds: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(),
                            "command": " ".join(sys.argv),
                            "config": {"models": args.models, "conditions": args.conditions, "splits": args.splits, "d_pos": args.d_pos,
                                       "d_neg": args.d_neg, "per_type": args.per_type, "dtype": args.dtype},
                            "seeds": f"split {args.split}; frame sample seed {args.seed}", "n_episodes": n_rows,
                            "wall_seconds": round(wall_seconds, 1), "outcome": outcome}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank", required=True)
    p.add_argument("--split", default=str(REPO / "configs" / "bank_split.yaml"))
    p.add_argument("--splits", nargs="+", default=["val"], help="never 'test' before the end (CLAUDE.md rule 3)")
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, REFERENCE_MODEL])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--d-pos", type=float, default=15.0, help="positive if the nearest true surface in the region is closer than this")
    p.add_argument("--d-neg", type=float, default=30.0, help="negative if farther than this")
    p.add_argument("--per-type", type=int, help="cap frames per obstacle type (seeded sample); all if omitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=str(REPO / "results" / "vlm_eval_val.jsonl"))
    p.add_argument("--examples", help="directory for eyeball sheets; off if omitted")
    p.add_argument("--n-examples", type=int, default=6)
    args = p.parse_args()
    if "test" in args.splits:
        raise SystemExit("refusing to evaluate on the test split before the end of the project (CLAUDE.md rule 3)")

    frames = select_frames(args.bank, args.split, tuple(args.splits), args.per_type, args.seed)
    print(f"{len(frames)} frames from splits {args.splits}", flush=True)
    judges = {m: QwenVLJudge(model_id=m, dtype=args.dtype) for m in args.models}
    start, outcome, rows = time.time(), "failed", []
    try:
        rows = evaluate(judges, args.bank, frames, args.out, args.d_pos, args.d_neg, tuple(args.conditions), args.examples, args.n_examples)
        outcome = "success"
    finally:
        _log_run(args, len(rows), time.time() - start, outcome)
    summary = summarize(rows)
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"d_pos": args.d_pos, "d_neg": args.d_neg, "splits": args.splits, "summary": summary}, indent=1))
    print("\n" + markdown_tables(summary, args.d_pos, args.d_neg))
    print(f"\nrows: {args.out}\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
