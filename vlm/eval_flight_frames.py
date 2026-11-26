"""Is the yes/no read-out usable on pictures from real flights? (plan v6, step 1.)

    python -m vlm.eval_flight_frames --flights results/videos/city_*_always_qwen8b --out results/flight_frames_8b.jsonl
    python -m vlm.eval_flight_frames --summarize-only --out results/flight_frames_8b.jsonl      # re-score, no model

Re-asks the judge about every (stride-th) frame of saved flights (``flight.jsonl`` plus the ``raw/`` pictures) and
labels each frame from the simulator's true depth, offline only (CLAUDE.md rule 5):

* ``placed``  a placed obstacle in the path, closer than ``--d-pos`` (the bank's positive);
* ``scenery`` a map surface (tree, lamppost, building, rising ground) closer than ``--d-pos`` and no placed obstacle;
* ``clear``   nothing at all in the window within ``--d-neg`` (the bank's negative).

Why: the cut (-4.02) was tuned on bank pictures; in four city flights it said yes on 84% of calls with no placed
obstacle. This table shows the false-alarm and recall rates per cut on the flight pictures themselves. Resumable:
rows are keyed (model, episode, t).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm.eval_bank import CONDITIONS, REPO, auroc

CUTS = tuple(float(c) for c in np.arange(-6.0, 4.01, 1.0))
CLASSES = ("placed", "scenery", "clear")


def frame_class(d_all: float, d_placed: float, family: str | None, d_pos: float = 15.0, d_neg: float = 30.0) -> str | None:
    """Label from true distances; ``None`` for the ambiguous middle (excluded, as in the bank benchmark)."""
    if family and d_placed < d_pos:
        return "placed"
    if d_all < d_pos:
        return "scenery"
    if d_all >= d_neg:
        return "clear"
    return None


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def summarize(rows: list[dict[str, Any]], cuts: tuple[float, ...] = CUTS) -> dict[str, Any]:
    """Per class: n, score quartiles, and the share of frames above each cut (yes-rate). AUROC of each class vs clear."""
    by = {c: np.array([r["confidence"] for r in rows if r["cls"] == c], dtype=float) for c in CLASSES}
    out: dict[str, Any] = {"n": {c: int(len(v)) for c, v in by.items()}, "yes_rate": {}, "quartiles": {}, "auroc_vs_clear": {}}
    for c, v in by.items():
        out["yes_rate"][c] = {f"{t:g}": (round(float((v > t).mean()), 3) if len(v) else None) for t in cuts}
        out["quartiles"][c] = [round(float(q), 2) for q in np.percentile(v, [25, 50, 75])] if len(v) else None
    for c in ("placed", "scenery"):
        out["auroc_vs_clear"][c] = round(auroc(by[c], by["clear"]), 3) if len(by[c]) and len(by["clear"]) else None
    return out


def table(summary: dict[str, Any]) -> str:
    """Markdown: yes-rate by cut. 'clear' is the false-alarm rate; 'placed' is the recall."""
    cuts = list(next(iter(summary["yes_rate"].values())).keys())
    lines = ["Yes-rate per cut (clear = false alarms, placed = recall, scenery = map objects nearer than the bank positive).", "",
             "n: " + ", ".join(f"{c} {summary['n'][c]}" for c in CLASSES),
             "score quartiles (25/50/75): " + ", ".join(f"{c} {summary['quartiles'][c]}" for c in CLASSES),
             "AUROC vs clear: " + ", ".join(f"{c} {v}" for c, v in summary["auroc_vs_clear"].items()), "",
             "| cut | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for t in cuts:
        lines.append(f"| {t} | " + " | ".join(str(summary["yes_rate"][c][t]) for c in CLASSES) + " |")
    return "\n".join(lines) + "\n"


def flight_frames(flight_dir: Path, stride: int) -> list[dict[str, Any]]:
    lines = (flight_dir / "flight.jsonl").read_text().splitlines()
    return [json.loads(line) for i, line in enumerate(lines) if line.strip() and i % stride == 0]


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def evaluate(judge: Any, model: str, flights: list[Path], out_path: str | Path, stride: int, d_pos: float, d_neg: float,
             load_rgb=None, load_depth=None) -> list[dict[str, Any]]:
    from perception.depth_utils import corridor_min
    from vlm.judge import INVERTED_PROMPTS, prompt_text
    from vlm.overlay import render

    if load_rgb is None:
        from PIL import Image

        def load_rgb(p):  # noqa: E306
            return np.asarray(Image.open(_path(p)).convert("RGB"))
    load_depth = load_depth or (lambda p: np.load(_path(p)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    done = {(r["model"], r["episode"], r["t"]) for r in rows}
    use_overlay, use_crop, prompt_key = CONDITIONS["both"]  # what the flights used (prompt_key "cross", overlay + crop)
    judge.use_overlay, judge.use_crop = use_overlay, use_crop
    judge.prompt = prompt_text(prompt_key, use_overlay)
    judge.invert = prompt_key in INVERTED_PROMPTS
    n = 0
    with open(out_path, "a") as f:
        for fd in flights:
            for fr in flight_frames(fd, stride):
                if (model, fr["episode_id"], fr["t"]) in done:
                    continue
                cls = frame_class(corridor_min(load_depth(fr["depth_gt_path"])), float(fr["corridor_min_gt"]),
                                  fr.get("obstacle_family"), d_pos, d_neg)
                if cls is None:
                    continue
                rgb = load_rgb(fr["rgb_path"])
                overlay, crop = render(rgb)
                ans = judge.ask(rgb, overlay, crop)
                row = {"model": model, "episode": fr["episode_id"], "t": fr["t"], "cls": cls,
                       "d_all": round(float(corridor_min(load_depth(fr["depth_gt_path"]))), 2),
                       "d_placed": round(float(fr["corridor_min_gt"]), 2), "action_taken": fr["action_taken"],
                       "confidence": round(float(ans.confidence), 4), "latency_s": round(float(ans.latency_s), 4)}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
                n += 1
                if n % 100 == 0:
                    f.flush()
                    print(f"  {n} frames", flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    from vlm.judge import REFERENCE_MODEL

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flights", nargs="+", help="flight folders (each has flight.jsonl and raw/)")
    p.add_argument("--model", default=REFERENCE_MODEL)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--d-pos", type=float, default=15.0)
    p.add_argument("--d-neg", type=float, default=30.0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=str(REPO / "results" / "flight_frames.jsonl"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args(argv)
    if not args.summarize_only:
        if not args.flights:
            raise SystemExit("--flights is required unless --summarize-only")
        from vlm.judge import QwenVLJudge

        start = time.time()
        evaluate(QwenVLJudge(model_id=args.model, dtype=args.dtype), args.model, [Path(f) for f in args.flights], args.out,
                 args.stride, args.d_pos, args.d_neg)
        print(f"[{time.strftime('%H:%M:%S')}] done in {time.time() - start:.0f} s", flush=True)
    rows = [r for r in read_rows(args.out)]
    if not rows:
        raise SystemExit(f"no rows in {args.out}")
    summary = summarize(rows)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    text = table(summary)
    Path(args.out).with_suffix(".report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
