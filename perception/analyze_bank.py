"""First look at a bank run: miss rates, scale check, eyeball sheet (tasks W-B2, W-B3).

Reads the JSONL written by ``perception.run_bank``. Only train and val frames
are ever loaded; test episodes stay closed (CLAUDE.md rule 4). The scale is
fitted on train and reported on val.

The scale check compares corridor minimum depths, not full depth maps. The bank
stores distance along the pixel ray and the model predicts planar depth, but in
the corridor window (central 10% by 20% of the image) the two differ by about
1% or less for any plausible field of view, so no conversion is applied here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gate.labels import Outcome, outcomes
from perception.bank_split import load_split, split_of_episode
from perception.depth_utils import MAX_RANGE_M, corridor_window
from perception.frame_bank import load_rgb, read_jsonl

# gt_along_m is measured to the obstacle's centre. Below this the camera is
# inside or past the obstacle and the frame says nothing about perception.
MIN_ALONG_M = 1.0


def load_run(
    run_path: str | Path, split: dict[str, Any], min_along: float = MIN_ALONG_M, gt_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Run rows from train and val episodes, each tagged with ``split``.

    With ``gt_path`` (from ``gate.bank_gt``), ``corridor_min_gt`` is replaced by
    the clean value; the raw one is kept as ``corridor_min_gt_raw``.
    """
    of_episode = split_of_episode(split)
    clean = {r["stem"]: r for r in read_jsonl(gt_path)} if gt_path else None
    rows = []
    for row in read_jsonl(run_path):
        name = of_episode.get(int(row["episode"]))
        if name not in ("train", "val"):
            continue
        along = row.get("gt_along_m")
        if along is not None and along < min_along:
            continue
        if clean is not None:
            row = {**row, "corridor_min_gt_raw": row["corridor_min_gt"], "corridor_min_gt": clean[row["stem"]]["corridor_min_gt_clean"]}
        rows.append({**row, "split": name})
    return rows


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    gt = np.array([r["corridor_min_gt"] for r in rows], dtype=np.float64)
    est = np.array([r["corridor_min_est"] for r in rows], dtype=np.float64)
    return gt, est


def correct(est: np.ndarray, scale: float = 1.0, shift: float = 0.0, max_range: float = MAX_RANGE_M) -> np.ndarray:
    """Apply the global correction to corridor minimums, then re-clip, as ``DepthAnythingV2.perceive`` does."""
    out = np.maximum(scale * np.asarray(est, dtype=np.float64) + shift, 1e-3)
    out[out > max_range] = max_range
    return out


def outcome_table(
    rows: list[dict[str, Any]], d_safe: float, key: str = "obstacle", scale: float = 1.0, shift: float = 0.0
) -> dict[str, dict[str, Any]]:
    """Outcome counts per value of ``key``. ``miss_rate`` = missed / frames with a GT obstacle."""
    if not rows:
        return {}
    gt, est = _arrays(rows)
    codes = outcomes(gt, correct(est, scale, shift), d_safe)
    counts: dict[str, Counter] = defaultdict(Counter)
    for row, code in zip(rows, codes):
        counts[str(row[key])][Outcome(int(code)).name] += 1
        counts["ALL"][Outcome(int(code)).name] += 1
    table = {}
    for name, c in sorted(counts.items()):
        danger = c["MISSED"] + c["AGREE_AVOID"]
        clear = c["FALSE_STOP"] + c["AGREE_CLEAR"]
        table[name] = {
            "frames": danger + clear,
            "gt_obstacle": danger,
            "missed": c["MISSED"],
            "false_stop": c["FALSE_STOP"],
            "miss_rate": c["MISSED"] / danger if danger else None,
            "false_stop_rate": c["FALSE_STOP"] / clear if clear else None,
        }
    return table


def _pairs(rows: list[dict[str, Any]], fit_max: float) -> tuple[np.ndarray, np.ndarray]:
    gt, est = _arrays(rows)
    keep = (gt <= fit_max) & (est < MAX_RANGE_M)
    return gt[keep], est[keep]


def abs_rel(gt: np.ndarray, est: np.ndarray) -> float:
    return float(np.mean(np.abs(est - gt) / gt)) if len(gt) else float("nan")


def scale_check(rows: list[dict[str, Any]], fit_max: float = 40.0) -> dict[str, Any]:
    """Fit ``gt = scale * est`` (median ratio) and ``gt = scale * est + shift`` on train; report AbsRel on val.

    Only frames whose GT corridor minimum is within ``fit_max`` metres are used,
    since near range is what the controller acts on.
    """
    gt_tr, est_tr = _pairs([r for r in rows if r["split"] == "train"], fit_max)
    gt_va, est_va = _pairs([r for r in rows if r["split"] == "val"], fit_max)
    if len(gt_tr) < 2:
        raise ValueError("not enough train frames in range to fit a scale")
    median_scale = float(np.median(gt_tr / est_tr))
    slope, intercept = np.polyfit(est_tr, gt_tr, 1)
    return {
        "fit_max_m": fit_max,
        "n_train": int(len(gt_tr)),
        "n_val": int(len(gt_va)),
        "median_scale": median_scale,
        "affine": {"scale": float(slope), "shift": float(intercept)},
        "val_abs_rel": {
            "raw": abs_rel(gt_va, est_va),
            "median_scale": abs_rel(gt_va, median_scale * est_va),
            "affine": abs_rel(gt_va, slope * est_va + intercept),
        },
    }


def eyeball_sheet(
    bank: str | Path, rows: list[dict[str, Any]], d_safe: float, out: str | Path, n: int = 20, seed: int = 0, cols: int = 5
) -> dict[str, int]:
    """Contact sheet of up to ``n`` L1 positives (red) then ``n`` negatives (green), from train frames.

    Each tile shows the corridor box, GT and estimated corridor minimum.
    """
    from PIL import Image, ImageDraw

    train = [r for r in rows if r["split"] == "train"]
    gt, est = _arrays(train)
    missed = outcomes(gt, est, d_safe) == Outcome.MISSED
    rng = np.random.default_rng(seed)
    picks = []
    for flag, colour in ((True, (255, 40, 40)), (False, (40, 220, 40))):
        idx = np.flatnonzero(missed == flag)
        for i in sorted(rng.choice(idx, size=min(n, len(idx)), replace=False)):
            picks.append((train[i], colour))
    if not picks:
        raise ValueError("no train frames to draw")
    first = load_rgb(bank, picks[0][0]["stem"])
    h, w = first.shape[:2]
    sheet = Image.new("RGB", (cols * w, -(-len(picks) // cols) * h))
    win_rows, win_cols = corridor_window((h, w))
    for k, (row, colour) in enumerate(picks):
        tile = Image.fromarray(load_rgb(bank, row["stem"]))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([win_cols.start, win_rows.start, win_cols.stop, win_rows.stop], outline=colour, width=2)
        draw.rectangle([0, 0, w - 1, h - 1], outline=colour, width=3)
        text = f"{row['stem']} {row['obstacle']} {row['condition']}\ngt {row['corridor_min_gt']:.1f} m  est {row['corridor_min_est']:.1f} m"
        draw.rectangle([4, 4, 4 + 190, 4 + 30], fill=(0, 0, 0))
        draw.text((7, 5), text, fill=(255, 255, 255))
        sheet.paste(tile, ((k % cols) * w, (k // cols) * h))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return {"positives": int(sum(c[0] == 255 for _, c in picks)), "negatives": int(sum(c[0] != 255 for _, c in picks))}


def _fmt(rate: float | None) -> str:
    return "   n/a" if rate is None else f"{100 * rate:5.1f}%"


def print_table(title: str, table: dict[str, dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"  {'group':<14}{'frames':>8}{'gt_obst':>9}{'missed':>8}{'miss':>8}{'f_stop':>8}{'f_stop':>8}")
    for name, t in table.items():
        print(
            f"  {name:<14}{t['frames']:>8}{t['gt_obstacle']:>9}{t['missed']:>8}{_fmt(t['miss_rate']):>8}"
            f"{t['false_stop']:>8}{_fmt(t['false_stop_rate']):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="JSONL from perception.run_bank")
    parser.add_argument("--split", default="configs/bank_split.yaml")
    parser.add_argument("--d-safe", nargs="+", type=float, default=[5.0, 8.0, 10.0, 12.0])
    parser.add_argument("--gt", help="clean ground truth from gate.bank_gt; replaces the raw corridor minimum")
    parser.add_argument("--min-along", type=float, default=MIN_ALONG_M)
    parser.add_argument("--fit-max", type=float, default=40.0)
    parser.add_argument("--out", help="write the full report here as JSON")
    parser.add_argument("--bank", help="frame-bank directory; needed only for --sheet")
    parser.add_argument("--sheet", help="PNG contact sheet of positives and negatives to write")
    parser.add_argument("--sheet-d-safe", type=float, default=8.0)
    args = parser.parse_args()

    rows = load_run(args.run, load_split(args.split), args.min_along, args.gt)
    by_split = Counter(r["split"] for r in rows)
    print(f"{len(rows)} frames (train {by_split['train']}, val {by_split['val']}); test and excluded episodes not loaded")

    scale = scale_check(rows, args.fit_max)
    print(f"\nScale check on corridor minimum, GT <= {args.fit_max:g} m (fit on {scale['n_train']} train, scored on {scale['n_val']} val)")
    print(f"  median scale {scale['median_scale']:.3f}; affine scale {scale['affine']['scale']:.3f} shift {scale['affine']['shift']:+.2f} m")
    print("  val AbsRel: " + ", ".join(f"{k} {v:.3f}" for k, v in scale["val_abs_rel"].items()))

    report: dict[str, Any] = {"run": str(args.run), "frames": dict(by_split), "scale_check": scale, "tables": {}}
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    for d_safe in args.d_safe:
        tables = {
            "train_by_obstacle": outcome_table(train, d_safe, "obstacle"),
            "train_by_condition": outcome_table(train, d_safe, "condition"),
            "val_by_obstacle": outcome_table(val, d_safe, "obstacle"),
            "val_by_obstacle_scaled": outcome_table(val, d_safe, "obstacle", scale=scale["median_scale"]),
        }
        report["tables"][f"d_safe_{d_safe:g}"] = tables
        for name, table in tables.items():
            print_table(f"d_safe {d_safe:g} m | {name}", table)

    if args.sheet:
        if not args.bank:
            raise SystemExit("--sheet needs --bank")
        n = eyeball_sheet(args.bank, rows, args.sheet_d_safe, args.sheet)
        print(f"\nwrote {args.sheet}: {n['positives']} positives (red), {n['negatives']} negatives (green), d_safe {args.sheet_d_safe:g} m")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
