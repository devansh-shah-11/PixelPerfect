"""Run the fast stack over the frame bank (task W-B1).

Writes one JSONL row per frame with the ground-truth and the estimated corridor
minimum, plus the gate features. Labels are not stored: they follow from the
two minimums for any ``d_safe`` (gate.labels). Resumable: frames already in the
output file are skipped, so a requeued job continues where it stopped.

    python -m perception.run_bank --bank /scratch/$USER/frame_bank/CityEnviron \
        --out /scratch/$USER/bank_runs/da2_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from interfaces import FastStack
from perception.depth_utils import corridor_min
from perception.frame_bank import frame_rows, load_depth_gt, load_rgb, read_jsonl

RUNS_LOG = Path(__file__).resolve().parent.parent / "logs" / "runs.jsonl"


def run_bank(
    bank: str | Path,
    out_path: str | Path,
    model: FastStack,
    save_depth_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Process every frame not yet in ``out_path``. Returns counts of done and skipped frames."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["stem"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in frame_rows(bank) if r["stem"] not in done]
    if limit is not None:
        todo = todo[:limit]
    if save_depth_dir is not None:
        Path(save_depth_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as f:
        for i, row in enumerate(todo):
            stem = row["stem"]
            start = time.perf_counter()
            estimate = model.perceive(load_rgb(bank, stem))
            infer_ms = 1000 * (time.perf_counter() - start)
            record: dict[str, Any] = {
                "stem": stem,
                "episode": row["episode"],
                "step": row["step"],
                "condition": row["condition"],
                "obstacle": row["obstacle"],
                "gt_along_m": row.get("gt_along_m"),
                "gt_lateral_m": row.get("gt_lateral_m"),
                "corridor_min_gt": corridor_min(load_depth_gt(bank, stem)),
                "corridor_min_est": estimate.features["corridor_min"],
                "features": {k: v for k, v in estimate.features.items() if k != "embedding"},
                "infer_ms": round(infer_ms, 2),
            }
            # Depth map first: a frame counts as done only once its row is written.
            if save_depth_dir is not None:
                np.save(Path(save_depth_dir) / f"{stem}_depth_est.npy", estimate.depth.astype(np.float16))
            f.write(json.dumps(record) + "\n")
            f.flush()
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(todo)} frames", flush=True)
    return {"done": len(todo), "skipped": len(done)}


def _log_run(args: argparse.Namespace, counts: dict[str, int], wall_seconds: float, outcome: str) -> None:
    """Append one line to logs/runs.jsonl (CLAUDE.md, "Monitoring and logging")."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RUNS_LOG.parent.parent,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": commit,
            "env": socket.gethostname(),
            "command": " ".join(sys.argv),
            "config": {"checkpoint": args.checkpoint, "scale": args.scale, "shift": args.shift},
            "seeds": f"frame bank {Path(args.bank).name} (episode seeds 10000 + episode id)",
            "n_frames": counts,
            "wall_seconds": round(wall_seconds, 1),
            "outcome": outcome,
        }) + "\n")


def main() -> None:
    from perception.depth_anything import DEFAULT_CHECKPOINT, DepthAnythingV2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bank", required=True, help="frame-bank directory")
    parser.add_argument("--out", required=True, help="JSONL file to append to")
    parser.add_argument("--save-depth", help="directory for estimated depth maps (float16 .npy); off by default")
    parser.add_argument("--limit", type=int, help="process at most this many new frames")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    model = DepthAnythingV2(checkpoint=args.checkpoint, scale=args.scale, shift=args.shift)
    start = time.time()
    counts, outcome = {"done": 0, "skipped": 0}, "failed"
    try:
        counts = run_bank(args.bank, args.out, model, args.save_depth, args.limit)
        outcome = "success"
    finally:
        _log_run(args, counts, time.time() - start, outcome)
    print(f"{counts['done']} frames processed, {counts['skipped']} already done -> {args.out}")


if __name__ == "__main__":
    main()
