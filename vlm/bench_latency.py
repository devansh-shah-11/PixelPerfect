"""Chunk M2.1: serve Qwen3-VL and measure latency (median, p95) per model.

    python -m vlm.bench_latency --frames /scratch/$USER/frame_bank/CityEnviron/frames --n 50
    python -m vlm.bench_latency --models Qwen/Qwen3-VL-2B-Instruct --n 20 --no-crop

Reads ``*_rgb.png`` frames from a directory (any bank), or uses random noise
frames if ``--frames`` is not given. Each model is loaded, warmed up, timed for
``--n`` calls, then freed. Results are written after every model, so a later
model running out of memory does not lose the earlier numbers. Appends one line
per model to ``logs/runs.jsonl`` (CLAUDE.md) and prints a Markdown table to
paste into ``README.md``.

Latency here is one question = one forward pass with the overlay and the crop
(two images). It is the real wall-clock number; the delay charged in flights
is a separate, controlled knob (D10).
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
from PIL import Image

from vlm.judge import DEFAULT_MODEL, REFERENCE_MODEL, QwenVLJudge
from vlm.overlay import render

REPO = Path(__file__).resolve().parent.parent
RUNS_LOG = REPO / "logs" / "runs.jsonl"


def summarize(latencies_s: list[float]) -> dict[str, float]:
    a = np.asarray(latencies_s, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_s": float(np.median(a)),
        "p95_s": float(np.percentile(a, 95)),
        "p99_s": float(np.percentile(a, 99)),
        "mean_s": float(a.mean()),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
    }


def load_frames(frames_dir: str | Path | None, n: int, seed: int, shape: tuple[int, int] = (225, 400)) -> list[np.ndarray]:
    """``n`` RGB frames: a seeded sample of ``*_rgb.png`` under ``frames_dir``, or noise if no directory."""
    rng = np.random.default_rng(seed)
    if frames_dir is None:
        return [rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8) for _ in range(n)]
    paths = sorted(Path(frames_dir).glob("*_rgb.png"))
    if not paths:
        raise FileNotFoundError(f"no *_rgb.png under {frames_dir}")
    picks = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    out = []
    for i in sorted(picks):
        with Image.open(paths[i]) as im:
            out.append(np.asarray(im.convert("RGB")))
    return out


def bench_model(model_id: str, frames: list[np.ndarray], warmup: int, dtype: str, use_overlay: bool, use_crop: bool) -> dict[str, Any]:
    import torch

    judge = QwenVLJudge(model_id=model_id, dtype=dtype, use_overlay=use_overlay, use_crop=use_crop)
    rendered = [render(f) for f in frames]
    result: dict[str, Any] = {"model": model_id, "dtype": dtype, "use_overlay": use_overlay, "use_crop": use_crop}
    try:
        warm = [judge.warmup() for _ in range(max(1, warmup))]
        result["load_s"] = judge.load_seconds
        result["warmup_s"] = warm
        lat, answers = [], []
        for frame, (overlay, crop) in zip(frames, rendered):
            ans = judge.ask(frame, overlay, crop)
            lat.append(ans.latency_s)
            answers.append({"answer": ans.answer, "confidence": round(ans.confidence, 3)})
        result.update(summarize(lat))
        result["latencies_s"] = [round(x, 4) for x in lat]
        result["yes_fraction"] = float(np.mean([a["answer"] for a in answers]))
        result["confidence_median"] = float(np.median([a["confidence"] for a in answers]))
        if torch.cuda.is_available():
            result["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            result["gpu"] = torch.cuda.get_device_name(0)
        result["outcome"] = "success"
    except Exception as e:  # noqa: BLE001  (an OOM on the 8B must not kill the 2B numbers)
        result["outcome"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
    finally:
        del judge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return result


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_run(result: dict[str, Any], args: argparse.Namespace, wall_seconds: float) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": _commit(),
            "env": socket.gethostname(),
            "command": " ".join(sys.argv),
            "config": {"model": result["model"], "dtype": args.dtype, "use_overlay": not args.no_overlay, "use_crop": not args.no_crop, "frames": args.frames},
            "seeds": f"frame sample seed {args.seed}",
            "n_episodes": result.get("n", 0),
            "wall_seconds": round(wall_seconds, 1),
            "outcome": result["outcome"],
        }) + "\n")


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = ["| Model | dtype | n | median (s) | p95 (s) | p99 (s) | load (s) | peak GPU (GB) | outcome |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["outcome"] == "success":
            lines.append(f"| {r['model']} | {r['dtype']} | {r['n']} | {r['median_s']:.3f} | {r['p95_s']:.3f} | {r['p99_s']:.3f} | {r['load_s']:.1f} | {r.get('gpu_peak_mem_gb', 'n/a')} | success |")
        else:
            lines.append(f"| {r['model']} | {r['dtype']} | | | | | | | {r['outcome']} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, REFERENCE_MODEL])
    p.add_argument("--frames", help="directory with *_rgb.png (frame bank); random noise frames if omitted")
    p.add_argument("--n", type=int, default=50, help="timed calls per model")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--out", default=str(REPO / "results" / "vlm_latency.json"))
    args = p.parse_args()

    frames = load_frames(args.frames, args.n, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for model_id in args.models:
        print(f"[{time.strftime('%H:%M:%S')}] {model_id}: loading", flush=True)
        start = time.time()
        r = bench_model(model_id, frames, args.warmup, args.dtype, not args.no_overlay, not args.no_crop)
        r["frames"] = args.frames or "random noise"
        results.append(r)
        log_run(r, args, time.time() - start)
        out_path.write_text(json.dumps(results, indent=1))
        if r["outcome"] == "success":
            print(f"  median {r['median_s']:.3f} s  p95 {r['p95_s']:.3f} s  load {r['load_s']:.1f} s  yes-fraction {r['yes_fraction']:.2f}", flush=True)
        else:
            print(f"  {r['outcome']}", flush=True)
    print("\nPaste into README.md:\n")
    print(markdown_table(results))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
