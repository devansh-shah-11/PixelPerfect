"""Hidden states of the VLM judge on saved flight pictures, for a learned read-out (probe) on the frozen model.

    python -m vlm.features --logs 'results/sweeps/cone_vlm_data/flights/*.jsonl' --out results/vlm_features/qwen8b \\
        --model Qwen/Qwen3-VL-8B-Instruct --layers -1 -9 -18

Why: in flight the 8B said "yes" on nearly every call, also on frames with nothing within 30 m (two v3 gate flights:
43 of 43 calls). A cut on the yes/no score cannot fix that. A small head trained on the model's hidden state, with
labels from the simulator's true depth (offline only, CLAUDE.md rule 5), may. At run time the probe reads only the
picture, through the VLM: no ground truth.

Each saved frame (``rgb_path`` set; ``eval.run_episode`` with ``loop.save_dir`` and ``save_every``) gets one forward pass
with the flights' inputs (prompt "cross", path overlay + crop, ``vlm.eval_bank.CONDITIONS["both"]``). Written:

* ``index.jsonl``: one row per frame: episode, split, seed, t, true distances (``d_placed`` placed obstacles, glass
  included; ``d_scenery`` map surfaces in the drone's flight tube, ``sim.raycast.scenery_truth``; ``d_all`` every surface
  in the centre window, which also catches buildings beside the path and the ground, so it is a looser check), family, the yes/no score, and where its vectors are (shard, row);
* ``shard_00000.npz``: one float16 array per layer, rows in index order.

Labels are not fixed here: the training script derives them from ``d_all`` / ``d_placed``, so the distance cuts (D23:
8 m positive, 30 m clear; sensitivity at 6 and 10 m) can change without a GPU. Resumable: frames already in the index
are skipped; a shard is written before its index rows, so a killed job loses at most one unfinished shard.
Split by seed downstream, never by frame (CLAUDE.md rule 4).
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from vlm.eval_bank import CONDITIONS, REPO


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def saved_frames(log_paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """Frame records with a saved picture, from per-flight logs (``flights/<episode>.jsonl`` or ``<video>/flight.jsonl``)."""
    for lp in log_paths:
        for line in Path(lp).read_text().splitlines():
            if line.strip():
                fr = json.loads(line)
                if fr.get("rgb_path"):
                    fr["_log"] = str(lp)
                    yield fr


def read_index(out: Path) -> list[dict[str, Any]]:
    p = out / "index.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def extract(judge: Any, log_paths: Sequence[Path], out: str | Path, layers: Sequence[int], shard_size: int = 500,
            load_rgb=None, load_depth=None) -> int:
    """Run the judge over every saved frame not yet in ``out``; returns the number of new frames."""
    from perception.depth_utils import corridor_min
    from vlm.judge import INVERTED_PROMPTS, prompt_text
    from vlm.overlay import render

    if load_rgb is None:
        from PIL import Image

        def load_rgb(p):  # noqa: E306
            return np.asarray(Image.open(_path(p)).convert("RGB"))
    load_depth = load_depth or (lambda p: np.load(_path(p)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(out)
    done = {(r["episode"], r["t"]) for r in index}
    shard_no = 1 + max((r["shard"] for r in index), default=-1)
    use_overlay, use_crop, prompt_key = CONDITIONS["both"]
    judge.use_overlay, judge.use_crop = use_overlay, use_crop
    judge.prompt = prompt_text(prompt_key, use_overlay)
    judge.invert = prompt_key in INVERTED_PROMPTS
    rows: list[dict[str, Any]] = []
    vecs: dict[int, list[np.ndarray]] = {i: [] for i in layers}
    n = 0

    def flush() -> None:
        nonlocal shard_no, rows, vecs
        if not rows:
            return
        np.savez(out / f"shard_{shard_no:05d}.npz", **{f"layer_{i}": np.stack(v).astype(np.float16) for i, v in vecs.items()})
        with open(out / "index.jsonl", "a") as f:
            for j, r in enumerate(rows):
                f.write(json.dumps({**r, "shard": shard_no, "row": j}) + "\n")
        shard_no += 1
        rows, vecs = [], {i: [] for i in layers}

    for fr in saved_frames(log_paths):
        key = (fr["episode_id"], fr["t"])
        if key in done:
            continue
        rgb = load_rgb(fr["rgb_path"])
        overlay, crop = render(rgb)
        score, hv = judge.features(rgb, overlay, crop, layers)
        rows.append({"episode": fr["episode_id"], "split": fr.get("split"), "seed": fr.get("seed"), "t": fr["t"],
                     "d_all": round(float(corridor_min(load_depth(fr["depth_gt_path"]))), 2),
                     "d_placed": round(float(fr["corridor_min_gt"]), 2), "family": fr.get("obstacle_family"),
                     "d_scenery": fr.get("scenery_min_gt"),  # map surface in the flight tube; the label to use (d_all is the window)
                     "alt": round(-float(fr["pose"][2]), 2) if fr.get("pose") else None,
                     "score": round(float(score), 4), "log": fr["_log"]})
        for i in layers:
            vecs[i].append(np.asarray(hv[i], dtype=np.float32))
        done.add(key)
        n += 1
        if len(rows) >= shard_size:
            flush()
            print(f"[{time.strftime('%H:%M:%S')}] {n} frames", flush=True)
    flush()
    return n


def load(out: str | Path, layer: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Index rows and the matching (n, hidden) matrix for one layer, in index order."""
    out = Path(out)
    index = read_index(out)
    shards: dict[int, np.ndarray] = {}
    X = []
    for r in index:
        if r["shard"] not in shards:
            shards[r["shard"]] = np.load(out / f"shard_{r['shard']:05d}.npz")[f"layer_{layer}"]
        X.append(shards[r["shard"]][r["row"]])
    return index, (np.stack(X).astype(np.float32) if X else np.zeros((0, 0), dtype=np.float32))


def main(argv: list[str] | None = None) -> None:
    from vlm.judge import REFERENCE_MODEL

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--logs", nargs="+", required=True, help="per-flight logs or globs (quote globs)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=REFERENCE_MODEL)
    p.add_argument("--layers", type=int, nargs="+", default=[-1, -9, -18], help="hidden_states indices (negative from the top)")
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args(argv)
    logs = sorted({Path(x) for g in a.logs for x in (glob.glob(g) or [g])})
    test = [lp for lp in logs if "test-" in lp.name]
    if test:
        raise SystemExit(f"refusing test-split flights ({len(test)} logs): probe data is train/val only")
    from vlm.judge import QwenVLJudge

    start = time.time()
    n = extract(QwenVLJudge(model_id=a.model, dtype=a.dtype), logs, a.out, a.layers, a.shard_size)
    print(f"[{time.strftime('%H:%M:%S')}] {n} new frames from {len(logs)} logs in {time.time() - start:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
