"""Cheap thin-structure features from the RGB corridor crop (PLAN_DETAILED.md §4.2; for the gate, M3.4).

The depth model misses thin wires, poles and glass; their *image* edges are often
still there (the 4x VLM crop shows the 1 cm wire at 8 to 13 m). These features
are run-time legal (RGB only) and use numpy alone:

- ``img_grad_mean``: mean Sobel gradient magnitude in the crop
- ``img_edge_frac``: share of pixels with a strong gradient
- ``img_hline`` / ``img_vline``: strongest row-mean of the vertical gradient (a horizontal line,
  which is how the bank's "axis_z" poles and wires render) and strongest column-mean of the
  horizontal gradient (a vertical line), each relative to the crop's median
- ``img_contrast`` / ``img_brightness``: grey-level spread and mean (dusk and white walls)

Offline: one CPU pass over a bank, resumable JSONL keyed by stem.

    python -m gate.image_features --bank /scratch/dns5508/frame_bank/CityEnviron --out /scratch/$USER/bank_runs/image_features.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from perception.depth_utils import corridor_window

EDGE_THRESHOLD = 40.0  # grey levels per pixel, on 0..255 images


def _grey(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _sobel(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx, gy


def corridor_crop(rgb: np.ndarray, margin: float = 0.5) -> np.ndarray:
    rows, cols = corridor_window(rgb.shape)
    h, w = rgb.shape[:2]
    mr, mc = int(margin * (rows.stop - rows.start)), int(margin * (cols.stop - cols.start))
    return rgb[max(0, rows.start - mr): min(h, rows.stop + mr), max(0, cols.start - mc): min(w, cols.stop + mc)]


def image_features(rgb: np.ndarray) -> dict[str, float]:
    g = _grey(corridor_crop(rgb))
    gx, gy = _sobel(g)
    mag = np.hypot(gx, gy) / 4.0  # Sobel gain is 4 for a unit step
    row_h = np.abs(gy).mean(axis=1) / 4.0
    col_v = np.abs(gx).mean(axis=0) / 4.0
    eps = 1e-6
    return {
        "img_grad_mean": float(mag.mean()),
        "img_edge_frac": float((mag > EDGE_THRESHOLD).mean()),
        "img_hline": float(row_h.max() / (np.median(row_h) + eps + 1.0)),
        "img_vline": float(col_v.max() / (np.median(col_v) + eps + 1.0)),
        "img_contrast": float(g.std()),
        "img_brightness": float(g.mean()),
    }


def run(bank: str | Path, out_path: str | Path, limit: int | None = None) -> dict[str, int]:
    from perception.frame_bank import frame_rows, load_rgb, read_jsonl

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["stem"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in frame_rows(bank) if r["stem"] not in done][:limit]
    with open(out_path, "a") as f:
        for k, r in enumerate(todo, 1):
            f.write(json.dumps({"stem": r["stem"], **image_features(load_rgb(bank, r["stem"]))}) + "\n")
            if k % 2000 == 0:
                f.flush()
                print(f"{k}/{len(todo)} frames", flush=True)
    return {"done": len(todo), "skipped": len(done)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    c = run(a.bank, a.out, a.limit)
    print(f"{c['done']} frames processed, {c['skipped']} already done -> {a.out}")


if __name__ == "__main__":
    main()
