"""Reader for the simulator frame bank (hpc/instructions.md §3.4).

Layout written by hpc/capability/capability_test.py:
    <bank>/episodes.jsonl                one row per finished episode
    <bank>/frames.jsonl                  one row per frame (or an "error" row)
    <bank>/frames/<stem>_rgb.png         <stem> is "<episode:04d>_<step:03d>"
    <bank>/frames/<stem>_seg.png
    <bank>/frames/<stem>_depth.npy       float16 metres, distance along the pixel ray

The bank holds ground truth, so it is for labels and metrics only (CLAUDE.md rule 5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_rows(bank: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(bank) / "episodes.jsonl")


def frame_rows(bank: str | Path) -> list[dict[str, Any]]:
    """Rows of frames.jsonl that have saved images; rows that only record an error are dropped."""
    return [r for r in read_jsonl(Path(bank) / "frames.jsonl") if "stem" in r]


def load_rgb(bank: str | Path, stem: str) -> np.ndarray:
    """(H, W, 3) uint8 in RGB order."""
    with Image.open(Path(bank) / "frames" / f"{stem}_rgb.png") as im:
        return np.asarray(im.convert("RGB"))


def load_depth_gt(bank: str | Path, stem: str) -> np.ndarray:
    """(H, W) float32 metres, unclipped (sky is saturated; see perception.depth_utils)."""
    return np.load(Path(bank) / "frames" / f"{stem}_depth.npy").astype(np.float32)
