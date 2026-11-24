"""Chunk M1.9: is the fast stack's depth broken, or are its misses real?

The scale check in ``perception.analyze_bank`` reports a val AbsRel of 2.42 on the
corridor minimum. That number is a *mean over frames* of ``|est - gt| / gt`` and
includes the missed obstacles themselves: a frame with the truth at 2 m and the
estimate at 35 m contributes 16.5 on its own. So it cannot tell "the model is
wrong everywhere" from "the model is right on ordinary surfaces and blind to
thin obstacles". This script separates the two.

Part A, offline (seconds, no model, no images): decomposes the corridor AbsRel
from a bank run (``perception.run_bank`` JSONL), using the clean ground truth
from ``gate.bank_gt`` to tell frames whose nearest corridor surface is the
spawned obstacle from frames where it is the scene (ground, buildings).

    python -m perception.diagnose_depth --run /scratch/$USER/bank_runs/da2_small.jsonl \
        --gt /scratch/$USER/bank_runs/bank_gt.jsonl --out results/depth_diag

Part B, per pixel (needs the model and the bank; a GPU job or about 0.1 s a frame on CPU):
runs Depth Anything V2 on a sample of train frames and scores every pixel,
split into scene pixels and obstacle pixels, after converting the simulator's
ray distance to planar depth (the model predicts planar depth; at the image
edge of a 90 degree camera the two differ by up to 41%). Writes a contact sheet
of RGB, ground truth, estimate and error.

    python -m perception.diagnose_depth --run ... --gt ... --bank /scratch/dns5508/frame_bank/CityEnviron \
        --pixels 8 --out results/depth_diag

Only train episodes are read (val and test stay untouched by diagnosis). Ground
truth is used offline only (CLAUDE.md rule 5).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from perception.bank_split import load_split, split_of_episode
from perception.depth_utils import CAMERA_HFOV_DEG, MAX_RANGE_M, corridor_window, valid_depth_mask
from perception.frame_bank import read_jsonl

HFOV_DEG = CAMERA_HFOV_DEG
FIT_MAX_M = 40.0  # same range as analyze_bank.scale_check
MIN_ALONG_M = 1.0


# ---- Part A: corridor AbsRel decomposition ---------------------------------------------


def rel_errors(gt: np.ndarray, est: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(est, float) - np.asarray(gt, float)) / np.asarray(gt, float)


def describe(err: np.ndarray) -> dict[str, float | int]:
    if err.size == 0:
        return {"n": 0}
    s = np.sort(err)
    top = s[int(0.95 * s.size):]
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "p90": float(np.percentile(s, 90)),
        "trimmed_mean_5pct": float(s[: max(1, int(0.95 * s.size))].mean()),
        "share_of_error_from_worst_5pct": float(top.sum() / s.sum()) if s.sum() > 0 else 0.0,
        "delta_1.25": float(np.mean(s < 0.25)),
    }


def load_frames(run_path: str | Path, gt_path: str | Path | None, split_path: str | Path, splits: tuple[str, ...] = ("train",)) -> list[dict[str, Any]]:
    of_episode = split_of_episode(load_split(split_path))
    clean = {r["stem"]: r for r in read_jsonl(gt_path)} if gt_path else {}
    rows = []
    for r in read_jsonl(run_path):
        name = of_episode.get(int(r["episode"]))
        if name not in splits:
            continue
        along = r.get("gt_along_m")
        if along is not None and along < MIN_ALONG_M:
            continue
        c = clean.get(r["stem"])
        rows.append({**r, "split": name, **({"gt_clean": c} if c else {})})
    return rows


def nearest_is_obstacle(r: dict[str, Any]) -> bool | None:
    """True if the nearest true corridor surface is the spawned obstacle; None without clean ground truth."""
    c = r.get("gt_clean")
    if c is None:
        return None
    return c["obstacle_px"] > 0 and c["corridor_min_gt_obstacle"] <= c["corridor_min_gt_scene"]


def decompose(rows: list[dict[str, Any]], fit_max: float = FIT_MAX_M) -> dict[str, Any]:
    """Corridor-minimum relative error, overall and split by what the nearest surface is."""
    kept = [r for r in rows if r["corridor_min_gt"] <= fit_max and r["corridor_min_est"] < MAX_RANGE_M]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        kind = nearest_is_obstacle(r)
        groups["all"].append(r)
        groups["nearest = obstacle" if kind else "nearest = scene" if kind is False else "nearest = unknown (no --gt)"].append(r)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        if nearest_is_obstacle(r) is False:
            by_type[str(r["obstacle"])].append(r)
    err = lambda rs: rel_errors([r["corridor_min_gt"] for r in rs], [r["corridor_min_est"] for r in rs])  # noqa: E731
    est_over_gt = lambda rs: float(np.median([r["corridor_min_est"] / r["corridor_min_gt"] for r in rs])) if rs else float("nan")  # noqa: E731
    return {
        "fit_max_m": fit_max,
        "groups": {k: {**describe(err(v)), "median_est_over_gt": est_over_gt(v)} for k, v in groups.items()},
        "scene_frames_by_obstacle_type": {k: {**describe(err(v)), "median_est_over_gt": est_over_gt(v)} for k, v in sorted(by_type.items())},
    }


# ---- Part B: per-pixel check -------------------------------------------------------------


def ray_to_planar(ray: np.ndarray, hfov_deg: float = HFOV_DEG) -> np.ndarray:
    """Convert distance along each pixel's ray to planar depth (distance along the optical axis)."""
    h, w = ray.shape
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = (np.arange(h) + 0.5 - h / 2) / f
    norm = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
    return ray / norm


def pixel_scores(gt_planar: np.ndarray, est: np.ndarray, obstacle_mask: np.ndarray, max_range: float = FIT_MAX_M) -> dict[str, Any]:
    """AbsRel and delta<1.25 on scene pixels and on obstacle pixels within ``max_range``; corridor separately."""
    valid = valid_depth_mask(gt_planar, max_range)
    out: dict[str, Any] = {}
    rows, cols = corridor_window(gt_planar.shape)
    corridor = np.zeros_like(valid)
    corridor[rows, cols] = True
    for name, m in (("px_scene", valid & ~obstacle_mask), ("px_obstacle", valid & obstacle_mask), ("px_corridor_scene", valid & ~obstacle_mask & corridor)):
        if m.sum() == 0:
            out[name] = None
            continue
        e = np.abs(est[m] - gt_planar[m]) / gt_planar[m]
        out[name] = {"px": int(m.sum()), "abs_rel_mean": float(e.mean()), "abs_rel_median": float(np.median(e)),
                     "delta_1.25": float(np.mean(np.maximum(est[m] / gt_planar[m], gt_planar[m] / est[m]) < 1.25)),
                     "median_est_over_gt": float(np.median(est[m] / gt_planar[m]))}
    return out


def pixel_check(bank: str | Path, rows: list[dict[str, Any]], per_type: int, out_dir: Path, seed: int = 0, model: Any = None) -> dict[str, Any]:
    from PIL import Image

    from perception.frame_bank import load_depth_gt, load_obstacle_mask, load_rgb

    if model is None:
        from perception.depth_anything import DepthAnythingV2

        model = DepthAnythingV2()
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[str(r["obstacle"])].append(r)
    picks = []
    for t in sorted(by_type):
        g = by_type[t]
        picks += [g[i] for i in sorted(rng.choice(len(g), size=min(per_type, len(g)), replace=False))]

    per_frame, tiles = [], []
    for r in picks:
        rgb = load_rgb(bank, r["stem"])
        gt = ray_to_planar(load_depth_gt(bank, r["stem"]))
        mask = load_obstacle_mask(bank, r["stem"])
        est = model.perceive(rgb).depth
        s = pixel_scores(gt, est, mask)
        per_frame.append({"stem": r["stem"], "obstacle": r["obstacle"], "condition": r["condition"], **s})
        tiles.append(_tile(rgb, gt, est, mask, f"{r['stem']} {r['obstacle']} {r['condition']}"))

    agg: dict[str, Any] = {}
    for key in ("px_scene", "px_obstacle", "px_corridor_scene"):
        vals = [f[key] for f in per_frame if f.get(key)]
        agg[key] = {
            "frames": len(vals),
            "abs_rel_median_of_frame_medians": float(np.median([v["abs_rel_median"] for v in vals])) if vals else None,
            "delta_1.25_mean": float(np.mean([v["delta_1.25"] for v in vals])) if vals else None,
            "median_est_over_gt": float(np.median([v["median_est_over_gt"] for v in vals])) if vals else None,
        }
    by_t: dict[str, Any] = {}
    for t in sorted(by_type):
        fr = [f for f in per_frame if f["obstacle"] == t]
        sc = [f["px_scene"]["abs_rel_median"] for f in fr if f.get("px_scene")]
        ob = [f["px_obstacle"]["median_est_over_gt"] for f in fr if f.get("px_obstacle")]
        by_t[t] = {"frames": len(fr), "scene_abs_rel_median": float(np.median(sc)) if sc else None,
                   "obstacle_px_frames": len(ob), "obstacle_median_est_over_gt": float(np.median(ob)) if ob else None}

    out_dir.mkdir(parents=True, exist_ok=True)
    if tiles:
        w, h = tiles[0].size
        cols = 2
        sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h), "white")
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * w, (k // cols) * h))
        sheet.save(out_dir / "depth_sheet.png")
    return {"frames": len(per_frame), "aggregate": agg, "by_obstacle": by_t, "per_frame": per_frame}


def _colour(depth: np.ndarray, max_m: float = FIT_MAX_M) -> np.ndarray:
    x = np.clip(depth / max_m, 0, 1)
    return (np.stack([1 - x, 0.5 * (1 - np.abs(2 * x - 1)), x], axis=-1) * 255).astype(np.uint8)


def _tile(rgb: np.ndarray, gt: np.ndarray, est: np.ndarray, mask: np.ndarray, title: str):
    from PIL import Image, ImageDraw

    err = np.clip(np.abs(est - gt) / np.maximum(gt, 1e-3), 0, 1)
    err_rgb = (np.stack([err, 1 - err, np.zeros_like(err)], axis=-1) * 255).astype(np.uint8)
    err_rgb[mask] = (255, 255, 255)
    panels = [rgb, _colour(gt), _colour(est), err_rgb]
    h, w = rgb.shape[:2]
    im = Image.new("RGB", (4 * w + 6, h + 16), "white")
    for i, p in enumerate(panels):
        im.paste(Image.fromarray(p), (i * (w + 2), 16))
    ImageDraw.Draw(im).text((3, 2), f"{title}   [RGB | GT planar | estimate | rel. error, white = obstacle]", fill="black")
    return im


# ---- report ----------------------------------------------------------------------------------


def report(a: dict[str, Any], b: dict[str, Any] | None) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}" if isinstance(x, float) else str(x)

    L = [f"## Part A: corridor-minimum relative error, train frames, GT <= {a['fit_max_m']:g} m", "",
         "| Frames | n | mean | median | trimmed mean (drop worst 5%) | share of total error from worst 5% | median est/gt |", "|---|---|---|---|---|---|---|"]
    for k, g in a["groups"].items():
        if g["n"]:
            L.append(f"| {k} | {g['n']} | {f(g['mean'])} | {f(g['median'])} | {f(g['trimmed_mean_5pct'])} | {f(g['share_of_error_from_worst_5pct'])} | {f(g['median_est_over_gt'])} |")
    L += ["", "Frames where the nearest corridor surface is the scene, by episode obstacle type:", "",
          "| Obstacle type | n | mean | median | median est/gt |", "|---|---|---|---|---|"]
    for k, g in a["scene_frames_by_obstacle_type"].items():
        L.append(f"| {k} | {g['n']} | {f(g.get('mean'))} | {f(g.get('median'))} | {f(g.get('median_est_over_gt'))} |")
    if b:
        L += ["", f"## Part B: per-pixel check on {b['frames']} train frames (ray distance converted to planar)", "",
              "| Pixels | frames | median AbsRel | delta < 1.25 | median est/gt |", "|---|---|---|---|---|"]
        for k, g in b["aggregate"].items():
            L.append(f"| {k} | {g['frames']} | {f(g['abs_rel_median_of_frame_medians'])} | {f(g['delta_1.25_mean'])} | {f(g['median_est_over_gt'])} |")
        L += ["", "| Obstacle type | frames | scene median AbsRel | frames with obstacle pixels | obstacle median est/gt |", "|---|---|---|---|---|"]
        for k, g in b["by_obstacle"].items():
            L.append(f"| {k} | {g['frames']} | {f(g['scene_abs_rel_median'])} | {g['obstacle_px_frames']} | {f(g['obstacle_median_est_over_gt'])} |")
        L += ["", "Reading: scene median AbsRel around 0.1 to 0.3 with delta<1.25 above about 0.6 means the model is sane on ordinary surfaces, "
              "and an obstacle est/gt well above 1 means it sees through the obstacle (a real perception miss). "
              "Scene AbsRel above about 0.5 everywhere would point at a pipeline bug (units, scale, or ray vs planar)."]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="JSONL from perception.run_bank")
    p.add_argument("--gt", help="clean ground truth JSONL from gate.bank_gt (strongly recommended)")
    p.add_argument("--split", default="configs/bank_split.yaml")
    p.add_argument("--fit-max", type=float, default=FIT_MAX_M)
    p.add_argument("--bank", help="frame bank, for Part B")
    p.add_argument("--pixels", type=int, default=0, help="Part B: frames per obstacle type (0 = skip Part B)")
    p.add_argument("--out", default="results/depth_diag")
    args = p.parse_args(argv)

    rows = load_frames(args.run, args.gt, args.split)
    print(f"{len(rows)} train frames", flush=True)
    a = decompose(rows, args.fit_max)
    b = None
    if args.pixels:
        if not args.bank:
            raise SystemExit("--pixels needs --bank")
        b = pixel_check(args.bank, rows, args.pixels, Path(args.out))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth_diag.json").write_text(json.dumps({"part_a": a, "part_b": b}, indent=1))
    text = report(a, b)
    (out / "depth_diag.md").write_text(text + "\n")
    print("\n" + text + f"\n\nwritten: {out}/depth_diag.md, depth_diag.json" + (", depth_sheet.png" if b else ""))


if __name__ == "__main__":
    main()
