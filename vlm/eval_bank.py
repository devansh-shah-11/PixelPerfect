"""Chunk M2.4: standalone VLM evaluation on labelled frame-bank frames, and the M2 gate check.

    python -m vlm.eval_bank --bank /scratch/dns5508/frame_bank/CityEnviron --examples results/vlm_examples
    python -m vlm.eval_bank --summarize-only --out results/vlm_eval.jsonl      # re-score saved rows, no GPU

Asks each judge one question per frame under several conditions (overlay,
crop and prompt wording) and reports, per model and condition: recall,
precision and false-positive rate at the default cut (confidence > 0), AUROC
of the confidence, recall at a cut tuned on val, per obstacle type with 95%
intervals from resampling whole flights, and the M2 gate verdict. Resumable:
rows already in ``--out`` are skipped, keyed by (model, condition, stem).

Frames. Train and val flights by default (test stays closed, CLAUDE.md rule
3). The VLM is frozen, so train frames are as unseen to it as val frames; the
only tuned quantity, the cut on the confidence, is chosen on val and reported
on train. Consecutive frames of one flight are near-duplicates, so
``--stride`` keeps every k-th step, and every interval is computed over
flights, not frames.

Ground truth (offline only, CLAUDE.md rule 5). The distance to the nearest
true surface inside the highlighted window, from ``gate.labels.clean_corridor_min``
(segmentation-masked depth). Glass is invisible to both the depth and the
segmentation cameras (job 33462: no mask pixels in any of 50 glass frames, depth
reads the background behind the pane), so for the types in
``METADATA_GT_TYPES`` the obstacle distance comes from the episode metadata
instead: the pane is in the window when its recorded lateral offset is within
its half-width plus the window's half-width at that distance (90 degree
horizontal field of view, verified in the shipped Project AirSim camera config
that hpc/capability/make_configs.py copies). Positive: nearest surface closer
than ``--d-pos``. Negative: farther than ``--d-neg``. In between: ambiguous,
left out of the scores but counted.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
RUNS_LOG = REPO / "logs" / "runs.jsonl"

# name -> (use_overlay, use_crop, prompt key in vlm.judge.PROMPTS)
CONDITIONS: dict[str, tuple[bool, bool, str]] = {
    "both": (True, True, "cross"),
    "overlay": (True, False, "cross"),
    "crop": (False, True, "cross"),
    "raw": (False, False, "cross"),
    "both_blocked": (True, True, "blocked"),
    "both_hit": (True, True, "hit"),
}
BUCKETS = ((0, 3), (3, 6), (6, 10), (10, 20), (20, float("inf")))
MIN_ALONG_M = 1.0  # as in perception.analyze_bank: closer and the camera is inside the obstacle

# Ground truth for see-through obstacles (glass) and the window geometry live in gate.labels,
# shared with the gate labels and gate.bank_gt.
from gate.labels import HALF_WIDTH_M, METADATA_GT_TYPES, metadata_in_window, window_half_width, with_metadata_fallback  # noqa: E402
from perception.depth_utils import CAMERA_HFOV_DEG as HFOV_DEG  # noqa: E402

GATE_RECALL = 0.60  # TASKS.md M2 kill switch: recall below about 60% on the visible types
TUNE_MAX_FPR = 0.10  # the tuned cut is the lowest one that keeps val FPR at or below this
MIN_GATE_FLIGHTS = 10  # fewer positive flights than this: the verdict is reported as thin


# ---- frames and ground truth -----------------------------------------------------------


def bucket(distance_m: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= distance_m < hi:
            return f"{lo}-{hi:g} m" if hi != float("inf") else f">{lo} m"
    return "n/a"


def frame_label(gt_clean_m: float, d_pos: float, d_neg: float) -> str:
    """'pos', 'neg' or 'ambiguous' from the distance to the nearest true surface in the window."""
    if gt_clean_m < d_pos:
        return "pos"
    if gt_clean_m > d_neg:
        return "neg"
    return "ambiguous"


def select_frames(
    bank: str | Path, split_path: str | Path, splits: Iterable[str], stride: int = 1,
    per_type: int | None = None, seed: int = 0,
) -> list[dict[str, Any]]:
    """Frames of the chosen splits, every ``stride``-th step, minus frames where the camera is past the obstacle.

    Each returned row carries its ``split``. ``per_type`` caps frames per obstacle type (seeded sample).
    """
    from perception.bank_split import load_split, split_of_episode
    from perception.frame_bank import frame_rows

    splits = tuple(splits)
    if "test" in splits:
        raise ValueError("the test split stays closed until the end of the project (CLAUDE.md rule 3)")
    of_episode = split_of_episode(load_split(split_path))
    rows = []
    for r in frame_rows(bank):
        name = of_episode.get(int(r["episode"]))
        if name not in splits or int(r["step"]) % max(1, stride):
            continue
        along = r.get("gt_along_m")
        if along is not None and along < MIN_ALONG_M:
            continue
        rows.append({**r, "split": name})
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


def ground_truth(bank: str | Path, frame: dict[str, Any], hfov_deg: float = HFOV_DEG) -> dict[str, Any]:
    """Clean corridor distances for one frame, with the metadata fallback for see-through obstacles (gate.labels)."""
    from gate.labels import clean_corridor_min
    from perception.depth_utils import MAX_RANGE_M
    from perception.frame_bank import load_depth_gt, load_obstacle_mask

    stem = frame["stem"]
    gt = clean_corridor_min(load_depth_gt(bank, stem), load_obstacle_mask(bank, stem), MAX_RANGE_M)
    return with_metadata_fallback(gt, frame, hfov_deg)


# ---- evaluation loop -----------------------------------------------------------------------


def evaluate(
    judges: dict[str, Any],
    bank: str | Path,
    frames: list[dict[str, Any]],
    out_path: str | Path,
    conditions: Iterable[str] = tuple(CONDITIONS),
    examples_dir: str | Path | None = None,
    n_examples: int = 6,
    hfov_deg: float = HFOV_DEG,
) -> list[dict[str, Any]]:
    """Ask every (model, condition) about every frame not yet in ``out_path``; return all rows, old and new.

    Frame-major: each frame is loaded and rendered once and asked under every
    model and condition, so all judges stay loaded together (2B + 8B peak about
    22 GB on the 40 GB A100, job 33341). Labels are not stored; they follow from
    ``gt_clean_m`` at scoring time, so other ``d_pos`` / ``d_neg`` need no rerun.
    """
    from perception.frame_bank import load_rgb
    from vlm.judge import prompt_text
    from vlm.overlay import render

    conditions = tuple(conditions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out_path)
    done = {(r["model"], r["condition"], r["stem"]) for r in rows}
    sheets = {(m, c): ExampleSheet(Path(examples_dir) / f"{_slug(m)}_{c}", n_examples) for m in judges for c in conditions} if examples_dir else {}
    todo = [fr for fr in frames if any((m, c, fr["stem"]) not in done for m in judges for c in conditions)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(frames)} frames, {len(todo)} with work left; {len(judges)} model(s) x {len(conditions)} condition(s)", flush=True)

    with open(out_path, "a") as f:
        for k, fr in enumerate(todo, 1):
            stem = fr["stem"]
            gt = ground_truth(bank, fr, hfov_deg)
            rgb = load_rgb(bank, stem)
            overlay, crop = render(rgb)
            for model_name, judge in judges.items():
                for cond in conditions:
                    if (model_name, cond, stem) in done:
                        continue
                    use_overlay, use_crop, prompt_key = CONDITIONS[cond]
                    judge.use_overlay, judge.use_crop = use_overlay, use_crop
                    judge.prompt = prompt_text(prompt_key, use_overlay)
                    ans = judge.ask(rgb, overlay, crop)
                    row = {
                        "model": model_name, "condition": cond, "prompt": prompt_key, "stem": stem, "split": fr["split"],
                        "episode": int(fr["episode"]), "step": int(fr["step"]), "obstacle": fr["obstacle"], "weather": fr["condition"],
                        "gt_clean_m": round(float(gt["corridor_min_gt_clean"]), 3),
                        "gt_obstacle_m": round(float(gt["corridor_min_gt_obstacle"]), 3),
                        "gt_scene_m": round(float(gt["corridor_min_gt_scene"]), 3),
                        "obstacle_px": int(gt["obstacle_px"]), "gt_source": gt["gt_source"],
                        "gt_along_m": fr.get("gt_along_m"), "gt_lateral_m": fr.get("gt_lateral_m"), "meta_in_window": gt["meta_in_window"],
                        "answer": bool(ans.answer), "confidence": round(float(ans.confidence), 4), "latency_s": round(float(ans.latency_s), 4),
                    }
                    f.write(json.dumps(row) + "\n")
                    rows.append(row)
                    if sheets:
                        sheets[(model_name, cond)].maybe_add(row, overlay, crop)
            f.flush()
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} frames", flush=True)
    for sheet in sheets.values():
        sheet.save()
    return rows


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- scoring -------------------------------------------------------------------------------


def labelled(rows: list[dict[str, Any]], d_pos: float, d_neg: float) -> list[dict[str, Any]]:
    """Rows with ``label`` and ``bucket`` attached. Rows from the first run (job 33462) have no split: they were val."""
    out = []
    for r in rows:
        d = float(r["gt_clean_m"])
        out.append({**r, "split": r.get("split", "val"), "label": frame_label(d, d_pos, d_neg), "bucket": bucket(d)})
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a positive's confidence exceeds a negative's (ties count half). NaN if either is empty."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def flight_ci(rows: list[dict[str, Any]], tau: float = 0.0, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for recall over positive ``rows`` by resampling whole flights. NaN with fewer than 2 flights."""
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["episode"]][0] += int(r["confidence"] > tau)
        per[r["episode"]][1] += 1
    if len(per) < 2:
        return float("nan"), float("nan")
    hits, n = np.array(list(per.values()), dtype=np.float64).T
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    rec = hits[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))


def scores(rows: list[dict[str, Any]], tau: float = 0.0) -> dict[str, Any]:
    pos = [r for r in rows if r["label"] == "pos"]
    neg = [r for r in rows if r["label"] == "neg"]
    tp = sum(r["confidence"] > tau for r in pos)
    fp = sum(r["confidence"] > tau for r in neg)
    lo, hi = flight_ci(pos, tau)
    return {
        "n_pos": len(pos), "n_neg": len(neg), "n_ambiguous": sum(r["label"] == "ambiguous" for r in rows),
        "pos_flights": len({r["episode"] for r in pos}), "neg_flights": len({r["episode"] for r in neg}),
        "tp": int(tp), "fp": int(fp),
        "recall": tp / len(pos) if pos else float("nan"), "recall_ci": [lo, hi],
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fpr": fp / len(neg) if neg else float("nan"),
    }


def tune_tau(val_rows: list[dict[str, Any]], max_fpr: float = TUNE_MAX_FPR) -> float:
    """Lowest cut on the confidence whose FPR on val negatives is at most ``max_fpr``. NaN without val negatives."""
    neg = np.sort(np.array([r["confidence"] for r in val_rows if r["label"] == "neg"], dtype=np.float64))
    if neg.size == 0:
        return float("nan")
    allowed = int(math.floor(max_fpr * neg.size))  # negatives allowed above the cut
    return float(neg[neg.size - 1 - allowed]) if allowed < neg.size else float(neg[0] - 1e-6)


def visible(obstacle: str) -> bool:
    """Types the M2 gate counts: everything except wires (unverified until M4.1) and empty frames."""
    return not obstacle.startswith("wire") and obstacle != "none"


def summarize(rows: list[dict[str, Any]], d_pos: float, d_neg: float) -> dict[str, Any]:
    rows = labelled(rows, d_pos, d_neg)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    out: dict[str, Any] = {}
    for (model, cond), g in sorted(groups.items()):
        val = [r for r in g if r["split"] == "val"]
        train = [r for r in g if r["split"] == "train"]
        tau = tune_tau(val)
        negs = np.array([r["confidence"] for r in g if r["label"] == "neg"])
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in g:
            by_type[str(r["obstacle"])].append(r)
            by_bucket[r["bucket"]].append(r)
        vis = [r for r in g if visible(str(r["obstacle"]))]
        vis_train = [r for r in vis if r["split"] == "train"]
        entry = {
            "overall": scores(g),
            "auroc": auroc([r["confidence"] for r in g if r["label"] == "pos"], negs),
            "tau_val": tau,
            "train_at_tau": scores(train, tau) if train and tau == tau else None,
            "visible": scores(vis),
            "visible_train_at_tau": scores(vis_train, tau) if vis_train and tau == tau else None,
            "by_obstacle": {
                t: {**scores(v), "auroc_vs_all_neg": auroc([r["confidence"] for r in v if r["label"] == "pos"], negs)}
                for t, v in sorted(by_type.items())
            },
            "by_bucket": {b: scores(v) for b, v in sorted(by_bucket.items(), key=lambda kv: _bucket_key(kv[0]))},
        }
        entry["gate"] = gate_verdict(entry)
        out[f"{model} / {cond}"] = entry
    return out


def gate_verdict(entry: dict[str, Any]) -> str:
    v = entry["visible"]
    if v["n_pos"] == 0:
        return "NO DATA (no positive frames of visible types)"
    thin = f"; THIN: {v['pos_flights']} positive flights" if v["pos_flights"] < MIN_GATE_FLIGHTS else ""
    if v["recall"] >= GATE_RECALL:
        return f"PASS at cut 0 (recall {v['recall']:.2f}){thin}"
    t = entry.get("visible_train_at_tau")
    if t and t["n_pos"] and t["recall"] >= GATE_RECALL and t["fpr"] <= 1.5 * TUNE_MAX_FPR:
        return f"PASS at val-tuned cut {entry['tau_val']:+.2f} (train recall {t['recall']:.2f}, FPR {t['fpr']:.2f}){thin}"
    return f"FAIL (recall {v['recall']:.2f} at cut 0){thin}"


def agreement(rows: list[dict[str, Any]], d_pos: float) -> dict[str, dict[str, int]]:
    """Per type: frames where metadata and segmentation agree on 'obstacle in the window and closer than d_pos'.

    One condition's rows are enough (ground truth does not depend on the question). Large disagreement
    on a type means its segmentation (or the half-width assumption) is off.
    """
    seen, out = set(), defaultdict(lambda: {"both": 0, "meta_only": 0, "seg_only": 0, "neither": 0, "no_meta": 0})
    for r in rows:
        if r["stem"] in seen:
            continue
        seen.add(r["stem"])
        t = str(r["obstacle"])
        if t == "none":
            continue
        meta = r.get("meta_in_window")
        if meta is None:
            out[t]["no_meta"] += 1
            continue
        meta_near = bool(meta) and float(r["gt_along_m"]) < d_pos
        # For metadata-labelled rows gt_obstacle_m holds the metadata distance, so only the pixel count speaks for segmentation.
        seg_near = r.get("obstacle_px", 0) > 0 and (r.get("gt_source") == "metadata" or float(r.get("gt_obstacle_m", 1e9)) < d_pos)
        key = "both" if meta_near and seg_near else "meta_only" if meta_near else "seg_only" if seg_near else "neither"
        out[t][key] += 1
    return dict(sorted(out.items()))


# ---- report --------------------------------------------------------------------------------


def _bucket_key(name: str) -> float:
    return float(name.lstrip(">").split("-")[0].split()[0]) if name != "n/a" else float("inf")


def _f(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{x:.2f}"


def _ci(c: list[float]) -> str:
    return "n/a" if c[0] != c[0] else f"{c[0]:.2f} to {c[1]:.2f}"


def report(summary: dict[str, Any], agree: dict[str, dict[str, int]], d_pos: float, d_neg: float) -> str:
    L = [
        f"Positive: nearest true surface in the window closer than {d_pos:g} m. Negative: farther than {d_neg:g} m. "
        f"In between: ambiguous, excluded. Intervals: 95%, resampling whole flights.",
        "",
        "### All frames, cut at confidence 0",
        "",
        "| Model / condition | n_pos (flights) | n_neg | recall [95% CI] | precision | FPR | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, s in summary.items():
        o = s["overall"]
        L.append(f"| {k} | {o['n_pos']} ({o['pos_flights']}) | {o['n_neg']} | {_f(o['recall'])} [{_ci(o['recall_ci'])}] | {_f(o['precision'])} | {_f(o['fpr'])} | {_f(s['auroc'])} |")
    L += ["", f"### M2 gate: recall on visible types (not wires, not empty) >= {GATE_RECALL:.2f}", "",
          "| Model / condition | visible n_pos (flights) | recall at 0 [95% CI] | cut tuned on val | train recall / FPR at that cut | verdict |",
          "|---|---|---|---|---|---|"]
    for k, s in summary.items():
        v, t = s["visible"], s["visible_train_at_tau"]
        at = f"{_f(t['recall'])} / {_f(t['fpr'])}" if t else "n/a"
        L.append(f"| {k} | {v['n_pos']} ({v['pos_flights']}) | {_f(v['recall'])} [{_ci(v['recall_ci'])}] | {_f(s['tau_val'])} | {at} | {s['gate']} |")
    best = max(summary, key=lambda k: (summary[k]["visible"]["recall"] if summary[k]["visible"]["recall"] == summary[k]["visible"]["recall"] else -1))
    for k in dict.fromkeys([best] + [x for x in summary if x.endswith("/ both")]):
        s = summary[k]
        L += ["", f"### {k}: by obstacle type", "",
              "| Obstacle | n_pos (flights) | recall [95% CI] | n_neg | FPR | AUROC vs all negatives |", "|---|---|---|---|---|---|"]
        for t, p in s["by_obstacle"].items():
            L.append(f"| {t} | {p['n_pos']} ({p['pos_flights']}) | {_f(p['recall'])} [{_ci(p['recall_ci'])}] | {p['n_neg']} | {_f(p['fpr'])} | {_f(p['auroc_vs_all_neg'])} |")
        L += ["", f"### {k}: by distance of the nearest true surface", "",
              "| Distance | n_pos (flights) | recall | n_neg | FPR |", "|---|---|---|---|---|"]
        for b, p in s["by_bucket"].items():
            L.append(f"| {b} | {p['n_pos']} ({p['pos_flights']}) | {_f(p['recall'])} | {p['n_neg']} | {_f(p['fpr'])} |")
    if agree:
        L += ["", f"### Ground-truth check: metadata vs segmentation, 'obstacle in window and closer than {d_pos:g} m'", "",
              "| Obstacle | both | metadata only | segmentation only | neither | no metadata |", "|---|---|---|---|---|---|"]
        for t, a in agree.items():
            L.append(f"| {t} | {a['both']} | {a['meta_only']} | {a['seg_only']} | {a['neither']} | {a['no_meta']} |")
        L.append("")
        L.append("Glass uses the metadata label (its segmentation is see-through). A large 'metadata only' count on another type means its segmentation is see-through too, or its half-width in `HALF_WIDTH_M` is wrong.")
    return "\n".join(L)


# ---- eyeball sheet -------------------------------------------------------------------------


class ExampleSheet:
    """Up to ``n`` positive and ``n`` negative frames per obstacle type, at most ``per_flight`` from one flight."""

    def __init__(self, out_dir: Path, n: int, per_flight: int = 2) -> None:
        self.out_dir, self.n, self.per_flight = out_dir, n, per_flight
        self.tiles: dict[str, list[Image.Image]] = defaultdict(list)
        self.counts: dict[tuple, int] = defaultdict(int)

    def maybe_add(self, row: dict[str, Any], overlay: np.ndarray, crop: np.ndarray, d_pos: float = 15.0, d_neg: float = 30.0) -> None:
        label = row.get("label") or frame_label(float(row["gt_clean_m"]), d_pos, d_neg)
        t = str(row["obstacle"])
        if label == "ambiguous" or self.counts[(t, label)] >= self.n or self.counts[(t, label, row["episode"])] >= self.per_flight:
            return
        self.counts[(t, label)] += 1
        self.counts[(t, label, row["episode"])] += 1
        self.tiles[t].append(tile({**row, "label": label}, overlay, crop))

    def save(self) -> list[Path]:
        paths = []
        if not self.tiles:
            return paths
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
    """Overlay and crop side by side with the truth and the judge's answer written above."""
    ov, cr = Image.fromarray(overlay), Image.fromarray(crop)
    cr = cr.resize((int(cr.width * ov.height / cr.height), ov.height))
    im = Image.new("RGB", (ov.width + cr.width + 8, ov.height + text_h), "white")
    im.paste(ov, (0, text_h))
    im.paste(cr, (ov.width + 8, text_h))
    verdict = "correct" if (row["answer"] == (row["label"] == "pos")) else "WRONG"
    src = " meta" if row.get("gt_source") == "metadata" else ""
    msg = (f"{row['stem']} {row.get('split', '')} {row['obstacle']} truth={row['label']} ({row['gt_clean_m']} m{src}) "
           f"judge={'yes' if row['answer'] else 'no'} ({row['confidence']:+.2f}) {verdict}")
    ImageDraw.Draw(im).text((4, 2), msg, fill="red" if verdict == "WRONG" else "black")
    return im


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-Instruct", "").lower()


# ---- CLI -----------------------------------------------------------------------------------


def _log_run(args: argparse.Namespace, n_rows: int, wall_seconds: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(),
                            "command": " ".join(sys.argv),
                            "config": {"models": args.models, "conditions": args.conditions, "splits": args.splits, "stride": args.stride,
                                       "d_pos": args.d_pos, "d_neg": args.d_neg, "per_type": args.per_type, "dtype": args.dtype, "hfov": args.hfov},
                            "seeds": f"split {args.split}; frame sample seed {args.seed}", "n_episodes": n_rows,
                            "wall_seconds": round(wall_seconds, 1), "outcome": outcome}) + "\n")


def main(argv: list[str] | None = None) -> None:
    from vlm.judge import DEFAULT_MODEL, REFERENCE_MODEL

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank", help="frame-bank directory (not needed with --summarize-only)")
    p.add_argument("--split", default=str(REPO / "configs" / "bank_split.yaml"))
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="never 'test' before the end (CLAUDE.md rule 3)")
    p.add_argument("--stride", type=int, default=3, help="keep every k-th step of a flight (neighbours are near-duplicates)")
    p.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, REFERENCE_MODEL])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--d-pos", type=float, default=15.0)
    p.add_argument("--d-neg", type=float, default=30.0)
    p.add_argument("--per-type", type=int, help="cap frames per obstacle type (seeded sample); all if omitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hfov", type=float, default=HFOV_DEG)
    p.add_argument("--out", default=str(REPO / "results" / "vlm_eval.jsonl"))
    p.add_argument("--examples", help="directory for eyeball sheets; off if omitted")
    p.add_argument("--n-examples", type=int, default=6)
    p.add_argument("--summarize-only", action="store_true", help="score the rows already in --out; no model, no bank")
    args = p.parse_args(argv)
    if "test" in args.splits:
        raise SystemExit("refusing to evaluate on the test split before the end of the project (CLAUDE.md rule 3)")

    if args.summarize_only:
        rows = read_rows(args.out)
        if not rows:
            raise SystemExit(f"no rows in {args.out}")
    else:
        if not args.bank:
            raise SystemExit("--bank is required unless --summarize-only")
        from vlm.judge import QwenVLJudge

        frames = select_frames(args.bank, args.split, args.splits, args.stride, args.per_type, args.seed)
        judges = {m: QwenVLJudge(model_id=m, dtype=args.dtype) for m in args.models}
        start, outcome, rows = time.time(), "failed", []
        try:
            rows = evaluate(judges, args.bank, frames, args.out, args.conditions, args.examples, args.n_examples, args.hfov)
            outcome = "success"
        finally:
            _log_run(args, len(rows), time.time() - start, outcome)

    summary = summarize(rows, args.d_pos, args.d_neg)
    first_cond = rows[0]["condition"]
    agree = agreement([r for r in rows if r["condition"] == first_cond and r["model"] == rows[0]["model"]], args.d_pos)
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"d_pos": args.d_pos, "d_neg": args.d_neg, "summary": summary, "agreement": agree}, indent=1))
    text = report(summary, agree, args.d_pos, args.d_neg)
    Path(args.out).with_suffix(".report.md").write_text(text + "\n")
    print("\n" + text)
    print(f"\nrows: {args.out}\nsummary: {summary_path}\nreport: {Path(args.out).with_suffix('.report.md')}")


if __name__ == "__main__":
    main()
