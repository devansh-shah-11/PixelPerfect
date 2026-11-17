"""Benchmark table: depth only, VLM only, depth + VLM, per VLM variant (PLAN_DETAILED.md D22, D23).

    python -m eval.benchmark --vlm results/vlm_eval.jsonl --run /scratch/$USER/bank_runs/da2_small.jsonl \
        --latency results/vlm_latency_*.json --out results/benchmark.md
    python -m eval.benchmark ... --flights results/sweeps/bench_da2_8b results/sweeps/bench_clear_8b

CPU only: it reads result files, loads no model and no frame.

Frame-bank part. One task for every system: is there an obstacle within
``--d-safe`` in the corridor window? Positive: the nearest true surface is
closer than ``--d-safe`` (the distance at which the controller must react).
Negative: nothing closer than ``--d-neg``. In between: left out. These are the
labels of the "fast stack's misses" table in ``vlm.eval_bank``, so recall on
depth's misses here is the M2 gate number (D23).

- depth only: the fast stack says "obstacle" when its corridor minimum is below ``--d-safe``;
- VLM only: the judge says "obstacle" when its confidence is above its cut;
- depth + VLM: either says "obstacle" (what always-ask does).

Nothing is tuned on the frames that are scored. Each VLM variant's condition
(overlay, crop, prompt) is the one with the best val AUROC, its cut is the
lowest with val false alarms at most ``--max-fpr`` (``vlm.eval_bank.tune_tau``),
and the table is scored on train flights: the VLM is frozen, so train frames
are as unseen to it as val frames. Test frames are refused (CLAUDE.md rule 3).
Every system is scored on the same frames. Intervals resample whole flights.

Latency and peak GPU memory come from ``vlm.bench_latency`` files (one model
loaded at a time, same GPU); depth + VLM is the sum of the two. Without a
latency file the per-call latency logged during the evaluation is used and
memory is "n/a"; the report says which.

Flight part (``--flights``, optional): one row per sweep directory and policy,
from ``eval.sweep`` results. ``never`` is the system without the VLM, ``always``
is with it; the stack tells depth + VLM from VLM only. Surrogate-judge rows
keep their "surrogate VLM" label (CLAUDE.md rule 7).
"""
from __future__ import annotations

import argparse
import glob
import json
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from eval.metrics import by_policy, policy_summary, read_results
from vlm.eval_bank import GATE_RECALL, TUNE_MAX_FPR, auroc, flight_ci, labelled, read_rows, tune_tau

REPO = Path(__file__).resolve().parent.parent
MAX_GATE_FPR = 1.5 * TUNE_MAX_FPR  # the false-alarm cap of the M2 gate (D23), as in vlm.eval_bank.gate_verdict
CLEAR_STACK = "none (no depth model)"  # perception.stand_ins.ClearStack.name


# ---- inputs --------------------------------------------------------------------------------


def load_latency(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """``{model spec: bench row}`` from ``vlm.bench_latency`` files; the depth row is under the key "depth". Later files win."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            if r.get("outcome") == "success":
                out["depth" if r.get("kind") == "depth" else r["model"]] = {**r, "source": Path(p).name}
    return out


def load_depth(path: str | Path) -> dict[str, dict[str, float]]:
    """``{stem: {"est": corridor minimum estimate, "infer_ms": ...}}`` from a ``perception.run_bank`` file."""
    return {r["stem"]: {"est": float(r["corridor_min_est"]), "infer_ms": float(r.get("infer_ms", float("nan")))} for r in read_rows(path)}


# ---- frame-bank table ----------------------------------------------------------------------


def _score(frames: list[dict[str, Any]], pred: dict[str, bool], conf: dict[str, float] | None = None) -> dict[str, Any]:
    pos = [f for f in frames if f["label"] == "pos"]
    neg = [f for f in frames if f["label"] == "neg"]
    tp, fp = sum(pred[f["stem"]] for f in pos), sum(pred[f["stem"]] for f in neg)
    lo, hi = flight_ci([{"episode": f["episode"], "confidence": 1.0 if pred[f["stem"]] else -1.0} for f in pos])
    by_type: dict[str, list[bool]] = defaultdict(list)
    for f in pos:
        by_type[str(f["obstacle"])].append(pred[f["stem"]])
    return {
        "recall": tp / len(pos) if pos else float("nan"), "recall_ci": [lo, hi],
        "fpr": fp / len(neg) if neg else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "auroc": auroc([conf[f["stem"]] for f in pos], [conf[f["stem"]] for f in neg]) if conf else float("nan"),
        "by_obstacle": {t: {"n": len(v), "recall": sum(v) / len(v)} for t, v in sorted(by_type.items())},
    }


def choose_settings(rows: list[dict[str, Any]], max_fpr: float, condition: str | None = None) -> dict[str, dict[str, Any]]:
    """Per model: the condition with the best val AUROC (or ``condition``) and its val-tuned cut. ``rows`` are labelled."""
    by_mc: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mc[(r["model"], r["condition"])].append(r)
    out: dict[str, dict[str, Any]] = {}
    for model in sorted({m for m, _ in by_mc}):
        best: tuple[float, str] | None = None
        for (m, c), g in sorted(by_mc.items()):
            if m != model or (condition is not None and c != condition):
                continue
            val = [r for r in g if r["split"] == "val"]
            a = auroc([r["confidence"] for r in val if r["label"] == "pos"], [r["confidence"] for r in val if r["label"] == "neg"])
            if best is None or (a == a and a > best[0]):
                best = (a if a == a else -1.0, c)
        if best is None:
            continue
        val = [r for r in by_mc[(model, best[1])] if r["split"] == "val"]
        tau = tune_tau(val, max_fpr)
        out[model] = {"condition": best[1], "val_auroc": best[0] if best[0] >= 0 else float("nan"),
                      "tau": tau if tau == tau else 0.0, "tau_tuned": tau == tau, "conditions_tried": sorted({c for m, c in by_mc if m == model})}
    return out


def build(
    vlm_rows: list[dict[str, Any]], depth: dict[str, dict[str, float]], latency: dict[str, dict[str, Any]] | None = None,
    d_safe: float = 8.0, d_neg: float = 30.0, max_fpr: float = TUNE_MAX_FPR, condition: str | None = None, score_split: str = "train",
) -> dict[str, Any]:
    """The frame-bank benchmark as a dict (``report`` renders it)."""
    if score_split == "test" or any(r.get("split") == "test" for r in vlm_rows):
        raise SystemExit("refusing to benchmark on test frames before the end of the project (CLAUDE.md rule 3)")
    latency = latency or {}
    rows = labelled(vlm_rows, d_safe, d_neg)
    settings = choose_settings(rows, max_fpr, condition)
    if not settings:
        raise SystemExit("no VLM rows to benchmark" + (f" for condition {condition!r}" if condition else ""))
    per_model = {m: {r["stem"]: r for r in rows if r["model"] == m and r["condition"] == s["condition"]} for m, s in settings.items()}
    stems = set(depth)
    for m in per_model:
        stems &= set(per_model[m])
    any_model = next(iter(per_model.values()))
    frames = [any_model[s] for s in sorted(stems) if any_model[s]["split"] == score_split and any_model[s]["label"] != "ambiguous"]
    n_union = len({s for m in per_model.values() for s in m})

    depth_pred = {f["stem"]: depth[f["stem"]]["est"] < d_safe for f in frames}
    misses = [f for f in frames if f["label"] == "pos" and not depth_pred[f["stem"]]]
    d_lat = latency.get("depth")
    infer = np.array([depth[f["stem"]]["infer_ms"] for f in frames], dtype=np.float64)
    infer = infer[~np.isnan(infer)]
    depth_cost = ({"median_ms": 1000 * d_lat["median_s"], "p95_ms": 1000 * d_lat["p95_s"], "gpu_gb": d_lat.get("gpu_peak_mem_gb"), "source": d_lat["source"]}
                  if d_lat else {"median_ms": float(np.median(infer)) if infer.size else float("nan"),
                                 "p95_ms": float(np.percentile(infer, 95)) if infer.size else float("nan"), "gpu_gb": None,
                                 "source": "perception.run_bank infer_ms (includes loading the image; may be another machine)"})
    systems: list[dict[str, Any]] = [{"system": "depth only", "kind": "depth", "model": None, **depth_cost,
                                      **_score(frames, depth_pred, {f["stem"]: -depth[f["stem"]]["est"] for f in frames})}]
    fused: list[dict[str, Any]] = []
    for m, s in settings.items():
        conf = {f["stem"]: float(per_model[m][f["stem"]]["confidence"]) for f in frames}
        pred = {k: v > s["tau"] for k, v in conf.items()}
        v_lat = latency.get(m)
        logged = np.array([float(per_model[m][f["stem"]].get("latency_s", float("nan"))) for f in frames], dtype=np.float64)
        logged = logged[~np.isnan(logged)]
        cost = ({"median_ms": 1000 * v_lat["median_s"], "p95_ms": 1000 * v_lat["p95_s"], "gpu_gb": v_lat.get("gpu_peak_mem_gb"), "source": v_lat["source"]}
                if v_lat else {"median_ms": 1000 * float(np.median(logged)) if logged.size else float("nan"),
                               "p95_ms": 1000 * float(np.percentile(logged, 95)) if logged.size else float("nan"), "gpu_gb": None,
                               "source": "latency logged by vlm.eval_bank (other models may have been loaded too)"})
        miss_recall = sum(pred[f["stem"]] for f in misses) / len(misses) if misses else float("nan")
        sc = _score(frames, pred, conf)
        passes = bool(miss_recall == miss_recall and miss_recall >= GATE_RECALL and sc["fpr"] <= MAX_GATE_FPR)
        systems.append({"system": f"VLM only: {m}", "kind": "vlm", "model": m, **cost, **sc, "miss_recall": miss_recall, "d23_pass": passes, **s})
        both = {k: depth_pred[k] or pred[k] for k in pred}
        gb = None if cost["gpu_gb"] is None or depth_cost["gpu_gb"] is None else round(cost["gpu_gb"] + depth_cost["gpu_gb"], 2)
        fused.append({"system": f"depth + VLM: {m}", "kind": "both", "model": m, "median_ms": depth_cost["median_ms"] + cost["median_ms"],
                      "p95_ms": depth_cost["p95_ms"] + cost["p95_ms"], "gpu_gb": gb, "source": "sum of the two", **_score(frames, both)})
    systems += fused

    passing = [s for s in systems if s["kind"] == "vlm" and s["d23_pass"]]
    if passing:  # smallest memory first; unknown memory last; then the better recall on the misses
        pick = min(passing, key=lambda s: (s["gpu_gb"] is None, s["gpu_gb"] or 0.0, -s["miss_recall"]))
        pick_text = (f"{pick['model']} ({pick['condition']}, cut {pick['tau']:+.2f}): the smallest-memory variant that passes the M2 gate (D23), "
                     f"recall on depth's misses {pick['miss_recall']:.2f}, false alarms {pick['fpr']:.2f}"
                     + ("" if pick["gpu_gb"] is not None else "; memory unknown, so 'smallest' is not established"))
    else:
        vl = [s for s in systems if s["kind"] == "vlm" and s["miss_recall"] == s["miss_recall"]]
        best = max(vl, key=lambda s: s["miss_recall"]) if vl else None
        pick = None
        pick_text = ("no variant passes the M2 gate (D23)" + (f"; the closest is {best['model']} ({best['condition']}), recall on depth's misses "
                                                              f"{best['miss_recall']:.2f} at false alarms {best['fpr']:.2f}" if best else ""))
    pos = [f for f in frames if f["label"] == "pos"]
    return {
        "d_safe": d_safe, "d_neg": d_neg, "max_fpr": max_fpr, "score_split": score_split,
        "n_frames": len(frames), "n_pos": len(pos), "n_neg": len(frames) - len(pos), "pos_flights": len({f["episode"] for f in pos}),
        "n_depth_misses": len(misses), "n_common_stems": len(stems), "n_union_stems": n_union,
        "settings": settings, "systems": systems, "pick": pick["model"] if pick else None, "pick_text": pick_text,
    }


# ---- flights -------------------------------------------------------------------------------


def flight_rows(dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """One row per (sweep directory, policy): which system it is, and its flight metrics."""
    out = []
    for d in dirs:
        path = Path(d) / "results.jsonl"
        if not path.exists():
            raise SystemExit(f"no results.jsonl in {d}")
        for policy, rs in sorted(by_policy(read_results(path)).items()):
            stacks = sorted({str(r.extra.get("stack")) for r in rs})
            backends = sorted({str(r.extra.get("backend")) for r in rs})
            judges = sorted({r.judge for r in rs if r.n_calls})
            no_depth = stacks == [CLEAR_STACK]
            asks = any(r.n_calls for r in rs)
            system = ("VLM only" if no_depth else "depth + VLM") if asks else ("no perception" if no_depth else "depth only")
            out.append({"dir": Path(d).name, "policy": policy, "system": system, "stack": ", ".join(stacks), "backend": ", ".join(backends),
                        "judge": ", ".join(judges) or "none", "splits": sorted({r.split for r in rs}), **policy_summary(rs)})
    if any("test" in r["splits"] for r in out):
        raise SystemExit("refusing to tabulate test flights here; the final test run has its own report (PLAN_DETAILED.md D14)")
    return out


# ---- report --------------------------------------------------------------------------------


def _f(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def report(b: dict[str, Any] | None, flights: list[dict[str, Any]] | None = None) -> str:
    L = ["# Benchmark: depth only, VLM only, depth + VLM"]
    if b:
        L += ["", "## Frame bank", "",
              f"Task: an obstacle within {b['d_safe']:g} m in the corridor window. Positive: nearest true surface closer than {b['d_safe']:g} m. "
              f"Negative: nothing closer than {b['d_neg']:g} m. In between: left out.",
              f"Scored on {b['n_frames']} {b['score_split']} frames ({b['n_pos']} positive from {b['pos_flights']} flights, {b['n_neg']} negative), "
              f"the same frames for every system. The fast stack misses {b['n_depth_misses']} of the positives. Intervals: 95%, resampling whole flights.",
              f"Each VLM's condition and cut were chosen on val (cut: lowest with val false alarms at most {b['max_fpr']:.2f}), never on the frames scored here."]
        if b["n_common_stems"] < b["n_union_stems"]:
            L.append(f"**Note:** only {b['n_common_stems']} of {b['n_union_stems']} evaluated frames have a row for every system; the table uses those.")
        L += ["", "| System | recall [95% CI] | false alarms | precision | AUROC | recall on depth's misses | latency median / p95 (ms) | peak GPU (GB) |",
              "|---|---|---|---|---|---|---|---|"]
        for s in b["systems"]:
            ci = "n/a" if s["recall_ci"][0] != s["recall_ci"][0] else f"{s['recall_ci'][0]:.2f} to {s['recall_ci'][1]:.2f}"
            L.append(f"| {s['system']} | {_f(s['recall'])} [{ci}] | {_f(s['fpr'])} | {_f(s['precision'])} | {_f(s['auroc'])} | "
                     f"{_f(s.get('miss_recall')) if s['kind'] == 'vlm' else ''} | {_f(s['median_ms'], 0)} / {_f(s['p95_ms'], 0)} | {_f(s['gpu_gb'], 1)} |")
        L += ["", "### VLM settings (chosen on val)", "", "| Model | condition | val AUROC | cut on the confidence | passes the M2 gate (D23) |", "|---|---|---|---|---|"]
        for s in b["systems"]:
            if s["kind"] == "vlm":
                cut = f"{s['tau']:+.2f}" + ("" if s["tau_tuned"] else " (not tuned: no val negatives)")
                L.append(f"| {s['model']} | {s['condition']} (of {', '.join(s['conditions_tried'])}) | {_f(s['val_auroc'])} | {cut} | {'yes' if s['d23_pass'] else 'no'} |")
        types = sorted({t for s in b["systems"] for t in s["by_obstacle"]})
        if types:
            counts = b["systems"][0]["by_obstacle"]
            L += ["", "### Recall by obstacle type (positive frames)", "",
                  "| System | " + " | ".join(f"{t} (n={counts.get(t, {}).get('n', 0)})" for t in types) + " |", "|---|" + "---|" * len(types)]
            for s in b["systems"]:
                L.append(f"| {s['system']} | " + " | ".join(_f(s["by_obstacle"].get(t, {}).get("recall")) for t in types) + " |")
        L += ["", f"**Pick, by rule (the team decides):** {b['pick_text']}.",
              f"Rule: recall on depth's misses at least {GATE_RECALL:.2f} with false alarms at most {MAX_GATE_FPR:.2f}, then the least GPU memory.",
              "", "Latency and memory sources: " + "; ".join(dict.fromkeys(f"{s['system']}: {s['source']}" for s in b["systems"] if s["kind"] != "both")) + "."]
    if flights:
        L += ["", "## Flights", "", "One row per sweep and policy. Compare rows only when the backend, the seeds and the delay match.", "",
              "| System | sweep | backend | fast stack | judge | flights | crash rate [95% CI] | success | calls/flight | mission time (s) | real latency p50 (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in flights:
            lo, hi = r["crash_rate_ci"]
            L.append(f"| {r['system']} | {r['dir']} / {r['policy']} | {r['backend']} | {r['stack']} | {r['judge']} | {r['flights']} | "
                     f"{_f(r['crash_rate'])} [{_f(lo)} to {_f(hi)}] | {_f(r['success_rate'])} | {_f(r['calls_per_flight'], 1)} | "
                     f"{_f(r['mission_time_s_mean'], 1)} | {_f(r['latency_s']['p50'], 3)} |")
    return "\n".join(L)


# ---- CLI -----------------------------------------------------------------------------------


def _log_run(args: argparse.Namespace, n: int, wall: float, outcome: str) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    (REPO / "logs").mkdir(exist_ok=True)
    with open(REPO / "logs" / "runs.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": commit, "env": socket.gethostname(), "command": " ".join(sys.argv),
                            "config": {"vlm": args.vlm, "run": args.run, "latency": args.latency, "flights": args.flights, "d_safe": args.d_safe,
                                       "d_neg": args.d_neg, "max_fpr": args.max_fpr, "condition": args.condition, "score_split": args.score_split},
                            "seeds": "frame bank, configs/bank_split.yaml; test closed", "n_episodes": n, "wall_seconds": round(wall, 1), "outcome": outcome}) + "\n")


def _expand(paths: list[str]) -> list[str]:
    return [q for p in paths for q in (sorted(glob.glob(p)) or [p])]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vlm", nargs="*", default=[], help="vlm.eval_bank row files (JSONL); several are merged")
    p.add_argument("--run", help="perception.run_bank JSONL (the fast stack over the bank)")
    p.add_argument("--latency", nargs="*", default=[], help="vlm.bench_latency JSON files (globs allowed); later files win")
    p.add_argument("--flights", nargs="*", default=[], help="eval.sweep output directories")
    p.add_argument("--d-safe", type=float, default=8.0)
    p.add_argument("--d-neg", type=float, default=30.0)
    p.add_argument("--max-fpr", type=float, default=TUNE_MAX_FPR, help="val false-alarm budget for each VLM's cut")
    p.add_argument("--condition", help="force one eval_bank condition for every VLM (default: the best on val for each)")
    p.add_argument("--score-split", default="train", choices=["train", "val"], help="'val' scores on the frames the cut was tuned on: optimistic")
    p.add_argument("--out", default=str(REPO / "results" / "benchmark.md"))
    p.add_argument("--no-log", action="store_true", help="do not append to logs/runs.jsonl (tests)")
    a = p.parse_args(argv)
    if bool(a.vlm) != bool(a.run):
        raise SystemExit("the frame-bank table needs both --vlm and --run")
    if not a.vlm and not a.flights:
        raise SystemExit("nothing to do: give --vlm and --run, and/or --flights")
    start, outcome, n = time.time(), "failed", 0
    try:
        bank = None
        if a.vlm:
            rows = [r for path in _expand(a.vlm) for r in read_rows(path)]
            bank = build(rows, load_depth(a.run), load_latency(_expand(a.latency)), a.d_safe, a.d_neg, a.max_fpr, a.condition, a.score_split)
            n = bank["n_frames"]
        flights = flight_rows(_expand(a.flights)) if a.flights else None
        outcome = "success"
    finally:
        if not a.no_log:
            _log_run(a, n, time.time() - start, outcome)
    text = report(bank, flights)
    if bank and a.score_split == "val":
        text += "\n\n**Scored on val, the frames the cuts were tuned on: these numbers are optimistic.**"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps({"frame_bank": bank, "flights": flights}, indent=1, default=float))
    print(text)
    print(f"\nwritten: {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
