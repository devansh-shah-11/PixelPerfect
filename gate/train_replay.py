"""Headline 1 (E4, chunk M6.3): gate-L3 against gate-L1, trained on the same flight frames.

Both gates see identical inputs (the run-time features stored by ``eval.replay``); only the label
differs. They are compared on val flights **at equal query rates**, by what asking is for:

- ``l3_recall``: share of "worth asking" frames (L3 = 1) that the gate asks on;
- ``flights_saved``: of the val flights that crash without the VLM and have at least one frame
  where asking saves them, the share where the gate asks on at least one such frame (an offline
  estimate: in flight, the first ask also changes what follows);
- ``wasted``: share of asks on frames where asking changes nothing (L3 = 0).

The closed-loop check of the same two pickles is a normal sweep with ``policy: gate`` (M5).

    python -m gate.train_replay --train results/replay/airsim_da2_surrogate/replay_labels.jsonl \\
        --val results/replay/airsim_da2_surrogate_val/replay_labels.jsonl --out results/gate_replay
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gate.features import DEPTH_KEYS, IMAGE_KEYS, TEMPORAL_KEYS, matrix
from gate.train import auroc, auroc_ci, make_model

QUERY_RATES = (0.02, 0.05, 0.10, 0.20)
FAMILY_RATES = (0.05, 0.10)
LABELS = ("l1", "l3")


def load_rows(paths: Sequence[str | Path], split: str) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    bad = {r["split"] for r in rows} - {split}
    if bad:
        raise SystemExit(f"--{split} files contain {sorted(bad)} rows: train and val flights must not mix")
    return rows


def ask_mask(scores: np.ndarray, rate: float) -> np.ndarray:
    """Ask on the top ``rate`` share of frames by score (ties broken by order, so the rate is exact)."""
    n = int(round(rate * scores.size))
    mask = np.zeros(scores.size, dtype=bool)
    if n > 0:
        mask[np.argsort(-scores, kind="stable")[:n]] = True
    return mask


def _recall_saved(l3: np.ndarray, flight: np.ndarray, ask: np.ndarray) -> tuple[float, float, int]:
    """(L3 recall, share of savable flights with an ask on a savable frame, number of savable flights)."""
    savable = np.unique(flight[l3])
    hit = np.unique(flight[ask & l3])
    recall = float((ask & l3).sum() / l3.sum()) if l3.any() else float("nan")
    return recall, (hit.size / savable.size if savable.size else float("nan")), int(savable.size)


def at_rate(rows: Sequence[dict[str, Any]], scores: np.ndarray, rate: float) -> dict[str, float]:
    ask = ask_mask(scores, rate)
    l3 = np.array([r["l3"] for r in rows], dtype=bool)
    recall, saved, n_savable = _recall_saved(l3, np.array([r["seed"] for r in rows]), ask)
    return {
        "query_rate": rate, "asks": int(ask.sum()), "l3_recall": recall, "flights_saved": saved, "savable_flights": n_savable,
        "wasted": float((ask & ~l3).sum() / ask.sum()) if ask.any() else float("nan"),
    }


def paired_diff(rows: Sequence[dict[str, Any]], s_a: np.ndarray, s_b: np.ndarray, rate: float,
                n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Gate a minus gate b at one query rate, with a paired bootstrap over val flights.

    Frames of one flight are strongly correlated, so flights (not frames) are resampled; both gates
    see the same resample, and each resample re-applies the query rate to the frames it drew. A
    flight drawn twice counts as two flights. ``p_a_better``: share of resamples where a saves more
    flights than b (ties count half), a one-sided bootstrap check, not a test corrected for the four rates.
    """
    l3 = np.array([r["l3"] for r in rows], dtype=bool)
    seeds = np.array([r["seed"] for r in rows])
    uniq = np.unique(seeds)
    idx_of = {u: np.flatnonzero(seeds == u) for u in uniq}
    ra, fa, _ = _recall_saved(l3, seeds, ask_mask(s_a, rate))
    rb, fb, _ = _recall_saved(l3, seeds, ask_mask(s_b, rate))
    rng = np.random.default_rng(seed)
    d_rec, d_saved = [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_of[u] for u in draw])
        flight = np.concatenate([np.full(idx_of[u].size, j) for j, u in enumerate(draw)])
        y = l3[idx]
        if not y.any():
            continue
        a = _recall_saved(y, flight, ask_mask(s_a[idx], rate))
        b = _recall_saved(y, flight, ask_mask(s_b[idx], rate))
        d_rec.append(a[0] - b[0])
        d_saved.append(a[1] - b[1])
    d_rec_arr, d_saved_arr = np.array(d_rec), np.array(d_saved)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan")] * 2  # noqa: E731
    return {"query_rate": rate, "d_l3_recall": ra - rb, "d_l3_recall_ci": ci(d_rec_arr),
            "d_flights_saved": fa - fb, "d_flights_saved_ci": ci(d_saved_arr),
            "p_a_better": float((d_saved_arr > 0).mean() + 0.5 * (d_saved_arr == 0).mean()) if d_saved_arr.size else float("nan"),
            "n_boot": int(d_saved_arr.size)}


def compare(train: list[dict], val: list[dict], names: list[str], kind: str, out_dir: Path) -> dict[str, Any]:
    x_tr, x_va = matrix([r["features"] for r in train], names), matrix([r["features"] for r in val], names)
    y3_va = np.array([r["l3"] for r in val])
    eps = np.array([r["seed"] for r in val])
    scores: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {"gates": {}}
    for label in LABELS:
        y_tr = np.array([r[label] for r in train])
        if not 0 < y_tr.sum() < y_tr.size:
            out["gates"][f"gate_{label}"] = {"skipped": f"train has {int(y_tr.sum())} positives of {y_tr.size}"}
            continue
        model = make_model(kind).fit(x_tr, y_tr)
        s = model.predict_proba(x_va)[:, 1]
        scores[f"gate_{label}"] = s
        path = out_dir / f"gate_{label}_{kind}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "names": list(names), "kind": kind, "feature_set": f"replay-{label}"}, f)
        out["gates"][f"gate_{label}"] = {"path": str(path), "train_pos": int(y_tr.sum()), "auroc_vs_l3": auroc(y3_va, s),
                                         "auroc_vs_l3_ci": list(auroc_ci(y3_va, s, eps))}
    # Reference points on the same frames: a seeded random asker and the strongest single depth feature (sign on train).
    scores["random"] = np.random.default_rng(0).random(len(val))
    y3_tr = np.array([r["l3"] for r in train])
    best, best_auc, sign = None, -1.0, 1.0
    for k in DEPTH_KEYS:
        a = auroc(y3_tr, np.array([r["features"][k] for r in train]))
        if a == a and max(a, 1 - a) > best_auc:
            best, best_auc, sign = k, max(a, 1 - a), (1.0 if a >= 0.5 else -1.0)
    if best is not None:
        scores[f"threshold({'-' if sign < 0 else ''}{best})"] = sign * np.array([r["features"][best] for r in val])
    out["at_rate"] = {name: [at_rate(val, s, q) for q in QUERY_RATES] for name, s in scores.items()}
    if "gate_l3" in scores and "gate_l1" in scores:  # the headline difference, with its flight-level uncertainty
        out["l3_minus_l1"] = [paired_diff(val, scores["gate_l3"], scores["gate_l1"], q) for q in QUERY_RATES]
    # Score cut that asks on the top `rate` share of val replay frames: the starting tau for the closed-loop sweep
    # (val is the tuning split; the cooldown and the asks themselves change the rate in flight, so the sweep reports calls/flight).
    out["tau_at_rate"] = {name: {str(q): float(np.sort(s)[::-1][max(int(round(q * s.size)) - 1, 0)]) for q in QUERY_RATES}
                          for name, s in scores.items() if name.startswith("gate_")}
    out["by_family"] = by_family(val, {k: v for k, v in scores.items() if k.startswith("gate_")}, FAMILY_RATES)
    # Val scores per frame, so later analysis (per family, per flight, plots) needs no retraining.
    out["val_scores"] = {"seed": [r["seed"] for r in val], "k": [r["k"] for r in val],
                         **{name: [float(x) for x in s] for name, s in scores.items()}}
    return out


def flight_family(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    """Obstacle family per flight: the families its frames saw (one obstacle per flight in these configs)."""
    fams: dict[int, set] = {}
    for r in rows:
        fams.setdefault(r["seed"], set())
        if r.get("obstacle_family"):
            fams[r["seed"]].add(r["obstacle_family"])
    return {sd: ",".join(sorted(f)) or "none" for sd, f in fams.items()}


def by_family(rows: Sequence[dict[str, Any]], scores: dict[str, np.ndarray], rates: Sequence[float]) -> list[dict[str, Any]]:
    """Savable val flights saved by each gate, split by obstacle family. Asks are chosen over all val frames
    at the given rate (the same asks as the main table), then counted per family."""
    fam = flight_family(rows)
    l3 = np.array([r["l3"] for r in rows], dtype=bool)
    seeds = np.array([r["seed"] for r in rows])
    out = []
    for rate in rates:
        asks = {name: ask_mask(s, rate) for name, s in scores.items()}
        for f in sorted(set(fam.values())):
            in_f = np.isin(seeds, [sd for sd, x in fam.items() if x == f])
            savable = np.unique(seeds[in_f & l3])
            row = {"query_rate": rate, "family": f, "flights": int(len({sd for sd, x in fam.items() if x == f})), "savable": int(savable.size)}
            for name, a in asks.items():
                row[name] = int(np.unique(seeds[in_f & l3 & a]).size)
            out.append(row)
    return out


def report(meta: dict[str, Any], res: dict[str, Any]) -> str:
    L = [f"Replay frames: train {meta['train_frames']} ({meta['train_l3']} L3, {meta['train_l1']} L1 positive), "
         f"val {meta['val_frames']} ({meta['val_l3']} L3 positive, {meta['val_flights']} flights). Judge: {meta['judge']}.", ""]
    for g, v in res["gates"].items():
        L.append(f"- {g}: " + (v["skipped"] if "skipped" in v else
                 f"val AUROC against L3 {v['auroc_vs_l3']:.3f} [{v['auroc_vs_l3_ci'][0]:.3f} to {v['auroc_vs_l3_ci'][1]:.3f}]"))
    L += ["", "| Policy | query rate | L3 recall | flights saved (of savable) | wasted asks |", "|---|---|---|---|---|"]
    for name, rows in res["at_rate"].items():
        for r in rows:
            L.append(f"| {name} | {r['query_rate']:.2f} | {r['l3_recall']:.3f} | {r['flights_saved']:.3f} ({r['savable_flights']}) | {r['wasted']:.3f} |")
    if "l3_minus_l1" in res:
        L += ["", "gate_l3 minus gate_l1, 95% paired bootstrap over val flights:", "",
              "| query rate | L3 recall difference | flights saved difference | share of resamples gate_l3 saves more |", "|---|---|---|---|"]
        for d in res["l3_minus_l1"]:
            L.append(f"| {d['query_rate']:.2f} | {d['d_l3_recall']:+.3f} [{d['d_l3_recall_ci'][0]:+.3f}, {d['d_l3_recall_ci'][1]:+.3f}] | "
                     f"{d['d_flights_saved']:+.3f} [{d['d_flights_saved_ci'][0]:+.3f}, {d['d_flights_saved_ci'][1]:+.3f}] | {d['p_a_better']:.2f} |")
    if res.get("tau_at_rate"):
        L += ["", "Score cut for each query rate on val frames (tau for the closed-loop sweep):", "",
              "| gate | " + " | ".join(f"{q:.2f}" for q in QUERY_RATES) + " |", "|---|" + "---|" * len(QUERY_RATES)]
        for g, taus in res["tau_at_rate"].items():
            L.append(f"| {g} | " + " | ".join(f"{taus[str(q)]:.4f}" for q in QUERY_RATES) + " |")
    if res.get("by_family"):
        gates = [k for k in res["by_family"][0] if k.startswith("gate_")]
        L += ["", "Savable val flights saved, by obstacle family:", "",
              "| query rate | family | flights | savable | " + " | ".join(gates) + " |", "|---|---|---|---|" + "---|" * len(gates)]
        for r in res["by_family"]:
            L.append(f"| {r['query_rate']:.2f} | {r['family']} | {r['flights']} | {r['savable']} | " + " | ".join(str(r[g]) for g in gates) + " |")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", nargs="+", required=True, help="replay_labels.jsonl from train-seed runs")
    p.add_argument("--val", nargs="+", required=True, help="replay_labels.jsonl from val-seed runs")
    p.add_argument("--model", default="mlp", choices=["logreg", "gbt", "mlp"])
    p.add_argument("--no-image", action="store_true", help="depth + temporal features only")
    p.add_argument("--out", default="results/gate_replay")
    a = p.parse_args(argv)
    train, val = load_rows(a.train, "train"), load_rows(a.val, "val")
    if {r["seed"] for r in train} & {r["seed"] for r in val}:
        raise SystemExit("a seed appears in both train and val")
    names = list(DEPTH_KEYS + TEMPORAL_KEYS) + ([] if a.no_image else list(IMAGE_KEYS))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = compare(train, val, names, a.model, out)
    meta = {"train_frames": len(train), "train_l3": sum(r["l3"] for r in train), "train_l1": sum(r["l1"] for r in train),
            "val_frames": len(val), "val_l3": sum(r["l3"] for r in val), "val_flights": len({r["seed"] for r in val}),
            "judge": sorted({r["judge"] for r in train + val}), "model": a.model, "features": names}
    text = report(meta, res)
    (out / "replay_gate_report.md").write_text(text + "\n")
    (out / "replay_gate_results.json").write_text(json.dumps({"meta": meta, **res}, indent=1, default=float))
    print(text)
    print(f"\nwritten: {out}/replay_gate_report.md, replay_gate_results.json, gate_l1_{a.model}.pkl, gate_l3_{a.model}.pkl")


if __name__ == "__main__":
    main()
