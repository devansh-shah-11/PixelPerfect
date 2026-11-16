"""Train and evaluate the gate on the frame bank (PLAN_DETAILED.md §4.2 to §4.3, E2; chunks M3.3 to M3.5).

    python -m gate.train --run /scratch/$USER/bank_runs/da2_small.jsonl --gt /scratch/$USER/bank_runs/bank_gt.jsonl \
        [--image-features /scratch/$USER/bank_runs/image_features.jsonl] [--holdout-family glass_pane_alpha20] \
        --d-safe 8 --out results/gate

Label: L1, "missed obstacle" (``gate.labels.l1_labels``): the clean ground-truth
corridor distance (``gate.bank_gt``, with the metadata fallback for glass) is
inside ``d_safe`` while the estimate says the corridor is clear.

Trained on train episodes, scored on val episodes (test stays closed). Compared
models, in PLAN order:

- ``baseline_var``: the hand-tuned depth-variance threshold (the score is the corridor variance);
- ``best_single``: the best single feature with its sign, both chosen on train (the strongest
  "tuned threshold" a careful person would pick);
- ``logreg``, ``gbt`` (gradient boosting) and ``mlp`` (small MLP), each on depth-only, depth +
  temporal, and, if given, depth + temporal + image features.

Reported on val: AUROC with a 95% interval from resampling whole flights, AUPRC,
ECE (probabilistic models), recall and FPR at the cut that gives 10% FPR on
train, and AUROC per obstacle family. AUROC above 0.99 is flagged as a likely
leak (CLAUDE.md escalation). Learned models are saved as pickles that
``gate.model.GateModel`` loads behind ``interfaces.Gate``.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gate.features import DEPTH_KEYS, IMAGE_KEYS, TEMPORAL_KEYS, add_temporal_offline, matrix
from gate.labels import l1_labels
from perception.bank_split import load_split, split_of_episode
from perception.frame_bank import read_jsonl

MIN_ALONG_M = 1.0
TRAIN_FPR = 0.10
LEAK_AUROC = 0.99


# ---- data ----------------------------------------------------------------------------------


def load_dataset(run_path: str | Path, gt_path: str | Path, split_path: str | Path, image_path: str | Path | None, d_safe: float) -> list[dict[str, Any]]:
    """Train and val rows with ``features`` (depth + temporal [+ image]), ``label`` (L1) and ``split``."""
    of_episode = split_of_episode(load_split(split_path))
    clean = {r["stem"]: r for r in read_jsonl(gt_path)}
    img = {r["stem"]: r for r in read_jsonl(image_path)} if image_path else {}
    rows = [r for r in read_jsonl(run_path) if of_episode.get(int(r["episode"])) in ("train", "val")]
    rows = add_temporal_offline(rows)  # before any frame is dropped, so the previous step is the real one
    out = []
    for r in rows:
        along = r.get("gt_along_m")
        if along is not None and along < MIN_ALONG_M:
            continue
        c = clean.get(r["stem"])
        if c is None:
            raise KeyError(f"no clean ground truth for {r['stem']}; rerun gate.bank_gt over the whole bank")
        f = dict(r["features"])
        if img:
            f.update({k: v for k, v in img[r["stem"]].items() if k != "stem"})
        gt = float(c["corridor_min_gt_clean"])
        out.append({"stem": r["stem"], "episode": int(r["episode"]), "step": int(r["step"]), "obstacle": r["obstacle"],
                    "condition": r["condition"], "split": of_episode[int(r["episode"])], "features": f,
                    "gt": gt, "gt_source": c.get("gt_source", "seg"), "est": float(r["corridor_min_est"]),
                    "label": int(l1_labels(gt, float(r["corridor_min_est"]), d_safe))})
    return out


# ---- metrics -------------------------------------------------------------------------------


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float("nan")


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on predicted probability."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def auroc_ci(y: np.ndarray, s: np.ndarray, episodes: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% interval for AUROC, resampling whole flights."""
    eps = np.unique(episodes)
    if eps.size < 2:
        return float("nan"), float("nan")
    by = {e: np.flatnonzero(episodes == e) for e in eps}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([by[e] for e in rng.choice(eps, size=eps.size, replace=True)])
        a = auroc(y[pick], s[pick])
        if a == a:
            vals.append(a)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def cut_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = TRAIN_FPR) -> float:
    neg = np.sort(s[y == 0])
    if neg.size == 0:
        return float("nan")
    k = int(np.floor(fpr * neg.size))
    return float(neg[neg.size - 1 - k]) if k < neg.size else float(neg[0] - 1e-9)


def evaluate_scores(rows: Sequence[dict[str, Any]], s: np.ndarray, cut: float, prob: bool) -> dict[str, Any]:
    y = np.array([r["label"] for r in rows])
    eps = np.array([r["episode"] for r in rows])
    lo, hi = auroc_ci(y, s, eps)
    pos, neg = y == 1, y == 0
    fam: dict[str, Any] = {}
    for t in sorted({r["obstacle"] for r in rows}):
        m = np.array([r["obstacle"] == t for r in rows]) & pos
        fam[t] = {"pos": int(m.sum()), "pos_flights": len(set(eps[m])),
                  "auroc_vs_all_neg": auroc(np.r_[np.ones(m.sum()), np.zeros(neg.sum())], np.r_[s[m], s[neg]]) if m.any() else float("nan"),
                  "recall_at_cut": float((s[m] > cut).mean()) if m.any() and cut == cut else float("nan")}
    return {
        "n": int(y.size), "pos": int(pos.sum()), "pos_flights": len(set(eps[pos])),
        "auroc": auroc(y, s), "auroc_ci": [lo, hi], "auprc": auprc(y, s), "base_rate": float(pos.mean()),
        "ece": ece(y, s) if prob else None,
        "recall_at_cut": float((s[pos] > cut).mean()) if pos.any() and cut == cut else float("nan"),
        "fpr_at_cut": float((s[neg] > cut).mean()) if neg.any() and cut == cut else float("nan"),
        "by_family": fam,
    }


# ---- models --------------------------------------------------------------------------------


def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0))
    raise ValueError(kind)


def train_and_score(train: list[dict], val: list[dict], feature_sets: dict[str, list[str]], kinds: Sequence[str], out_dir: Path) -> dict[str, Any]:
    y_tr = np.array([r["label"] for r in train])
    results: dict[str, Any] = {}

    # Hand-tuned baselines: one feature, sign chosen on train.
    var_tr = np.array([r["features"]["corridor_var"] for r in train])
    var_va = np.array([r["features"]["corridor_var"] for r in val])
    results["baseline_var"] = {"features": ["corridor_var"], **evaluate_scores(val, var_va, cut_at_fpr(y_tr, var_tr), prob=False)}
    best, best_auc, best_sign = None, -1.0, 1.0
    for k in DEPTH_KEYS + TEMPORAL_KEYS + tuple(k for k in IMAGE_KEYS if k in train[0]["features"]):
        x = np.array([r["features"][k] for r in train])
        a = auroc(y_tr, x)
        if a == a and max(a, 1 - a) > best_auc:
            best, best_auc, best_sign = k, max(a, 1 - a), (1.0 if a >= 0.5 else -1.0)
    if best is not None:
        s_tr = best_sign * np.array([r["features"][best] for r in train])
        s_va = best_sign * np.array([r["features"][best] for r in val])
        results["best_single"] = {"features": [f"{'' if best_sign > 0 else '-'}{best}"], "train_auroc": best_auc,
                                  **evaluate_scores(val, s_va, cut_at_fpr(y_tr, s_tr), prob=False)}

    for fs_name, names in feature_sets.items():
        x_tr, x_va = matrix([r["features"] for r in train], names), matrix([r["features"] for r in val], names)
        for kind in kinds:
            model = make_model(kind).fit(x_tr, y_tr)
            p_tr, p_va = model.predict_proba(x_tr)[:, 1], model.predict_proba(x_va)[:, 1]
            key = f"{kind}/{fs_name}"
            results[key] = {"features": list(names), **evaluate_scores(val, p_va, cut_at_fpr(y_tr, p_tr), prob=True)}
            path = out_dir / f"gate_{kind}_{fs_name}.pkl"
            with open(path, "wb") as f:
                pickle.dump({"model": model, "names": list(names), "kind": kind, "feature_set": fs_name}, f)
            results[key]["path"] = str(path)
    return results


# ---- report --------------------------------------------------------------------------------


def report(meta: dict[str, Any], results: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"

    L = [f"L1 labels at d_safe {meta['d_safe']:g} m. Train {meta['train_n']} frames ({meta['train_pos']} positive), "
         f"val {meta['val_n']} frames ({meta['val_pos']} positive, {meta['val_pos_flights']} flights). "
         f"Glass labelled from metadata: {meta['metadata_rows']} rows." + (f" Held out of training: {meta['holdout']}." if meta.get("holdout") else ""),
         "", "| Model / features | val AUROC [95% CI over flights] | AUPRC (base rate) | ECE | recall / FPR at train 10%-FPR cut |", "|---|---|---|---|---|"]
    for k, r in results.items():
        lo, hi = r["auroc_ci"]
        L.append(f"| {k} | {f(r['auroc'])} [{f(lo)} to {f(hi)}] | {f(r['auprc'])} ({f(r['base_rate'])}) | {f(r['ece'])} | {f(r['recall_at_cut'])} / {f(r['fpr_at_cut'])} |")
    learned = {k: r for k, r in results.items() if "/" in k and r["auroc"] == r["auroc"]}
    if learned:
        best = max(learned, key=lambda k: learned[k]["auroc"])
        base = results["baseline_var"]["auroc"]
        L += ["", f"Best learned gate: **{best}**, val AUROC {f(learned[best]['auroc'])} against the depth-variance baseline {f(base)}"
              f" (M3 gate: {'beats' if learned[best]['auroc'] > base else 'does NOT beat'} the baseline).", "",
              f"### {best}: by obstacle family (positives vs all val negatives)", "",
              "| Family | positives (flights) | AUROC | recall at cut |", "|---|---|---|---|"]
        for t, r in learned[best]["by_family"].items():
            L.append(f"| {t} | {r['pos']} ({r['pos_flights']}) | {f(r['auroc_vs_all_neg'])} | {f(r['recall_at_cut'])} |")
        if learned[best]["auroc"] > LEAK_AUROC:
            L += ["", f"**WARNING: val AUROC above {LEAK_AUROC}. Suspect leakage (ground truth in a feature, or split by frame) before believing it.**"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="perception.run_bank JSONL")
    p.add_argument("--gt", required=True, help="gate.bank_gt JSONL (clean ground truth)")
    p.add_argument("--split", default="configs/bank_split.yaml")
    p.add_argument("--image-features", help="gate.image_features JSONL (optional)")
    p.add_argument("--d-safe", type=float, default=8.0)
    p.add_argument("--models", nargs="+", default=["logreg", "gbt", "mlp"])
    p.add_argument("--holdout-family", help="leave this obstacle type out of training (E6 generalisation)")
    p.add_argument("--out", default="results/gate")
    a = p.parse_args(argv)

    rows = load_dataset(a.run, a.gt, a.split, a.image_features, a.d_safe)
    train = [r for r in rows if r["split"] == "train" and r["obstacle"] != a.holdout_family]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val or sum(r["label"] for r in train) == 0:
        raise SystemExit("need train and val frames with at least one positive")
    fsets = {"depth": list(DEPTH_KEYS), "depth+temporal": list(DEPTH_KEYS + TEMPORAL_KEYS)}
    if a.image_features:
        fsets["depth+temporal+image"] = list(DEPTH_KEYS + TEMPORAL_KEYS + IMAGE_KEYS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = train_and_score(train, val, fsets, a.models, out)
    meta = {"d_safe": a.d_safe, "train_n": len(train), "train_pos": sum(r["label"] for r in train), "val_n": len(val),
            "val_pos": sum(r["label"] for r in val), "val_pos_flights": len({r["episode"] for r in val if r["label"]}),
            "metadata_rows": sum(r["gt_source"] == "metadata" for r in rows), "holdout": a.holdout_family,
            "positives_by_family_train": dict(sorted(Counter(r["obstacle"] for r in train if r["label"]).items()))}
    (out / "gate_results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    text = report(meta, results)
    (out / "gate_report.md").write_text(text + "\n")
    print(text + f"\n\nwritten: {out}/gate_report.md, gate_results.json, gate_*.pkl")


if __name__ == "__main__":
    main()
