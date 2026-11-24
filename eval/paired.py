"""Paired comparison of two policies from one sweep, and a repeatability check of its never-ask flights.

    python -m eval.paired --results results/sweeps/replay_gates_surrogate/results.jsonl \\
        --pairs gate_l3_q05:gate_l1_q05 gate_l3_q10:gate_l1_q10 \\
        --replay results/replay/airsim_da2_surrogate_val/replay_labels.jsonl

Every policy flies the same seeds (D15), so compare crashes seed by seed: the crash-rate and
calls-per-flight differences (a minus b) with a 95% paired bootstrap over seeds, and an exact McNemar
test on the seeds where only one of the two crashed. ``--replay`` compares the sweep's never-ask
flights with the replay's base flights on the same seeds; on a deterministic simulator they are equal.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

import numpy as np

from eval.metrics import EpisodeResult, by_policy, read_results


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts (1.0 when there are none)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a: Sequence[EpisodeResult], b: Sequence[EpisodeResult], n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    ra, rb = {r.seed: r for r in a}, {r.seed: r for r in b}
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        raise ValueError("the two policies share no seeds")
    ca, cb = (np.array([float(d[s].crashed) for s in seeds]) for d in (ra, rb))
    qa, qb = (np.array([float(d[s].n_calls) for s in seeds]) for d in (ra, rb))
    i = np.random.default_rng(seed).integers(0, len(seeds), size=(n_boot, len(seeds)))
    ci = lambda v: [float(np.percentile(v[i].mean(1), q)) for q in (2.5, 97.5)]  # noqa: E731
    only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    return {"seeds": len(seeds), "crash_a": float(ca.mean()), "crash_b": float(cb.mean()),
            "d_crash": float((ca - cb).mean()), "d_crash_ci": ci(ca - cb), "only_a_crashed": only_a, "only_b_crashed": only_b,
            "mcnemar_p": mcnemar_exact(only_a, only_b), "calls_a": float(qa.mean()), "calls_b": float(qb.mean()),
            "d_calls": float((qa - qb).mean()), "d_calls_ci": ci(qa - qb)}


def repeatability(never: Sequence[EpisodeResult], replay_rows: Sequence[dict[str, Any]], tol_s: float = 0.05) -> dict[str, Any]:
    """Seeds whose never-ask flight ended differently in the sweep and in the replay's base flight."""
    base = {r["seed"]: r["base_crash_t"] for r in replay_rows}
    sweep = {r.seed: (r.mission_time_s if r.crashed else None) for r in never}
    seeds = sorted(set(base) & set(sweep))
    flipped = [s for s in seeds if (base[s] is None) != (sweep[s] is None)]
    shifted = [s for s in seeds if base[s] is not None and sweep[s] is not None and abs(base[s] - sweep[s]) > tol_s]
    return {"seeds": len(seeds), "crash_replay": sum(base[s] is not None for s in seeds), "crash_sweep": sum(sweep[s] is not None for s in seeds),
            "flipped": [(s, base[s], sweep[s]) for s in flipped], "time_shifted": [(s, base[s], sweep[s]) for s in shifted]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", required=True)
    p.add_argument("--pairs", nargs="*", default=[], metavar="A:B")
    p.add_argument("--replay", help="replay_labels.jsonl flown on the same seeds")
    p.add_argument("--never", default="never")
    a = p.parse_args(argv)
    policies = by_policy(read_results(a.results))
    print(f"judge: {sorted({r.judge for rs in policies.values() for r in rs})}")
    print(f"{'policy':18s} flights crashes  by obstacle (crashes/flights)")
    for name, rs in policies.items():
        fam: dict[str, list[int]] = {}
        for r in rs:
            f = fam.setdefault(r.obstacle_family or "none", [0, 0])
            f[0] += int(r.crashed)
            f[1] += 1
        print(f"{name:18s} {len(rs):7d} {sum(r.crashed for r in rs):7d}  " + " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(fam.items())))
    for pair in a.pairs:
        x, _, y = pair.partition(":")
        d = compare(policies[x], policies[y])
        print(f"\n{x} minus {y} on {d['seeds']} seeds: crash {d['crash_a']:.3f} vs {d['crash_b']:.3f}, difference {d['d_crash']:+.3f} "
              f"[{d['d_crash_ci'][0]:+.3f}, {d['d_crash_ci'][1]:+.3f}]; only {x} crashed {d['only_a_crashed']}, only {y} {d['only_b_crashed']}, "
              f"McNemar p {d['mcnemar_p']:.2f}; calls/flight {d['calls_a']:.2f} vs {d['calls_b']:.2f}, difference {d['d_calls']:+.2f} "
              f"[{d['d_calls_ci'][0]:+.2f}, {d['d_calls_ci'][1]:+.2f}]")
    if a.replay:
        with open(a.replay) as f:
            r = repeatability(policies[a.never], [json.loads(line) for line in f if line.strip()])
        print(f"\nnever-ask repeatability on {r['seeds']} seeds: replay {r['crash_replay']} crashes, sweep {r['crash_sweep']}")
        print(f"  ended differently (seed, replay crash time, sweep crash time): {r['flipped']}")
        print(f"  crashed at different times: {len(r['time_shifted'])} {r['time_shifted'][:10]}")


if __name__ == "__main__":
    main()
