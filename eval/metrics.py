"""Flight metrics and statistics (PLAN_DETAILED.md §5, chunk M3.1).

Input is one summary per flight (``EpisodeResult``), written by
``eval.run_episode``. Everything is computed per policy:

- crash rate per flight with a 95% Wilson interval, and crashes per km
- success rate (goal reached, no crash) with a Wilson interval
- mission time and average speed (a drone that never moves is "safe")
- VLM calls per flight and query rate (calls per frame)
- real VLM latency p50 / p95 / p99 (pooled over calls) and simulated delay charged per flight
- safety retained (the headline): ``(crash_never - crash_policy) / (crash_never - crash_reference)``,
  with a paired bootstrap over seeds, because every policy flies the same seeds (D15). The reference is the
  oracle by default: always-ask is "safe" partly by never finishing a mission (0 of 60 val flights reached the
  goal), so it is a poor ceiling. Pass ``reference="always"`` to get the old ratio. ``success`` (goal reached and
  no crash) is printed beside every crash rate so a policy cannot look safe by stalling

Surrogate-judge flights carry ``judge`` = "surrogate VLM" and are reported as such
(CLAUDE.md rule 7); ``results_table`` refuses to mix judges within one table.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class EpisodeResult:
    """One flight's summary. ``seed`` identifies the scenario; the same seed is flown by every policy."""

    episode_id: str
    seed: int
    split: str
    policy: str
    judge: str  # "qwen3-vl-2b", "surrogate VLM", "none", ...
    crashed: bool
    reached_goal: bool
    distance_m: float
    mission_time_s: float  # simulated seconds, delays included
    n_frames: int
    n_calls: int
    delay_charged_s: float
    latencies_s: list[float] = field(default_factory=list)  # real wall-clock per VLM call
    obstacle_family: str | None = None
    crash_cause: str | None = None  # filled by the crash-cause tool (M5.4)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeResult:
        return cls(**d)


def read_results(path: str | Path) -> list[EpisodeResult]:
    with open(path) as f:
        return [EpisodeResult.from_dict(json.loads(line)) for line in f if line.strip()]


# ---- basic statistics -------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion ``k / n``. NaN when ``n`` is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentiles(x: Sequence[float], qs: Iterable[float] = (50, 95, 99)) -> dict[str, float]:
    a = np.asarray(list(x), dtype=np.float64)
    return {f"p{int(q)}": (float(np.percentile(a, q)) if a.size else float("nan")) for q in qs}


def safety_retained(crash_never: float, crash_policy: float, crash_reference: float) -> float:
    """Share of the never-to-reference crash reduction a policy keeps. NaN if the reference is not safer than never-ask."""
    gain = crash_never - crash_reference
    if not gain > 0:
        return float("nan")
    return (crash_never - crash_policy) / gain


# ---- per-policy summary -----------------------------------------------------------------


def policy_summary(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"flights": 0}
    crashes = sum(r.crashed for r in results)
    success = sum(r.reached_goal and not r.crashed for r in results)
    km = sum(r.distance_m for r in results) / 1000.0
    frames = sum(r.n_frames for r in results)
    calls = sum(r.n_calls for r in results)
    time_s = np.array([r.mission_time_s for r in results], dtype=np.float64)
    dist = np.array([r.distance_m for r in results], dtype=np.float64)
    lat = [x for r in results for x in r.latencies_s]
    return {
        "flights": n,
        "judges": sorted({r.judge for r in results}),
        "crash_rate": crashes / n, "crash_rate_ci": wilson_ci(crashes, n),
        "crashes_per_km": crashes / km if km > 0 else float("nan"),
        "success_rate": success / n, "success_rate_ci": wilson_ci(success, n),
        "mission_time_s_mean": float(time_s.mean()), "mission_time_s_median": float(np.median(time_s)),
        "avg_speed_mps": float(dist.sum() / time_s.sum()) if time_s.sum() > 0 else float("nan"),
        "calls_per_flight": calls / n,
        "query_rate": calls / frames if frames else float("nan"),
        "delay_charged_s_per_flight": float(np.mean([r.delay_charged_s for r in results])),
        "latency_s": percentiles(lat),
    }


def by_policy(results: Iterable[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    out: dict[str, list[EpisodeResult]] = {}
    for r in results:
        out.setdefault(r.policy, []).append(r)
    return out


def paired_bootstrap_safety(
    policy: Sequence[EpisodeResult], never: Sequence[EpisodeResult], reference: Sequence[EpisodeResult],
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, float]:
    """Safety retained and its 95% interval, resampling the seeds that all three policies flew."""
    idx = lambda rs: {r.seed: r.crashed for r in rs}  # noqa: E731
    p, nv, al = idx(policy), idx(never), idx(reference)
    seeds = sorted(set(p) & set(nv) & set(al))
    if not seeds:
        return {"seeds": 0, "safety_retained": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    P = np.array([p[s] for s in seeds], float)
    N = np.array([nv[s] for s in seeds], float)
    A = np.array([al[s] for s in seeds], float)
    point = safety_retained(N.mean(), P.mean(), A.mean())
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    boot = np.array([safety_retained(N[i].mean(), P[i].mean(), A[i].mean()) for i in idxs])
    boot = boot[~np.isnan(boot)]
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot.size else (float("nan"), float("nan"))
    return {"seeds": len(seeds), "safety_retained": point, "ci_low": lo, "ci_high": hi}


def results_table(results: Sequence[EpisodeResult], never: str = "never", reference: str = "oracle") -> dict[str, dict[str, Any]]:
    """Per-policy metrics, plus safety retained between ``never`` and ``reference`` (oracle by default) when both were flown."""
    judges = {r.judge for r in results if r.n_calls}
    if len(judges) > 1:
        raise ValueError(f"results mix judges {sorted(judges)}; report surrogate and real-VLM results in separate tables (CLAUDE.md rule 7)")
    groups = by_policy(results)
    table = {name: policy_summary(rs) for name, rs in sorted(groups.items())}
    if never in groups and reference in groups:
        for name, rs in groups.items():
            table[name]["safety_retained"] = {**paired_bootstrap_safety(rs, groups[never], groups[reference]), "reference": reference}
    return table


def markdown(table: dict[str, dict[str, Any]]) -> str:
    def f(x: float, d: int = 2) -> str:
        return "n/a" if x != x else f"{x:.{d}f}"

    judge = sorted({j for t in table.values() for j in t.get("judges", [])} - {"none"})
    L = []
    if judge:
        L += [f"Judge: {', '.join(judge)}", ""]
    ref = next((t["safety_retained"]["reference"] for t in table.values() if t.get("safety_retained")), "reference")
    L += [f"| Policy | flights | crash rate [95% CI] | crashes/km | success | calls/flight | query rate | mission time (s) | safety retained, {ref} = 1 [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, t in table.items():
        sr = t.get("safety_retained")
        srs = f"{f(sr['safety_retained'])} [{f(sr['ci_low'])} to {f(sr['ci_high'])}]" if sr else "n/a"
        lo, hi = t["crash_rate_ci"]
        L.append(f"| {name} | {t['flights']} | {f(t['crash_rate'])} [{f(lo)} to {f(hi)}] | {f(t['crashes_per_km'])} | {f(t['success_rate'])} | "
                 f"{f(t['calls_per_flight'], 1)} | {f(t['query_rate'], 3)} | {f(t['mission_time_s_mean'], 1)} | {srs} |")
    return "\n".join(L)
