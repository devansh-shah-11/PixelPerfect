"""Crash-cause tool (PLAN_DETAILED.md §5 "Crash-cause breakdown", chunk M5.4). CPU only, reads sweep outputs.

Every crashed flight gets exactly one cause:

1. ``controller``: the same seed also crashes with the true-depth reference (never-ask on true depth). The
   perception and the VLM are not the problem; the rule controller or the scenario is.
2. ``gate_did_not_ask``: no VLM call in the last ``window_s`` simulated seconds before the crash.
3. ``vlm_wrong``: the policy asked in that window and every answer was "no".
4. ``delay``: the policy asked and the VLM said "yes", but the drone crashed anyway (right answer, too late or
   too weak an avoidance; the two are not separated here).
5. ``unknown``: no flight log for this flight (set ``flight_logs: true`` in the sweep config).

Rules are checked in that order. ``never`` crashes are all ``gate_did_not_ask`` by construction.

    python -m eval.crash_causes --sweep results/sweeps/mvp_gate [--sweep ...] \
        --reference results/sweeps/true_depth/results.jsonl [--window-s 3.0] [--examples 30]

Writes ``<sweep>/crash_causes.jsonl`` (one row per crash) and ``<sweep>/crash_causes.md`` (table by policy and
cause, then example crashes). ``results.jsonl`` is never modified. Ground truth is used here for the reference
lookup only, which is analysis, not runtime (CLAUDE.md rule 5).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from eval.metrics import EpisodeResult, read_results
from interfaces import FrameRecord

CAUSES = ("gate_did_not_ask", "vlm_wrong", "delay", "controller", "unknown")


def read_flight(path: Path) -> list[FrameRecord]:
    with open(path) as f:
        return [FrameRecord.from_json(line) for line in f if line.strip()]


def classify(records: Sequence[FrameRecord] | None, reference_crashed: bool | None, window_s: float = 3.0) -> tuple[str, dict]:
    """(cause, detail) for one crashed flight. ``reference_crashed`` is None when the seed is not in the reference."""
    if reference_crashed:
        return "controller", {"reference_crashed": True}
    if not records:
        return "unknown", {"reason": "no flight log"}
    t_crash = records[-1].t
    asked = [r for r in records if r.asked and r.t >= t_crash - window_s]
    detail = {"t_crash": t_crash, "asks_in_window": len(asked), "reference_checked": reference_crashed is not None}
    if not asked:
        return "gate_did_not_ask", detail
    detail["last_ask_t"] = asked[-1].t
    detail["last_answer"] = asked[-1].vlm_answer
    if any(r.vlm_answer for r in asked):
        return "delay", detail
    return "vlm_wrong", detail


def reference_crashes(path: Path | None, policy: str = "never") -> dict[tuple[str, int], bool]:
    """``(split, seed) -> crashed`` from a true-depth sweep's results (rows of ``policy`` only)."""
    if path is None:
        return {}
    return {(r.split, r.seed): r.crashed for r in read_results(path) if r.policy == policy}


def analyse(sweep: Path, reference: dict[tuple[str, int], bool], window_s: float = 3.0) -> list[dict]:
    rows = []
    for r in read_results(sweep / "results.jsonl"):
        if not r.crashed:
            continue
        log = sweep / "flights" / f"{r.episode_id}.jsonl"
        cause, detail = classify(read_flight(log) if log.exists() else None, reference.get((r.split, r.seed)), window_s)
        rows.append({"episode_id": r.episode_id, "policy": r.policy, "judge": r.judge, "seed": r.seed, "split": r.split,
                     "cause": cause, "obstacle_family": r.obstacle_family,
                     "collision_source": r.extra.get("collision_source"), **detail})
    return rows


def markdown(rows: Sequence[dict], n_flights: dict[str, int], examples: int = 30) -> str:
    label = lambda r: r["policy"] if r["judge"] in ("none", None) else f"{r['policy']} ({r['judge']})"  # noqa: E731
    by: dict[str, Counter] = {}
    for r in rows:
        by.setdefault(label(r), Counter())[r["cause"]] += 1
    out = ["| policy | flights | crashes | " + " | ".join(CAUSES) + " |", "|---|---:|---:|" + "---:|" * len(CAUSES)]
    for name in sorted(set(by) | set(n_flights)):
        c = by.get(name, Counter())
        out.append(f"| {name} | {n_flights.get(name, '?')} | {sum(c.values())} | " + " | ".join(str(c.get(k, 0)) for k in CAUSES) + " |")
    out += ["", f"### Example crashes (first {min(examples, len(rows))})", "",
            "| episode | cause | obstacle | collision source | asks in window | last answer |", "|---|---|---|---|---:|---|"]
    for r in rows[:examples]:
        out.append(f"| {r['episode_id']} | {r['cause']} | {r['obstacle_family']} | {r['collision_source']} | "
                   f"{r.get('asks_in_window', '')} | {r.get('last_answer', '')} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", type=Path, action="append", required=True, help="sweep output dir (has results.jsonl); repeatable")
    p.add_argument("--reference", type=Path, help="results.jsonl of the true-depth never-ask sweep (same seeds)")
    p.add_argument("--reference-policy", default="never")
    p.add_argument("--window-s", type=float, default=3.0, help="simulated seconds before the crash that count as 'the decision'")
    p.add_argument("--examples", type=int, default=30)
    a = p.parse_args(argv)
    ref = reference_crashes(a.reference, a.reference_policy)
    for sweep in a.sweep:
        rows = analyse(sweep, ref, a.window_s)
        flights: Counter = Counter()
        for r in read_results(sweep / "results.jsonl"):
            flights[r.policy if r.judge in ("none", None) else f"{r.policy} ({r.judge})"] += 1
        (sweep / "crash_causes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        text = markdown(rows, dict(flights), a.examples)
        (sweep / "crash_causes.md").write_text(text)
        print(f"== {sweep}\n{text}")
        if rows and not ref:
            print("note: no --reference given, so 'controller' was not checked (all crashes are attributed to perception or the VLM)")
        if rows and all(r["cause"] == "unknown" for r in rows):
            print("note: no flight logs found; rerun the sweep with flight_logs: true")


if __name__ == "__main__":
    main()
