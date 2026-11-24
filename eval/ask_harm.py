"""Crashes on seeds that never-ask flies safely: simulator noise or harm from asking? CPU only, reads sweep outputs.

    python -m eval.ask_harm --sweep results/sweeps/replay_gates_surrogate --sweep results/sweeps/replay_gates_surrogate_extra

1. Noise check: a flight with no calls should end exactly like never-ask on its seed, so any zero-call flight that
   ends differently is simulator noise. Reported per policy kind; a short flight has fewer chances to ask, so
   zero-call flights lean towards early crashes and the rate is a rough guide, not an unbiased estimate.
2. Every crash after at least one call on a seed never-ask survives, with each ask and what the drone did next.
   A crash is ``asked_and_harmed`` only as a candidate: noise and harm cannot be told apart from one flight.

``always`` and ``oracle`` are references and are left out of the totals. Ground truth is printed for analysis only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from eval.crash_causes import read_flight
from eval.metrics import EpisodeResult, read_results
from interfaces import FrameRecord

REFERENCES = ("always", "oracle")


def split_flights(results: Sequence[EpisodeResult], never: str = "never") -> dict[str, Any]:
    base = {r.seed: r for r in results if r.policy == never}
    others = [r for r in results if r.policy != never and r.seed in base]
    zero = [r for r in others if r.n_calls == 0]
    flips = [r for r in zero if r.crashed != base[r.seed].crashed]
    new = [r for r in others if r.crashed and not base[r.seed].crashed]
    ranked = lambda rs: [r for r in rs if r.policy not in REFERENCES]  # noqa: E731
    safe_asked = [r for r in ranked(others) if r.n_calls > 0 and not base[r.seed].crashed]
    return {"base": base, "zero": zero, "flips": flips, "zero_safe": [r for r in zero if not base[r.seed].crashed],
            "zero_crash": [r for r in zero if base[r.seed].crashed], "candidates": [r for r in new if r.n_calls > 0],
            "asked_safe_n": len(safe_asked), "asked_safe_crashed": sum(r.crashed for r in safe_asked)}


def crash_table(results: Sequence[EpisodeResult]) -> list[str]:
    """Crashes by collision source and start pose, per policy and in total. A pose whose crashes are mostly "scenery" is suspect."""
    pose = lambda r: r.extra.get("start_index")  # noqa: E731
    poses = sorted({pose(r) for r in results}, key=str)
    out = ["policy             source    " + " ".join(f"pose {p}" for p in poses)]
    for name in dict.fromkeys(r.policy for r in results):
        rs = [r for r in results if r.policy == name]
        for src in sorted({str(r.extra.get("collision_source")) for r in rs if r.crashed}):
            hits = Counter(pose(r) for r in rs if r.crashed and str(r.extra.get("collision_source")) == src)
            out.append(f"{name:18s} {src:9s} " + " ".join(f"{hits.get(p, 0):6d}" for p in poses))
    flown = Counter(pose(r) for r in results if r.policy == results[0].policy)
    out.append(f"{'flights per policy':18s} {'':9s} " + " ".join(f"{flown.get(p, 0):6d}" for p in poses))
    fams = Counter((r.obstacle_family, str(r.extra.get("collision_source"))) for r in results if r.crashed)
    out += ["", "crashes by obstacle in the scenario and collision source (all policies):"]
    out += [f"  {f or 'none':26s} " + " ".join(f"{src}:{n}" for (g, src), n in sorted(fams.items(), key=str) if g == f) for f in sorted({k[0] for k in fams}, key=str)]
    return out


def guard_split(results: Sequence[EpisodeResult], where: dict[tuple[str, int], Path], near_m: float = 1.5) -> list[str]:
    """Split "scenery" crashes using the last logged frame: a placed obstacle within ``near_m`` ahead means the depth guard
    fired on that obstacle (the camera is 0.3 m ahead of the drone); anything else is the map or a bad capture."""
    placed, other = Counter(), []
    for r in results:
        if not r.crashed or r.extra.get("collision_source") != "scenery":
            continue
        log = where[(r.policy, r.seed)]
        recs = read_flight(log) if log.exists() else []  # empty when the guard fired on the first frame
        last = recs[-1] if recs else None
        if last is not None and last.obstacle_family and last.corridor_min_gt < near_m:
            placed[r.policy] += 1
        else:
            end = "no log" if last is None else f"x={last.pose[0]:.1f} alt={-last.pose[2]:.1f} {getattr(last.action_taken, 'name', last.action_taken)}"
            other.append(f"  {r.seed} {r.policy} pose {r.extra.get('start_index')} {r.obstacle_family} calls {r.n_calls} {end}")
    return [f"scenery crashes on the placed obstacle (guard fired first): {sum(placed.values())} {dict(placed)}",
            f"scenery crashes elsewhere (map or bad capture): {len(other)}", *sorted(other)]


def ask_lines(records: Sequence[FrameRecord]) -> list[str]:
    """One line per ask, then the last frames before the crash."""
    out = []
    for i, r in enumerate(records):
        if r.asked:
            after = Counter(str(getattr(x.action_taken, "name", x.action_taken)) for x in records[i:i + 20])
            out.append(f"    ask t={r.t:.1f} vlm={'yes' if r.vlm_answer else 'no'} score={r.gate_score} corridor gt={r.corridor_min_gt:.1f} est={r.corridor_min_est:.1f} "
                       f"next 2 s: {', '.join(f'{k} {v}' for k, v in after.most_common())}")
    for r in records[-3:]:
        out.append(f"    end t={r.t:.1f} action={getattr(r.action_taken, 'name', r.action_taken)} corridor gt={r.corridor_min_gt:.1f} est={r.corridor_min_est:.1f} "
                   f"x={r.pose[0]:.1f} alt={-r.pose[2]:.1f} yaw={r.pose[3]:.2f}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep", type=Path, action="append", required=True, help="sweep output dir (has results.jsonl); repeatable")
    p.add_argument("--never", default="never")
    a = p.parse_args(argv)
    results, where = [], {}
    for sweep in a.sweep:
        for r in read_results(sweep / "results.jsonl"):
            results.append(r)
            where[(r.policy, r.seed)] = sweep / "flights" / f"{r.episode_id}.jsonl"
    d = split_flights(results, a.never)
    if not d["base"]:
        raise SystemExit(f"no {a.never!r} flights in the given sweeps")
    print("\n".join(crash_table(results)) + "\n")
    print("\n".join(guard_split(results, where)) + "\n")
    fc = sum(not d["base"][r.seed].crashed for r in d["flips"])
    print(f"zero-call flights: {len(d['zero'])}; ended unlike {a.never}: {len(d['flips'])} "
          f"(safe seeds that crashed {fc} of {len(d['zero_safe'])}, crash seeds that survived {len(d['flips']) - fc} of {len(d['zero_crash'])})")
    print("  flips: " + ", ".join(f"{r.seed}/{r.policy}" for r in d["flips"]))
    print(f"flights with calls on seeds {a.never} survives (references left out): {d['asked_safe_n']}, crashed {d['asked_safe_crashed']}")
    for r in sorted(d["candidates"], key=lambda r: (r.seed, r.policy)):
        print(f"\n{r.seed} {r.policy}: {r.obstacle_family}, pose {r.extra.get('start_index')}, {r.n_calls} calls, crashed at {r.mission_time_s:.1f} s into {r.extra.get('collision_source')}")
        log = where[(r.policy, r.seed)]
        print("\n".join(ask_lines(read_flight(log))[-9:]) if log.exists() else "    no flight log")


if __name__ == "__main__":
    main()
