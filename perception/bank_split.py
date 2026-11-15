"""Train/val/test split of the frame bank by episode (task W-S1, PLAN_DETAILED.md D13).

Episodes, never frames, are assigned to splits, so frames of one flight cannot
leak across splits (CLAUDE.md rule 4). Episodes are stratified by obstacle type
and condition.

    python -m perception.bank_split --bank /scratch/$USER/frame_bank/CityEnviron --out configs/bank_split.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from perception.frame_bank import episode_rows

SPLITS = ("train", "val", "test")
# hpc/instructions.md §3.5: fog renders near-black and snow flakes land in the
# depth image, so both are left out; rain looks the same as clear and is kept.
KEPT_CONDITIONS = ("clear", "dusk", "rain")
# 70 / 15 / 15, interleaved so that small strata still reach val and test.
_PATTERN = ("train", "train", "train", "val", "train", "train", "test", "train", "train", "train",
            "val", "train", "train", "test", "train", "train", "val", "train", "train", "test")


def split_checksum(splits: dict[str, list[int]]) -> str:
    canonical = json.dumps({name: sorted(splits[name]) for name in SPLITS}, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_split(
    episodes: list[dict[str, Any]],
    kept_conditions: tuple[str, ...] = KEPT_CONDITIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Assign each kept episode to train, val or test. Same input and seed give the same split."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    excluded: list[int] = []
    for row in episodes:
        if row["condition"] in kept_conditions:
            strata.setdefault((row["obstacle"], row["condition"]), []).append(int(row["episode"]))
        else:
            excluded.append(int(row["episode"]))

    splits: dict[str, list[int]] = {name: [] for name in SPLITS}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        offset = int(rng.integers(len(_PATTERN)))
        for j, episode in enumerate(ids):
            splits[_PATTERN[(j + offset) % len(_PATTERN)]].append(episode)
    for name in SPLITS:
        splits[name].sort()

    return {
        "seed": seed,
        "kept_conditions": list(kept_conditions),
        "n_episodes": {name: len(splits[name]) for name in SPLITS} | {"excluded": len(excluded)},
        "checksum": split_checksum(splits),
        "splits": splits,
        "excluded": sorted(excluded),
    }


def check_split(split: dict[str, Any]) -> None:
    """Raise if an episode is in two splits or the checksum does not match the lists."""
    seen: dict[int, str] = {}
    for name in SPLITS:
        for episode in split["splits"][name]:
            if episode in seen:
                raise ValueError(f"episode {episode} is in both {seen[episode]} and {name}")
            seen[episode] = name
    for episode in split.get("excluded", []):
        if episode in seen:
            raise ValueError(f"episode {episode} is excluded but also in {seen[episode]}")
    if split_checksum(split["splits"]) != split["checksum"]:
        raise ValueError("split lists do not match the stored checksum")


def split_of_episode(split: dict[str, Any]) -> dict[int, str]:
    """Map episode id to split name; excluded episodes are absent."""
    return {episode: name for name in SPLITS for episode in split["splits"][name]}


def load_split(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        split = yaml.safe_load(f)
    check_split(split)
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bank", required=True, help="frame-bank directory holding episodes.jsonl")
    parser.add_argument("--out", required=True, help="YAML file to write, e.g. configs/bank_split.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", nargs="+", default=list(KEPT_CONDITIONS), help="conditions to keep")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists. The split is frozen once written (PLAN_DETAILED.md D13); delete it by hand to redo it.")
    episodes = episode_rows(args.bank)
    split = make_split(episodes, tuple(args.keep), args.seed)
    check_split(split)
    split = {"bank": Path(args.bank).name, **split}
    out.write_text(yaml.safe_dump(split, sort_keys=False, default_flow_style=None, width=100))
    print(f"wrote {out}: {split['n_episodes']} checksum {split['checksum'][:12]}")


if __name__ == "__main__":
    main()
