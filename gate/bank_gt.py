"""Clean ground-truth corridor distances for every frame of a bank (task W-B3).

Reads depth and segmentation only, so it needs no GPU. Output: one JSONL row
per frame, see ``gate.labels.clean_corridor_min``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate.labels import clean_corridor_min
from perception.frame_bank import frame_rows, load_depth_gt, load_obstacle_mask


def write_bank_gt(bank: str | Path, out_path: str | Path, limit: int | None = None) -> int:
    rows = frame_rows(bank)[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for k, row in enumerate(rows, 1):
            stem = row["stem"]
            gt = clean_corridor_min(load_depth_gt(bank, stem), load_obstacle_mask(bank, stem))
            f.write(json.dumps({"stem": stem, **gt}) + "\n")
            if k % 2000 == 0:
                print(f"{k}/{len(rows)} frames", flush=True)
    tmp.replace(out_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(f"{write_bank_gt(args.bank, args.out, args.limit)} frames -> {args.out}")


if __name__ == "__main__":
    main()
