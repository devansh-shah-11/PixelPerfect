import json

import numpy as np
import pytest
from PIL import Image

from perception.analyze_bank import correct, eyeball_sheet, load_run, outcome_table, scale_check
from perception.run_bank import run_bank
from tests.test_frame_bank import BlindModel, write_bank


def row(episode, gt, est, split="train", obstacle="pole", along=10.0):
    return {
        "episode": episode, "stem": f"{episode:04d}_000", "split": split, "obstacle": obstacle,
        "condition": "clear", "corridor_min_gt": gt, "corridor_min_est": est, "gt_along_m": along,
    }


def test_outcome_table_counts_and_rates():
    rows = [row(0, 5, 50), row(0, 5, 4), row(0, 5, 6), row(1, 40, 3, obstacle="none"), row(1, 40, 40, obstacle="none")]
    table = outcome_table(rows, d_safe=8.0)
    assert table["pole"]["gt_obstacle"] == 3 and table["pole"]["missed"] == 1
    assert table["pole"]["miss_rate"] == pytest.approx(1 / 3)
    assert table["none"]["miss_rate"] is None
    assert table["none"]["false_stop_rate"] == pytest.approx(0.5)
    assert table["ALL"]["frames"] == 5


def test_scale_correction_changes_outcomes():
    rows = [row(0, 9, 7)]  # estimate too short: a false stop at d_safe 8
    assert outcome_table(rows, 8.0)["ALL"]["false_stop"] == 1
    assert outcome_table(rows, 8.0, scale=1.3)["ALL"]["false_stop"] == 0
    assert correct(np.array([90.0]), scale=2.0)[0] == 100.0


def test_scale_check_recovers_scale_from_train_only():
    rng = np.random.default_rng(0)
    est = rng.uniform(3, 25, 200)
    rows = [row(i, 1.2 * e, e) for i, e in enumerate(est)]
    rows += [row(1000 + i, 1.2 * e, e, split="val") for i, e in enumerate(est[:50])]
    rows += [row(2000, 90.0, 10.0)]  # beyond fit_max: ignored
    result = scale_check(rows)
    assert result["median_scale"] == pytest.approx(1.2)
    assert result["affine"]["scale"] == pytest.approx(1.2) and abs(result["affine"]["shift"]) < 1e-6
    assert result["val_abs_rel"]["median_scale"] == pytest.approx(0.0, abs=1e-9)
    assert result["val_abs_rel"]["raw"] == pytest.approx(1 / 6)


def test_load_run_drops_test_episodes_and_frames_past_the_obstacle(tmp_path):
    split = {"splits": {"train": [0], "val": [1], "test": [2]}}
    run = tmp_path / "run.jsonl"
    rows = [row(0, 5, 5), row(0, 5, 5, along=0.4), row(1, 5, 5, along=None), row(2, 5, 5), row(3, 5, 5)]
    run.write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = load_run(run, split)
    assert [(r["episode"], r["split"]) for r in loaded] == [(0, "train"), (1, "val")]


def test_eyeball_sheet_from_a_real_run(tmp_path):
    bank = write_bank(tmp_path / "bank", n_episodes=4, steps=3)
    run = tmp_path / "run.jsonl"
    run_bank(bank, run, BlindModel())
    split = {"splits": {"train": [0, 1], "val": [2], "test": [3]}}
    rows = load_run(run, split)
    out = tmp_path / "sheets" / "sheet.png"
    counts = eyeball_sheet(bank, rows, d_safe=8.0, out=out, n=2, cols=2)
    assert counts == {"positives": 2, "negatives": 2}  # episode 1 has the missed pole
    assert Image.open(out).size == (2 * 400, 2 * 225)
