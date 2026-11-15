"""Tests for vlm/eval_bank.py on a tiny fake bank with a fake judge. No model, no GPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from perception.bank_split import make_split
from perception.frame_bank import OBSTACLE_SEG_RGB
from vlm import eval_bank
from vlm.eval_bank import ExampleSheet, bucket, evaluate, frame_label, markdown_tables, prf, select_frames, summarize
from vlm.overlay import corridor_box

H, W = 225, 400


class ConfidentJudge:
    """Says yes iff the overlay's corridor box contains dark pixels (how a good judge should behave on this bank)."""

    name = "fake"
    use_overlay = True
    use_crop = True

    def ask(self, image, path_overlay, crop):
        from interfaces import VLMAnswer

        r0, r1, c0, c1 = corridor_box(image.shape)
        dark = (image[r0:r1, c0:c1].mean(axis=-1) < 60).any()
        return VLMAnswer(answer=bool(dark), confidence=2.0 if dark else -2.0, latency_s=0.01)


def make_bank(root: Path, episodes: dict[int, tuple[str, str, bool, float]]) -> Path:
    """episodes: id -> (obstacle, condition, obstacle_in_corridor, distance_m). One frame per episode."""
    bank = root / "bank"
    (bank / "frames").mkdir(parents=True)
    r0, r1, c0, c1 = corridor_box((H, W, 3))
    ep_rows, fr_rows = [], []
    for ep, (obstacle, cond, in_corridor, dist) in episodes.items():
        stem = f"{ep:04d}_000"
        rgb = np.full((H, W, 3), 200, np.uint8)
        seg = np.zeros((H, W, 3), np.uint8)
        depth = np.full((H, W), 80.0, np.float32)  # far scene everywhere
        if in_corridor:
            rr, cc = slice(r0 + 5, r0 + 15), slice(c0 + 5, c0 + 15)
            rgb[rr, cc] = 20
            seg[rr, cc] = OBSTACLE_SEG_RGB
            depth[rr, cc] = dist
        Image.fromarray(rgb).save(bank / "frames" / f"{stem}_rgb.png")
        Image.fromarray(seg).save(bank / "frames" / f"{stem}_seg.png")
        np.save(bank / "frames" / f"{stem}_depth.npy", depth.astype(np.float16))
        ep_rows.append({"episode": ep, "condition": cond, "obstacle": obstacle})
        fr_rows.append({"episode": ep, "step": 0, "stem": stem, "condition": cond, "obstacle": obstacle, "gt_along_m": dist if in_corridor else None})
    (bank / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_rows))
    (bank / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fr_rows))
    return bank


@pytest.fixture
def bank(tmp_path):
    episodes = {}
    k = 0
    for obstacle in ("pole", "box"):
        for in_corr, dist in ((True, 5.0), (True, 12.0), (False, 80.0), (True, 22.0)):
            for _ in range(3):
                episodes[k] = (obstacle, "clear", in_corr, dist)
                k += 1
    bank = make_bank(tmp_path, episodes)
    split = make_split([{"episode": e, "condition": v[1], "obstacle": v[0]} for e, v in episodes.items()], seed=0)
    split_path = tmp_path / "bank_split.yaml"
    split_path.write_text(yaml.safe_dump(split))
    return bank, split_path, episodes


def test_bucket_and_label():
    assert bucket(2.9) == "0-3 m" and bucket(6.0) == "6-10 m" and bucket(25) == ">20 m"
    assert frame_label({"corridor_min_gt_clean": 5.0}, 15, 30) == "pos"
    assert frame_label({"corridor_min_gt_clean": 50.0}, 15, 30) == "neg"
    assert frame_label({"corridor_min_gt_clean": 20.0}, 15, 30) == "ambiguous"


def test_select_frames_respects_split_and_cap(bank):
    b, split_path, episodes = bank
    rows = select_frames(b, split_path, ("val",), None, 0)
    of = {e for e in yaml.safe_load(split_path.read_text())["splits"]["val"]}
    assert rows and {int(r["episode"]) for r in rows} <= of
    capped = select_frames(b, split_path, ("train", "val"), 2, 0)
    per_type = {}
    for r in capped:
        per_type[r["obstacle"]] = per_type.get(r["obstacle"], 0) + 1
    assert all(v <= 2 for v in per_type.values())


def test_evaluate_is_resumable_and_labels_from_ground_truth(bank, tmp_path):
    b, split_path, _ = bank
    frames = select_frames(b, split_path, ("train", "val"), None, 0)
    out = tmp_path / "rows.jsonl"
    judge = ConfidentJudge()
    rows = evaluate({"fake": judge}, b, frames, out, d_pos=15, d_neg=30, conditions=("both",), examples_dir=tmp_path / "ex", n_examples=2)
    assert len(rows) == len(frames)
    labels = {r["label"] for r in rows}
    assert labels == {"pos", "neg", "ambiguous"}
    for r in rows:  # the fake judge is perfect on unambiguous frames
        if r["label"] != "ambiguous":
            assert r["answer"] == (r["label"] == "pos")
    # second call does nothing new
    again = evaluate({"fake": judge}, b, frames, out, d_pos=15, d_neg=30, conditions=("both",))
    assert len(again) == len(frames)
    assert sum(1 for _ in open(out)) == len(frames)
    sheets = list((tmp_path / "ex" / "fake_both").glob("*.png"))
    assert {p.stem for p in sheets} == {"pole", "box"}
    assert Image.open(sheets[0]).size[1] > 225  # several tiles stacked


def test_summary_tables():
    rows = [
        {"model": "m", "condition": "both", "obstacle": "pole", "bucket": "3-6 m", "label": "pos", "answer": True},
        {"model": "m", "condition": "both", "obstacle": "pole", "bucket": "3-6 m", "label": "pos", "answer": False},
        {"model": "m", "condition": "both", "obstacle": "pole", "bucket": ">20 m", "label": "neg", "answer": True},
        {"model": "m", "condition": "both", "obstacle": "pole", "bucket": ">20 m", "label": "neg", "answer": False},
        {"model": "m", "condition": "both", "obstacle": "pole", "bucket": "10-20 m", "label": "ambiguous", "answer": False},
    ]
    s = summarize(rows)["m / both"]
    o = s["overall"]
    assert (o["n_pos"], o["n_neg"], o["n_ambiguous"]) == (2, 2, 1)
    assert o["recall"] == 0.5 and o["precision"] == 0.5 and o["fpr"] == 0.5
    assert list(s["by_bucket"]) == ["3-6 m", "10-20 m", ">20 m"]  # sorted by distance; the ambiguous bucket shows with n/a
    assert s["by_bucket"]["10-20 m"]["n_pos"] == 0 and s["by_bucket"]["10-20 m"]["n_ambiguous"] == 1
    text = markdown_tables(summarize(rows), 15, 30)
    assert "| m / both | 2 | 2 | 0.50 | 0.50 | 0.50 |" in text and "pole" in text
    assert prf([])["recall"] != prf([])["recall"]  # nan when empty


def test_example_sheet_caps_per_type(tmp_path):
    sheet = ExampleSheet(tmp_path / "s", n=1)
    ov = np.zeros((H, W, 3), np.uint8)
    cr = np.zeros((100, 80, 3), np.uint8)
    base = {"stem": "x", "obstacle": "pole", "gt_clean_m": 3.0, "answer": True, "confidence": 1.0}
    sheet.maybe_add({**base, "label": "pos"}, ov, cr)
    sheet.maybe_add({**base, "label": "pos"}, ov, cr)  # over the cap
    sheet.maybe_add({**base, "label": "neg"}, ov, cr)
    sheet.maybe_add({**base, "label": "ambiguous"}, ov, cr)
    assert len(sheet.tiles["pole"]) == 2
    assert sheet.save()[0].name == "pole.png"


def test_refuses_test_split(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["eval_bank", "--bank", "x", "--splits", "test"])
    with pytest.raises(SystemExit):
        eval_bank.main()
