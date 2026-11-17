"""Tests for vlm/eval_bank.py on a tiny fake bank with a fake judge. No model, no GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from interfaces import VLMAnswer
from perception.bank_split import make_split
from perception.frame_bank import OBSTACLE_SEG_RGB
from vlm import eval_bank
from vlm.eval_bank import (
    CONDITIONS, ExampleSheet, agreement, auroc, bucket, evaluate, evaluate_models, flight_ci, frame_label, gate_verdict, labelled,
    metadata_in_window, report, scores, select_frames, summarize, tune_tau, window_half_width,
)
from vlm.judge import PROMPTS, prompt_text
from vlm.overlay import corridor_box

H, W = 225, 400


class DarkPixelJudge:
    """Says yes iff the corridor box of the raw frame has dark pixels. Records the prompts it was given."""

    def __init__(self):
        self.use_overlay, self.use_crop, self.prompt = True, True, ""
        self.prompts: list[str] = []

    def ask(self, image, path_overlay, crop):
        self.prompts.append(self.prompt)
        r0, r1, c0, c1 = corridor_box(image.shape)
        dark = bool((image[r0:r1, c0:c1].mean(axis=-1) < 60).any())
        return VLMAnswer(answer=dark, confidence=2.0 if dark else -2.0, latency_s=0.01)


def make_bank(root: Path, episodes: dict[int, dict]) -> Path:
    """One flight per episode, ``steps`` frames each. Spec keys: obstacle, condition, in_corr, dist, seg (bool), steps."""
    bank = root / "bank"
    (bank / "frames").mkdir(parents=True)
    r0, r1, c0, c1 = corridor_box((H, W, 3))
    ep_rows, fr_rows = [], []
    for ep, s in episodes.items():
        for step in range(s.get("steps", 1)):
            stem = f"{ep:04d}_{step:03d}"
            rgb = np.full((H, W, 3), 200, np.uint8)
            seg = np.zeros((H, W, 3), np.uint8)
            depth = np.full((H, W), 80.0, np.float32)
            if s["in_corr"]:
                rr, cc = slice(r0 + 5, r0 + 15), slice(c0 + 5, c0 + 15)
                rgb[rr, cc] = 20
                if s.get("seg", True):
                    seg[rr, cc] = OBSTACLE_SEG_RGB
                    depth[rr, cc] = s["dist"]
            Image.fromarray(rgb).save(bank / "frames" / f"{stem}_rgb.png")
            Image.fromarray(seg).save(bank / "frames" / f"{stem}_seg.png")
            np.save(bank / "frames" / f"{stem}_depth.npy", depth.astype(np.float16))
            row = {"episode": ep, "step": step, "stem": stem, "condition": s["condition"], "obstacle": s["obstacle"]}
            if s["obstacle"] != "none":
                row.update(gt_along_m=s["dist"], gt_lateral_m=0.0 if s["in_corr"] else 20.0)
            fr_rows.append(row)
        ep_rows.append({"episode": ep, "condition": s["condition"], "obstacle": s["obstacle"]})
    (bank / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_rows))
    (bank / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fr_rows))
    return bank


@pytest.fixture
def bank(tmp_path):
    episodes, k = {}, 0
    for obstacle in ("pole_10cm_axis_z", "box_1m_grey"):
        for in_corr, dist in ((True, 5.0), (True, 12.0), (False, 80.0), (True, 22.0)):
            for _ in range(3):
                episodes[k] = {"obstacle": obstacle, "condition": "clear", "in_corr": in_corr, "dist": dist, "steps": 3}
                k += 1
    for _ in range(6):  # glass: visible in RGB, invisible to depth and segmentation
        episodes[k] = {"obstacle": "glass_pane_alpha20", "condition": "clear", "in_corr": True, "dist": 6.0, "seg": False, "steps": 3}
        k += 1
    bank = make_bank(tmp_path, episodes)
    split = make_split([{"episode": e, "condition": v["condition"], "obstacle": v["obstacle"]} for e, v in episodes.items()], seed=0)
    split_path = tmp_path / "bank_split.yaml"
    split_path.write_text(yaml.safe_dump(split))
    return bank, split_path, split


# ---- prompts -----------------------------------------------------------------------------


def test_prompt_wording_follows_overlay():
    assert "highlighted region" in prompt_text("cross", True)
    assert "highlighted" not in prompt_text("cross", False) and "centre of the image" in prompt_text("cross", False)
    assert prompt_text("cross", True) == "Does any object, including thin wires, cables, poles or glass, cross the highlighted region? Answer yes or no."
    assert set(PROMPTS) == {c[2] for c in CONDITIONS.values()}
    assert all(p.endswith("Answer yes or no.") for p in PROMPTS.values())


# ---- geometry and labels ---------------------------------------------------------------


def test_window_geometry_and_metadata():
    assert window_half_width(10.0) == pytest.approx(1.0)  # 10% of the width at 90 degrees: 0.1 x 10 m
    assert metadata_in_window("glass_pane_alpha20", 10.0, 2.0) is True  # 2.0 < 1.25 + 1.0
    assert metadata_in_window("glass_pane_alpha20", 10.0, 2.4) is False
    assert metadata_in_window("glass_pane_alpha20", None, 0.0) is None
    assert metadata_in_window("unknown_type", 5.0, 0.0) is None


def test_bucket_and_label():
    assert bucket(2.9) == "0-3 m" and bucket(6.0) == "6-10 m" and bucket(25) == ">20 m"
    assert frame_label(5.0, 15, 30) == "pos" and frame_label(50.0, 15, 30) == "neg" and frame_label(20.0, 15, 30) == "ambiguous"


def test_select_frames_split_stride_and_no_test(bank):
    b, split_path, split = bank
    rows = select_frames(b, split_path, ("train", "val"), stride=1)
    allowed = set(split["splits"]["train"]) | set(split["splits"]["val"])
    assert rows and {int(r["episode"]) for r in rows} <= allowed
    assert {r["split"] for r in rows} <= {"train", "val"}
    strided = select_frames(b, split_path, ("train", "val"), stride=2)
    assert all(r["step"] % 2 == 0 for r in strided) and len(strided) < len(rows)
    with pytest.raises(ValueError):
        select_frames(b, split_path, ("test",))


def test_glass_label_comes_from_metadata(bank):
    b, split_path, _ = bank
    frames = [f for f in select_frames(b, split_path, ("train", "val")) if f["obstacle"] == "glass_pane_alpha20"]
    assert frames
    gt = eval_bank.ground_truth(b, frames[0])
    assert gt["obstacle_px"] == 0  # segmentation sees nothing
    assert gt["gt_source"] == "metadata" and gt["corridor_min_gt_clean"] == pytest.approx(6.0)
    pole = [f for f in select_frames(b, split_path, ("train", "val")) if f["obstacle"] == "pole_10cm_axis_z" and f["gt_along_m"] == 5.0][0]
    gt_pole = eval_bank.ground_truth(b, pole)
    assert gt_pole["gt_source"] == "seg" and gt_pole["corridor_min_gt_clean"] == pytest.approx(5.0)


# ---- evaluation loop ---------------------------------------------------------------------


def test_evaluate_frame_major_resumable_with_prompts(bank, tmp_path):
    b, split_path, _ = bank
    frames = select_frames(b, split_path, ("train", "val"))
    out = tmp_path / "rows.jsonl"
    judge = DarkPixelJudge()
    conds = ("both", "raw", "both_blocked")
    rows = evaluate({"fake": judge}, b, frames, out, conds, examples_dir=tmp_path / "ex", n_examples=2)
    assert len(rows) == len(frames) * len(conds)
    assert set(judge.prompts) == {prompt_text("cross", True), prompt_text("cross", False), prompt_text("blocked", True)}
    lab = labelled(rows, 15, 30)
    assert {r["label"] for r in lab} == {"pos", "neg", "ambiguous"}
    for r in lab:  # the fake judge sees every dark patch, glass included, so it is right on every clear-cut frame
        if r["label"] != "ambiguous":
            assert r["answer"] == (r["label"] == "pos"), r
    glass = [r for r in lab if r["obstacle"] == "glass_pane_alpha20"]
    assert glass and all(r["label"] == "pos" and r["gt_source"] == "metadata" for r in glass)
    again = evaluate({"fake": judge}, b, frames, out, conds)  # nothing left to do
    assert len(again) == len(rows) and sum(1 for _ in open(out)) == len(rows)
    sheets = {p.parent.name for p in (tmp_path / "ex").rglob("*.png")}
    assert sheets == {"fake_both", "fake_raw", "fake_both_blocked"}


def test_example_sheet_caps_per_type_and_per_flight(tmp_path):
    sheet = ExampleSheet(tmp_path / "s", n=3, per_flight=1)
    ov, cr = np.zeros((H, W, 3), np.uint8), np.zeros((100, 80, 3), np.uint8)
    base = {"stem": "x", "obstacle": "pole", "gt_clean_m": 3.0, "answer": True, "confidence": 1.0}
    sheet.maybe_add({**base, "episode": 1}, ov, cr)
    sheet.maybe_add({**base, "episode": 1}, ov, cr)  # same flight: skipped
    sheet.maybe_add({**base, "episode": 2}, ov, cr)
    sheet.maybe_add({**base, "episode": 3, "gt_clean_m": 20.0}, ov, cr)  # ambiguous: skipped
    assert len(sheet.tiles["pole"]) == 2
    assert sheet.save()[0].name == "pole.png"


# ---- scoring -----------------------------------------------------------------------------


def test_auroc():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert math.isnan(auroc([], [1]))
    rng = np.random.default_rng(0)
    p, n = rng.normal(1, 1, 400), rng.normal(0, 1, 400)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n])
    assert auroc(p, n) == pytest.approx(brute)


def _row(ep, label_dist, conf, split="val", obstacle="pole", model="m", cond="both"):
    return {"model": model, "condition": cond, "episode": ep, "stem": f"{ep}_{conf}", "split": split, "obstacle": obstacle,
            "gt_clean_m": label_dist, "confidence": conf, "answer": conf > 0}


def test_flight_ci_resamples_flights_not_frames():
    lab = labelled([_row(1, 5, 1.0), _row(1, 5, 1.0), _row(2, 5, -1.0), _row(2, 5, -1.0)], 15, 30)
    lo, hi = flight_ci([r for r in lab if r["label"] == "pos"])
    assert lo == 0.0 and hi == 1.0  # two flights, one all-hit and one all-miss: anything is possible
    assert all(math.isnan(x) for x in flight_ci(lab[:2]))  # one flight: no interval


def test_tune_tau_controls_val_fpr():
    negs = labelled([_row(i, 80, c) for i, c in enumerate(np.linspace(-3, 1, 20))], 15, 30)
    tau = tune_tau(negs, max_fpr=0.10)
    assert sum(r["confidence"] > tau for r in negs) <= 2
    assert math.isnan(tune_tau([], 0.1))


def test_summary_gate_and_report():
    rows = []
    for ep in range(12):  # 12 visible flights, recall 2/3 at cut 0
        rows += [_row(ep, 5, 1.0), _row(ep, 5, 0.5), _row(ep, 5, -0.5), _row(100 + ep, 80, -1.0)]
    rows += [_row(200, 5, 1.0, obstacle="wire_1cm_axis_z"), _row(300, 20, 0.2)]  # wire: not in the gate; one ambiguous
    rows += [_row(400 + ep, 5, 0.8, split="train") for ep in range(3)] + [_row(500, 80, -2.0, split="train")]
    s = summarize(rows, 15, 30)["m / both"]
    assert s["visible"]["n_pos"] == 39 and s["visible"]["pos_flights"] == 15
    assert s["overall"]["n_ambiguous"] == 1
    assert s["gate"].startswith("PASS at cut 0")
    assert s["train_at_tau"] is not None and s["auroc"] > 0.5
    thin = summarize([_row(1, 5, -1.0), _row(2, 80, -2.0)], 15, 30)["m / both"]
    assert thin["gate"].startswith("FAIL") and "THIN" in thin["gate"]
    assert gate_verdict({"visible": scores([])}).startswith("NO DATA")
    text = report(summarize(rows, 15, 30), {}, 15, 30)
    assert "M2 gate" in text and "| m / both |" in text and "by obstacle type" in text


def test_agreement_flags_see_through_types():
    rows = [
        {"stem": "a", "obstacle": "box", "meta_in_window": True, "gt_along_m": 5.0, "obstacle_px": 30, "gt_obstacle_m": 5.0, "gt_source": "seg"},
        {"stem": "b", "obstacle": "glass", "meta_in_window": True, "gt_along_m": 5.0, "obstacle_px": 0, "gt_obstacle_m": 5.0, "gt_source": "metadata"},
        {"stem": "c", "obstacle": "box", "meta_in_window": None, "obstacle_px": 0, "gt_obstacle_m": 100.0},
        {"stem": "c", "obstacle": "box", "meta_in_window": None, "obstacle_px": 0, "gt_obstacle_m": 100.0},  # duplicate stem: counted once
    ]
    a = agreement(rows, 15)
    assert a["box"]["both"] == 1 and a["box"]["no_meta"] == 1 and a["glass"]["meta_only"] == 1


def test_old_rows_without_split_count_as_val():
    old = [{"model": "m", "condition": "both", "episode": 1, "stem": "s", "obstacle": "pole", "gt_clean_m": 5.0, "confidence": 1.0, "answer": True}]
    assert labelled(old, 15, 30)[0]["split"] == "val"


def test_cli_refuses_test_split_and_summarizes_saved_rows(tmp_path, capsys):
    with pytest.raises(SystemExit):
        eval_bank.main(["--bank", "x", "--splits", "test"])
    out = tmp_path / "rows.jsonl"
    rows = [_row(ep, 5, 1.0) for ep in range(3)] + [_row(10 + ep, 80, -1.0) for ep in range(3)]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    eval_bank.main(["--summarize-only", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "M2 gate" in printed and out.with_suffix(".report.md").exists()


def test_gate_rejects_a_yes_machine():
    rows = []
    for ep in range(12):  # says yes to everything: recall 1.0, FPR 1.0
        rows += [_row(ep, 5, 1.0), _row(100 + ep, 80, 1.0)]
    s = summarize(rows, 15, 30)["m / both"]
    assert s["gate"].startswith("FAIL") and "says yes too often" in s["gate"]


def test_recall_on_fast_stack_misses(tmp_path, capsys):
    from vlm.eval_bank import fast_miss_report, fast_miss_scores

    rows = [_row(1, 5, 1.0), _row(2, 5, -1.0), _row(3, 5, 1.0), _row(10, 80, -1.0), _row(11, 80, -1.0)]
    for r in rows:
        r["stem"] = f"s{r['episode']}"
    est = {"s1": 30.0, "s2": 30.0, "s3": 4.0, "s10": 90.0, "s11": 90.0}  # s3: the fast stack already sees it, not a miss
    summ = summarize(rows, 15, 30)
    fm = fast_miss_scores(rows, est, summ, d_safe=8.0, d_pos=15, d_neg=30)["m / both"]
    assert fm["n"] == 2 and fm["recall"] == 0.5 and fm["auroc"] == pytest.approx(0.75)
    assert "fast stack's misses" in fast_miss_report({"m / both": fm}, 8.0)
    out = tmp_path / "rows.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "run.jsonl"
    run.write_text("".join(json.dumps({"stem": k, "corridor_min_est": v}) + "\n" for k, v in est.items()))
    eval_bank.main(["--summarize-only", "--out", str(out), "--run", str(run)])
    assert "fast stack's misses" in out.with_suffix(".report.md").read_text()


# ---- second prompt round and one model at a time (D22, D23) --------------------------------


class SignJudge(DarkPixelJudge):
    """DarkPixelJudge that honours ``invert`` the way QwenVLJudge does, answering the question it was asked."""

    def __init__(self):
        super().__init__()
        self.invert, self._model, self._processor = False, object(), object()

    def ask(self, image, path_overlay, crop):
        a = super().ask(image, path_overlay, crop)
        raw = -a.confidence if "free of obstacles" in self.prompt else a.confidence  # the model's own yes/no to this wording
        conf = -raw if self.invert else raw
        return VLMAnswer(answer=conf > 0, confidence=conf, latency_s=a.latency_s)


def test_inverted_prompt_condition_still_means_obstacle(bank, tmp_path):
    b, split_path, _ = bank
    frames = select_frames(b, split_path, ("train", "val"))
    assert CONDITIONS["both_free"] == (True, True, "free") and CONDITIONS["both_fill"] == (True, True, "fill")
    judge = SignJudge()
    rows = evaluate({"fake": judge}, b, frames, tmp_path / "rows.jsonl", ("both", "both_free", "both_fill"))
    assert prompt_text("free", True) in judge.prompts and prompt_text("fill", True) in judge.prompts
    by = {(r["condition"], r["stem"]): r for r in rows}
    for r in rows:  # "is it free?" answered yes means clear: after the flip all three conditions agree frame by frame
        assert r["confidence"] == by[("both", r["stem"])]["confidence"] and r["answer"] == by[("both", r["stem"])]["answer"]
    assert not judge.invert  # the last condition asked ("both_fill") is not inverted: the flag follows the prompt


def test_models_one_at_a_time_gives_the_same_rows_and_releases_each(bank, tmp_path):
    b, split_path, _ = bank
    frames = select_frames(b, split_path, ("val",))
    made: list[SignJudge] = []

    def make(spec):
        made.append(SignJudge())
        if len(made) > 1:  # the previous judge must be released before the next one loads
            assert made[-2]._model is None and made[-2]._processor is None
        return made[-1]

    specs = ["org/M-8B-Instruct", "org/M-8B-Instruct@4bit"]
    seq = evaluate_models(specs, make, True, b, frames, tmp_path / "seq.jsonl", conditions=("both",))
    assert len(made) == 2 and made[-1]._model is None
    together = evaluate_models(specs, lambda spec: SignJudge(), False, b, frames, tmp_path / "all.jsonl", conditions=("both",))
    key = lambda rows: sorted((r["model"], r["condition"], r["stem"], r["confidence"]) for r in rows)  # noqa: E731
    assert key(seq) == key(together) and {r["model"] for r in seq} == set(specs)
    assert len(evaluate_models(specs, make, True, b, frames, tmp_path / "seq.jsonl", conditions=("both",))) == len(seq)  # resumable
