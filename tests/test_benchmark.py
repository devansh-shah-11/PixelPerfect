"""Tests for eval/benchmark.py on synthetic result rows. No model, no GPU, no simulator."""
from __future__ import annotations

import json

import pytest

from eval.benchmark import CLEAR_STACK, build, choose_settings, flight_rows, load_latency, main, report
from eval.metrics import EpisodeResult
from vlm.eval_bank import labelled

BIG, SMALL = "org/Big-8B-Instruct", "org/Big-8B-Instruct@4bit"


def vlm_row(model, cond, ep, step, split, obstacle, gt, conf):
    return {"model": model, "condition": cond, "episode": ep, "step": step, "stem": f"{ep:04d}_{step:03d}", "split": split,
            "obstacle": obstacle, "gt_clean_m": gt, "confidence": conf, "answer": conf > 0, "latency_s": 0.09}


def make_rows():
    """Per split: 4 wire flights (depth blind, VLM sees), 4 wall flights (depth sees, VLM blind), 6 empty flights.

    BIG separates wires from empty frames under "both"; its "raw" condition is worse than chance. SMALL is a weaker
    copy of BIG with a shifted confidence scale, as a quantized model may be.
    """
    vlm, depth, ep = [], {}, 0
    for split in ("val", "train"):
        for obstacle, gt, est, big_conf in (("wire_1cm_axis_z", 5.0, 60.0, 4.0), ("wall_white_featureless", 5.0, 4.0, -6.0), ("none", 80.0, 80.0, -5.0)):
            for i in range(6 if obstacle == "none" else 4):
                for step in range(2):
                    stem = f"{ep:04d}_{step:03d}"
                    depth[stem] = {"est": est, "infer_ms": 12.0}
                    conf = big_conf - 2.0 * (i % 2) if obstacle == "none" else big_conf  # empty frames at -5 and -7; walls (-6) sit among them
                    vlm.append(vlm_row(BIG, "both", ep, step, split, obstacle, gt, conf))
                    vlm.append(vlm_row(BIG, "raw", ep, step, split, obstacle, gt, -0.01 * conf))  # worse than chance: never chosen
                    seen = obstacle == "wire_1cm_axis_z" and i < 2  # SMALL sees half of the wire flights
                    vlm.append(vlm_row(SMALL, "both", ep, step, split, obstacle, gt, -1.0 if seen else -9.0 if obstacle == "none" else -9.5))
                ep += 1
    return vlm, depth


LAT = {"depth": {"model": "da2", "kind": "depth", "median_s": 0.010, "p95_s": 0.012, "gpu_peak_mem_gb": 0.5, "source": "f"},
       BIG: {"model": BIG, "median_s": 0.089, "p95_s": 0.091, "gpu_peak_mem_gb": 17.7, "source": "f"},
       SMALL: {"model": SMALL, "median_s": 0.200, "p95_s": 0.250, "gpu_peak_mem_gb": 6.0, "source": "f"}}


def test_settings_are_chosen_on_val_per_model():
    vlm, _ = make_rows()
    s = choose_settings(labelled(vlm, 8.0, 30.0), max_fpr=0.10)
    assert s[BIG]["condition"] == "both" and s[BIG]["val_auroc"] > 0.6  # walls are positives the VLM misses, so not 1.0
    assert -5.0 <= s[BIG]["tau"] < 4.0  # just above the empty frames
    assert s[SMALL]["tau"] < -1.0  # its own, shifted cut: a shared cut of 0 would give it zero recall
    assert choose_settings(labelled(vlm, 8.0, 30.0), 0.10, condition="raw")[BIG]["condition"] == "raw"


def test_table_scores_each_system_on_train_and_fusion_covers_both_failures():
    vlm, depth = make_rows()
    b = build(vlm, depth, LAT)
    sys_ = {s["system"]: s for s in b["systems"]}
    assert b["score_split"] == "train" and b["n_pos"] == 16 and b["n_neg"] == 12 and b["n_depth_misses"] == 8
    d, v, both = sys_["depth only"], sys_[f"VLM only: {BIG}"], sys_[f"depth + VLM: {BIG}"]
    assert d["recall"] == pytest.approx(0.5) and d["by_obstacle"]["wire_1cm_axis_z"]["recall"] == 0.0
    assert v["recall"] == pytest.approx(0.5) and v["by_obstacle"]["wall_white_featureless"]["recall"] == 0.0
    assert both["recall"] == pytest.approx(1.0) and both["fpr"] == 0.0  # each covers what the other misses
    assert v["miss_recall"] == pytest.approx(1.0) and v["d23_pass"]
    small = sys_[f"VLM only: {SMALL}"]
    assert small["miss_recall"] == pytest.approx(0.5) and not small["d23_pass"]
    assert both["median_ms"] == pytest.approx(99.0) and both["gpu_gb"] == pytest.approx(18.2)  # depth + VLM: the sum
    assert b["pick"] == BIG  # the only variant that passes, although it is not the smallest
    text = report(b)
    assert "depth + VLM" in text and "Pick, by rule" in text and "wire_1cm_axis_z (n=8)" in text


def test_pick_prefers_the_smallest_passing_variant_and_reports_when_none_pass():
    vlm, depth = make_rows()
    for r in vlm:  # make SMALL as good as BIG
        if r["model"] == SMALL and r["obstacle"] == "wire_1cm_axis_z":
            r["confidence"] = -1.0
    assert build(vlm, depth, LAT)["pick"] == SMALL  # both pass: least memory wins
    for r in vlm:
        if r["obstacle"] == "wire_1cm_axis_z":
            r["confidence"] = -9.5  # now nobody sees wires
    b = build(vlm, depth, LAT)
    assert b["pick"] is None and "no variant passes" in b["pick_text"]


def test_refuses_test_frames_and_falls_back_to_logged_latency():
    vlm, depth = make_rows()
    b = build(vlm, depth, {})
    v = next(s for s in b["systems"] if s["system"] == f"VLM only: {BIG}")
    assert v["median_ms"] == pytest.approx(90.0) and v["gpu_gb"] is None and "eval_bank" in v["source"]
    assert "n/a" in report(b)
    with pytest.raises(SystemExit):
        build(vlm + [vlm_row(BIG, "both", 999, 0, "test", "none", 80.0, -5.0)], depth, {})
    with pytest.raises(SystemExit):
        build(vlm, depth, {}, score_split="test")


def test_same_frames_for_every_system():
    vlm, depth = make_rows()
    partial = [r for r in vlm if not (r["model"] == SMALL and r["episode"] % 2)]  # SMALL was run on half the flights
    b = build(partial, depth, LAT)
    assert b["n_common_stems"] < b["n_union_stems"] and "only" in report(b)
    assert b["n_frames"] == 14  # train frames of the even flights only, for every system


def _result(policy, seed, crashed, calls, stack, judge, backend="raycast-planB", split="val"):
    return EpisodeResult(episode_id=f"{split}-{seed}-{policy}", seed=seed, split=split, policy=policy, judge=judge, crashed=crashed,
                         reached_goal=not crashed, distance_m=40.0, mission_time_s=10.0 + calls, n_frames=80, n_calls=calls,
                         delay_charged_s=float(calls), latencies_s=[0.09] * calls, extra={"stack": stack, "backend": backend})


def test_flight_rows_name_the_system_and_keep_the_surrogate_label(tmp_path):
    a, c = tmp_path / "bench_da2", tmp_path / "bench_clear"
    for d, stack in ((a, "depth-anything-small"), (c, CLEAR_STACK)):
        d.mkdir()
        rs = [_result("never", s, s % 2 == 0, 0, stack, "surrogate VLM") for s in range(6)] + [_result("always", s, False, 12, stack, "surrogate VLM") for s in range(6)]
        (d / "results.jsonl").write_text("".join(r.to_json() + "\n" for r in rs))
    rows = flight_rows([a, c])
    assert [(r["dir"], r["policy"], r["system"]) for r in rows] == [
        ("bench_da2", "always", "depth + VLM"), ("bench_da2", "never", "depth only"),
        ("bench_clear", "always", "VLM only"), ("bench_clear", "never", "no perception")]
    assert rows[0]["judge"] == "surrogate VLM" and rows[1]["judge"] == "none" and rows[1]["crash_rate"] == pytest.approx(0.5)
    assert "surrogate VLM" in report(None, rows)
    (a / "results.jsonl").write_text(_result("never", 1, False, 0, "x", "none", split="test").to_json() + "\n")
    with pytest.raises(SystemExit):
        flight_rows([a])


def test_cli_writes_report_and_json(tmp_path, capsys):
    vlm, depth = make_rows()
    (tmp_path / "vlm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in vlm))
    (tmp_path / "run.jsonl").write_text("".join(json.dumps({"stem": k, "corridor_min_est": v["est"], "infer_ms": v["infer_ms"]}) + "\n" for k, v in depth.items()))
    (tmp_path / "lat_1.json").write_text(json.dumps([
        {"model": "da2", "kind": "depth", "outcome": "success", "median_s": 0.01, "p95_s": 0.012, "gpu_peak_mem_gb": 0.5},
        {"model": BIG, "outcome": "success", "median_s": 0.089, "p95_s": 0.091, "gpu_peak_mem_gb": 17.7},
        {"model": SMALL, "outcome": "failed: OutOfMemoryError"}]))
    assert set(load_latency([tmp_path / "lat_1.json"])) == {"depth", BIG}  # failed rows are not costs
    out = tmp_path / "benchmark.md"
    main(["--vlm", str(tmp_path / "vlm.jsonl"), "--run", str(tmp_path / "run.jsonl"), "--latency", str(tmp_path / "lat_*.json"), "--out", str(out), "--no-log"])
    assert "| depth only |" in out.read_text() and json.loads(out.with_suffix(".json").read_text())["frame_bank"]["pick"] == BIG
    with pytest.raises(SystemExit):
        main(["--vlm", str(tmp_path / "vlm.jsonl"), "--no-log"])  # the table needs the depth run too
