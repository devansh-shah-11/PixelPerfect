"""Tests for vlm/: overlay geometry, the judge on a fake model, the surrogate, the benchmark summary.

No network, no GPU, no model download. The real model path is exercised by
hpc/vlm_latency.sbatch on the A100.
"""
from __future__ import annotations

import numpy as np
import pytest

from interfaces import VLMAnswer, VLMJudge
from vlm import judge as judge_mod
from vlm.bench_latency import bench_depth, load_frames, markdown_table, summarize
from vlm.judge import INVERTED_PROMPTS, PROMPTS, QwenVLJudge, _slug_model, parse_model_spec, variant_ids, yes_no_score
from vlm.overlay import CORRIDOR_COLS, CORRIDOR_ROWS, corridor_box, crop_region, draw_overlay, render
from vlm.surrogate import SurrogateJudge

H, W = 225, 400


def frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(H, W, 3), dtype=np.uint8)


# ---- overlay -----------------------------------------------------------------------


def test_corridor_box_matches_label_window():
    r0, r1, c0, c1 = corridor_box((H, W, 3))
    assert (r0, r1, c0, c1) == (90, 135, 180, 220)
    try:
        from perception import depth_utils  # on workstream B's branch
    except ImportError:
        pytest.skip("perception.depth_utils not on this branch")
    assert (CORRIDOR_ROWS, CORRIDOR_COLS) == (depth_utils.CORRIDOR_ROWS, depth_utils.CORRIDOR_COLS)
    rows, cols = depth_utils.corridor_window((H, W))
    assert (rows.start, rows.stop, cols.start, cols.stop) == (r0, r1, c0, c1)


def test_overlay_touches_only_the_box_and_its_outline():
    rgb = frame()
    box = corridor_box(rgb.shape)
    out = draw_overlay(rgb, box, thickness=2, fill_alpha=0.15)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    changed = (out != rgb).any(axis=-1)
    r0, r1, c0, c1 = box
    assert not changed[: r0 - 2].any() and not changed[r1 + 2 :].any()
    assert not changed[:, : c0 - 2].any() and not changed[:, c1 + 2 :].any()
    assert changed[r0 - 1, c0:c1].all()  # outline row
    assert changed[r0:r1, c0:c1].mean() > 0.9  # tinted inside (a few pixels may round to the same value)
    assert rgb.sum() == frame().sum()  # input untouched


def test_crop_is_full_resolution_region_upscaled():
    rgb = frame()
    box = corridor_box(rgb.shape)  # 45 x 40 px
    crop = crop_region(rgb, box, margin=0.5, upscale=4)
    assert crop.shape == (4 * (45 + 2 * 22), 4 * (40 + 2 * 20), 3)  # margin rounds to 22 rows, 20 cols
    crop1 = crop_region(rgb, box, margin=0.0, upscale=1)
    assert np.array_equal(crop1, rgb[90:135, 180:220])


def test_render_returns_pair():
    overlay, crop = render(frame())
    assert overlay.shape == (H, W, 3) and crop.ndim == 3 and crop.dtype == np.uint8


def test_overlay_rejects_non_rgb():
    with pytest.raises(ValueError):
        draw_overlay(np.zeros((H, W), dtype=np.uint8), (0, 1, 0, 1))


# ---- judge on a fake model ------------------------------------------------------------


class FakeTokenizer:
    """'yes'/'no' and their capitalised forms are single tokens; the spaced forms are two tokens."""

    table = {"yes": [1], "Yes": [2], "no": [3], "No": [4], "YES": [5], "NO": [6]}

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return self.table.get(s, [0, 0])


def test_variant_ids_keeps_single_token_forms_only():
    tok = FakeTokenizer()
    assert variant_ids(tok, judge_mod.YES_FORMS) == [1, 2, 5]
    assert variant_ids(tok, judge_mod.NO_FORMS) == [3, 4, 6]
    with pytest.raises(ValueError):
        variant_ids(tok, (" yes",))


def test_yes_no_score_sign_and_pooling():
    logits = np.zeros(10)
    logits[1], logits[3] = 2.0, 1.0
    assert yes_no_score(logits, [1], [3]) == pytest.approx(1.0)
    logits[2] = 2.0  # a second yes form adds probability mass, never removes it
    assert yes_no_score(logits, [1, 2], [3]) > 1.0


class FakeJudge(QwenVLJudge):
    """Skips the model: returns logits favouring yes when the first image's top-left pixel is bright."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._yes_ids, self._no_ids = [1], [3]
        self.seen: list[int] = []

    def _logits(self, images):
        self.seen.append(len(images))
        logits = np.zeros(8)
        bright = images[0][0, 0, 0] > 127
        logits[1], logits[3] = (3.0, 0.0) if bright else (0.0, 3.0)
        return logits


def test_judge_implements_interface_and_reads_answer_from_logits():
    j = FakeJudge()
    assert isinstance(j, VLMJudge)
    bright = np.full((H, W, 3), 255, dtype=np.uint8)
    dark = np.zeros((H, W, 3), dtype=np.uint8)
    yes = j.ask(bright, bright, bright)
    no = j.ask(dark, dark, dark)
    assert isinstance(yes, VLMAnswer) and yes.answer and yes.confidence == pytest.approx(3.0)
    assert not no.answer and no.confidence == pytest.approx(-3.0)
    assert yes.latency_s > 0


def test_judge_ablation_switches_control_what_the_model_sees():
    bright = np.full((H, W, 3), 255, dtype=np.uint8)
    dark = np.zeros((H, W, 3), dtype=np.uint8)
    both = FakeJudge()
    both.ask(dark, bright, bright)  # overlay (bright) is image 0 when use_overlay
    assert both.seen == [2]
    raw_only = FakeJudge(use_overlay=False, use_crop=False)
    ans = raw_only.ask(dark, bright, bright)  # raw frame (dark) is image 0
    assert raw_only.seen == [1] and not ans.answer


# ---- surrogate ---------------------------------------------------------------------


def test_surrogate_is_labelled_and_needs_truth():
    s = SurrogateJudge()
    assert "surrogate" in s.name.lower()
    assert isinstance(s, VLMJudge)
    img = frame()
    with pytest.raises(RuntimeError):
        s.ask(img, img, img)


def test_surrogate_rates_match_configuration():
    s = SurrogateJudge(tpr={"wire": 0.3}, fpr={"wire": 0.05}, default_tpr=1.0, default_fpr=0.0, seed=1)
    img = frame()
    n = 4000
    yes_on_wire = 0
    for _ in range(n):
        s.set_truth("wire", crosses=True)
        yes_on_wire += s.ask(img, img, img).answer
    assert abs(yes_on_wire / n - 0.3) < 0.03
    false_on_wire = 0
    for _ in range(n):
        s.set_truth("wire", crosses=False)
        false_on_wire += s.ask(img, img, img).answer
    assert abs(false_on_wire / n - 0.05) < 0.02
    s.set_truth("pole", crosses=True)
    a = s.ask(img, img, img)
    assert a.answer and a.confidence > 0 and a.latency_s == 1.0
    s.set_truth("pole", crosses=False)
    assert not s.ask(img, img, img).answer
    with pytest.raises(ValueError):
        SurrogateJudge(tpr={"x": 1.5})


# ---- benchmark helpers --------------------------------------------------------------


def test_summarize_and_table():
    stats = summarize([0.5, 1.0, 1.5, 2.0])
    assert stats["n"] == 4 and stats["median_s"] == pytest.approx(1.25) and stats["max_s"] == 2.0
    table = markdown_table([
        {"model": "m", "dtype": "bf16", "outcome": "success", "n": 4, "load_s": 3.0, "gpu_peak_mem_gb": 5.1, **stats},
        {"model": "big", "dtype": "bf16", "outcome": "failed: OutOfMemoryError"},
    ])
    assert "| m |" in table and "failed: OutOfMemoryError" in table


def test_load_frames_noise_and_dir(tmp_path):
    noise = load_frames(None, 3, seed=0)
    assert len(noise) == 3 and noise[0].shape == (H, W, 3)
    from PIL import Image

    for i in range(5):
        Image.fromarray(frame(i)).save(tmp_path / f"{i:04d}_000_rgb.png")
    picked = load_frames(tmp_path, 2, seed=0)
    assert len(picked) == 2 and picked[0].dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        load_frames(tmp_path / "empty", 2, seed=0)


# ---- quantized variants, inverted prompts, the cut (D22, D23) ------------------------


def test_model_spec_parsing_and_labels():
    assert parse_model_spec("Qwen/Qwen3-VL-8B-Instruct") == ("Qwen/Qwen3-VL-8B-Instruct", None)
    assert parse_model_spec("Qwen/Qwen3-VL-8B-Instruct@4bit") == ("Qwen/Qwen3-VL-8B-Instruct", "4bit")
    with pytest.raises(ValueError):
        parse_model_spec("Qwen/Qwen3-VL-8B-Instruct@2bit")
    assert _slug_model("Qwen/Qwen3-VL-8B-Instruct") == "qwen3-vl-8b"
    assert _slug_model("Qwen/Qwen3-VL-8B-Instruct@8bit") == "qwen3-vl-8b-8bit"  # results tell the variants apart
    j = QwenVLJudge(model_id="Qwen/Qwen3-VL-8B-Instruct@4bit")
    assert (j.model_id, j.quant, j.spec) == ("Qwen/Qwen3-VL-8B-Instruct", "4bit", "Qwen/Qwen3-VL-8B-Instruct@4bit")
    k = QwenVLJudge(model_id="Qwen/Qwen3-VL-8B-Instruct", quant="8bit")
    assert k.spec == "Qwen/Qwen3-VL-8B-Instruct@8bit"
    assert QwenVLJudge().quant is None and QwenVLJudge().spec == QwenVLJudge().model_id
    with pytest.raises(ValueError):
        QwenVLJudge(quant="3bit")


def test_factory_builds_a_quantized_judge_from_a_prompt_key():
    from eval.factory import make_judge
    from vlm.judge import prompt_text

    j, label = make_judge("qwen", model_id="Qwen/Qwen3-VL-8B-Instruct@4bit", prompt_key="free", tau=-2.5)
    assert label == "qwen3-vl-8b-4bit" and j.invert and j.tau == -2.5 and j.prompt == prompt_text("free", True)
    j, label = make_judge("qwen", prompt_key="blocked")
    assert label == "qwen3-vl-2b" and not j.invert and j.prompt == prompt_text("blocked", True)


def test_inverted_prompt_flips_the_sign_and_the_cut_moves_the_answer():
    assert INVERTED_PROMPTS <= set(PROMPTS) and "free" in INVERTED_PROMPTS
    bright = np.full((H, W, 3), 255, dtype=np.uint8)  # FakeJudge: logits favour "yes" by 3
    assert FakeJudge().ask(bright, bright, bright).confidence == pytest.approx(3.0)
    inv = FakeJudge(invert=True).ask(bright, bright, bright)  # "yes, it is free" means no obstacle
    assert inv.confidence == pytest.approx(-3.0) and not inv.answer
    assert not FakeJudge(tau=5.0).ask(bright, bright, bright).answer  # above the cut only
    dark = np.zeros((H, W, 3), dtype=np.uint8)
    low = FakeJudge(tau=-4.0).ask(dark, dark, dark)  # a conservative model needs a cut below 0 to say yes at all
    assert low.confidence == pytest.approx(-3.0) and low.answer


def test_quantized_judge_refuses_cpu():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    j = QwenVLJudge(model_id="Qwen/Qwen3-VL-2B-Instruct@4bit", device="cpu")
    with pytest.raises((RuntimeError, OSError)):  # OSError: no weights and no network on this machine
        j._load()


def test_bench_depth_times_a_fast_stack_and_the_table_shows_quantization():
    class FakeDepth:
        checkpoint = "fake/depth-small"

        def __init__(self):
            self.calls = 0

        def perceive(self, rgb):
            self.calls += 1

    model = FakeDepth()
    r = bench_depth([frame(i) for i in range(4)], warmup=2, model=model)
    assert r["outcome"] == "success" and r["kind"] == "depth" and r["model"] == "fake/depth-small" and r["n"] == 4 and model.calls == 6

    class Broken:
        def perceive(self, rgb):
            raise MemoryError("boom")

    assert bench_depth([frame(0)], warmup=1, model=Broken())["outcome"].startswith("failed: MemoryError")
    stats = summarize([0.1, 0.2])
    table = markdown_table([{"model": "m@4bit", "dtype": "bfloat16", "quant": "4bit", "outcome": "success", "load_s": 1.0, **stats}, {**r, "load_s": 0.1}])
    assert "4bit (bfloat16 compute)" in table and "| fake/depth-small | float32 |" in table
