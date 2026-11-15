"""The VLM judge: Qwen3-VL answering one yes/no question about the flight path (PLAN_DETAILED.md D8, D9; chunks M2.1, M2.3).

Implements ``interfaces.VLMJudge``. One forward pass per question, no
``generate()``: the answer is read from the next-token logits as
``logit(yes) - logit(no)``, which is the ``confidence`` field of
``interfaces.VLMAnswer`` and can be thresholded later.

Needs the ``vlm`` extra: ``pip install -e ".[vlm]"``. Run it on the A100 in the
same job as the simulator (D17); never as a separate idle server.

Model ids verified on Hugging Face on 2026-11-15 (all Apache 2.0, architecture
``Qwen3VLForConditionalGeneration``): Qwen/Qwen3-VL-2B-Instruct,
Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from interfaces import VLMAnswer

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # D8 default
REFERENCE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # D8 "does size help?" reference

# Fixed prompt (D9). Variants only in the prompt ablation (M2.4).
PROMPT = (
    "Does any object, including thin wires, cables, poles or glass, cross the "
    "highlighted region? Answer yes or no."
)

# Surface forms of the two answers. Several spellings are single tokens in most
# tokenizers; the ones that are not are skipped, so the score is robust to how
# the tokenizer splits them.
YES_FORMS = ("yes", "Yes", " yes", " Yes", "YES")
NO_FORMS = ("no", "No", " no", " No", "NO")


def variant_ids(tokenizer: Any, forms: Sequence[str]) -> list[int]:
    """Token ids of the forms that encode to exactly one token. Raises if none do."""
    ids: list[int] = []
    for form in forms:
        toks = tokenizer.encode(form, add_special_tokens=False)
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(int(toks[0]))
    if not ids:
        raise ValueError(f"none of {forms!r} is a single token in this tokenizer")
    return ids


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def yes_no_score(logits: np.ndarray, yes_ids: Sequence[int], no_ids: Sequence[int]) -> float:
    """``logit(yes) - logit(no)`` with each side pooled over its surface forms (log-sum-exp)."""
    return _logsumexp(np.asarray(logits, dtype=np.float64)[list(yes_ids)]) - _logsumexp(
        np.asarray(logits, dtype=np.float64)[list(no_ids)]
    )


class QwenVLJudge:
    """Qwen3-VL as the slow system. ``ask`` returns ``(answer, confidence, latency_s)``.

    ``use_overlay`` and ``use_crop`` are the M2.4 ablation switches: with both
    off the model sees only the raw frame.
    """

    name = "qwen3-vl"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "bfloat16",
        use_overlay: bool = True,
        use_crop: bool = True,
        prompt: str = PROMPT,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_overlay = use_overlay
        self.use_crop = use_crop
        self.prompt = prompt
        self.load_seconds: float | None = None
        self._model = None
        self._processor = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []

    # ---- loading -----------------------------------------------------------------

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        start = time.perf_counter()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, self.dtype) if self.device == "cuda" else torch.float32
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls  # type: ignore
        except ImportError:  # older transformers: generic image-text class [UNVERIFIED for qwen3_vl]
            from transformers import AutoModelForImageTextToText as ModelCls  # type: ignore
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = ModelCls.from_pretrained(self.model_id, dtype=torch_dtype).to(self.device).eval()
        tok = self._processor.tokenizer
        self._yes_ids = variant_ids(tok, YES_FORMS)
        self._no_ids = variant_ids(tok, NO_FORMS)
        self.load_seconds = time.perf_counter() - start

    def warmup(self, shape: tuple[int, int] = (225, 400)) -> float:
        """One throwaway call on a blank frame (the first call pays for kernels and caches). Returns its latency."""
        blank = np.zeros((*shape, 3), dtype=np.uint8)
        return self.ask(blank, blank, blank).latency_s

    # ---- inference ---------------------------------------------------------------

    def _logits(self, images: list[np.ndarray]) -> np.ndarray:
        """Next-token logits (vocab,) after the prompt. Overridden by tests with a fake."""
        import torch

        if self._model is None:
            self._load()
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self.prompt})
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        pil = [Image.fromarray(im) for im in images]
        inputs = self._processor(text=[text], images=pil, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)
            logits = out.logits[0, -1]
            if self.device == "cuda":
                torch.cuda.synchronize()
        return logits.float().cpu().numpy()

    def ask(self, image: np.ndarray, path_overlay: np.ndarray, crop: np.ndarray) -> VLMAnswer:
        images = [path_overlay if self.use_overlay else image]
        if self.use_crop:
            images.append(crop)
        start = time.perf_counter()
        logits = self._logits(images)
        latency = time.perf_counter() - start
        conf = yes_no_score(logits, self._yes_ids, self._no_ids)
        return VLMAnswer(answer=conf > 0, confidence=float(conf), latency_s=latency)
