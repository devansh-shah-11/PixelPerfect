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

# Quantized variants for the on-prem comparison (PLAN_DETAILED.md D22). A model spec is a Hugging Face
# id with an optional suffix: "Qwen/Qwen3-VL-8B-Instruct@4bit". bitsandbytes, no calibration data:
# 4-bit NF4 and 8-bit LLM.int8 (arXiv 2208.07339). 4-bit can shift the confidence scale (arXiv
# 2607.24440, 2609.06922), so every variant gets its own cut tuned on val (vlm.eval_bank.tune_tau).
# Needs the ``quant`` extra and a CUDA GPU. [UNVERIFIED on a live model until the A100 job runs.]
QUANT_MODES = ("4bit", "8bit")


def parse_model_spec(spec: str) -> tuple[str, str | None]:
    """``"<hf id>@4bit"`` -> ``("<hf id>", "4bit")``; a plain id -> ``(id, None)``."""
    model_id, _, quant = spec.partition("@")
    if not quant:
        return model_id, None
    if quant not in QUANT_MODES:
        raise ValueError(f"unknown quantization {quant!r} in {spec!r}; use one of {QUANT_MODES}")
    return model_id, quant

# Fixed prompt (D9). Variants only in the prompt ablation (M2.4).
PROMPT = (
    "Does any object, including thin wires, cables, poles or glass, cross the "
    "highlighted region? Answer yes or no."
)

# Prompt ablation (M2.4). "cross" is the D9 default. The variants target the
# failure seen in job 33462: a surface that fills the whole region (a building
# face at 6 to 9 m) does not "cross" it, and the 2B answered no with high
# confidence. "{where}" becomes "the highlighted region" when the overlay is
# shown and "the centre of the image" when it is not, so the question never
# refers to a box the model cannot see.
PROMPTS = {
    "cross": "Does any object, including thin wires, cables, poles or glass, cross {where}? Answer yes or no.",
    "blocked": (
        "Is {where} blocked by any object or surface, including walls, buildings, glass, "
        "poles or thin wires? Answer yes or no."
    ),
    "hit": (
        "A drone is flying straight ahead towards {where}. Would it hit any object or surface "
        "there, including walls, glass, poles or thin wires? Answer yes or no."
    ),
    # Second and last prompt round (PLAN_DETAILED.md D23), aimed at what job 33620 showed.
    # "fill": the VLM misses surfaces that fill the whole region (wall 0.44, building 0.02), so say they count.
    "fill": (
        "Is {where} blocked by any object or surface, including walls, buildings, glass, "
        "poles or thin wires? A flat wall or building face that fills the whole region counts as blocked. "
        "Answer yes or no."
    ),
    # "free": the opposite question, so "yes" means clear. Separates what the model sees from its yes/no bias.
    "free": "Is {where} completely free of obstacles, with nothing in the way? Answer yes or no.",
}
# Prompts where "yes" means the path is clear: the judge flips the sign so confidence > 0 still means "obstacle".
INVERTED_PROMPTS = frozenset({"free"})


def _slug_model(model_id: str) -> str:
    """Short judge label for results, e.g. "qwen3-vl-2b" or "qwen3-vl-8b-4bit" for a quantized spec."""
    name, quant = parse_model_spec(model_id)
    slug = name.split("/")[-1].replace("-Instruct", "").lower()
    return f"{slug}-{quant}" if quant else slug


def prompt_text(key: str, overlay: bool) -> str:
    """The question for prompt ``key``, worded for whether the highlighted region is drawn."""
    where = "the highlighted region" if overlay else "the centre of the image"
    return PROMPTS[key].format(where=where)

# Surface forms of the two answers. Several spellings are single tokens in most
# tokenizers; the ones that are not are skipped, so the score is robust to how
# the tokenizer splits them.
YES_FORMS = ("yes", "Yes", " yes", " Yes", "YES")
NO_FORMS = ("no", "No", " no", " No", "NO")
# Direction question (plan v7, DS 2026-11-25), asked only after a "yes" when ``direction=True``: a second forward
# pass, and the answer is the most likely of the three words. Its accuracy is not measured yet [UNVERIFIED].
DIRECTION_PROMPT = ("Something blocks {where}. To get past it safely, should the drone go left, right or over? "
                    "Answer with one word: left, right or over.")
DIRECTION_FORMS = {"left": ("left", "Left", " left", " Left"), "right": ("right", "Right", " right", " Right"),
                   "over": ("over", "Over", " over", " Over")}


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


def quant_load_kwargs(quant: str, dtype: str = "bfloat16") -> dict[str, Any]:
    """``from_pretrained`` keyword arguments for a bitsandbytes-quantized load on GPU 0."""
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "4bit":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False,
                                 bnb_4bit_compute_dtype=getattr(torch, dtype))
    elif quant == "8bit":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unknown quantization {quant!r}; use one of {QUANT_MODES}")
    return {"quantization_config": cfg, "device_map": {"": 0}}


class QwenVLJudge:
    """Qwen3-VL as the slow system. ``ask`` returns ``(answer, confidence, latency_s)``.

    ``use_overlay`` and ``use_crop`` are the M2.4 ablation switches: with both
    off the model sees only the raw frame. ``model_id`` may be a spec with a
    quantization suffix (``"...-8B-Instruct@4bit"``); ``spec`` keeps the full
    string and ``model_id`` the plain Hugging Face id. ``invert`` is for prompts
    where "yes" means clear (``INVERTED_PROMPTS``). ``tau`` is the cut on the
    confidence: each model and prompt has its own, tuned on val (the 8B says
    yes to almost nothing at 0, job 33620).
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
        quant: str | None = None,
        invert: bool = False,
        tau: float = 0.0,
        direction: bool = False,
    ) -> None:
        plain_id, spec_quant = parse_model_spec(model_id)
        if quant is not None and quant not in QUANT_MODES:
            raise ValueError(f"unknown quantization {quant!r}; use one of {QUANT_MODES}")
        self.model_id = plain_id
        self.quant = quant or spec_quant
        self.spec = f"{plain_id}@{self.quant}" if self.quant else plain_id
        self.device = device
        self.dtype = dtype
        self.use_overlay = use_overlay
        self.use_crop = use_crop
        self.prompt = prompt
        self.invert = invert
        self.tau = tau  # "yes" when confidence > tau; the cut is tuned offline on val (vlm.eval_bank.tune_tau), never at run time
        self.direction = direction
        self.last_direction: str | None = None
        self._dir_ids: dict[str, list[int]] = {}
        self._dir4_ids: dict[str, list[int]] = {}  # straight, left, right, over (vlm.direction), for ask_direction
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
        if self.quant:
            if self.device != "cuda":
                raise RuntimeError(f"{self.spec}: bitsandbytes quantization needs a CUDA GPU (got device {self.device!r})")
            self._model = ModelCls.from_pretrained(self.model_id, **quant_load_kwargs(self.quant, self.dtype)).eval()
        else:
            self._model = ModelCls.from_pretrained(self.model_id, dtype=torch_dtype).to(self.device).eval()
        tok = self._processor.tokenizer
        self._yes_ids = variant_ids(tok, YES_FORMS)
        self._no_ids = variant_ids(tok, NO_FORMS)
        self._dir_ids = {k: variant_ids(tok, forms) for k, forms in DIRECTION_FORMS.items()}
        from vlm.direction import OPTION_FORMS

        self._dir4_ids = {k: variant_ids(tok, forms) for k, forms in OPTION_FORMS.items()}
        self.load_seconds = time.perf_counter() - start

    def warmup(self, shape: tuple[int, int] = (225, 400)) -> float:
        """One throwaway call on a blank frame (the first call pays for kernels and caches). Returns its latency."""
        blank = np.zeros((*shape, 3), dtype=np.uint8)
        return self.ask(blank, blank, blank).latency_s

    # ---- inference ---------------------------------------------------------------

    def _logits(self, images: list[np.ndarray], prompt: str | None = None) -> np.ndarray:
        """Next-token logits (vocab,) after ``prompt`` (default: the judge's prompt). Overridden by tests with a fake."""
        import torch

        if self._model is None:
            self._load()
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self.prompt if prompt is None else prompt})
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

    def ask_direction(self, images: list[np.ndarray], prompt: str) -> tuple[dict[str, float], float]:
        """One pass, no prior yes/no: log-probabilities over straight / left / right / over, and the latency in seconds.

        ``images`` and ``prompt`` come from ``vlm.direction.render_inputs`` (goal marker and text included). The answer
        is read from the next-token logits like ``ask``, so there is no free text to hallucinate (D8).
        """
        from vlm.direction import option_logprobs

        if not self._dir4_ids and self._model is None:
            self._load()
        start = time.perf_counter()
        logits = self._logits(images, prompt)
        return option_logprobs(logits, self._dir4_ids), time.perf_counter() - start

    def _generate(self, images: list[np.ndarray], prompt: str, max_new_tokens: int) -> str:
        """Greedy free text after ``prompt``. Overridden by tests with a fake."""
        import torch

        if self._model is None:
            self._load()
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        text = self._processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=[text], images=[Image.fromarray(im) for im in images], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            if self.device == "cuda":
                torch.cuda.synchronize()
        new = out[0, inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(new, skip_special_tokens=True)

    def generate(self, images: list[np.ndarray], prompt: str, max_new_tokens: int = 200) -> tuple[str, float]:
        """Free-text reply (greedy) and the latency in seconds. Used for the JSON scene read (``vlm.scene``); the caller parses and checks it."""
        start = time.perf_counter()
        text = self._generate(images, prompt, max_new_tokens)
        return text, time.perf_counter() - start

    def score_yes_no(self, images: list[np.ndarray], prompt: str) -> tuple[float, float]:
        """``logit(yes) - logit(no)`` for a custom question, and the latency in seconds (the grid format of ``vlm.scene``)."""
        start = time.perf_counter()
        logits = self._logits(images, prompt)  # loads the model on first use, which also fills the yes/no ids
        return yes_no_score(logits, self._yes_ids, self._no_ids), time.perf_counter() - start

    def ask(self, image: np.ndarray, path_overlay: np.ndarray, crop: np.ndarray) -> VLMAnswer:
        images = [path_overlay if self.use_overlay else image]
        if self.use_crop:
            images.append(crop)
        start = time.perf_counter()
        logits = self._logits(images)
        conf = yes_no_score(logits, self._yes_ids, self._no_ids)
        if self.invert:  # the question asked whether the path is clear
            conf = -conf
        answer = conf > self.tau
        self.last_direction = None
        if answer and self.direction:
            where = "the highlighted region" if self.use_overlay else "the centre image"
            dl = np.asarray(self._logits(images, DIRECTION_PROMPT.format(where=where)), dtype=np.float64)
            self.last_direction = max(self._dir_ids, key=lambda k: _logsumexp(dl[self._dir_ids[k]]))
        latency = time.perf_counter() - start  # both passes: the drone waits for the direction too
        return VLMAnswer(answer=bool(answer), confidence=float(conf), latency_s=latency)
