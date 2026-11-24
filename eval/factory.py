"""Build backends, fast stacks and judges from names (used by eval.run_episode and eval.sweep)."""
from __future__ import annotations

import dataclasses
from typing import Any


def make_sim(name: str, **kw: Any) -> Any:
    if name == "raycast":
        from sim.raycast import Camera, RaycastSim

        return RaycastSim(Camera(kw.get("width", 400), kw.get("height", 225)))
    if name == "airsim":
        from sim.airsim_backend import AirSimBackend

        return AirSimBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")


def make_stack(name: str, **kw: Any) -> Any:
    from perception.stand_ins import BlindStack, ClearStack, RgbStack, TrueDepthStack

    if name == "true-depth":
        return TrueDepthStack()
    if name == "blind":
        return BlindStack(**kw)
    if name == "clear":
        return ClearStack()
    if name == "da2":
        from perception.depth_anything import DepthAnythingV2

        return RgbStack(DepthAnythingV2(**kw))
    raise ValueError(f"unknown fast stack {name!r}")


def make_judge(name: str, **kw: Any) -> tuple[Any, str]:
    """(judge, label for the results). Surrogate results are always labelled "surrogate VLM" (CLAUDE.md rule 7)."""
    if name == "none":
        return None, "none"
    if name == "surrogate":
        from vlm.surrogate import SurrogateJudge

        j = SurrogateJudge(**kw)
        return j, j.name
    if name == "qwen":
        from vlm.judge import QwenVLJudge, _slug_model

        kw = dict(kw)
        key = kw.pop("prompt_key", None)  # a name from vlm.judge.PROMPTS instead of the raw text
        if key is not None:
            from vlm.judge import INVERTED_PROMPTS, prompt_text

            kw["prompt"] = prompt_text(key, kw.get("use_overlay", True))
            kw["invert"] = key in INVERTED_PROMPTS
        j = QwenVLJudge(**kw)  # model_id may carry a quantization suffix, e.g. "...-8B-Instruct@4bit"
        return j, _slug_model(j.spec)
    raise ValueError(f"unknown judge {name!r}")


def fork_judge(judge: Any, seed: int, k: int) -> Any:
    """A judge for one fork (or, with ``k = 0``, for one sweep flight). A seeded judge (the surrogate) gets its own seed per (flight, frame),
    so forks neither share one random draw nor depend on the order they run in."""
    fields = {f.name for f in dataclasses.fields(judge)} if dataclasses.is_dataclass(judge) else set()
    return dataclasses.replace(judge, seed=int(seed) * 100_003 + int(k)) if "seed" in fields else judge
