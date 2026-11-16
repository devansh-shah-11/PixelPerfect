"""Build backends, fast stacks and judges from names (used by eval.run_episode and eval.sweep)."""
from __future__ import annotations

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
    from perception.stand_ins import BlindStack, RgbStack, TrueDepthStack

    if name == "true-depth":
        return TrueDepthStack()
    if name == "blind":
        return BlindStack(**kw)
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

        j = QwenVLJudge(**kw)
        return j, _slug_model(j.model_id)
    raise ValueError(f"unknown judge {name!r}")
