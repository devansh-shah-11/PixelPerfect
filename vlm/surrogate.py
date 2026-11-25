"""Surrogate judge: a noisy oracle with the real VLM's measured error rates (PLAN_DETAILED.md §4.4, chunk M2.5).

Same ``ask`` interface as the real judge, so the flight loop and the sweeps can
run before the VLM is wired in, and fast sweeps can run without a GPU. The
rates come from the standalone evaluation (M2.4), per obstacle family.

Every result produced with this class must be labelled "surrogate VLM"
(CLAUDE.md rule 7); ``name`` carries that label, so put it in file names and
plot titles.

Ground truth: ``ask`` cannot see the obstacle (its inputs are images), so the
evaluation loop, which owns the ground truth, calls ``set_truth`` before each
``ask``. Nothing at run time in the real pipeline does this; the surrogate is an
evaluation tool only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interfaces import VLMAnswer


@dataclass
class SurrogateJudge:
    """``tpr`` / ``fpr`` per obstacle family; ``None`` family uses ``default_tpr`` / ``default_fpr``."""

    tpr: dict[str, float] = field(default_factory=dict)
    fpr: dict[str, float] = field(default_factory=dict)
    default_tpr: float = 0.8
    default_fpr: float = 0.1
    latency_s: float = 1.0  # reported, not real: the delay charged is a separate knob (D10)
    seed: int = 0
    name: str = "surrogate VLM"
    # Direction (plan v7, DS 2026-11-25): with ``p_dir`` set, a "yes" also says where to go ("left", "right", "over"):
    # the side the evaluation loop passed to ``set_truth`` with probability ``p_dir``, otherwise one of the other two
    # at random (any of the three after a false alarm). ``p_dir`` is NOT measured yet: 0.8 is a placeholder
    # [UNVERIFIED] until the real VLM's direction accuracy is measured on val frames. ``None`` (default) keeps the
    # yes/no judge and its random stream exactly as before.
    p_dir: float | None = None

    def __post_init__(self) -> None:
        if self.p_dir is not None and not 0.0 <= self.p_dir <= 1.0:
            raise ValueError(f"p_dir must be in [0, 1], got {self.p_dir}")
        for d in (self.tpr, self.fpr):
            for k, v in d.items():
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"rate for {k!r} must be in [0, 1], got {v}")
        self._rng = np.random.default_rng(self.seed)
        self._family: str | None = None
        self._crosses: bool | None = None
        self._side: str | None = None
        self.last_direction: str | None = None

    def set_truth(self, obstacle_family: str | None, crosses: bool, side: str | None = None) -> None:
        """Called by the evaluation loop before ``ask``: what is actually in the highlighted region, and which way past
        it is shortest (``sim.geometry.best_side``; only used when ``p_dir`` is set)."""
        self._family = obstacle_family
        self._crosses = bool(crosses)
        self._side = side

    def ask(self, image: np.ndarray, path_overlay: np.ndarray, crop: np.ndarray) -> VLMAnswer:
        if self._crosses is None:
            raise RuntimeError("SurrogateJudge.ask called before set_truth(); the evaluation loop must supply the truth")
        if self._crosses:
            p_yes = self.tpr.get(self._family, self.default_tpr)
        else:
            p_yes = self.fpr.get(self._family, self.default_fpr)
        answer = bool(self._rng.random() < p_yes)
        # A plausible confidence: positive for yes, negative for no, with spread.
        conf = float(self._rng.normal(1.0 if answer else -1.0, 0.5))
        if (conf > 0) != answer:
            conf = -conf
        self.last_direction = None
        if answer and self.p_dir is not None:
            sides = ("left", "right", "over")
            if self._crosses and self._side in sides and self._rng.random() < self.p_dir:
                self.last_direction = self._side
            else:
                others = [d for d in sides if d != self._side] if self._crosses and self._side in sides else list(sides)
                self.last_direction = str(others[int(self._rng.integers(len(others)))])
        self._crosses = None  # one truth per question; forgetting to set it again is an error
        return VLMAnswer(answer=answer, confidence=conf, latency_s=self.latency_s)
