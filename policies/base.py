"""Asking policies behind ``interfaces.Policy`` (PLAN_DETAILED.md §4.4, chunks M3.2 and M5.1).

Every policy shares one decision path: a cooldown first (no question within
``cooldown_s`` simulated seconds of the last one, which prevents query storms),
then the policy's own rule. All are compared at matched query rates by
sweeping their single knob (``p``, ``every_s`` or ``tau``).

Run-time rule (CLAUDE.md rule 5): ``should_ask`` sees only estimated features,
the drone's own state and its own history. The oracle is the documented
exception: the evaluation loop hands it the label through ``set_truth`` before
each step, exactly as with the surrogate judge.

Policies that hold state (random, gate) implement ``reset(seed)``; the episode
runner calls it at the start of every flight so a seed always replays the same
decisions.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from interfaces import DroneState, Gate, RuntimeStep


class CooldownPolicy:
    """Base class: cooldown, then ``_decide``. Subclasses set ``name`` and implement ``_decide``."""

    name = "base"

    def __init__(self, cooldown_s: float = 1.0) -> None:
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        self.cooldown_s = cooldown_s
        self.last_score: float | None = None  # for the flight log (RuntimeStep.gate_score)

    def reset(self, seed: int = 0) -> None:
        self.last_score = None

    def should_ask(self, t: float, features: dict[str, Any], drone_state: DroneState, history: Sequence[RuntimeStep]) -> bool:
        self.last_score = None
        last = next((h.t for h in reversed(history) if h.asked), None)
        if last is not None and t - last < self.cooldown_s:
            self._observe(t, features, drone_state, history)
            return False
        return bool(self._decide(t, features, drone_state, history))

    def _observe(self, t: float, features: dict[str, Any], drone_state: DroneState, history: Sequence[RuntimeStep]) -> None:
        """Hook for policies that must see every frame even during the cooldown (the gate logs its score)."""

    def _decide(self, t: float, features: dict[str, Any], drone_state: DroneState, history: Sequence[RuntimeStep]) -> bool:
        raise NotImplementedError


class Never(CooldownPolicy):
    name = "never"

    def __init__(self, **_: Any) -> None:
        super().__init__(cooldown_s=0.0)

    def _decide(self, *_: Any) -> bool:
        return False


class Always(CooldownPolicy):
    """Ask at every frame the cooldown allows. With ``cooldown_s=0`` this is literally every frame."""

    name = "always"

    def __init__(self, cooldown_s: float = 0.0) -> None:
        super().__init__(cooldown_s)

    def _decide(self, *_: Any) -> bool:
        return True


class Random(CooldownPolicy):
    name = "random"

    def __init__(self, p: float, cooldown_s: float = 1.0, seed: int = 0) -> None:
        super().__init__(cooldown_s)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p, self._base_seed = p, seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._rng = np.random.default_rng([self._base_seed, seed])

    def _decide(self, *_: Any) -> bool:
        return bool(self._rng.random() < self.p)


class FixedInterval(CooldownPolicy):
    """Ask every ``every_s`` simulated seconds (the cooldown is the interval)."""

    name = "fixed_interval"

    def __init__(self, every_s: float) -> None:
        if every_s <= 0:
            raise ValueError("every_s must be > 0")
        super().__init__(cooldown_s=every_s)

    def _decide(self, *_: Any) -> bool:
        return True


class Threshold(CooldownPolicy):
    """Hand-tuned threshold on one estimated feature, e.g. ask if ``corridor_min < tau`` or ``corridor_var > tau``.

    Tuned on val as carefully as the gate (PLAN_DETAILED.md §4.4: the strongest simple baseline).
    """

    name = "threshold"

    def __init__(self, feature: str = "corridor_min", tau: float = 10.0, below: bool = True, cooldown_s: float = 1.0) -> None:
        super().__init__(cooldown_s)
        self.feature, self.tau, self.below = feature, tau, below

    def _decide(self, t: float, features: dict[str, Any], *_: Any) -> bool:
        x = float(features[self.feature])
        self.last_score = x
        return x < self.tau if self.below else x > self.tau


class GatePolicy(CooldownPolicy):
    """Learned gate: ask if ``gate.score(features, state) > tau``. The score is logged on every frame."""

    name = "gate"

    def __init__(self, gate: Gate, tau: float = 0.5, cooldown_s: float = 1.0, label: str = "gate") -> None:
        super().__init__(cooldown_s)
        self.gate, self.tau = gate, tau
        self.name = label

    @property
    def feature_names(self) -> list[str]:
        """The inputs the gate was trained on (empty if the gate does not say). The flight loop reads this to
        decide whether it must compute the RGB image features for each frame."""
        return list(getattr(self.gate, "names", []))

    def _score(self, features: dict[str, Any], drone_state: DroneState) -> float:
        self.last_score = float(self.gate.score(features, drone_state))
        return self.last_score

    def _observe(self, t: float, features: dict[str, Any], drone_state: DroneState, history: Sequence[RuntimeStep]) -> None:
        self._score(features, drone_state)

    def _decide(self, t: float, features: dict[str, Any], drone_state: DroneState, history: Sequence[RuntimeStep]) -> bool:
        return self._score(features, drone_state) > self.tau


class Oracle(CooldownPolicy):
    """Asks exactly when the label says the fast stack is wrong (upper bound). The evaluation loop calls ``set_truth``."""

    name = "oracle"

    def __init__(self, cooldown_s: float = 0.0) -> None:
        super().__init__(cooldown_s)
        self._truth: bool | None = None

    def set_truth(self, fast_stack_wrong: bool) -> None:
        self._truth = bool(fast_stack_wrong)

    def _decide(self, *_: Any) -> bool:
        if self._truth is None:
            raise RuntimeError("Oracle.should_ask called before set_truth(); the evaluation loop must supply the label")
        truth, self._truth = self._truth, None
        return truth

    def _observe(self, *_: Any) -> None:
        self._truth = None  # a label handed over during the cooldown is spent, not carried to the next frame


REGISTRY: dict[str, Callable[..., CooldownPolicy]] = {
    "never": Never, "always": Always, "random": Random, "fixed_interval": FixedInterval,
    "threshold": Threshold, "gate": GatePolicy, "oracle": Oracle,
}


def make_policy(spec: dict[str, Any]) -> CooldownPolicy:
    """Build a policy from a config dict, e.g. ``{"policy": "random", "p": 0.05, "cooldown_s": 1.0}``.

    The learned gate is given by file: ``{"policy": "gate", "model_path": "results/gate/gate_mlp_depth+temporal+image.pkl",
    "tau": 0.5}``. ``model_path`` may name environment variables (``$SCRATCH/...``); the pickle is a ``gate.train``
    output loaded by ``gate.model.GateModel``.
    """
    spec = dict(spec)
    kind = spec.pop("policy")
    if kind not in REGISTRY:
        raise ValueError(f"unknown policy {kind!r}; known: {sorted(REGISTRY)}")
    if kind == "gate" and "model_path" in spec:
        import os

        from gate.model import GateModel

        spec["gate"] = GateModel(os.path.expandvars(str(spec.pop("model_path"))))
    return REGISTRY[kind](**spec)
