"""Shared interfaces between workstreams (PLAN.md §6, task S-04).

Components talk to each other only through the types below. Changing a
signature here breaks everyone's code, so change it only after all three
members agree.

Ground-truth rule (CLAUDE.md rule 5): nothing passed to ``perceive``, ``act``,
``score`` or ``should_ask`` may contain ground truth about obstacles (GT depth,
segmentation, collision). Ground truth appears only in ``FrameRecord``, which
is written to the flight log for labels and metrics. The oracle policy is the
one exception and gets its labels from the evaluation code, not from here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, NamedTuple, Protocol, Sequence, runtime_checkable

import numpy as np


class Action(str, Enum):
    """Discrete actions of the kinematic drone (PLAN.md §4.2)."""

    CONTINUE = "continue"
    STOP = "stop"
    YAW_LEFT = "yaw_left"
    YAW_RIGHT = "yaw_right"
    CLIMB = "climb"


@dataclass(frozen=True)
class DroneState:
    """What the drone knows about itself at run time.

    Pose comes from the simulator (standing in for odometry); it says nothing
    about obstacles, so it is allowed at run time.
    """

    t: float  # simulated seconds since the episode started
    position: tuple[float, float, float]  # metres, world frame
    yaw: float  # radians
    speed: float  # m/s along the heading


@dataclass
class DepthEstimate:
    """Output of the fast stack for one frame.

    ``features`` holds the gate's inputs: scalar corridor statistics,
    thin-structure scores, temporal changes, and optionally the encoder
    embedding under the key ``"embedding"`` (an ``np.ndarray``).
    """

    depth: np.ndarray  # (H, W) float32, metres, estimated (never ground truth)
    features: dict[str, Any]


class VLMAnswer(NamedTuple):
    """Result of one VLM call. Unpacks as ``(answer, confidence, latency_s)``."""

    answer: bool  # True = "an object crosses the highlighted flight path"
    confidence: float  # logit(yes) - logit(no)
    latency_s: float  # measured wall-clock seconds for this call


@dataclass(frozen=True)
class RuntimeStep:
    """One past step as policies see it (``should_ask`` history). No ground truth."""

    t: float
    asked: bool
    vlm_answer: bool | None  # None if the VLM was not asked at this step
    action_taken: Action
    gate_score: float | None  # None for policies that don't use a gate


@dataclass
class FrameRecord:
    """One row of the flight log (JSONL, one line per frame). PLAN.md §6.

    Contains ground truth, so it is for labels and metrics only. Fields beyond
    PLAN.md §6: ``split`` (episodes are keyed by ``(split, seed)``, CLAUDE.md
    rule 3) and ``gate_score`` (needed by the crash-cause tool).
    """

    episode_id: str
    split: str  # "train" | "val" | "test"
    seed: int
    t: float
    rgb_path: str
    depth_gt_path: str
    seg_gt_path: str
    depth_est_path: str
    pose: tuple[float, float, float, float]  # x, y, z (m), yaw (rad)
    velocity: tuple[float, float, float]  # m/s, world frame
    corridor_min_gt: float  # metres
    corridor_min_est: float  # metres
    action_proposed: Action
    asked: bool
    vlm_answer: bool | None
    vlm_conf: float | None
    vlm_latency_real_s: float | None
    delay_charged_s: float  # simulated seconds charged for the VLM call (0 if not asked)
    action_taken: Action
    collision: bool
    obstacle_family: str | None  # e.g. "wire"; None if no obstacle in the corridor
    gate_score: float | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["action_proposed"] = self.action_proposed.value
        d["action_taken"] = self.action_taken.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, line: str) -> FrameRecord:
        d = json.loads(line)
        d["action_proposed"] = Action(d["action_proposed"])
        d["action_taken"] = Action(d["action_taken"])
        d["pose"] = tuple(d["pose"])
        d["velocity"] = tuple(d["velocity"])
        return cls(**d)


@runtime_checkable
class FastStack(Protocol):
    """Fast perception, every frame (owner: B)."""

    def perceive(self, rgb: np.ndarray) -> DepthEstimate: ...


@runtime_checkable
class Controller(Protocol):
    """Rule controller: estimated depth to proposed action (owner: B)."""

    def act(self, depth: DepthEstimate, drone_state: DroneState) -> Action: ...


@runtime_checkable
class Gate(Protocol):
    """Scores how much reason there is to ask the VLM; higher = ask (owner: B)."""

    def score(self, features: dict[str, Any], drone_state: DroneState) -> float: ...


@runtime_checkable
class Policy(Protocol):
    """When to ask. Every baseline shares this interface; cooldown lives inside (owner: C)."""

    name: str

    def should_ask(
        self,
        t: float,
        features: dict[str, Any],
        drone_state: DroneState,
        history: Sequence[RuntimeStep],
    ) -> bool: ...


@runtime_checkable
class VLMJudge(Protocol):
    """Slow system (owner: C). Real VLM and surrogate judge both implement this.

    ``image`` is the raw RGB frame, ``path_overlay`` the same frame with the next
    ``T_h`` seconds of flight path drawn, and ``crop`` a full-resolution crop of
    that region.
    """

    def ask(self, image: np.ndarray, path_overlay: np.ndarray, crop: np.ndarray) -> VLMAnswer: ...
