"""Direction benchmark (L2): which move is safe and goal-aware when an obstacle is ahead? Labels, goal inputs, prompts,
depth baselines and scoring. No model here; ``vlm.eval_direction`` runs the models.

Question. When the gate fires, the slow system may be asked to steer, not only to say yes or no. This module defines
what a correct move is on a saved frame, and how depth-only, VLM-only and combined choices are scored, so "the VLM
steers" can be judged by a number (working plan v3, DS 2026-11-26).

Options (relative to the drone's heading): ``straight`` (nothing to avoid), ``left``, ``right``, ``over`` (climb).

Ground truth (offline only, CLAUDE.md rule 5). The frame bank flies straight at one obstacle: each frame has the
obstacle's distance ahead and lateral offset (``gt_along_m``, ``gt_lateral_m``). The obstacle is taken to be an
axis-aligned box of its catalog size (``sim.scenario.half_extents``) centred at the drone's altitude [UNVERIFIED:
the bank places obstacles at flight height; the glb yaw is not recorded]. As in ``sim.geometry.best_side``, the move
that clears it is the lateral distance to its edge plus the drone radius (left, right) or its half height plus the
radius (over).

- ``avoid``: the straight line hits the box (``|lateral| < half_width + radius``) and it is within ``d_pos``.
- ``straight``: the line misses it, or it is farther than ``d_neg`` (the clear frames that measure false steering).
- anything else is ambiguous and left out, as in M2.4.
- ``valid`` options: those needing at most ``need_cap_m`` of motion (one option, the cheapest, if none do).
- ``best``: the valid option with the smallest ``need - bonus``, the bonus being ``goal_bonus_m`` for the side the
  goal lies on (ties: over, left, right, as in ``best_side``).

Goal. Bank flights have no goal, so each episode gets a synthetic goal bearing (0, +25 or -25 degrees, positive =
right), seeded by the episode id and given to the model only through the prompt and the picture. The frame itself is
unchanged. The goal is ``GOAL_DIST_M`` away.

Scoring, per (model, condition): ``safe`` rate on ``avoid`` frames (choice in ``valid``), false-steer rate on
``straight`` frames, exact match with ``best``, regret in metres (``need[choice] - need[best]``), with 95% intervals
from resampling whole flights.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

OPTIONS = ("straight", "left", "right", "over")
OPTION_FORMS = {
    "straight": ("straight", "Straight", " straight", " Straight"),
    "left": ("left", "Left", " left", " Left"),
    "right": ("right", "Right", " right", " Right"),
    "over": ("over", "Over", " over", " Over"),
}
GOAL_INPUTS = ("none", "arrow", "text", "arrow_text")
IMAGES = ("both", "raw")  # both = overlay with the corridor box + full-resolution crop; raw = the frame alone
GOAL_BEARINGS_DEG = (0.0, 25.0, -25.0)  # positive: the goal lies to the right of the heading
GOAL_DIST_M = 40.0
DRONE_RADIUS_M = 0.3
NEED_CAP_M = 6.0
GOAL_BONUS_M = 3.0
D_POS_M = 15.0
D_NEG_M = 30.0
TIE_ORDER = ("over", "left", "right")  # as sim.geometry.best_side


# ---- ground truth ----------------------------------------------------------------------------------------------


def goal_bearing_deg(episode: int, seed: int = 0) -> float:
    """The synthetic goal bearing of an episode: deterministic in (seed, episode), the same for every frame of it."""
    rng = np.random.default_rng([seed, int(episode)])
    return float(GOAL_BEARINGS_DEG[int(rng.integers(len(GOAL_BEARINGS_DEG)))])


def goal_side(bearing_deg: float) -> str | None:
    return None if bearing_deg == 0 else ("right" if bearing_deg > 0 else "left")


def moves_needed(along_m: float, lateral_m: float, family: str, radius: float = DRONE_RADIUS_M) -> dict[str, float]:
    """Motion (metres) the drone needs to clear the obstacle by going left, right or over it (can be negative: already clear)."""
    from sim.scenario import half_extents

    _, half_lat, half_down = half_extents(family)
    return {"left": half_lat + radius - lateral_m, "right": lateral_m + half_lat + radius, "over": half_down + radius}


def direction_label(frame: dict[str, Any], goal_deg: float, d_pos: float = D_POS_M, d_neg: float = D_NEG_M,
                    need_cap_m: float = NEED_CAP_M, goal_bonus_m: float = GOAL_BONUS_M) -> dict[str, Any]:
    """Label one frame-bank row (``frames.jsonl``): ``{"kind": "avoid"|"straight"|"skip", "valid", "best", "need"}``."""
    along, lateral, family = frame.get("gt_along_m"), frame.get("gt_lateral_m"), str(frame.get("obstacle"))
    if along is None or lateral is None or along <= 0:
        return {"kind": "skip", "valid": [], "best": None, "need": {}}
    need = moves_needed(float(along), float(lateral), family)
    on_course = need["left"] > 0 and need["right"] > 0
    if not on_course or along > d_neg:
        return {"kind": "straight", "valid": ["straight"], "best": "straight", "need": need}
    if along > d_pos:
        return {"kind": "skip", "valid": [], "best": None, "need": need}
    valid = [k for k in ("left", "right", "over") if need[k] <= need_cap_m] or [min(need, key=need.get)]
    side = goal_side(goal_deg)
    best = min(valid, key=lambda k: (need[k] - (goal_bonus_m if k == side else 0.0), TIE_ORDER.index(k)))
    return {"kind": "avoid", "valid": valid, "best": best, "need": need}


# ---- what the model is shown -------------------------------------------------------------------------------------


def goal_text(bearing_deg: float, dist_m: float = GOAL_DIST_M) -> str:
    """The goal as numbers in the drone's own frame (x forward, y right, metres), drone at the origin."""
    th = math.radians(bearing_deg)
    return (f"Your position is (0, 0) and you face along +x (x forward, y to the right). "
            f"The goal is at ({dist_m * math.cos(th):.0f}, {dist_m * math.sin(th):.0f}).")


def draw_goal_marker(rgb: np.ndarray, bearing_deg: float, hfov_deg: float = 90.0) -> np.ndarray:
    """Copy of ``rgb`` with a green marker above the horizon at the goal's bearing (pinhole camera, ``hfov_deg``)."""
    h, w = rgb.shape[:2]
    x = w / 2 + (w / 2) * math.tan(math.radians(bearing_deg)) / math.tan(math.radians(hfov_deg) / 2)
    x = float(min(max(x, 12.0), w - 12.0))
    im = Image.fromarray(np.ascontiguousarray(rgb))
    d = ImageDraw.Draw(im)
    top, size = 0.12 * h, max(8.0, 0.03 * w)
    d.polygon([(x, top + 2 * size), (x - size, top), (x + size, top)], fill=(0, 255, 0), outline=(0, 90, 0))
    d.text((x - 14, top + 2 * size + 2), "GOAL", fill=(0, 255, 0))
    return np.asarray(im)


def direction_prompt(goal_input: str, overlay: bool) -> str:
    """The question for one goal input. ``overlay``: whether the corridor box is drawn (else "the centre of the image")."""
    if goal_input not in GOAL_INPUTS:
        raise ValueError(f"goal_input must be one of {GOAL_INPUTS}, got {goal_input!r}")
    where = "the highlighted region" if overlay else "the centre of the image"
    parts = [f"A drone is flying straight ahead towards {where}."]
    if goal_input in ("arrow", "arrow_text"):
        parts.append("The green GOAL marker shows the direction of its goal.")
    if goal_input in ("text", "arrow_text"):
        parts.append("{goal_text}")
    parts.append("If nothing blocks the way, answer straight. Otherwise choose the safest move past the obstacle, "
                 "left, right or over (climb)" + (", preferring the side towards the goal when it is safe" if goal_input != "none" else "")
                 + ". Answer with one word: straight, left, right or over.")
    return " ".join(parts)


def render_inputs(rgb: np.ndarray, goal_input: str, goal_deg: float, image: str, hfov_deg: float = 90.0) -> tuple[list[np.ndarray], str]:
    """``(images, prompt)`` for one condition: the pictures to show and the filled-in question."""
    from vlm.overlay import render

    if image not in IMAGES:
        raise ValueError(f"image must be one of {IMAGES}, got {image!r}")
    overlay, crop = render(rgb)
    first = overlay if image == "both" else rgb
    if goal_input in ("arrow", "arrow_text"):
        first = draw_goal_marker(first, goal_deg, hfov_deg)
    prompt = direction_prompt(goal_input, overlay=image == "both").format(goal_text=goal_text(goal_deg))
    return ([first, crop] if image == "both" else [first]), prompt


def option_logprobs(logits: np.ndarray, ids: dict[str, Sequence[int]]) -> dict[str, float]:
    """Log-probabilities over the four options, each pooled over its single-token spellings and renormalised among the four."""
    raw = {k: _logsumexp(np.asarray(logits, dtype=np.float64)[list(v)]) for k, v in ids.items()}
    z = _logsumexp(np.array(list(raw.values())))
    return {k: float(v - z) for k, v in raw.items()}


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


# ---- depth baselines and combinations -----------------------------------------------------------------------------


def controller_choice(depth: np.ndarray, goal_deg: float, mode: str = "choose", alt_m: float = 3.0, d_safe: float = 8.0) -> str:
    """What the rule controller (``control.rules``) does on one depth map: straight, left, right or over.

    The controller is the depth-only baseline, unchanged. It starts at the origin facing +x with the goal
    ``GOAL_DIST_M`` away at ``goal_deg``; "straight" is any step that is not avoidance (cruise or turning to the goal).
    """
    from control.rules import RuleController
    from interfaces import Action, DepthEstimate, DroneState
    from perception.depth_utils import corridor_features

    c = RuleController(d_safe=d_safe, avoid_mode=mode)
    th = math.radians(goal_deg)
    c.set_mission((GOAL_DIST_M * math.cos(th), GOAL_DIST_M * math.sin(th)), cruise_alt=alt_m, land_alt=1.0)
    act = c.act(DepthEstimate(depth=depth, features=corridor_features(depth)), DroneState(t=0.0, position=(0.0, 0.0, -alt_m), yaw=0.0, speed=0.0))
    if c.phase != "avoid":
        return "straight"
    return {Action.YAW_LEFT: "left", Action.YAW_RIGHT: "right", Action.CLIMB: "over"}[act]


def depth_bands(depth: np.ndarray) -> dict[str, float]:
    """Free distance left, right and over (``control.rules.free_space``) and the corridor minimum, for the combinations."""
    from control.rules import free_space
    from perception.depth_utils import corridor_min

    return {**free_space(depth), "corridor_min": float(corridor_min(depth))}


def depth_veto(vlm_choice: str, bands: dict[str, float], fallback: str, d_safe: float = 8.0) -> str:
    """The VLM's move unless depth positively reads that side as blocked; then the depth choice. Depth reading "clear" never vetoes."""
    if vlm_choice in ("left", "right", "over") and bands[vlm_choice] < d_safe:
        return fallback
    return vlm_choice


# ---- scoring ---------------------------------------------------------------------------------------------------


def judge_choice(row: dict[str, Any]) -> dict[str, Any]:
    """Score one row that has ``choice`` and the label fields (``kind``, ``valid``, ``best``, ``need``)."""
    choice, kind = row["choice"], row["kind"]
    safe = choice in row["valid"]
    regret = None
    if kind == "avoid" and choice in row["need"]:
        regret = row["need"][choice] - row["need"][row["best"]]
    return {"safe": bool(safe), "exact": choice == row["best"], "regret": regret}


def _flight_ci(by_flight: dict[Any, list[float]], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    flights = [np.asarray(v, dtype=np.float64) for v in by_flight.values() if len(v)]
    if len(flights) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(len(flights), size=len(flights))
        stats.append(float(np.concatenate([flights[i] for i in pick]).mean()))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per ``model|condition``: safe rate on avoid frames (with flight-level CI), false-steer rate on straight frames, exact, regret, by family."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"{r['model']}|{r['condition']}"].append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, rs in groups.items():
        av, st = [r for r in rs if r["kind"] == "avoid"], [r for r in rs if r["kind"] == "straight"]
        scored = [(r, judge_choice(r)) for r in av]
        safe_by_flight: dict[int, list[float]] = defaultdict(list)
        for r, j in scored:
            safe_by_flight[int(r["episode"])].append(float(j["safe"]))
        fam: dict[str, dict[str, float]] = {}
        for name in sorted({r["obstacle"] for r in av}):
            sub = [j for r, j in scored if r["obstacle"] == name]
            fam[name] = {"n": len(sub), "safe": float(np.mean([j["safe"] for j in sub]))}
        regrets = [j["regret"] for _, j in scored if j["regret"] is not None]
        out[key] = {
            "n_avoid": len(av), "n_straight": len(st),
            "safe": float(np.mean([j["safe"] for _, j in scored])) if scored else float("nan"),
            "safe_ci": _flight_ci(safe_by_flight),
            "exact": float(np.mean([j["exact"] for _, j in scored])) if scored else float("nan"),
            "regret_m": float(np.mean(regrets)) if regrets else float("nan"),
            "false_steer": float(np.mean([r["choice"] != "straight" for r in st])) if st else float("nan"),
            "by_family": fam,
        }
    return out


def table(summary: dict[str, dict[str, Any]]) -> str:
    lines = ["| model | condition | avoid n | safe [95% CI] | exact | regret m | straight n | false steer |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for key, s in sorted(summary.items()):
        model, cond = key.split("|", 1)
        lo, hi = s["safe_ci"]
        lines.append(f"| {model} | {cond} | {s['n_avoid']} | {s['safe']:.2f} [{lo:.2f}, {hi:.2f}] | {s['exact']:.2f} | "
                     f"{s['regret_m']:.2f} | {s['n_straight']} | {s['false_steer']:.2f} |")
    return "\n".join(lines) + "\n"
