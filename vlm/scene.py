"""Scene read: ask the VLM what is ahead and where, and score it against the true depth (plan v7, step A).

The VLM returns a short list of objects, each with a box (Qwen3-VL's grounding format: x0 y0 x1 y1 on a 0-1000 scale), a kind
and a range bucket. Several objects are several list entries; nothing is an empty list. Boxes are scored against ground truth
built from the simulator's true depth of EVERY surface, placed obstacle or map scenery (a real building or tree is an obstacle,
not a false alarm), plus the obstacle mask for see-through things. Ground truth is offline only (CLAUDE.md rule 5): no prompt,
box rule or controller input here reads it.

Pure numpy; the model call lives in ``vlm.judge`` and the driver in ``vlm.eval_scene``.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

RANGES = ("close", "near", "far")  # under CLOSE_M, CLOSE_M to NEAR_M, over NEAR_M
KINDS = ("wire", "pole", "glass", "wall", "building", "tree", "vehicle", "other")
THIN_KINDS = ("wire", "pole", "glass")
CLOSE_M = 5.0
DEPTH_SAFE_M = 8.0  # the fast stack acts below this; a depth "miss" is an object nearer than this that the estimate shows farther
CORRIDOR_BOX = [450.0, 400.0, 550.0, 600.0]  # the highlighted window of vlm.overlay in 0-1000 units (what a yes/no question is about)
NEAR_M = 15.0
BOX_SCALE = 1000.0
HUGE_AREA = 0.60  # a box covering more of the picture than this is "everything" (all-ones failure)
MAX_NEAR_OBJECTS = 4  # this many near/close boxes in one answer counts as all-ones
PLACED_KIND = {  # frame-bank obstacle family -> coarse kind
    "wire_1cm_axis_z": "wire", "wire_3cm_axis_z": "wire", "pole_10cm_axis_z": "pole", "glass_pane_alpha20": "glass",
    "wall_white_featureless": "wall", "building_block": "building", "box_1m_black": "other", "box_1m_grey": "other",
    "ball_20cm_red": "other"}

GRID_COLS = ("left", "centre", "right")
GRID_ROWS = ("upper", "lower")


def range_bucket(dist_m: float) -> str:
    return "close" if dist_m < CLOSE_M else ("near" if dist_m < NEAR_M else "far")


# ---- what the model is shown ------------------------------------------------------------------------------------


def scene_prompt() -> str:
    return (
        "You are the eyes of a small drone flying straight ahead. List every object that the drone could hit within 15 metres, "
        "including thin wires, poles, glass panes, walls and buildings that fill the view. Ignore the sky and the ground. "
        'Reply with JSON only, a list like [{"box": [x0, y0, x1, y1], "kind": "wire", "range": "near"}]. '
        "The box uses coordinates from 0 to 1000 across the picture (0,0 is the top left corner). "
        f'kind is one of {", ".join(KINDS)}. range is "close" (under 5 m), "near" (5 to 15 m) or "far" (over 15 m). '
        "Reply [] if nothing is within 15 m."
    )


# Prompt v2 (plan v8): say what the drone is and what counts. Distances are NOT asked in metres (the model cannot measure them);
# "near" wording only shifts how readily it says yes. Depth keeps giving the distance wherever it sees the surface.
PATH_PROMPTS = {
    # no range at all: anything in the path counts
    "v2": ("A small drone is flying straight ahead, towards {where}. Is there anything in its path that it could crash into, including thin "
           "wires, poles, glass, or a wall or building that is the same colour as its surroundings? Answer yes or no."),
    # only what is close: the drone could hit it within a few seconds
    "v2near": ("A small drone is flying straight ahead, towards {where}. Is there anything close in its path, within about 10 metres, that it "
               "could crash into, including thin wires, poles, glass, or a wall that is the same colour as its surroundings? "
               "Ignore distant buildings. Answer yes or no."),
    # only the things a depth camera misses
    "v2thin": ("A small drone is flying straight ahead, towards {where}. Is there a thin or hard-to-see object in its path, such as a wire, "
               "cable, pole or pane of glass, or a wall that blends into its surroundings? Answer yes or no."),
}


def path_prompt(key: str, overlay: bool) -> str:
    where = "the highlighted region" if overlay else "the centre of the image"
    return PATH_PROMPTS[key].format(where=where)


def scene_prompt_v2() -> str:
    return (
        "A small drone is flying straight ahead. List every object in its path that it could crash into, including thin wires, poles, "
        "glass panes, and walls or buildings that are the same colour as their surroundings. Ignore the sky and the ground. "
        'Reply with JSON only, a list like [{"box": [x0, y0, x1, y1], "kind": "wire", "range": "near"}]. '
        "The box uses coordinates from 0 to 1000 across the picture (0,0 is the top left corner). "
        f'kind is one of {", ".join(KINDS)}. range is "close" (under 5 m), "near" (5 to 15 m) or "far" (over 15 m). '
        "Reply [] if the path is clear."
    )


def cell_name(row: int, col: int) -> str:
    return f"{GRID_ROWS[row]} {GRID_COLS[col]}"


def grid_prompt(row: int, col: int) -> str:
    return (
        f"Look only at the {cell_name(row, col)} part of the picture. Is there anything, including thin wires, poles, glass, walls or "
        "buildings, within 15 metres of the drone in that part? Answer yes or no."
    )


def cell_box(row: int, col: int) -> list[float]:
    x0, x1 = col * BOX_SCALE / 3, (col + 1) * BOX_SCALE / 3
    y0, y1 = row * BOX_SCALE / 2, (row + 1) * BOX_SCALE / 2
    return [x0, y0, x1, y1]


def grid_objects(scores: dict[str, float], cut: float = 0.0) -> list[dict[str, Any]]:
    """Cells whose yes-score is above ``cut`` become near boxes (the cheap fallback format)."""
    out = []
    for r in range(len(GRID_ROWS)):
        for c in range(len(GRID_COLS)):
            if scores[cell_name(r, c)] > cut:
                out.append({"box": cell_box(r, c), "kind": "other", "range": "near"})
    return out


# ---- reading the answer -----------------------------------------------------------------------------------------


def parse_objects(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Boxes from the model's reply. Returns (objects, ok). Bad boxes are dropped; a reply with no JSON list is not ok.

    Accepts a fenced block, a bare list, or a list inside other words. Coordinates are clipped to 0-1000 and ordered.
    """
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [], False
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], False
    if not isinstance(raw, list):
        return [], False
    out: list[dict[str, Any]] = []
    for o in raw:
        try:
            x0, y0, x1, y1 = (float(v) for v in o["box"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, x1 = sorted((min(max(x0, 0.0), BOX_SCALE), min(max(x1, 0.0), BOX_SCALE)))
        y0, y1 = sorted((min(max(y0, 0.0), BOX_SCALE), min(max(y1, 0.0), BOX_SCALE)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        kind = o.get("kind") if o.get("kind") in KINDS else "other"
        rng = o.get("range") if o.get("range") in RANGES else "near"  # an unreadable range is treated as near: the safe side
        out.append({"box": [x0, y0, x1, y1], "kind": kind, "range": rng})
    return out, True


# ---- ground truth (offline) -----------------------------------------------------------------------------------


def ground_row(alt_m: float, d_m: float, shape: Sequence[int], hfov_deg: float) -> int:
    """First image row that can show flat ground nearer than ``d_m`` for a level camera at ``alt_m`` (rows below it may be ground)."""
    h, w = shape[:2]
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return int(min(h, h / 2.0 + fx * max(alt_m, 0.0) / d_m))


def _components(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    return [lab == i for i in range(1, n + 1) if int((lab == i).sum()) >= min_px]


def _bbox(mask: np.ndarray) -> list[float]:
    rows, cols = np.where(mask)
    h, w = mask.shape
    return [cols.min() / w * BOX_SCALE, rows.min() / h * BOX_SCALE, (cols.max() + 1) / w * BOX_SCALE, (rows.max() + 1) / h * BOX_SCALE]


def truth_objects(depth_gt: np.ndarray, obstacle_mask: np.ndarray, alt_m: float, placed_dist_m: float | None, family: str | None,
                  hfov_deg: float = 90.0, near_m: float = NEAR_M, min_px: int = 20) -> list[dict[str, Any]]:
    """Every real object in the picture: the placed obstacle (from its mask) and each near map surface (from the true depth).

    ``dist`` is the nearest true distance of the object. The placed obstacle keeps ``placed_dist_m`` (the clean distance,
    right for glass and wires the depth image sees through). Scenery is the connected near surface above the ground rows.
    """
    h, w = depth_gt.shape[:2]
    out: list[dict[str, Any]] = []
    obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
    if obstacle_mask.sum() >= 3:
        dist = float(placed_dist_m) if placed_dist_m is not None else float(np.nanmin(np.where(obstacle_mask, depth_gt, np.inf)))
        out.append({"src": "placed", "box": _bbox(obstacle_mask), "dist": dist, "bucket": range_bucket(dist),
                    "kind": PLACED_KIND.get(str(family), "other")})
    ok = np.isfinite(depth_gt) & (depth_gt > 0) & (depth_gt < near_m) & ~obstacle_mask
    ok[ground_row(alt_m, near_m, (h, w), hfov_deg):, :] = False
    for comp in _components(ok, min_px):
        dist = float(np.percentile(depth_gt[comp], 5))
        out.append({"src": "scenery", "box": _bbox(comp), "dist": dist, "bucket": range_bucket(dist), "kind": "scenery"})
    return out


# ---- scoring ----------------------------------------------------------------------------------------------------


def _area(b: Sequence[float]) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i) if i > 0 else 0.0


def _centre_in(box: Sequence[float], inner: Sequence[float]) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def near_pred(pred: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in pred if p["range"] in ("close", "near")]


def is_all_ones(pred: Iterable[dict[str, Any]]) -> bool:
    """The failure DS asked about: the answer flags (nearly) everything."""
    p = near_pred(pred)
    return len(p) >= MAX_NEAR_OBJECTS or any(_area(o["box"]) / BOX_SCALE**2 > HUGE_AREA for o in p)


def score_frame(pred: list[dict[str, Any]], truth: list[dict[str, Any]], near_m: float = NEAR_M) -> dict[str, Any]:
    """Frame-level flag vs truth, object-level found / supported, box overlap and range accuracy."""
    t_near = [t for t in truth if t["dist"] < near_m]
    p_near = near_pred(pred)
    found = []
    for t in t_near:
        match = [p for p in p_near if _centre_in(p["box"], t["box"]) or iou(p["box"], t["box"]) >= 0.3]
        best = max(match, key=lambda p: iou(p["box"], t["box"])) if match else None
        found.append({"src": t["src"], "kind": t["kind"], "found": best is not None, "in_path": _inter(CORRIDOR_BOX, t["box"]) > 0,
                      "iou": iou(best["box"], t["box"]) if best else 0.0,
                      "range_exact": bool(best and best["range"] == t["bucket"]),
                      "depth_misses": bool(t.get("depth_misses")) and t["dist"] < DEPTH_SAFE_M})
    supported = [any(_inter(p["box"], t["box"]) > 0 for t in t_near) for p in p_near]
    return {"positive": bool(t_near), "flagged": bool(p_near), "all_ones": is_all_ones(pred), "objects": found,
            "n_pred_near": len(p_near), "n_supported": int(sum(supported))}


# ---- from boxes to a move (the steering arm) -------------------------------------------------------------------


def boxes_to_move(pred: list[dict[str, Any]], goal_deg: float = 0.0, corridor: tuple[float, float] = (400.0, 600.0),
                  min_gap: float = 150.0, top_margin: float = 20.0, goal_bonus: float = 100.0) -> str:
    """Straight, left, right or over from the near boxes alone (no depth, no ground truth). Box units are 0-1000.

    Blocked width = union of the x-ranges of near/close boxes. Straight when nothing reaches the corridor. Otherwise the free
    span on each side of the blocked group is measured from the corridor edge; the wider side wins (goal side gets a small bonus).
    If both spans are under ``min_gap`` the drone goes over, unless a blocking box touches the top of the picture (then the
    wider side is taken anyway: the top is not free).
    """
    blocks = sorted((o["box"][0], o["box"][2], o["box"][1]) for o in near_pred(pred))
    if not blocks or not any(b0 < corridor[1] and b1 > corridor[0] for b0, b1, _ in blocks):
        return "straight"
    lo, hi = corridor
    changed = True
    while changed:  # grow the blocked group until nothing else touches it
        changed = False
        for b0, b1, _ in blocks:
            if b1 > lo and b0 < hi and (b0 < lo or b1 > hi):
                lo, hi, changed = min(lo, b0), max(hi, b1), True
    left_free, right_free = lo, BOX_SCALE - hi
    left_score = left_free + (goal_bonus if goal_deg < 0 else 0.0)  # the goal side gets a small bonus (negative bearing = left)
    right_score = right_free + (goal_bonus if goal_deg > 0 else 0.0)
    top_blocked = any(y0 <= top_margin for _, _, y0 in blocks)
    if max(left_free, right_free) < min_gap and not top_blocked:
        return "over"
    return "left" if left_score >= right_score else "right"


# ---- summary ----------------------------------------------------------------------------------------------------


def _rate(xs: list[bool]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame- and object-level numbers for the rows of one condition (each row carries its ``score``)."""
    pos = [r for r in rows if r["score"]["positive"]]
    neg = [r for r in rows if not r["score"]["positive"]]
    objs = [o for r in rows for o in r["score"]["objects"]]

    def found(sel: list[dict[str, Any]]) -> float:
        return _rate([o["found"] for o in sel])

    pred_near = sum(r["score"]["n_pred_near"] for r in rows)
    return {
        "n_frames": len(rows), "n_pos": len(pos), "n_neg": len(neg),
        "recall": _rate([r["score"]["flagged"] for r in pos]), "false_alarm": _rate([r["score"]["flagged"] for r in neg]),
        "all_ones": _rate([r["score"]["all_ones"] for r in rows]),
        "parse_fail": _rate([not r.get("ok", True) for r in rows]),
        "object_recall": found(objs), "n_objects": len(objs),
        "recall_thin": found([o for o in objs if o["kind"] in THIN_KINDS]),
        "recall_big": found([o for o in objs if o["kind"] not in THIN_KINDS]),
        "recall_placed": found([o for o in objs if o["src"] == "placed"]),
        "recall_scenery": found([o for o in objs if o["src"] == "scenery"]),
        "recall_depth_missed": found([o for o in objs if o["depth_misses"]]), "n_depth_missed": sum(1 for o in objs if o["depth_misses"]),
        "recall_path": found([o for o in objs if o["in_path"]]), "n_path": sum(1 for o in objs if o["in_path"]),
        "recall_path_thin": found([o for o in objs if o["in_path"] and o["kind"] in THIN_KINDS]),
        "recall_path_depth_missed": found([o for o in objs if o["in_path"] and o["depth_misses"]]),
        "n_path_depth_missed": sum(1 for o in objs if o["in_path"] and o["depth_misses"]),
        "box_precision": (sum(r["score"]["n_supported"] for r in rows) / pred_near) if pred_near else float("nan"),
        "median_iou_found": float(np.median([o["iou"] for o in objs if o["found"]])) if any(o["found"] for o in objs) else float("nan"),
        "range_exact": _rate([o["range_exact"] for o in objs if o["found"]]),
        "latency_median_s": float(np.median([r["latency_s"] for r in rows if r.get("latency_s") is not None])) if any(r.get("latency_s") is not None for r in rows) else float("nan"),
    }


def baseline_rows(truth_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """``oracle`` flags exactly the true near objects; ``all_near`` flags the whole picture. Both must be beaten / matched."""
    out: dict[str, list[dict[str, Any]]] = {"oracle": [], "all_near": []}
    for r in truth_rows:
        t = r["truth"]
        oracle = [{"box": o["box"], "kind": o["kind"], "range": o["bucket"]} for o in t if o["dist"] < NEAR_M]
        for name, pred in (("oracle", oracle), ("all_near", [{"box": [0.0, 0.0, BOX_SCALE, BOX_SCALE], "kind": "other", "range": "close"}])):
            out[name].append({**r, "pred": pred, "ok": True, "latency_s": None, "score": score_frame(pred, t)})
    return out


def fmt(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table(summary: dict[str, dict[str, Any]]) -> str:
    cols = [("n_pos", "pos n"), ("n_neg", "neg n"), ("recall", "recall"), ("false_alarm", "false alarm"), ("all_ones", "all ones"),
            ("parse_fail", "parse fail"), ("recall_path", "in-path found"), ("recall_path_thin", "in-path thin"),
            ("recall_path_depth_missed", "in-path depth-missed"), ("box_precision", "box precision"), ("median_iou_found", "IoU"),
            ("range_exact", "range exact"), ("latency_median_s", "latency s")]
    lines = ["| condition | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---:|" * len(cols)]
    for key, s in summary.items():
        lines.append(f"| {key} | " + " | ".join(str(s[k]) if k in ("n_pos", "n_neg") else fmt(s[k]) for k, _ in cols) + " |")
    return "\n".join(lines) + "\n"
