"""Seeded scenarios and the frozen seed lists (PLAN_DETAILED.md D6, D13, D15; chunk M4.3).

A scenario is everything a flight needs and nothing else: start pose, goal
distance, cruise speed, weather condition, and the obstacles as oriented boxes.
The same seed always produces the same scenario, on any machine, so every
policy flies the same scenarios (D15) and a flight can be replayed exactly.

Scenarios live in a local "path frame": start at the origin, heading +x
(north), altitude ``alt`` (z = -alt). A simulator backend places that frame at
a world start pose of its choice.

Obstacle sizes come from the catalog the frame bank used
(hpc/capability/capability_test.py CATALOG, glTF sizes ``(dx, dy, dz)``). The
glTF axes map to the box's forward, up and lateral axes as observed in the bank
(``GLTF_AXES``): the "axis_z" poles and wires render as horizontal bars across
the view, and the glass pane faces the camera. That mapping is [UNVERIFIED]
until M4.1 measures it; ``hpc/m4_checks.py`` does.

Seed lists: ``write_seed_lists`` writes ``configs/seeds.yaml`` once, with a
checksum, and refuses to overwrite it (D13: frozen at the simulator go/no-go).
Runs always take the first n seeds of a list, never a hand-picked subset (D24).

Scenario settings are a second, later freeze (D24): the seed lists hold only
seed numbers, so the difficulty knobs (``ScenarioConfig``) and the start poses
stay tunable on val seeds until the never-ask headroom check passes (M4.7).
``write_scenario_config`` then writes ``configs/scenario.yaml`` once, with a
checksum, and ``eval.sweep`` refuses to fly test seeds from anything else.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from sim.geometry import OBB

# glTF axis index that ends up along the box's (forward, lateral, up) axes. [UNVERIFIED, see M4.1]
GLTF_AXES = {"forward": 0, "lateral": 2, "up": 1}

# name -> (glTF size dx, dy, dz in metres, RGB 0..1 for the Plan B renderer, see-through)
CATALOG: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], bool]] = {
    "pole_10cm_axis_z": ((0.1, 0.1, 5.0), (0.5, 0.5, 0.5), False),
    "pole_10cm_axis_y": ((0.1, 5.0, 0.1), (0.5, 0.5, 0.5), False),
    "wire_3cm_axis_z": ((0.03, 0.03, 6.0), (0.05, 0.05, 0.05), False),
    "wire_1cm_axis_z": ((0.01, 0.01, 6.0), (0.05, 0.05, 0.05), False),
    "box_1m_grey": ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), False),
    "box_1m_black": ((1.0, 1.0, 1.0), (0.05, 0.05, 0.05), False),
    "wall_white_featureless": ((0.2, 8.0, 3.0), (0.95, 0.95, 0.95), False),
    "glass_pane_alpha20": ((0.05, 3.0, 2.5), (0.7, 0.85, 0.95), True),
    "building_block": ((12.0, 20.0, 25.0), (0.6, 0.45, 0.4), False),
    "ball_20cm_red": ((0.2, 0.2, 0.2), (0.9, 0.1, 0.1), False),
}
DEFAULT_FAMILIES = (
    "pole_10cm_axis_z", "wire_3cm_axis_z", "wire_1cm_axis_z", "box_1m_grey", "box_1m_black",
    "wall_white_featureless", "glass_pane_alpha20", "ball_20cm_red",
)


def half_extents(family: str) -> tuple[float, float, float]:
    """(forward, lateral, down) half sizes of a catalog obstacle."""
    size = CATALOG[family][0]
    return (size[GLTF_AXES["forward"]] / 2, size[GLTF_AXES["lateral"]] / 2, size[GLTF_AXES["up"]] / 2)


@dataclass
class Obstacle:
    family: str
    along_m: float  # path-frame x of the centre (distance ahead of the start)
    lateral_m: float  # path-frame y of the centre, signed to the right (east when heading north)
    alt_m: float  # centre altitude
    yaw: float = 0.0  # radians; "cone" courses turn the box to face along the start-to-goal line

    def obb(self, name: str = "") -> OBB:
        _, _, see_through = CATALOG[self.family]
        return OBB(center=(self.along_m, self.lateral_m, -self.alt_m), half=half_extents(self.family), yaw=self.yaw,
                   name=name or self.family, family=self.family, see_through=see_through)


@dataclass
class Scenario:
    seed: int
    split: str
    alt_m: float
    speed_mps: float
    goal_m: float
    condition: str
    obstacles: list[Obstacle] = field(default_factory=list)
    # "cone" courses only (None on "line" courses): the goal point in the path frame, and the height the drone
    # descends to there. On a cone course ``goal_m`` is the straight-line distance from the start to the goal.
    goal_xy: tuple[float, float] | None = None
    land_alt_m: float | None = None
    area_half_width_m: float | None = None  # mission area: this far either side of the start-to-goal line (geofence)

    def obbs(self) -> list[OBB]:
        return [o.obb(f"obs{i}_{o.family}") for i, o in enumerate(self.obstacles)]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        goal = d.get("goal_xy")
        return cls(**{**d, "obstacles": [Obstacle(**o) for o in d["obstacles"]], "goal_xy": tuple(goal) if goal is not None else None})


@dataclass
class ScenarioConfig:
    """Difficulty knobs. Tune on val so never-ask crashes in 15 to 30% of flights (PLAN_DETAILED.md §4.2)."""

    families: Sequence[str] = DEFAULT_FAMILIES
    holdout_family: str | None = None  # left out of train seeds (E6)
    goal_m: float = 60.0
    alt_range_m: tuple[float, float] = (1.5, 3.0)
    speed_mps: float = 5.0
    n_obstacles: tuple[int, int] = (1, 2)
    first_obstacle_m: tuple[float, float] = (15.0, 25.0)
    gap_m: tuple[float, float] = (12.0, 20.0)
    p_in_path: float = 0.8
    lateral_sd_m: float = 0.3  # spread of in-path obstacles
    off_path_m: tuple[float, float] = (3.0, 8.0)
    conditions: Sequence[str] = ("clear", "dusk")
    # Course layout. "line": the original course, goal = travel goal_m along +x, obstacles from first_obstacle_m
    # with gaps gap_m. "cone": a goal point drawn ahead of the start (DS 2026-11-25, PLAN_SCENARIO_FREEZE.md v7);
    # n_obstacles obstacles spread along the start-to-goal line, none within the last land_clear_m metres.
    layout: str = "line"
    goal_dist_m: tuple[float, float] = (50.0, 60.0)  # cone: straight-line distance start to goal
    goal_lateral_m: float = 10.0  # cone: goal at most this far to either side of the start heading
    along_min_m: float = 12.0  # cone: no obstacle closer to the start than this, along the line
    min_gap_m: float = 8.0  # cone: spacing between obstacles along the line (room for an ask plus avoidance)
    land_clear_m: float = 8.0  # cone: no obstacle within this distance of the goal along the line
    land_alt_m: float = 1.0  # cone: the drone descends to this height at the goal
    area_half_width_m: float = 15.0  # cone: mission area (geofence) either side of the start-to-goal line; tunable


def generate(seed: int, split: str, cfg: ScenarioConfig | None = None) -> Scenario:
    """The scenario for ``seed``. Pure function of (seed, split, cfg)."""
    cfg = cfg or ScenarioConfig()
    if cfg.layout == "cone":
        return _generate_cone(seed, split, cfg)
    if cfg.layout != "line":
        raise ValueError(f"unknown layout {cfg.layout!r}; use 'line' or 'cone'")
    rng = np.random.default_rng([seed, 7919])
    families = [f for f in cfg.families if not (split == "train" and f == cfg.holdout_family)]
    alt = float(rng.uniform(*cfg.alt_range_m))
    n = int(rng.integers(cfg.n_obstacles[0], cfg.n_obstacles[1] + 1))
    along = float(rng.uniform(*cfg.first_obstacle_m))
    obstacles = []
    for _ in range(n):
        if along > cfg.goal_m - 8.0:
            break
        fam = str(families[int(rng.integers(len(families)))])
        if rng.random() < cfg.p_in_path:
            lateral = float(rng.normal(0.0, cfg.lateral_sd_m))
        else:
            lateral = float(rng.choice([-1, 1]) * rng.uniform(*cfg.off_path_m))
        obstacles.append(Obstacle(fam, round(along, 3), round(lateral, 3), round(alt, 3)))
        along += float(rng.uniform(*cfg.gap_m))
    cond = str(cfg.conditions[int(rng.integers(len(cfg.conditions)))])
    return Scenario(seed=seed, split=split, alt_m=round(alt, 3), speed_mps=cfg.speed_mps, goal_m=cfg.goal_m, condition=cond, obstacles=obstacles)


def _generate_cone(seed: int, split: str, cfg: ScenarioConfig) -> Scenario:
    """A goal point ahead of the start and 1 to n obstacles spread along the start-to-goal line.

    Obstacle positions are drawn uniformly with a guaranteed minimum gap (sorted uniform draws on the span left
    after the gaps are taken out). If the line is too short for the drawn count at that gap, the count is reduced:
    ``len(scenario.obstacles)`` is the realised count. In-path obstacles sit on the line (spread ``lateral_sd_m``),
    the others ``off_path_m`` to one side of it; every box is turned to face along the line.
    """
    rng = np.random.default_rng([seed, 7919, 2])  # own stream: line courses for the same seed stay unchanged
    families = [f for f in cfg.families if not (split == "train" and f == cfg.holdout_family)]
    alt = float(rng.uniform(*cfg.alt_range_m))
    dist = float(rng.uniform(*cfg.goal_dist_m))
    lat = float(rng.uniform(-cfg.goal_lateral_m, cfg.goal_lateral_m))
    lat = max(-dist + 1e-6, min(dist - 1e-6, lat))
    gx, gy = float(np.sqrt(dist ** 2 - lat ** 2)), lat
    ux, uy = gx / dist, gy / dist  # unit vector along the line; (-uy, ux) points to its right
    heading = float(np.arctan2(uy, ux))
    lo, hi = cfg.along_min_m, dist - cfg.land_clear_m
    n = int(rng.integers(cfg.n_obstacles[0], cfg.n_obstacles[1] + 1))
    if hi < lo:
        n = 0
    else:
        n = min(n, 1 + int((hi - lo) // cfg.min_gap_m))
    slack = (hi - lo) - (n - 1) * cfg.min_gap_m if n else 0.0
    alongs = [lo + float(u) + i * cfg.min_gap_m for i, u in enumerate(np.sort(rng.uniform(0.0, slack, n)))] if n else []
    obstacles = []
    for t in alongs:
        fam = str(families[int(rng.integers(len(families)))])
        if rng.random() < cfg.p_in_path:
            off = float(rng.normal(0.0, cfg.lateral_sd_m))
        else:
            off = float(rng.choice([-1, 1]) * rng.uniform(*cfg.off_path_m))
        x, y = t * ux - off * uy, t * uy + off * ux
        obstacles.append(Obstacle(fam, round(x, 3), round(y, 3), round(alt, 3), round(heading, 5)))
    cond = str(cfg.conditions[int(rng.integers(len(cfg.conditions)))])
    return Scenario(seed=seed, split=split, alt_m=round(alt, 3), speed_mps=cfg.speed_mps, goal_m=round(dist, 3), condition=cond,
                    obstacles=obstacles, goal_xy=(round(gx, 3), round(gy, 3)), land_alt_m=cfg.land_alt_m,
                    area_half_width_m=cfg.area_half_width_m)


# ---- frozen seed lists (D13) -----------------------------------------------------------------


def seed_lists(n_train: int = 1000, n_val: int = 500, n_test: int = 500, base: int = 100_000) -> dict[str, list[int]]:
    """Disjoint, contiguous seed ranges. Contiguous on purpose: nothing about a seed's value encodes its split."""
    return {
        "train": list(range(base, base + n_train)),
        "val": list(range(base + n_train, base + n_train + n_val)),
        "test": list(range(base + n_train + n_val, base + n_train + n_val + n_test)),
    }


def seeds_checksum(lists: dict[str, list[int]]) -> str:
    return hashlib.sha256(json.dumps({k: lists[k] for k in ("train", "val", "test")}, separators=(",", ":")).encode()).hexdigest()


def write_seed_lists(path: str | Path, lists: dict[str, list[int]]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists and is frozen (PLAN_DETAILED.md D13); delete it by hand only with a Decision-log row")
    check = seeds_checksum(lists)
    path.write_text(yaml.safe_dump({"checksum": check, **{k: v for k, v in lists.items()}}, default_flow_style=None, width=100))
    return check


def load_seed_lists(path: str | Path) -> dict[str, list[int]]:
    d = yaml.safe_load(Path(path).read_text())
    lists = {k: d[k] for k in ("train", "val", "test")}
    if seeds_checksum(lists) != d["checksum"]:
        raise ValueError(f"{path}: seed lists do not match the stored checksum (edited after freezing?)")
    seen: dict[int, str] = {}
    for k, v in lists.items():
        for s in v:
            if s in seen:
                raise ValueError(f"seed {s} in both {seen[s]} and {k}")
            seen[s] = k
    return lists


# ---- frozen scenario settings (D24, the second freeze) -----------------------------------------


def _scenario_checksum(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_scenario_config(path: str | Path, scenario: dict[str, Any], starts: Sequence[Sequence[float]] | None = None) -> str:
    """Freeze the scenario settings and start poses. Refuses to overwrite.

    Every ``ScenarioConfig`` field is written out, defaults included, so a later change to a default
    in the code cannot silently change the frozen scenarios.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists and is frozen (PLAN_DETAILED.md D24); delete it by hand only with a Decision-log row")
    body = {"scenario": json.loads(json.dumps(asdict(ScenarioConfig(**scenario)))),
            "starts": [[float(v) for v in s] for s in starts] if starts else None}
    check = _scenario_checksum(body)
    path.write_text(yaml.safe_dump({"checksum": check, **body}, default_flow_style=None, width=100, sort_keys=False))
    return check


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    """``{"scenario": ScenarioConfig fields, "starts": poses or None}`` from a frozen file; checks the checksum."""
    d = yaml.safe_load(Path(path).read_text())
    body = {"scenario": d["scenario"], "starts": d.get("starts")}
    if _scenario_checksum(body) != d["checksum"]:
        raise ValueError(f"{path}: scenario settings do not match the stored checksum (edited after freezing?)")
    ScenarioConfig(**body["scenario"])  # unknown fields fail here, not mid-sweep
    return body


def main(argv: list[str] | None = None) -> None:
    """Freeze the seed lists or the scenario settings, or print a scenario.

        python -m sim.scenario --write-seeds configs/seeds.yaml      # once, at the simulator go/no-go (D13)
        python -m sim.scenario --freeze-scenario configs/sweeps/airsim_true_depth.yaml --to configs/scenario.yaml   # once, after M4.7 (D24)
        python -m sim.scenario --show 101000
    """
    import argparse

    p = argparse.ArgumentParser(description="seed lists and scenarios")
    p.add_argument("--write-seeds", help="path to write the frozen seed lists to (refuses to overwrite)")
    p.add_argument("--n", nargs=3, type=int, default=[1000, 500, 500], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--freeze-scenario", metavar="SWEEP_YAML", help="freeze this sweep config's scenario settings and start poses (refuses to overwrite)")
    p.add_argument("--to", default="configs/scenario.yaml", help="where --freeze-scenario writes")
    p.add_argument("--show", type=int, help="print the scenario for this seed")
    p.add_argument("--split", default="val")
    a = p.parse_args(argv)
    if a.write_seeds:
        print(f"wrote {a.write_seeds}, checksum {write_seed_lists(a.write_seeds, seed_lists(*a.n))[:12]}")
    if a.freeze_scenario:
        cfg = yaml.safe_load(Path(a.freeze_scenario).read_text())
        if not isinstance(cfg.get("scenario", {}), dict) or "file" in cfg.get("scenario", {}):
            raise SystemExit(f"{a.freeze_scenario}: 'scenario' must be inline settings to freeze, not a file reference")
        check = write_scenario_config(a.to, cfg.get("scenario", {}), cfg.get("backend_kw", {}).get("starts"))
        print(f"wrote {a.to}, checksum {check[:12]}")
    if a.show is not None:
        print(generate(a.show, a.split).to_json())


if __name__ == "__main__":
    main()
