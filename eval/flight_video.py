"""Video of one flight: camera, fast-stack depth, true depth and a top-down map, frame by frame.

Flies one seed with one policy from a sweep config (same simulator, stack, judge, controller and scenario
settings), saves every frame, then composes a video. Each video frame shows:

- top left: the camera image with the corridor window the controller and the VLM look at;
- top right: the fast stack's depth estimate (what the drone believes); bottom left: the true depth;
- bottom right: the course from above (obstacles, goal, mission area, track so far, heading);
- a status line: time, phase, action, the two corridor distances, and every VLM call with its answer and direction.
  A VLM call freezes the picture for the charged delay, so the cost of asking is visible.

    python -m eval.flight_video --config configs/sweeps/airsim_cone_smoke.yaml --seed 101060 --policy always --out results/videos/101060_always

Writes ``flight.mp4`` with ffmpeg if it is installed, else ``flight.gif``. ``--render-only`` rebuilds the video from
an earlier run's folder. Ground truth is drawn for the viewer only; the flight itself never sees it (rule 5).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from vlm.overlay import corridor_box, draw_overlay

TILE_W, TILE_H, BAR_H = 400, 225, 54
MAX_DEPTH = 40.0


def depth_rgb(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """Near = warm, far = dark blue; sky and beyond ``max_depth`` = darkest."""
    d = np.clip(np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=max_depth, posinf=max_depth), 0, max_depth) / max_depth
    near = 1.0 - d
    r = np.clip(1.6 * near - 0.2, 0, 1)
    g = np.clip(1.6 * near - 0.6, 0, 1) * 0.9 + 0.1 * near
    b = np.clip(0.35 + 0.65 * d * (1 - d) * 4, 0, 1) * (0.4 + 0.6 * near)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _tile(img: np.ndarray, label: str) -> Image.Image:
    t = Image.fromarray(img).resize((TILE_W, TILE_H), Image.NEAREST)
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, 8 + 6 * len(label), 14], fill=(0, 0, 0))
    d.text((4, 1), label, fill=(255, 255, 255))
    return t


def _map_tile(scenario: dict[str, Any], track: list[tuple[float, float, float]], k: int) -> Image.Image:
    from sim.scenario import Scenario

    sc = Scenario.from_dict(scenario)
    im = Image.new("RGB", (TILE_W, TILE_H), (24, 26, 30))
    d = ImageDraw.Draw(im)
    x_min, x_max, y_half = -3.0, 66.0, 22.0
    s = min((TILE_W - 10) / (x_max - x_min), (TILE_H - 20) / (2 * y_half))

    def p(x: float, y: float) -> tuple[float, float]:
        return 5 + (x - x_min) * s, TILE_H / 2 + y * s  # right of the start heading is down

    if sc.goal_xy is not None and sc.area_half_width_m:
        gx, gy = sc.goal_xy
        n = math.hypot(gx, gy)
        nx, ny = -gy / n * sc.area_half_width_m, gx / n * sc.area_half_width_m
        for sgn in (1, -1):
            d.line([p(sgn * nx, sgn * ny), p(gx * 1.1 + sgn * nx, gy * 1.1 + sgn * ny)], fill=(70, 90, 120), width=1)
        d.ellipse([*p(gx - 2, gy - 2), *p(gx + 2, gy + 2)], outline=(90, 200, 100), width=2)
    for b in sc.obbs():
        c, sn = math.cos(b.yaw), math.sin(b.yaw)
        hx, hy = max(b.half[0], 0.15), max(b.half[1], 0.15)
        pts = [(b.center[0] + c * ax * hx - sn * ay * hy, b.center[1] + sn * ax * hx + c * ay * hy) for ax, ay in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        d.polygon([p(*q) for q in pts], fill=(120, 170, 220) if b.see_through else (230, 140, 60))
    if len(track) > 1:
        d.line([p(x, y) for x, y, _ in track[: k + 1]], fill=(250, 250, 250), width=2)
    x, y, yaw = track[k]
    d.polygon([p(x + 1.8 * math.cos(yaw), y + 1.8 * math.sin(yaw)), p(x + 0.9 * math.cos(yaw + 2.5), y + 0.9 * math.sin(yaw + 2.5)),
               p(x + 0.9 * math.cos(yaw - 2.5), y + 0.9 * math.sin(yaw - 2.5))], fill=(255, 80, 80))
    d.text((4, 1), "top view (orange obstacle, blue glass, green goal)", fill=(255, 255, 255))
    return im


def compose(rec: dict[str, Any], scenario: dict[str, Any], track: list, k: int, phase: str, vlm: str, alt: float) -> Image.Image:
    rgb = np.asarray(Image.open(rec["rgb_path"]).convert("RGB"))
    rgb = draw_overlay(rgb, corridor_box(rgb.shape), fill_alpha=0.0)
    est = np.load(rec["depth_est_path"]).astype(np.float32)
    gt = np.load(rec["depth_gt_path"]).astype(np.float32)
    frame = Image.new("RGB", (2 * TILE_W, 2 * TILE_H + BAR_H), (10, 10, 12))
    frame.paste(_tile(rgb, "camera (red: corridor window)"), (0, 0))
    frame.paste(_tile(depth_rgb(est), "fast stack depth (what the drone believes)"), (TILE_W, 0))
    frame.paste(_tile(depth_rgb(gt), "true depth (viewer only)"), (0, TILE_H))
    frame.paste(_map_tile(scenario, track, k), (TILE_W, TILE_H))
    d = ImageDraw.Draw(frame)
    act = rec["action_taken"]
    line1 = (f"t {rec['t']:5.1f} s   height {alt:4.1f} m   phase {phase:<8}   action {act:<9}   "
             f"corridor: estimate {rec['corridor_min_est']:5.1f} m, true {rec['corridor_min_gt']:5.1f} m")
    d.text((8, 2 * TILE_H + 6), line1, fill=(235, 235, 235))
    d.text((8, 2 * TILE_H + 26), vlm, fill=(255, 210, 90) if vlm else (150, 150, 150))
    return frame


def render(folder: Path, fps: int = 10) -> Path:
    recs = [json.loads(line) for line in (folder / "flight.jsonl").read_text().splitlines() if line.strip()]
    result = json.loads((folder / "result.json").read_text())
    scenario = json.loads((folder / "scenario.json").read_text())
    extra = result.get("extra", {})
    phases = extra.get("phases") or []
    directions = list(extra.get("vlm_directions") or [])
    track = [(r["pose"][0], r["pose"][1], r["pose"][3]) for r in recs]
    frames_dir = folder / "video_frames"
    frames_dir.mkdir(exist_ok=True)
    n, ask_i, last_vlm = 0, 0, ""
    for k, rec in enumerate(recs):
        phase = "-"
        for t, ph in phases:
            if t <= rec["t"] + 1e-6:
                phase = ph
        repeat = 1
        if rec["asked"]:
            direction = directions[ask_i] if ask_i < len(directions) else None
            ask_i += 1
            ans = "YES" if rec["vlm_answer"] else "no"
            last_vlm = f"VLM asked at t {rec['t']:.1f} s: {ans}" + (f", go {direction}" if direction else "") + \
                       f"   (drone holds {rec['delay_charged_s']:.1f} s)"
            repeat = 1 + int(round(rec["delay_charged_s"] * fps))  # the pause the call costs
        img = compose(rec, scenario, track, k, phase, last_vlm, -rec["pose"][2])
        for _ in range(repeat):
            img.save(frames_dir / f"{n:05d}.png")
            n += 1
    end = Image.open(frames_dir / f"{n - 1:05d}.png")
    ImageDraw.Draw(end).text((8, 2 * TILE_H + 40), f"END: {extra.get('outcome') or ('crashed' if result['crashed'] else 'reached goal' if result['reached_goal'] else 'stopped')}"
                             f"   VLM calls {result['n_calls']}   time {result['mission_time_s']:.1f} s", fill=(255, 120, 120))
    for _ in range(2 * fps):
        end.save(frames_dir / f"{n:05d}.png")
        n += 1
    if shutil.which("ffmpeg"):
        out = folder / "flight.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)], check=True)
    else:
        out = folder / "flight.gif"
        imgs = [Image.open(frames_dir / f"{i:05d}.png") for i in range(0, n, 2)]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
    return out


def fly(cfg: dict[str, Any], seed: int, policy_name: str, folder: Path) -> None:
    from control.rules import RuleController
    from eval.factory import fork_judge, make_judge, make_sim, make_stack
    from eval.run_episode import LoopConfig, run_episode
    from eval.sweep import resolve_scenario
    from policies.base import make_policy
    from sim.scenario import generate

    scen_cfg, frozen_starts = resolve_scenario(cfg)
    backend_kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg.get("backend_kw", {}).items()}
    if frozen_starts and cfg.get("backend") == "airsim":
        backend_kw["starts"] = frozen_starts
    spec = next((dict(p) for p in cfg["policies"] if p["name"] == policy_name), None)
    if spec is None:
        raise SystemExit(f"policy {policy_name!r} is not in the config; choose one of {[p['name'] for p in cfg['policies']]}")
    spec.pop("name")
    policy = make_policy(spec)
    policy.name = policy_name
    sim = make_sim(cfg.get("backend", "raycast"), **backend_kw)
    stack = make_stack(cfg.get("stack", "true-depth"), **cfg.get("stack_kw", {}))
    judge, judge_name = make_judge(cfg.get("judge", "none"), **cfg.get("judge_kw", {}))
    loop = LoopConfig(delay_s=cfg.get("delay_s", 1.0), **{**cfg.get("loop", {}), "save_dir": str(folder / "raw")})
    scenario = generate(seed, cfg.get("split", "val"), scen_cfg)
    try:
        records, result = run_episode(scenario, sim, stack, RuleController(**{"d_safe": loop.d_safe, **cfg.get("controller", {})}),
                                      policy, fork_judge(judge, seed, 0), loop, judge_name=judge_name)
    finally:
        if hasattr(sim, "close"):
            sim.close()
    (folder / "flight.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
    (folder / "result.json").write_text(result.to_json())
    (folder / "scenario.json").write_text(scenario.to_json())
    print(result.to_json(), flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", help="sweep config (simulator, stack, judge, controller, scenario)")
    p.add_argument("--seed", type=int)
    p.add_argument("--policy", default="never", help="a policy name from the config")
    p.add_argument("--out", required=True, help="folder for frames, logs and the video")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a config key, dotted for nested ones, e.g. --set stack=true-depth")
    p.add_argument("--render-only", action="store_true", help="rebuild the video from --out without flying")
    p.add_argument("--fps", type=int, default=10, help="10 = real time at dt 0.1 s")
    a = p.parse_args(argv)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    if not a.render_only:
        if a.config is None or a.seed is None:
            raise SystemExit("--config and --seed are needed to fly")
        from eval.sweep import apply_overrides
        fly(apply_overrides(yaml.safe_load(Path(a.config).read_text()), a.overrides), a.seed, a.policy, folder)
    print(f"video: {render(folder, a.fps)}", flush=True)


if __name__ == "__main__":
    main()
