"""Generate capability-test scene/robot configs by patching the shipped Project AirSim examples.

Writes, for each resolution, a steppable scene that starts paused and a kinematic (non-physics)
robot with a forward-facing "FrontCam" (scene RGB, depth perspective, segmentation enabled).

Usage: python make_configs.py --src <ProjectAirSim>/client/python/example_user_scripts/sim_config \
                              --out /scratch/$USER/airsim/pa_configs/sim_config
"""

import argparse
import copy
import json
import os

import commentjson

RESOLUTIONS = [(400, 225, False), (1280, 720, False), (400, 225, True)]  # (w, h, depth noise on)
ENABLED_TYPES = {0, 2, 3}  # scene RGB, depth perspective, segmentation


def load(path: str) -> dict:
    with open(path) as f:
        return commentjson.load(f)


def make_robot(base: dict, width: int, height: int, noisy: bool = False) -> dict:
    robot = copy.deepcopy(base)
    sensors = robot["sensors"]
    down = next(s for s in sensors if s["id"] == "DownCamera")
    front = copy.deepcopy(down)
    front["id"] = "FrontCam"
    front["origin"] = {"xyz": "0.3 0.0 0.0", "rpy-deg": "0 0 0"}  # level, facing forward
    front["capture-interval"] = 0.02
    for cs in front["capture-settings"]:
        cs["width"], cs["height"] = width, height
        cs["capture-enabled"] = cs["image-type"] in ENABLED_TYPES
        cs["streaming-enabled"] = False
        cs["compress"] = False
        cs["pixels-as-float"] = cs["image-type"] in (1, 2)
    if noisy:  # shipped noise model, enabled for planar (1) and perspective (2) depth
        tmpl = next((n for n in down.get("noise-settings", []) if n.get("image-type") == 1), None)
        if tmpl is None:
            raise SystemExit("no depth noise template found in the shipped DownCamera config")
        front["noise-settings"] = [dict(copy.deepcopy(tmpl), enabled=True, **{"image-type": t}) for t in (1, 2)]
    else:
        front.pop("noise-settings", None)
    # Disable the other cameras so timings reflect FrontCam only
    for s in sensors:
        if s["id"] in ("DownCamera", "Chase"):
            s["enabled"] = False
    sensors.append(front)
    return robot


def make_scene(base: dict, robot_file: str) -> dict:
    scene = copy.deepcopy(base)
    actor = scene["actors"][0]
    actor["robot-config"] = robot_file
    actor["origin"] = {"xyz": "0.0 0.0 -1.5", "rpy-deg": "0 0 0"}
    scene["clock"] = {
        "type": "steppable",
        "step-ns": 5000000,
        "real-time-update-rate": 3000000,
        "pause-on-start": True,
    }
    scene["segmentation"] = {"initialize-ids": True, "ignore-existing": False, "use-owner-name": True}
    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base_robot = load(os.path.join(args.src, "robot_quadrotor_nonphysics.jsonc"))
    base_scene = load(os.path.join(args.src, "scene_manual_step_drone.jsonc"))
    for w, h, noisy in RESOLUTIONS:
        tag = f"{w}n" if noisy else f"{w}"
        robot_file, scene_file = f"robot_cap_{tag}.jsonc", f"scene_cap_{tag}.jsonc"
        with open(os.path.join(args.out, robot_file), "w") as f:
            json.dump(make_robot(base_robot, w, h, noisy), f, indent=2)
        with open(os.path.join(args.out, scene_file), "w") as f:
            json.dump(make_scene(base_scene, robot_file), f, indent=2)
        print(f"wrote {scene_file} and {robot_file} ({w}x{h}{', depth noise on' if noisy else ''})")
