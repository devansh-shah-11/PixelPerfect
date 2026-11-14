"""Project AirSim smoke client: connect, take off, save one RGB and one depth frame, land.

Headless-safe (no ImageDisplay / OpenCV windows). Based on example_user_scripts/hello_drone.py
(v1.0.1) and projectairsim.utils.unpack_image. Run from example_user_scripts/ so the scene
config resolves, with the sim already running.

Usage: python smoke_client.py --out /scratch/$USER/logs/smoke_frames
"""

import argparse
import asyncio
import os
import time

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.utils import projectairsim_log, unpack_image


async def main(out_dir: str, wait_sec: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
    log = projectairsim_log()
    latest: dict = {}

    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        client.subscribe(
            drone.sensors["DownCamera"]["scene_camera"],
            lambda _, msg: latest.__setitem__("rgb", msg),
        )
        client.subscribe(
            drone.sensors["DownCamera"]["depth_camera"],
            lambda _, msg: latest.__setitem__("depth", msg),
        )

        drone.enable_api_control()
        drone.arm()
        await (await drone.takeoff_async())
        await (await drone.move_by_velocity_async(0.0, 0.0, -1.0, 2.0))

        t0 = time.time()
        while time.time() - t0 < wait_sec and not {"rgb", "depth"} <= latest.keys():
            await asyncio.sleep(0.1)

        ok = True
        if "rgb" in latest:
            rgb = unpack_image(latest["rgb"])
            cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb)
            log.info(f"RGB saved: shape={rgb.shape} mean={rgb.mean():.1f} std={rgb.std():.1f}")
            if rgb.std() < 1.0:
                log.error("RGB frame looks constant (black or blank)")
                ok = False
        else:
            log.error("No RGB frame received")
            ok = False

        if "depth" in latest:
            depth = unpack_image(latest["depth"]).astype(np.float32)
            finite = np.isfinite(depth)
            np.save(os.path.join(out_dir, "depth.npy"), depth)
            log.info(
                f"Depth saved: shape={depth.shape} finite={finite.mean():.2f} "
                f"min={depth[finite].min() if finite.any() else float('nan'):.2f} "
                f"max={depth[finite].max() if finite.any() else float('nan'):.2f} (meters)"
            )
        else:
            log.error("No depth frame received")
            ok = False

        await (await drone.land_async())
        drone.disarm()
        drone.disable_api_control()
        return 0 if ok else 1
    except Exception as err:
        log.error(f"Exception occurred: {err}", exc_info=True)
        return 2
    finally:
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="directory for saved frames")
    parser.add_argument("--wait", type=float, default=10.0, help="max seconds to wait for frames")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.out, args.wait)))
