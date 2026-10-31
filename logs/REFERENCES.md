# External resources to consider (not part of PLAN.md; nothing here is adopted yet)

## UrbanScene3D (added 2026-10-31, suggested by the user as a possible extra test scene source)
- Repo: https://github.com/yilinliu77/UrbanScene3D (site moved to https://vcc.tech/UrbanScene3D)
- Paper: Liu, Xue, Huang, arXiv 2107.04286 (id taken from the repo page summary, [UNVERIFIED] until the paper is opened).
- What it is (from the repo page): large-scale urban scenes, both hand-modelled virtual cities and real scenes reconstructed
  from aerial photos (MVS) with realistic textures; original aerial images, 4K video sequences, instance segmentation,
  CAD models for the virtual scenes. Delivered as an **Unreal Engine 4 project** (4.24 recommended) built for Microsoft AirSim.
- License: research and education use only; cite the paper.
- Download: Baidu Netdisk / Dropbox split volumes, or a direct zip link (see repo README).
- Fit with our stack (assessment, not tested):
  - Project AirSim v1.0.1 prebuilts are Unreal 5.x packages. A UE4 project cannot be loaded into them directly.
  - Possible routes: (a) offline use of the images/meshes (no simulator) for perception-failure tests of Depth Anything V2 / SAM 2 on
    real urban imagery; (b) convert meshes to glTF and spawn with `World.spawn_object_from_file` (client API says only "gltf"; size
    limits unknown); (c) port to a custom UE5 environment with the Project AirSim plugin (heavy, needs Unreal + display/GPU build).
  - Route (a) is the cheapest and fits "test on a dataset like this". Decide after the Neighborhood/CityEnviron capability run.
- Status: not downloaded.

## Project AirSim prebuilt environments (v1.0.1 release assets, checked 2026-10-31)
Blocks, Neighborhood (1.8 GB), LandscapeMountains (0.6 GB), CityEnviron (3.2 GB, two parts). Windows and Linux builds.
v1.1.0 is a pre-release (wheeled vehicles, UE 5.8) with no environment packages attached yet.
