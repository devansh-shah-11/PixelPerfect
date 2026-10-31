# HPC guide (NYU Cloud Bursting), shared by the team

Account: `csci_ga_2271-2026fa` (300 GPU-hours each). Do not use Torch for coursework.
Portal: https://ood.burst.hpc.nyu.edu/ (NYU VPN off-campus).
Node OS is RHEL 9.8. Use `singularity` (no apptainer). Work in `/scratch/$USER`, not home.

## Gotchas
- The OOD Jupyter terminal is already inside a CPU job. `srun --gres=gpu:1` from it fails ("Unable to create step ... Invalid generic resource"). Use `sbatch` instead.
- GPU nodes are cloud VMs that boot on demand. A new job sits in `CF` for a few minutes. That is normal.
- Idle GPU for 20 min = auto-shutdown. Don't leave sessions idle. `scancel` your Jupyter job when done.
- GPU is `gpu:l4:1` on `g2-standard-12`, but those nodes failed to boot for us. The A100 partition (`c12m85-a100-1`, `--gres=gpu`) worked and renders fine.
- Running `Blocks.sh` on the CPU node cannot work (no Vulkan GPU). Use a GPU job.
- A job stuck in `CF` for more than ~10 minutes and then `CANCELLED` is a cluster-side boot failure. Don't keep resubmitting; try another partition or contact NYU HPC.

---

## 1. Initial setup (once per person, on the HPC terminal)

```bash
# folders
mkdir -p /scratch/$USER/{airsim,envs,logs}

# python 3.11 env (system python 3.9 is too old)
conda create -p /scratch/$USER/envs/pa python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh     # every new shell/job; never `conda init`
conda activate /scratch/$USER/envs/pa

# simulator binary + client (versions must match: 1.0.1)
cd /scratch/$USER/airsim
wget https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1/Blocks-Linux-1.0.1.zip
unzip Blocks-Linux-1.0.1.zip
chmod +x Blocks.sh
pip install projectairsim==1.0.1
git clone --branch v1.0.1 --depth 1 https://github.com/iamaisim/ProjectAirSim.git

# get the smoke test scripts (from this repo's hpc/ folder) into scratch
# scp hpc/smoke_test.sbatch hpc/smoke_client.py <netid>@<hpc-host>:/scratch/<netid>/
```
Check: `pip show projectairsim` says 1.0.1, and `ls /scratch/$USER/airsim` shows `Blocks.sh`.

---

## 2. Run the test

```bash
sbatch --partition=c12m85-a100-1 --gres=gpu --time=00:15:00 /scratch/$USER/smoke_test.sbatch
squeue -u $USER                                        # wait for ST=R
tail -f /scratch/$USER/logs/pa-smoke_<jobid>.out
```
- Sim log: `/scratch/$USER/logs/blocks_<jobid>.log`.
- **Pass** = `smoke_client exit code: 0` and a non-black frame in `/scratch/$USER/logs/smoke_frames_<jobid>/` (`rgb.png`, `depth.npy`). Copy `rgb.png` to your laptop and look at it.
- `hello_drone.py` crashes headless (it opens OpenCV windows), so the test uses `smoke_client.py`.
- Result so far: **passed on the A100 partition** (native on RHEL 9, no container; RGB 225x400 and depth 3.5 to 7.6 m). Segmentation, step time and pause/step are not tested yet.
- Job ended `F`/`CA`/`NF`? `sacct -j <jobid> --format=JobID,State,ExitCode,Reason`
- Paste the `.out` and `blocks_*.log` to the team and record pass/fail in `logs/AI_LOG.md`.

If it fails with GLIBC or missing `.so` errors, try a container (inside an sbatch job):
```bash
singularity pull ubuntu22.sif docker://ubuntu:22.04
singularity exec --nv ubuntu22.sif ./Blocks.sh -RenderOffScreen
```
If it still fails, stop and ask. That is the PLAN.md fallback decision.

---

## Notes
- Batch jobs need `#SBATCH --account=csci_ga_2271-2026fa --requeue` and must resume (append per-episode results, skip finished `(policy, seed)` pairs).
- PLAN.md default is Microsoft AirSim/Colosseum. Project AirSim is its listed fallback (Q4). Log this deviation in `logs/AI_LOG.md`.
- The binary runs natively on RHEL 9 (verified on the A100). The zip checksum is still unverified.

---

## 3. Capability test (what the simulator can do) and extra environments

Everything below was written 2026-10-31. Status is marked per step: DONE = actually run by us, TODO = not run yet.

### 3.1 What was downloaded, and where
| What | Where | Size | Status |
|---|---|---|---|
| `Blocks-Linux-1.0.1.zip` (release v1.0.1) | HPC `/scratch/$USER/airsim/` | 584 MB | DONE (section 1) |
| `projectairsim==1.0.1` (pip) and repo clone `ProjectAirSim` tag v1.0.1 | HPC conda env `pa`, `/scratch/$USER/airsim/ProjectAirSim` | small | DONE (section 1) |
| `Neighborhood-Linux-1.0.1.zip` | HPC `/scratch/$USER/airsim/downloads/` then unpacked to `/scratch/$USER/airsim/envs/Neighborhood` | 1.8 GB | DONE (job 16299 ran on it; sim segfaulted late, see 3.5) |
| `CityEnviron-Linux-1.0.1.zip.001` + `.002` | same, joined then unpacked to `.../envs/CityEnviron` | 1.9 GB + 1.3 GB | DONE (job 16300 completed) |
| `LandscapeMountains-Linux-1.0.1.zip` | same, `.../envs/LandscapeMountains` | 0.6 GB | TODO, optional |
| UrbanScene3D (UE4 dataset) | not downloaded; notes in `logs/REFERENCES.md` | n/a | TODO, undecided |

Release page: https://github.com/iamaisim/ProjectAirSim/releases (v1.0.1 is marked "Latest"; v1.1.0 is a pre-release with no environment packages). Sizes are from the GitHub release assets API. Prebuilt environments need no Unreal install.

### 3.2 Laptop-side steps we ran (only needed to read the client API or validate configs)
```bash
python3 -m pip install commentjson                      # make_configs.py dependency
python3 -m pip download projectairsim==1.0.1 --no-deps  # then unzip the .whl and read projectairsim/world.py
# shipped example configs used by make_configs.py (also present in the cloned repo on HPC):
B=https://raw.githubusercontent.com/iamaisim/ProjectAirSim/v1.0.1/client/python/example_user_scripts/sim_config
curl -sLO $B/robot_quadrotor_nonphysics.jsonc; curl -sLO $B/scene_manual_step_drone.jsonc
python3 hpc/capability/make_configs.py --src <dir with those two files> --out <outdir>   # validates the config generator
```
Docs we read (v1.0.1 tag): `docs/development/use_prebuilt.md`, `docs/scene/weather_visual_effects.md`, `docs/sensors/segmentation.md`.

### 3.3 Run the capability test on HPC
```bash
# laptop: copy the scripts (re-copy after any change)
scp hpc/capability/* <netid>@<hpc-host>:/scratch/<netid>/capability/

# HPC CPU terminal: download one environment at a time (resumable: wget -c; safe to re-run)
bash /scratch/$USER/capability/download_envs.sh Neighborhood        # 1.8 GB
bash /scratch/$USER/capability/download_envs.sh CityEnviron         # 3.2 GB, two parts, joined then unzipped
bash /scratch/$USER/capability/download_envs.sh LandscapeMountains  # 0.6 GB, optional
# if the launcher is not found later:
find /scratch/$USER/airsim/envs/Neighborhood -maxdepth 3 -name '*.sh'

# submit (A100, 1 h 45 min, python budget 90 min). Blocks is the default ENV and re-runs the old scene.
sbatch --export=ALL,ENV=Neighborhood /scratch/$USER/capability/capability_test.sbatch
squeue -u $USER
tail -f /scratch/$USER/logs/pa-capability_<jobid>.out
# a second environment (set EXTRA in the shell, not inside --export, so the space survives):
EXTRA="--scan-extent 150" sbatch --export=ALL,ENV=CityEnviron /scratch/$USER/capability/capability_test.sbatch
```
- Run one at a time at first; each job uses a full A100 for up to 1.75 h (billing against the 300 GPU-hours is UNVERIFIED).
- Resubmitting the same `ENV` continues the frame bank in `/scratch/$USER/frame_bank/<ENV>/` (finished episodes are skipped).
- Longer run: change `--time` and `BUDGET` together (`BUDGET` = time minus about 15 min, in seconds).
- Useful flags via `EXTRA`: `--only frame_bank`, `--skip video_rain_fog`, `--start x,y,z,yaw_deg`, `--bank-episodes N`, `--scan-extent M`.

### 3.4 Outputs
- `/scratch/$USER/logs/capability_<ENV>_<jobid>/`: `results.json`, `sim.log`, `unreal_*.log` (the real Unreal log), PNGs and mp4 videos under `400px/`, `400px_noisy_depth/`, `1280px/`.
- `/scratch/$USER/frame_bank/<ENV>/`: `frames/` (`<ep>_<step>_rgb.png`, `_seg.png`, `_depth.npy` float16), `frames.jsonl` (pose, condition, obstacle, ground-truth distance per frame), `episodes.jsonl`.
- Copy to the laptop to inspect: `rsync -av <netid>@<hpc-host>:/scratch/<netid>/logs/capability_<ENV>_<jobid>/ capability_<ENV>_<jobid>/` (not committed; do not add to git).
- Frame-bank episode seeds are 10000 + episode number. They are NOT the seeds in `configs/seeds.yaml`; the data is exploratory and must be split by episode if ever used for training.

### 3.5 Results so far
- Job 15689 (Blocks, A100, script v1): sim and API worked; found stale first frames after spawn/pose change, `get_images` times out while paused unless a step happens concurrently, `enable_api_control` unsupported for non-physics, 400 px loop 0.52x real time, 1280 px 0.10x, RGB not bit-repeatable, segmentation did not distinguish spawned objects. The v2 script was written to address these; it has not been run on a live sim yet.

#### Jobs 16299 (Neighborhood) and 16300 (CityEnviron), script v2, A100, 2026-10-31 (reanalysed afterwards)
Source: `pa-capability_*.out`, `capability_*/results.json`, the PNGs and mp4s. Items marked [UNVERIFIED] are inferred, not checked on the data.

**CityEnviron: completed (exit 0).**
- Frame bank: 411 episodes, 19,267 frames, no failures, 5009 s (the 5400 s budget ended it). Condition mix: clear 166, snow 77, rain 66, fog 53, dusk 49. Obstacle mix: none 71, nine types with 24 to 54 each. Only metadata (`episodes.jsonl`, `frames.jsonl`) is in `Framebank/` on the laptop; the PNGs are on HPC scratch.
- Speed: 400 px about 222 ms/step (0.45x real time); 1280 px about 975 ms/step (0.1x), so 1280 px is for spot checks only.
- `images_while_paused_concurrent_step` FAILED ("Canceled") at both resolutions. It passed on Neighborhood.
- Depth saturates at 65504 m (sky); Neighborhood 16416 m. Clip and mask before use.
- RGB is not repeatable: same pose twice differs in about 68k pixels (max diff 145 to 164). The bank is the fixed reference.

**Neighborhood: partial.** All 400 px sections up to `time_of_day_and_sun` passed. Then `video_clear`, `video_rain_fog` and `clean_depth_reference` timed out, the Unreal engine segfaulted (signal 11), and every later section (noisy depth, 1280 px, frame bank) failed. Cause not confirmed; the video section is the likely trigger [UNVERIFIED]. Rerun with `EXTRA="--skip video_clear --skip video_rain_fog"` if a second environment is wanted.

**Corrections to what the log first reported**
- `weather_visibly_changes_rgb: true` was a false pass: it counts changed pixels, and fog/dust blacking out the frame also counts.
- Rain 0.8 looks the same as clear (mean abs diff 1.35, noise level). Fog, dust and rain+fog render near-black. Snow gives a few flakes, and flakes land in depth (bank median corridor depth 1.5 m vs 32.8 m for clear).
- Effects contaminate each other: zeroing a parameter does not remove spawned particles (snow flakes in the fog image, leaves in rain+fog). Reload the scene between effects.
- Both videos have a pure white RGB panel; depth and seg panels are identical, so they say nothing about weather.
- Night and every lux setting give mean brightness 0.1 to 0.3 of 255; only about 06:00 to 19:00 is usable.
- Depth noise has no effect (0.0 m difference vs the clean session). Config not applied or measured wrongly [UNVERIFIED which].
- `after_destroy_rgb_changed_vs_empty` is about 1,400 px for every obstacle in CityEnviron, i.e. the render noise floor. It cannot show visibility.
- The 1 cm wire at 10 m is not visible in the 400 px or 1280 px images (not confirmed it was in frame). Whether wires render, or appear in GT depth/seg, is unknown; check via segmentation IDs and GT depth along the ray.
- 1280 px obstacle test covered only 5 obstacles (about 10 s each).

**Implication for the frame bank** [UNVERIFIED on bank PNGs]: keep `clear` and `dusk`; treat `rain` as clear; exclude `fog`; exclude or mask `snow` for depth labels. Obstacle geometry labels are usable. Do not report weather results from this bank.

**Known issue for new users:** do not run the video sections on Neighborhood until the segfault is understood.
