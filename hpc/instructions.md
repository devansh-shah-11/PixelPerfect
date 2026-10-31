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
