#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=pa-smoke
#SBATCH --output=/scratch/%u/logs/%x_%j.out

# Project AirSim smoke test: start the sim headless, run hello_drone.py, stop the sim.
# Paths use $USER so the same script works for every team member.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/pa

nvidia-smi
head -2 /etc/os-release
# Vulkan diagnostics (vulkaninfo is not installed on the nodes)
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
ldconfig -p | grep -i -E "libvulkan|libGLX_nvidia|libEGL_nvidia" | head

cd /scratch/$USER/airsim
chmod +x Blocks.sh
./Blocks.sh -RenderOffScreen -nosound > /scratch/$USER/logs/blocks_${SLURM_JOB_ID}.log 2>&1 &
SIM_PID=$!
sleep 60
pgrep -fa Blocks || echo "SIM NOT RUNNING: see blocks_${SLURM_JOB_ID}.log"

export QT_QPA_PLATFORM=offscreen   # no display on the node; stops OpenCV/Qt from aborting
cd ProjectAirSim/client/python/example_user_scripts
# hello_drone.py opens OpenCV windows (needs a display); smoke_client.py saves frames instead
python /scratch/$USER/smoke_client.py --out /scratch/$USER/logs/smoke_frames_${SLURM_JOB_ID}
echo "smoke_client exit code: $?"

kill $SIM_PID
