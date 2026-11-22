#!/bin/bash
# CPU-only Slurm job on Cloud Bursting: Plan B raycast flights, replay labels, gate training.
# No GPU, so no GPU-hours and no 20-minute idle-GPU shutdown. The CPU partition is the one the
# OOD Jupyter session runs on; batch use of it is [UNVERIFIED] until the first job runs.
#
#   sbatch --export=ALL,CMD="python -m eval.replay --config configs/replay/raycast_smoke.yaml" hpc/cpu_job.sbatch
#   sbatch --time=02:00:00 --export=ALL,CMD="..." hpc/cpu_job.sbatch
#
# Uses the conda env in $PA_ENV (default: your own /scratch/$USER/envs/pa).
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --job-name=pp-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PA_ENV:-/scratch/$USER/envs/pa}"
CMD="${CMD:?set CMD, e.g. --export=ALL,CMD=\"python -m eval.replay --config ...\"}"
echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname) cmd=$CMD"
eval "$CMD"
echo "[$(date -Is)] done"
