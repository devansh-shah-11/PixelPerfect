#!/bin/bash
# Chunk M2.1: serve Qwen3-VL-2B and 8B on an A100 and measure latency.
#
# One-time setup on the HPC (CPU terminal is fine; the download is a few GB):
#   cd /scratch/$USER && git clone https://github.com/devansh-shah-11/PixelPerfect.git && cd PixelPerfect
#   git checkout C/M2.1
#   conda create -y -p /scratch/$USER/envs/pp python=3.11
#   conda activate /scratch/$USER/envs/pp
#   pip install -e ".[dev,vlm]"
#   export HF_HOME=/scratch/$USER/hf        # weights go to scratch, never home
#   python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ('Qwen/Qwen3-VL-2B-Instruct','Qwen/Qwen3-VL-8B-Instruct')]"
#
# Submit (A100, ~30 min; ask the team before launching, CLAUDE.md):
#   sbatch hpc/vlm_latency.sbatch
# Bank frames instead of noise (needs read access to the bank):
#   sbatch --export=ALL,FRAMES=/scratch/dns5508/frame_bank/CityEnviron/frames hpc/vlm_latency.sbatch
# Only the 2B model:
#   sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-2B-Instruct" hpc/vlm_latency.sbatch
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-vlm-latency
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PP_ENV="${PP_ENV:-/scratch/$USER/envs/pp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-8B-Instruct}"
N="${N:-50}"
FRAMES_ARG=""
[ -n "${FRAMES:-}" ] && FRAMES_ARG="--frames ${FRAMES}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
eval "$(conda shell.bash hook)"
conda activate "${PP_ENV}"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# shellcheck disable=SC2086
python -m vlm.bench_latency --models ${MODELS} --n "${N}" ${FRAMES_ARG} --out "results/vlm_latency_${SLURM_JOB_ID:-local}.json"
echo "[$(date -Is)] done"
