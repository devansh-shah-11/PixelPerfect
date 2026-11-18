#!/bin/bash
# Chunk M2.1 and the benchmark table (D22): latency and peak GPU memory of the fast stack, Qwen3-VL-2B, 8B,
# and the 8B quantized to 8 and 4 bits. Each model is loaded alone, timed and freed.
#
# One-time setup on the HPC (CPU terminal is fine; the download is a few GB):
#   cd /scratch/$USER && git clone https://github.com/devansh-shah-11/PixelPerfect.git && cd PixelPerfect
#   git checkout C/M2.1
#   # env pa already exists on the HPC; create it only if missing:
#   conda create -y -p /scratch/$USER/envs/pa python=3.11
#   conda activate /scratch/$USER/envs/pa
#   pip install -e ".[dev,vlm,quant,perception]"     # quant = bitsandbytes, for the @8bit / @4bit variants
#   export HF_HOME=/scratch/$USER/hf        # weights go to scratch, never home
#   python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ('Qwen/Qwen3-VL-2B-Instruct','Qwen/Qwen3-VL-8B-Instruct')]"
#
# Submit (A100, ~30 min; ask the team before launching, CLAUDE.md):
#   sbatch hpc/vlm_latency.sbatch
# Bank frames instead of noise (needs read access to the bank):
#   sbatch --export=ALL,FRAMES=/scratch/dns5508/frame_bank/CityEnviron/frames hpc/vlm_latency.sbatch
# Only the 2B model, without the depth row:
#   sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-2B-Instruct",DEPTH=0 hpc/vlm_latency.sbatch
# A variant that fails to load (no bitsandbytes, out of memory) is recorded as failed; the other rows are kept.
# The output file feeds the benchmark table: python -m eval.benchmark ... --latency results/vlm_latency_<job>.json
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-vlm-latency
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PP_ENV="${PP_ENV:-/scratch/$USER/envs/pa}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-8B-Instruct@8bit Qwen/Qwen3-VL-8B-Instruct@4bit}"
DEPTH_ARG="--depth"
[ "${DEPTH:-1}" = "0" ] && DEPTH_ARG=""
N="${N:-50}"
FRAMES_ARG=""
[ -n "${FRAMES:-}" ] && FRAMES_ARG="--frames ${FRAMES}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
eval "$(conda shell.bash hook)"
conda activate "${PP_ENV}"
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

# shellcheck disable=SC2086
python -m vlm.bench_latency ${DEPTH_ARG} --models ${MODELS} --n "${N}" ${FRAMES_ARG} --out "results/vlm_latency_${SLURM_JOB_ID:-local}.json"
echo "[$(date -Is)] done"
