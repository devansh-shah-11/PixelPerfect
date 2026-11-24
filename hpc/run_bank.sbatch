#!/bin/bash
# Chunk M1.3 / M2.7: the fast stack (Depth Anything V2-Small, metric) over the frame bank, one JSONL row per frame.
# Output feeds `eval.benchmark --run` and `vlm.eval_bank --summarize-only --run` (the D23 recall on depth's misses).
# Resumable: frames already in OUT are skipped, so a requeue (or a resubmit) continues where it stopped.
# About 10 min on the A100 for ~19,000 frames. Ask the team before launching (CLAUDE.md).
#
# Run once from the repo root, so the log folder exists (Slurm does not create it):  mkdir -p logs/slurm
#   sbatch hpc/run_bank.sbatch
# Another bank or output file:
#   sbatch --export=ALL,BANK=/scratch/$USER/frame_bank/CityEnviron,OUT=/scratch/$USER/bank_runs/da2_small.jsonl hpc/run_bank.sbatch
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-run-bank
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PA_ENV="${PA_ENV:-/scratch/$USER/envs/pa}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
BANK="${BANK:-/scratch/dns5508/frame_bank/CityEnviron}"
OUT="${OUT:-/scratch/$USER/bank_runs/da2_small.jsonl}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PA_ENV}" || { echo "PREFLIGHT FAIL: no env ${PA_ENV}"; exit 1; }
[ -d "${BANK}" ] || { echo "PREFLIGHT FAIL: no bank ${BANK}"; exit 1; }

python -m perception.run_bank --bank "${BANK}" --out "${OUT}"
echo "[$(date -Is)] done: $(wc -l < "${OUT}") rows in ${OUT}"
