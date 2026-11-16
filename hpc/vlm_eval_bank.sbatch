#!/bin/bash
# Chunk M2.4: VLM standalone evaluation on the CityEnviron frame bank, and the M2 gate verdict.
#
# Same env as hpc/vlm_latency.sbatch (/scratch/$USER/envs/pp, weights under HF_HOME).
# Full run (default): train + val flights, every 3rd step (~3,600 frames), 2B and 8B together,
# 6 conditions (overlay/crop on-off with the D9 question, plus two reworded questions).
# About 0.5 s of model time per model per frame -> roughly 1 h 15 min. Ask the team before launching.
#   sbatch hpc/vlm_eval_bank.sbatch
# A quick look (2B, overlay+crop, 50 frames per type, ~5 min):
#   sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-2B-Instruct",CONDITIONS="both",PER_TYPE=50,OUT=results/vlm_eval_quick.jsonl hpc/vlm_eval_bank.sbatch
# Resubmitting with the same OUT continues where it stopped (rows are keyed by model, condition, frame).
# Re-score saved rows with other thresholds, no GPU (CPU node or laptop):
#   python -m vlm.eval_bank --summarize-only --out results/vlm_eval.jsonl --d-pos 10 --d-neg 30
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-vlm-eval
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PP_ENV="${PP_ENV:-/scratch/$USER/envs/pp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
BANK="${BANK:-/scratch/dns5508/frame_bank/CityEnviron}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-8B-Instruct}"
CONDITIONS="${CONDITIONS:-both overlay crop raw both_blocked both_hit}"
SPLITS="${SPLITS:-train val}"
STRIDE="${STRIDE:-3}"
OUT="${OUT:-results/vlm_eval.jsonl}"
EXAMPLES="${EXAMPLES:-results/vlm_examples}"
PER_TYPE_ARG=""
[ -n "${PER_TYPE:-}" ] && PER_TYPE_ARG="--per-type ${PER_TYPE}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
eval "$(conda shell.bash hook)"
conda activate "${PP_ENV}"

# shellcheck disable=SC2086
python -m vlm.eval_bank --bank "${BANK}" --models ${MODELS} --conditions ${CONDITIONS} --splits ${SPLITS} \
    --stride "${STRIDE}" ${PER_TYPE_ARG} --out "${OUT}" --examples "${EXAMPLES}"
echo "[$(date -Is)] done"
