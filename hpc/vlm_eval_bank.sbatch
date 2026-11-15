#!/bin/bash
# Chunk M2.4: VLM standalone evaluation on the val split of the CityEnviron frame bank.
#
# Same env as hpc/vlm_latency.sbatch (/scratch/$USER/envs/pp, weights under HF_HOME).
# Submit (A100; roughly 1 h for ~1,900 val frames x 4 conditions x 2 models at ~0.08 s a call,
# plus ground-truth loading; ask the team before launching):
#   sbatch hpc/vlm_eval_bank.sbatch
# A quick first look, 50 frames per obstacle type, 2B only, overlay+crop only (~5 min):
#   sbatch --export=ALL,MODELS="Qwen/Qwen3-VL-2B-Instruct",CONDITIONS="both",PER_TYPE=50 hpc/vlm_eval_bank.sbatch
# Resubmitting continues from the rows already in the output file.
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-vlm-eval
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PP_ENV="${PP_ENV:-/scratch/$USER/envs/pp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
BANK="${BANK:-/scratch/dns5508/frame_bank/CityEnviron}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-8B-Instruct}"
CONDITIONS="${CONDITIONS:-both overlay crop raw}"
OUT="${OUT:-results/vlm_eval_val.jsonl}"
EXAMPLES="${EXAMPLES:-results/vlm_examples}"
PER_TYPE_ARG=""
[ -n "${PER_TYPE:-}" ] && PER_TYPE_ARG="--per-type ${PER_TYPE}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
eval "$(conda shell.bash hook)"
conda activate "${PP_ENV}"

# shellcheck disable=SC2086
python -m vlm.eval_bank --bank "${BANK}" --models ${MODELS} --conditions ${CONDITIONS} ${PER_TYPE_ARG} \
    --out "${OUT}" --examples "${EXAMPLES}"
echo "[$(date -Is)] done"
