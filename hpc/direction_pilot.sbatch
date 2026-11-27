#!/bin/bash
# Plan v6 steps 1 and 2 on the A100: (1) the yes/no read-out on the four saved city flights, per cut;
# (2) the L2 direction pilot on the frame bank (val only, 8B). Same env as hpc/vlm_eval_bank.sbatch.
# Resubmitting continues where it stopped (rows are keyed). Ask the team before launching long variants.
#   sbatch hpc/direction_pilot.sbatch                       # both steps, about 1 h
#   sbatch --export=ALL,STEPS="flight" hpc/direction_pilot.sbatch   # only the flight-frame table (minutes)
#   sbatch --export=ALL,STEPS="pilot" hpc/direction_pilot.sbatch    # only the direction pilot
# Plan v7 step A (scene read: boxes, kind, range vs the true depth of every surface; val only; about 40 min with the depth model):
#   sbatch --time=01:30:00 --export=ALL,STEPS="scene" hpc/direction_pilot.sbatch
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp-direction
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PP_ENV="${PP_ENV:-/scratch/$USER/envs/pa}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
BANK="${BANK:-/scratch/dns5508/frame_bank/CityEnviron}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
STEPS="${STEPS:-flight pilot}"
FLIGHTS="${FLIGHTS:-results/videos/city_101062_always_qwen8b results/videos/city_101063_always_qwen8b results/videos/city_101070_always_qwen8b results/videos/city_101071_always_qwen8b}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} partition=${SLURM_JOB_PARTITION:-none} host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
eval "$(conda shell.bash hook)"
conda activate "${PP_ENV}"
python -c "import numpy, torch, transformers; print('env ok', torch.__version__, transformers.__version__)"

case " ${STEPS} " in *" flight "*)
    # shellcheck disable=SC2086
    python -m vlm.eval_flight_frames --flights ${FLIGHTS} --model "${MODEL}" --out results/flight_frames_8b.jsonl ;;
esac
case " ${STEPS} " in *" pilot "*)
    python -m vlm.eval_direction --bank "${BANK}" --models "${MODEL}" --splits val --stride 4 \
        --goal-inputs none arrow text --images both --out results/direction_pilot.jsonl ;;
esac
case " ${STEPS} " in *" scene "*)
    python -m vlm.eval_scene --bank "${BANK}" --models "${MODEL}" --splits val --stride 4 \
        --conditions json_raw grid_raw yesno_both --depth-model --out results/scene_val.jsonl ;;
esac
# Plan v8 step 1 (prompt v2): train + val, cuts tuned on train, numbers reported on val. About 30 to 40 min with the depth model.
#   sbatch --time=01:30:00 --export=ALL,STEPS="scene2" hpc/direction_pilot.sbatch
case " ${STEPS} " in *" scene2 "*)
    # shellcheck disable=SC2086
    python -m vlm.eval_scene --bank "${BANK}" --models "${MODEL}" --splits train val --stride 4 --report-split val \
        --conditions ${SCENE2_CONDITIONS:-yesno_both v2_both v2near_both v2thin_both v2_raw jsonv2_both jsonv2_raw} \
        --depth-model --out results/scene_v2.jsonl ;;
esac
echo "[$(date -Is)] done"
