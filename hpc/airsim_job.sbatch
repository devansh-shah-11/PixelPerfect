#!/bin/bash
# Start Project AirSim, then run any command against it (sweeps on the real simulator, chunks M4.7 and M5).
# Same setup as hpc/m4_checks.sbatch: PA_ENV may be someone else's env (read-only is fine), AIRSIM must be your own copy.
# The generated sim configs are exported as PP_SIM_CONFIGS, which sweep configs reference as "${PP_SIM_CONFIGS}".
#   sbatch --export=ALL,PA_ENV=/scratch/dns5508/envs/pa,CMD="python -m eval.sweep --config configs/sweeps/airsim_smoke.yaml" hpc/airsim_job.sbatch
# Time: about 30 s of rendering per 20 m flight (M4.2: 0.22 s per 100 ms step), plus 2 to 3 min for the sim to start.
# Override --time for long sweeps; sweeps are resumable, so a job that runs out of time can simply be resubmitted.
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=pp-airsim
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --requeue

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
PA_ENV="${PA_ENV:-/scratch/$USER/envs/pa}"
ENV=${ENV:-CityEnviron}
AIRSIM=${AIRSIM:-/scratch/$USER/airsim}
: "${CMD:?set CMD, e.g. CMD=\"python -m eval.sweep --config configs/sweeps/airsim_smoke.yaml\"}"
WORK="$REPO/results/airsim_job_${SLURM_JOB_ID:-local}"
export PP_SIM_CONFIGS=$WORK/sim_config
mkdir -p "$WORK/bin"
printf '#!/bin/sh\necho "$HOME"\n' > "$WORK/bin/xdg-user-dir"; chmod +x "$WORK/bin/xdg-user-dir"
export PATH="$WORK/bin:$PATH" QT_QPA_PLATFORM=offscreen

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PA_ENV" || { echo "PREFLIGHT FAIL: no env $PA_ENV"; exit 1; }
[ -w "$AIRSIM" ] || { echo "PREFLIGHT FAIL: $AIRSIM is not writable; copy the simulator to your scratch"; exit 1; }
python -c "import projectairsim, commentjson, cv2, numpy, yaml, PIL" || { echo "PREFLIGHT FAIL: python imports"; exit 1; }
LAUNCHER=$(find "$AIRSIM/envs/$ENV" -maxdepth 2 -name '*.sh' | head -1)
[ -n "$LAUNCHER" ] && [ -f "$LAUNCHER" ] || { echo "PREFLIGHT FAIL: no launcher for ENV=$ENV"; exit 1; }
echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} ENV=$ENV commit=$(git -C "$REPO" rev-parse --short HEAD) cmd: $CMD"

python "$REPO/hpc/capability/make_configs.py" --src $AIRSIM/ProjectAirSim/client/python/example_user_scripts/sim_config --out "$PP_SIM_CONFIGS" \
    || { echo "make_configs.py FAILED"; exit 1; }
cd "$(dirname "$LAUNCHER")" && chmod +x "$LAUNCHER"
"$LAUNCHER" -RenderOffScreen -nosound > "$WORK/sim.log" 2>&1 &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null' EXIT
sleep 60
pgrep -fa 'Linux-Shipping' > /dev/null || { echo "SIM NOT RUNNING after 60 s"; tail -20 "$WORK/sim.log"; exit 1; }
cd "$REPO"
eval "$CMD"
rc=$?
echo "[$(date -Is)] exit code: $rc"
exit $rc
