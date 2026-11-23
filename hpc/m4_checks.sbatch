#!/bin/bash
# Chunks M4.1 (wires, glTF axes) and M4.2 (pause/step, seconds per step) on a live Project AirSim.
#
# Needs the Project AirSim env and simulator from hpc/instructions.md §1 to §3 (env with projectairsim==1.0.1,
# commentjson, opencv; the Blocks or CityEnviron build). Both default to your own scratch; point elsewhere with
# PA_ENV (reading someone else's env is fine) and AIRSIM (must be writable: the sim writes logs into its folder).
# Using Devansh's env with your own copy of the simulator:
#   cp -r /scratch/dns5508/airsim /scratch/$USER/airsim        # once, ~3 to 5 GB
#   sbatch --export=ALL,ENV=CityEnviron,PA_ENV=/scratch/dns5508/envs/pa hpc/m4_checks.sbatch
# About 10 minutes of A100 (sim start ~2 to 3 min, checks a few minutes). Ask the team before launching.
#   sbatch --export=ALL,ENV=CityEnviron hpc/m4_checks.sbatch
#   sbatch --export=ALL,ENV=Blocks,START="0,0,-2,0" hpc/m4_checks.sbatch      # START = x,y,z,yaw_deg (NED world)
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=pp-m4-checks
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
PA_ENV="${PA_ENV:-/scratch/$USER/envs/pa}"
ENV=${ENV:-CityEnviron}
START=${START:-0,0,-2.0,0}
AIRSIM=${AIRSIM:-/scratch/$USER/airsim}
OUT="$REPO/results/m4_checks_${ENV}_${SLURM_JOB_ID:-local}"
CONFIGS=$OUT/sim_config   # written fresh per job, inside the results folder
mkdir -p "$OUT/bin"
printf '#!/bin/sh\necho "$HOME"\n' > "$OUT/bin/xdg-user-dir"; chmod +x "$OUT/bin/xdg-user-dir"
export PATH="$OUT/bin:$PATH" QT_QPA_PLATFORM=offscreen

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PA_ENV" || { echo "PREFLIGHT FAIL: no env $PA_ENV (see hpc/instructions.md)"; exit 1; }
[ -w "$AIRSIM" ] || { echo "PREFLIGHT FAIL: $AIRSIM is not writable; copy the simulator to your scratch (see header)"; exit 1; }
python -c "import projectairsim, commentjson, cv2, numpy, yaml, PIL" || { echo "PREFLIGHT FAIL: python imports (if only yaml or PIL are missing and the env is not yours: pip install --user pyyaml pillow)"; exit 1; }
if [ "$ENV" = "Blocks" ]; then LAUNCHER=$AIRSIM/Blocks.sh; else LAUNCHER=$(find "$AIRSIM/envs/$ENV" -maxdepth 2 -name '*.sh' | head -1); fi
[ -n "$LAUNCHER" ] && [ -f "$LAUNCHER" ] || { echo "PREFLIGHT FAIL: no launcher for ENV=$ENV"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} ENV=$ENV start=$START commit=$(git -C "$REPO" rev-parse --short HEAD)"

python "$REPO/hpc/capability/make_configs.py" --src $AIRSIM/ProjectAirSim/client/python/example_user_scripts/sim_config --out $CONFIGS \
    || { echo "make_configs.py FAILED"; exit 1; }
cd "$(dirname "$LAUNCHER")" && chmod +x "$LAUNCHER"
"$LAUNCHER" -RenderOffScreen -nosound > "$OUT/sim.log" 2>&1 &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null' EXIT
sleep 60
pgrep -fa 'Linux-Shipping' > /dev/null || { echo "SIM NOT RUNNING after 60 s"; tail -20 "$OUT/sim.log"; exit 1; }
cd "$REPO"
python hpc/m4_checks.py --configs $CONFIGS --out "$OUT" --start "$START"
rc=$?
echo "[$(date -Is)] m4_checks exit code: $rc"
exit $rc
