#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=01:45:00
#SBATCH --job-name=pa-capability
#SBATCH --output=/scratch/%u/logs/%x_%j.out

# Project AirSim capability test v2. Copy hpc/capability/* to /scratch/$USER/capability/ first.
# Choose the environment with:  sbatch --export=ALL,ENV=Neighborhood capability_test.sbatch
#   ENV = Blocks (default, installed in /scratch/$USER/airsim) | Neighborhood | CityEnviron | LandscapeMountains
#   (the others are unpacked by download_envs.sh into /scratch/$USER/airsim/envs/<ENV>)
# Time plan: --time 1:45 = 105 min. Node boot is not counted by python; sim start + map load take ~2-3 min, python then
# gets BUDGET (default 5400 s = 90 min) and stops starting new work after that, leaving ~10 min to write results.json,
# copy the Unreal logs and shut the sim down cleanly (SLURM would otherwise kill it mid-write).
# The fixed tests need roughly 30-50 min [UNVERIFIED]; the last section, frame_bank, then records flights (RGB + depth + seg
# + pose + ground-truth obstacle distance per frame) under /scratch/$USER/frame_bank/$ENV until the budget ends. Resumable:
# resubmitting continues the same folder. To change the length, change --time AND BUDGET together (BUDGET = time - ~15 min).
# Billing of A100 time against the 300 GPU-hours is UNVERIFIED.
# Optional extra python args: --export=ALL,ENV=CityEnviron,EXTRA="--scan-extent 150 --skip video_rain_fog"
# No hard timeout on python; each section has its own timeout and the python skips the remaining
# sections after --budget-sec; partial results are always written to results.json.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/pa
export QT_QPA_PLATFORM=offscreen

ENV=${ENV:-Blocks}
AIRSIM=/scratch/$USER/airsim
OUT=/scratch/$USER/logs/capability_${ENV}_${SLURM_JOB_ID}
CONFIGS=$AIRSIM/pa_configs/sim_config
mkdir -p "$OUT/bin"
# the sim calls xdg-user-dir at startup; it does not exist on the node (harmless warning in job 15689)
printf '#!/bin/sh\necho "$HOME"\n' > "$OUT/bin/xdg-user-dir"; chmod +x "$OUT/bin/xdg-user-dir"
export PATH="$OUT/bin:$PATH"

if [ "$ENV" = "Blocks" ]; then
    ENVDIR=$AIRSIM
    LAUNCHER=$AIRSIM/Blocks.sh
else
    ENVDIR=$AIRSIM/envs/$ENV
    LAUNCHER=$(find "$ENVDIR" -maxdepth 2 -name '*.sh' | head -1)
fi

SIM_PID=""
cleanup() {
    echo "--- sim alive at exit: $(pgrep -f 'Linux-Shipping' >/dev/null && echo yes || echo no)"
    echo "--- last lines of sim.log:"; tail -15 "$OUT/sim.log" 2>/dev/null
    # Unreal's own log (the stdout log is nearly empty): keep it and count warnings/errors
    for f in $(find "$ENVDIR" -path '*Saved/Logs/*.log' -newer "$OUT/bin/xdg-user-dir" 2>/dev/null | head -3); do
        cp "$f" "$OUT/unreal_$(basename "$f")"
    done
    for f in "$OUT"/unreal_*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f"): $(grep -c 'Warning:' "$f") warnings, $(grep -c 'Error:' "$f") errors; top error lines:"
        grep 'Error:' "$f" | cut -c1-200 | sort | uniq -c | sort -rn | head -10
    done
    [ -n "$SIM_PID" ] && kill $SIM_PID 2>/dev/null
    echo "results: $OUT/results.json"
}
trap cleanup EXIT

# preflight: fail fast (and cheaply) if something is missing
[ -n "$LAUNCHER" ] && [ -f "$LAUNCHER" ] || { echo "PREFLIGHT FAIL: no launcher for ENV=$ENV (run download_envs.sh $ENV)"; exit 1; }
for f in /scratch/$USER/capability/make_configs.py /scratch/$USER/capability/capability_test.py; do
    [ -f "$f" ] || { echo "PREFLIGHT FAIL: missing $f"; exit 1; }
done
python -c "import projectairsim, commentjson, cv2, numpy" || { echo "PREFLIGHT FAIL: python imports"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "ENV=$ENV launcher=$LAUNCHER"

python /scratch/$USER/capability/make_configs.py \
    --src $AIRSIM/ProjectAirSim/client/python/example_user_scripts/sim_config --out $CONFIGS \
    || { echo "make_configs.py FAILED"; exit 1; }

cd "$(dirname "$LAUNCHER")"
chmod +x "$LAUNCHER"
"$LAUNCHER" -RenderOffScreen -nosound > "$OUT/sim.log" 2>&1 &
SIM_PID=$!
sleep 60   # big maps (CityEnviron) load slowly; the python client also retries connecting for 5 min
pgrep -fa 'Linux-Shipping' || { echo "SIM NOT RUNNING after 60 s"; exit 1; }

python /scratch/$USER/capability/capability_test.py --env "$ENV" --configs $CONFIGS --out "$OUT" --budget-sec ${BUDGET:-5400} --bank-dir /scratch/$USER/frame_bank/$ENV $EXTRA
echo "capability_test exit code: $?"
