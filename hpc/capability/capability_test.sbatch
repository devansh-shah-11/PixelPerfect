#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=00:45:00
#SBATCH --job-name=pa-capability
#SBATCH --output=/scratch/%u/logs/%x_%j.out

# Project AirSim capability test. Copy hpc/capability/* to /scratch/$USER/capability/ first.
# Allow time for node boot in --time. No hard timeout on python; each test section has its
# own timeout, the python test skips remaining sections after a 25 min budget, and partial
# results are always written to results.json.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/pa
export QT_QPA_PLATFORM=offscreen

AIRSIM=/scratch/$USER/airsim
OUT=/scratch/$USER/logs/capability_${SLURM_JOB_ID}
CONFIGS=$AIRSIM/pa_configs/sim_config
mkdir -p "$OUT"
SIM_PID=""
cleanup() {
    echo "--- sim alive at exit: $(pgrep -f Blocks-Linux-Shipping >/dev/null && echo yes || echo no)"
    echo "--- last lines of blocks.log:"; tail -15 "$OUT/blocks.log" 2>/dev/null
    [ -n "$SIM_PID" ] && kill $SIM_PID 2>/dev/null
    echo "results: $OUT/results.json"
}
trap cleanup EXIT

# preflight: fail fast (and cheaply) if something is missing
for f in /scratch/$USER/capability/make_configs.py /scratch/$USER/capability/capability_test.py $AIRSIM/Blocks.sh; do
    [ -f "$f" ] || { echo "PREFLIGHT FAIL: missing $f"; exit 1; }
done
python -c "import projectairsim, commentjson, cv2, numpy" || { echo "PREFLIGHT FAIL: python imports"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv

python /scratch/$USER/capability/make_configs.py \
    --src $AIRSIM/ProjectAirSim/client/python/example_user_scripts/sim_config --out $CONFIGS \
    || { echo "make_configs.py FAILED"; exit 1; }

cd $AIRSIM
chmod +x Blocks.sh
./Blocks.sh -RenderOffScreen -nosound > "$OUT/blocks.log" 2>&1 &
SIM_PID=$!
sleep 45
pgrep -fa Blocks || { echo "SIM NOT RUNNING after 45 s"; exit 1; }

# the python client retries connecting for up to 2 min, so a slow sim start is fine
python /scratch/$USER/capability/capability_test.py --configs $CONFIGS --out "$OUT" --budget-sec 1500
echo "capability_test exit code: $?"
