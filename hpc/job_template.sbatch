#!/bin/bash
# Reusable Slurm job for NYU HPC Cloud Bursting (PLAN.md §8 and Appendix B).
#
# Default: runs the smoke test (task S-06), which finishes in seconds.
#   sbatch hpc/job_template.sbatch
# Any other command, with overrides on the command line:
#   sbatch --time=02:00:00 --export=ALL,CMD="python eval/sweep.py --config configs/sweep.yaml" hpc/job_template.sbatch
#
# Course rules: never use the Torch cluster. Nodes with low GPU use for 20 minutes
# are shut down, so jobs must keep the GPU busy and be resumable (CLAUDE.md rule 10).
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --job-name=pp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#
# Not known yet; fill in from `sinfo` after the first smoke test (S-06):
##SBATCH --partition=FILL_ME_IN
##SBATCH --array=0-9

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CMD="${CMD:-bash hpc/smoke_test.sh}"
echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} array_task=${SLURM_ARRAY_TASK_ID:-none} partition=${SLURM_JOB_PARTITION:-none}"
echo "cmd: ${CMD}"
eval "${CMD}"
