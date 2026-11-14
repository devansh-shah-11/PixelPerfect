#!/bin/bash
# Reusable Slurm job for NYU HPC Cloud Bursting (PLAN.md §8 and Appendix B).
#
# Default: runs the smoke test (task S-06), which finishes in seconds.
#   sbatch hpc/job_template.sbatch
# Any other command, with overrides on the command line:
#   sbatch --time=02:00:00 --export=ALL,CMD="python eval/sweep.py --config configs/sweep.yaml" hpc/job_template.sbatch
# L4 instead of A100 (g2-standard-12 nodes failed to boot on Oct 30; retry before relying on it):
#   sbatch --partition=g2-standard-12 --gres=gpu:1 hpc/job_template.sbatch
#
# Course rules: never use the Torch cluster. Nodes with low GPU use for 20 minutes
# are shut down, so jobs must keep the GPU busy and be resumable (CLAUDE.md rule 10).
# Partitions and setup are in hpc/instructions.md.
#
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=pp
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue
##SBATCH --array=0-9

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CMD="${CMD:-bash hpc/smoke_test.sh}"
echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} array_task=${SLURM_ARRAY_TASK_ID:-none} partition=${SLURM_JOB_PARTITION:-none}"
echo "cmd: ${CMD}"
eval "${CMD}"
