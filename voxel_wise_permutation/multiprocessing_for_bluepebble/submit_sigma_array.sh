#!/bin/bash
#SBATCH --job-name=vvprm_sigma
#SBATCH --partition=teach_cpu
#SBATCH --account=chem036964
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-13%14
#SBATCH --exclusive
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

mkdir -p logs test_datasets
module --force purge
module load languages/python/3.12.3

cd /user/home/lt22412/fmri

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SIGMAS=(0 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
SIGMA="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"

echo "Running sigma=${SIGMA}"
python -u /user/home/lt22412/fmri/multi_processing_single_sigma.py \
  --sigma "${SIGMA}" \
  --distribution t \
  --n-runs 200 \
  --n-perm 500 \
  --max-workers 14 \
  --output-dir /user/home/lt22412/fmri/test_datasets

