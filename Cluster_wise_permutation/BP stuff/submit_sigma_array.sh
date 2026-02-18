#!/bin/bash
#SBATCH --job-name=cluster_sigma
#SBATCH --partition=teach_cpu
#SBATCH --account=chem036964
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-13%14
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail


mkdir -p logs


module --force purge
module load languages/python/3.12.3


cd /user/home/tg22102/Fmri


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


SIGMAS=(0 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)


SIGMA="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"

echo "Running clusterwise permutation test for sigma=${SIGMA}"

python -u multi_processing_single_sigma.py \
    --sigma "${SIGMA}" \
    --output-dir /user/home/tg22102/Fmri/test_datasets

