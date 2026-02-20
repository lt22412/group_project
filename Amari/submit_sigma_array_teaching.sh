#!/bin/bash
#SBATCH --job-name=permutation
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --array=0-13%14
#SBATCH --output=logs/permutation_3d_%j.out
#SBATCH --error=logs/permutation_3d_%j.err


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

