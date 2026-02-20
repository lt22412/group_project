#!/bin/bash
#SBATCH --job-name=clusterwise_rft_3d
#SBATCH --partition=magma
#SBATCH --account=math022462
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=UNLIMITED
#SBATCH --output=logs/clusterwise_rft_3d_%j.out
#SBATCH --error=logs/clusterwise_rft_3d_%j.err

# ── Parameters ────────────────────────────────────────────────────────────────
THRESHOLD_U=4.4
OVERLAP_THRESHOLD=0
IMG_SIDE_LENGTH=32
NO_SIMULATION=100
METHODS=cauchy
# ──────────────────────────────────────────────────────────────────────────────

echo "Start time: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "MPI tasks: $SLURM_NTASKS"
echo "threshold_u=$THRESHOLD_U  overlap_threshold=$OVERLAP_THRESHOLD  img_side_length=$IMG_SIDE_LENGTH  no_simulation=$NO_SIMULATION"

mkdir -p logs

module load openmpi/4.1.6-xii4

cd /user/work/jq22530/RFT_Clusterwise/group_project/Clusterwise_RFT

mpirun -np $SLURM_NTASKS /user/work/jq22530/KidSat/.venv/bin/python run_3d_mpi.py \
    --threshold_u       $THRESHOLD_U \
    --overlap_threshold $OVERLAP_THRESHOLD \
    --img_side_length   $IMG_SIDE_LENGTH \
    --no_simulation     $NO_SIMULATION \
    --method            $METHODS

echo "Job completed at: $(date)"
