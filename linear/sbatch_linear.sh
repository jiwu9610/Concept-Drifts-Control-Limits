#!/bin/bash
#SBATCH --job-name=linear_demo
#SBATCH --output=logs/linear_demo_%A.out
#SBATCH --error=logs/linear_demo_%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --account=p32911

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$ROOT_DIR/logs"

echo "===== JOB INFO ====="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "===== PYTHON ====="
# Set PYTHON to your interpreter (e.g. "conda activate <env>" first)
PYTHON=${PYTHON:-python3}
$PYTHON -V

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

outdir="$ROOT_DIR/outputs"
mkdir -p "$outdir"

$PYTHON "$ROOT_DIR/linear_demo.py" \
  --n-tr 2000 \
  --B-outer 100 \
  --B-inner 200 \
  --R 2000 \
  --outdir "$outdir"

echo "Done. Results in $outdir/"
date
