#!/bin/bash
#SBATCH --job-name=kungang_far
#SBATCH --output=logs/kungang_far_%j.out
#SBATCH --error=logs/kungang_far_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --account=p32911

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/outputs/kungang_far"

# Set PYTHON to your interpreter (e.g. "conda activate <env>" first)
PYTHON=${PYTHON:-python3}
echo "===== JOB INFO ====="
date
hostname
echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket' || true
nvidia-smi || true
$PYTHON -V

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON "$ROOT_DIR/kungang_far.py" \
  --noise 0.03 0.25 \
  --N-db 50 \
  --N-future 1000 \
  --M 1000 \
  --n-total 2000 \
  --lam 0.01 \
  --alpha 0.01 \
  --lam-cov 0.1 \
  --perc 99.9 \
  --outdir "$ROOT_DIR/outputs/kungang_far" \
  "$@"
