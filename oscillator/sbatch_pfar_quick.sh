#!/bin/bash
#SBATCH --job-name=pfar_quick
#SBATCH --output=logs/pfar_quick_%A.out
#SBATCH --error=logs/pfar_quick_%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
echo "===== CPU ====="
lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket' || true
echo "===== GPU ====="
nvidia-smi || true
echo "===== PYTHON ====="
# Set PYTHON to your interpreter (e.g. "conda activate <env>" first)
PYTHON=${PYTHON:-python3}
$PYTHON -V

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Serial quick demo (adjust noise list as needed)
for noise in 0.03 0.20; do
  outdir="$ROOT_DIR/outputs/serial"
  mkdir -p "$outdir"
  $PYTHON "$ROOT_DIR/pfar_quick_demo.py" \
    --noise ${noise} \
    --n-tr 3000 \
    --alpha-lastlayer 0.1 \
    --outdir "$outdir"
done
