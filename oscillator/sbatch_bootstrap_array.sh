#!/bin/bash
#SBATCH --job-name=nl_boot
#SBATCH --output=logs/nl_boot_%A_%a.out
#SBATCH --error=logs/nl_boot_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --account=p32911
#SBATCH --array=0-9

# If you want GPU, uncomment next two lines (and adjust partition):
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$ROOT_DIR/logs"

echo "===== JOB INFO ====="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "===== CPU ====="
lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket' || true
echo "===== GPU ====="
nvidia-smi || true
echo "===== PYTHON ====="
# Set PYTHON to your interpreter (e.g. "conda activate <env>" first)
PYTHON=${PYTHON:-python3}
$PYTHON -V

# Avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run chunk
for noise in 0.03 0.20; do
  outdir="$ROOT_DIR/outputs/noise${noise}"
  mkdir -p "$outdir"
  $PYTHON "$ROOT_DIR/run_bootstrap_chunk.py" \
    --outdir "$outdir" \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --n-tasks 10 \
    --B-outer 50 \
    --B-inner 200 \
    --n-tr 3000 \
    --M 1000 \
    --perc 99.9 \
    --lam 0.01 \
    --lam-cov 1e-1 \
    --alpha-lastlayer 0.1 \
    --hidden 32 \
    --depth 4 \
    --noise-sd ${noise} \
    --fit-epochs-outer 500 \
    --fit-epochs-base 3000
done
