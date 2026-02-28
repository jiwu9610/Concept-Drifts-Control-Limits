#!/bin/bash
#SBATCH --job-name=nl_pfar
#SBATCH --output=logs/nl_pfar_%A_%a.out
#SBATCH --error=logs/nl_pfar_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --account=p32911
#SBATCH --array=0-9

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$ROOT_DIR/logs"

echo "===== JOB INFO ====="
date
hostname
lscpu | egrep 'Model name|CPU\\(s\\)' || true
nvidia-smi || true

# Set PYTHON to your interpreter (e.g. "conda activate <env>" first)
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

for noise in 0.03 0.15; do
  outdir="$ROOT_DIR/outputs/noise${noise}"
  $PYTHON "$ROOT_DIR/pfar_chunk.py" \
    --outdir "$outdir" \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --n-tasks 10 \
    --R 8000 \
    --n-tr 3000 \
    --M 1000 \
    --lam 0.01 \
    --alpha-lastlayer 0.1 \
    --hidden 32 \
    --depth 4 \
    --noise-sd ${noise}
done
