#!/bin/bash
# Train/evaluate the per-particle classifier on one GPU node.
# Usage (from the repository root):  sbatch scripts/particle_benchmark.slurm
# Edit the partition/account lines and the module/conda setup for your cluster.
#SBATCH --job-name=junkfinder-pb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=particle_benchmark_%j.log
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
# module load cuda             # if your cluster needs it
# source activate junkfinder   # environment with: pip install -e .[ml]

STEPS=${STEPS:-20000}
VIEWS=${VIEWS:-3000}
BOX=${BOX:-64}
APIX=${APIX:-6.0}

nvidia-smi || true
junkfinder particle-benchmark --cache structures \
    --out "particle_benchmark_${SLURM_JOB_ID:-local}" \
    --steps "$STEPS" --batch 256 --train-views "$VIEWS" --test-views 200 \
    --box "$BOX" --apix "$APIX" --threads "${SLURM_CPUS_PER_TASK:-8}"
