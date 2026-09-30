#!/bin/bash
# Figure 3 probes + plot over the extracted states (CPU-bound; one node share).
#   sbatch [--dependency=afterok:<extract jobs>] scripts/figure3_probe.sbatch [--states-dir ... --stem ...]
#SBATCH --job-name=prism-fig3-probe
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate prism
# 12 (model, dataset) cells in parallel, 6 BLAS threads each.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
python -W ignore scripts/figure3.py probe --jobs 12 "$@"
python scripts/figure3.py plot "$@"
