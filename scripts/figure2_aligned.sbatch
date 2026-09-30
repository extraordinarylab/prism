#!/bin/bash
# Aligned decoding-time trajectories (revision 2): one job per stage invocation.
#   trace:  sbatch scripts/figure2_aligned.sbatch trace --dataset math-qa --shard 0 --num-shards 8
#   smoke:  sbatch --partition=interactive --reservation=interactive --time=0:30:00 \
#             scripts/figure2_aligned.sbatch trace --dataset logiqa --limit 4 --out-dir <scratch>/smoke
#SBATCH --job-name=prism-fig2a
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate prism
export HF_HUB_CACHE=/lus/lfs1aip2/projects/u6sn/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -W ignore scripts/figure2_aligned.py "$@"
