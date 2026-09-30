#!/bin/bash
# Figure 3 probe-state extraction for one model.
#   smoke: sbatch --partition=interactive --time=0:30:00 scripts/figure3.sbatch qwen --limit 20 --states-dir <scratch>/smoke
#   full:  sbatch scripts/figure3.sbatch qwen
#SBATCH --job-name=prism-fig3
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
MODEL=$1; shift
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate prism
export HF_HUB_CACHE=/lus/lfs1aip2/projects/u6sn/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/figure3.py extract --model "$MODEL" "$@"
