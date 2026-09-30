#!/bin/bash
# Figure 1 length / token-composition controls (revision 3).
#   sbatch scripts/figure1_controls.sbatch extract --datasets sciq aime
#SBATCH --job-name=prism-fig1c
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate prism
export HF_HUB_CACHE=/lus/lfs1aip2/projects/u6sn/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -W ignore scripts/figure1_controls.py "$@"
