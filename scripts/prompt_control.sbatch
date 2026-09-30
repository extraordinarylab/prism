#!/bin/bash
# Prompt-structure control (revision 1): one job per stage invocation.
#   extract: sbatch scripts/prompt_control.sbatch extract --model qwen --dataset sciq --template all
#   probe:   sbatch --gres=gpu:1 --cpus-per-task=72 scripts/prompt_control.sbatch probe --jobs 12 --only qwen__sciq
#SBATCH --job-name=prism-pc
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate prism
export HF_HUB_CACHE=/lus/lfs1aip2/projects/u6sn/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
python -W ignore scripts/prompt_control.py "$@"
