# PRISM — TACL revision (submission 11673)

## Environment
- Conda env: `prism` (`source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh && conda activate prism`).
- Python 3.12, torch cu129 (aarch64 / GH200), transformers 4.x. Install vLLM only if figure6.py is needed.
- HF models/datasets live in `HF_HUB_CACHE=/lus/lfs1aip2/projects/u6sn/hf_cache/hub`; keep `huggingface_hub<1.0` (transformers 4.x).

## Slurm
- Smoke tests: `--partition=interactive`.
- Full-scale runs: `--partition=workq`.

## Git
- Commit as `penguinwang96825` only. Do NOT add a Claude `Co-Authored-By` line.

## Paper (Overleaf project 69173ecf21db233c754bc8d9)
- Mark every revision in colour (existing Okabe-Ito colour-blind-safe style); never edit silently:
  - `\rev{...}` (revcolor, #C1440E) and `\revb{...}` (revcolorb, #0072B2).
- Ask the user before deciding scope, experiment design, or wording of claims — but check the codebase/paper first instead of asking for things that can be looked up.

## Code
- `scripts/figureN.py` reproduce the paper figures; cached outputs live in `assets/`.
- Figure 1 static profile (`scripts/figure1.py`) feeds the raw question (+ choices/context), no chat template, metrics at the final input token.
