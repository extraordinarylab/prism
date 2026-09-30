# PRISM: Tracing Knowledge and Reasoning in LLMs

Code and data for the figures and tables of the paper.

## Setup

```bash
conda create -n prism python=3.12 -y && conda activate prism
pip install -r requirements.txt
```

Models and benchmarks download from the Hugging Face Hub on first use. Llama 3.1
is gated, so run `hf auth login` first.

## Reproducing the figures

Each script runs the models with `extract` / `trace` / `run` (GPU) and draws the
figure with `plot` (CPU). The data for `plot` is already in `assets/`, so every
figure except Figure 1 can be redrawn without a GPU. Run from the repository root.

| Paper figure | Command |
|---|---|
| `figure1.pdf` | `python scripts/figure1.py all` |
| `figure1_controls.pdf` | `python scripts/figure1_controls.py plot` |
| `figure2.pdf`, `figure2_reasoning.pdf` | `python scripts/figure2_aligned.py plot` |
| `figure3.pdf` | `python scripts/figure3.py plot` |
| `figure3_template_swap.pdf` | `python scripts/prompt_control.py plot-swap` |
| `figure4.pdf`, `figure8.pdf` | `python scripts/figure4.py plot` |
| `figure5.pdf` | `python scripts/figure5.py plot` |
| `figure6.pdf`, `figure7.pdf` | `python scripts/figure6.py plot` |
| Statistical tables | `python scripts/statistics.py`, `python scripts/figure2_aligned.py analyse` |

Run `python scripts/<script>.py --help` for the GPU stages and their options.
