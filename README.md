# PRISM: Tracing Knowledge and Reasoning in LLMs

Code and data behind the figures and statistical tables of the paper. Every
script lives in `scripts/`, writes its figures and tables to `assets/`, and is
split into stages: a GPU stage that runs the models, and CPU stages that
analyse and plot. The data needed for the CPU stages is committed in
`assets/`, so every figure can be redrawn without a GPU.

## Setup

```bash
conda create -n prism python=3.12 -y && conda activate prism
pip install -r requirements.txt     # pick the torch build that matches your CUDA
```

Models (`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`allenai/Olmo-3-7B-Instruct`) and benchmarks (`extraordinarylab/<name>`) are
downloaded from the Hugging Face Hub on first use. Llama 3.1 is gated: accept
its licence on the Hub and run `hf auth login` first. Set `HF_HUB_OFFLINE=1`
to run from a local cache only. GPU stages write intermediate results to
`runs/` (git-ignored).

Run every command from the repository root.

## Figures and tables

The file names are those under `figures/` in the paper source.

| Paper | Content | Redraw from `assets/` (CPU) | Re-run from the models (GPU) |
|---|---|---|---|
| `figure1.pdf` | Static profile: attention entropy and activation sparsity, 24 benchmarks | see note 1 | `python scripts/figure1.py all` |
| `figure1_controls.pdf` | Length and token-composition controls for Figure 1 | `python scripts/figure1_controls.py plot` | `python scripts/figure1_controls.py extract` then `analyse`, `plot` |
| `figure2.pdf` | Decoding-time trajectories on SciQ | `python scripts/figure2_aligned.py plot` | see note 2 |
| `figure2_reasoning.pdf` | Trajectory variability on SciQ, MathQA, LogiQA | `python scripts/figure2_aligned.py plot` | see note 2 |
| Table `tab:trajectory_variability` | Correct vs. incorrect trajectory variability | `python scripts/figure2_aligned.py analyse` | see note 2 |
| `figure3.pdf` | Layer-wise linear probing, 3 models x 4 benchmarks | `python scripts/figure3.py plot` | see note 3 |
| `figure3_template_swap.pdf` | Probing and attention under both prompt templates | `python scripts/prompt_control.py plot-swap` | see note 4 |
| `figure4.pdf`, `figure8.pdf` | Correct vs. distractor representation similarity | `python scripts/figure4.py plot` | `python scripts/figure4.py encode --model-path NAME=PATH ...` |
| `figure5.pdf` | Knowledge-reasoning capability landscape | `python scripts/figure5.py plot` | scores come from the evaluation tables (note 5) |
| `figure6.pdf`, `figure7.pdf` | NOTA perturbations on MathQA and SciQ | `python scripts/figure6.py plot` | `python scripts/figure6.py run --model NAME=PATH ...` (needs `vllm`) |
| Tables `tab:cot_effect`, `tab:nota_effect` | Wilcoxon and McNemar tests | `python scripts/statistics.py` | from the assets above |

### Notes

1. **Figure 1.** `figure1.py plot` reads per-benchmark features from
   `activation_analysis/question_and_choices/`, which `figure1.py extract`
   creates (one forward pass over 100 items per benchmark with
   Qwen2.5-7B-Instruct; a few minutes on one GPU). The resulting matrices are
   also committed as `assets/figure1_{attention_entropy,activation_sparsity}.npy`.
2. **Figure 2 and the trajectory table.** Sampling and tracing are sharded:
   ```bash
   for d in sciq math-qa logiqa; do
     for s in $(seq 0 15); do
       python scripts/figure2_aligned.py trace --dataset $d --shard $s --num-shards 16
     done
   done
   python scripts/figure2_aligned.py merge     # packs runs/ into assets/
   python scripts/figure2_aligned.py analyse
   python scripts/figure2_aligned.py plot
   ```
   Shards are independent and can run on separate GPUs. Generation is sampled
   (temperature 0.7, top-p 0.8, seed 42 plus the shard index), so a re-run on
   different hardware reproduces the statistics closely rather than exactly.
   The sampled responses are in `assets/figure2_aligned_generations.jsonl`.
3. **Figure 3.**
   ```bash
   for m in llama qwen olmo; do python scripts/figure3.py extract --model $m; done
   python scripts/figure3.py probe
   python scripts/figure3.py plot
   ```
4. **Template swap.** Every benchmark under every template, per model:
   ```bash
   for m in llama qwen olmo; do
     for d in biomix-qa sciq logiqa math-qa; do
       python scripts/prompt_control.py extract --model $m --dataset $d --template all
     done
   done
   python scripts/prompt_control.py probe --jobs 8
   python scripts/prompt_control.py attention
   python scripts/prompt_control.py plot-swap
   ```
   `probe --only PREFIX` restricts a run to some cells so the probes can be
   spread over several machines, and `probe --merge-only` joins the pieces.
   `report` prints the full template x benchmark summary.
5. **Figure 5.** The per-benchmark accuracies behind the landscape are the
   evaluation tables of the paper, stored in
   `assets/figure5_benchmark_scores.csv`; `figure5.py extract` re-parses them
   from the paper PDF.

## Repository layout

```
scripts/   one script per figure (see the table above) plus statistics.py
assets/    figures, and the tables and arrays they are drawn from
runs/      intermediate outputs of the GPU stages (created on demand)
```
