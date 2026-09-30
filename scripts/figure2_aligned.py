"""Decoding-time trajectories on SciQ, MathQA and LogiQA with aligned token indices.

Addresses TACL mandatory revision 2 (Reviewer A, W1 and general comment 2): the
published Figure 2 traces a single knowledge-centric benchmark, SciQ, so it is
open whether the higher within-trajectory variability of attention entropy on
incorrect generations generalises to reasoning.  This script traces SciQ and
two reasoning-centric benchmarks, MathQA and LogiQA, with Qwen2.5-7B-Instruct
under one shared prompt, and reads every signal at the model's own token
positions.

Differences from `scripts/figure2.py`, which reproduces the submitted figure:

  * One prompt for all three benchmarks, the retrieve-then-answer prompt of
    the paper's Figure 9, so the prompt format is not confounded with task.
  * Signals are read at the real positions of the generated tokens.  After
    sampling, the chat-formatted prompt plus the sampled token ids is run
    forward once; by causality the state at each generated token equals the
    state the model had while decoding it.  `figure2.py` instead re-tokenises
    whitespace-split prefixes as raw text without the chat template, so its
    positions are words and its inputs differ from the decoding context.
  * Eq. 6 is applied to generated-token indices: every generation contributes
    P = 101 uniformly spaced token positions (length normalisation).

The pipeline has three stages:

  trace    sample one answer per item, score it, and record attention entropy
           and activation sparsity at layers 3 / 14 / 22 for every generated
           token.  GPU; `--shard i --num-shards n` splits the items.
  analyse  within-trajectory SD per generation, correct vs incorrect
           (one-sided Mann-Whitney, BH across the three depths, Cliff's
           delta), with and without a control for generation length.  CPU.
  plot     Figure-2-style trajectories per benchmark, plus one summary figure
           of the within-trajectory SD.

Usage
-----
    python scripts/figure2_aligned.py trace --dataset math-qa --shard 0 --num-shards 8
    python scripts/figure2_aligned.py analyse
    python scripts/figure2_aligned.py plot
"""

import argparse
import csv
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Retrieve-then-answer prompt of the paper's Figure 9, shared by all benchmarks.
INSTRUCTION = (
    "Identify key facts and technical principles related to the following "
    "question. Based only on the knowledge retrieved above, analyze the "
    "following question. The last line of your response should be of the "
    "following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is "
    "one of {letters}."
)

# `context` prepends the passage; `limit` caps the split by shuffle(seed=42).
DATASETS: Dict[str, Dict[str, object]] = {
    "sciq": {"split": "test", "context": False, "limit": None, "type": "knowledge", "label": "SciQ"},
    "math-qa": {"split": "test", "context": False, "limit": 1000, "type": "reasoning", "label": "MathQA"},
    "logiqa": {"split": "test", "context": True, "limit": None, "type": "reasoning", "label": "LogiQA"},
}
HF_NAMESPACE = "extraordinarylab"
SUBSET_SEED = 42

# Representative lower / middle / upper depths (1-indexed Transformer layers).
LAYERS = [3, 14, 22]
LAYER_NAMES = ["Lower Layer", "Middle Layer", "Upper Layer"]
METRICS = {"attention_entropy": "Attention Entropy", "activation_sparsity": "Activation Sparsity"}

SPARSITY_THRESHOLD = 0.01
DELTA = 1e-10
N_POSITIONS = 101
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.8

DEFAULT_OUT_DIR = "/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/prism/figure2_aligned"
DEFAULT_STEM = "assets/figure2_aligned"


# --------------------------------------------------------------------------- #
# Prompt construction and scoring
# --------------------------------------------------------------------------- #

def answer_character(index: int) -> str:
    """Map an option index to its letter: 0 -> 'A', 1 -> 'B', ..."""
    if index < 26:
        return chr(ord("A") + index)
    return str(index - 25)


def build_prompt(question: str, choices: Sequence[str]) -> str:
    """The Figure 9 prompt for one multiple-choice item."""
    letters = ",".join(answer_character(i) for i in range(len(choices)))
    options = "\n".join(f"{answer_character(i)}) {c}" for i, c in enumerate(choices))
    return f"{INSTRUCTION.format(letters=letters)}\n{question}\n{options}"


def extract_predicted_answer(response: str) -> str:
    """The letter of the last 'ANSWER: X' in the response, '' if absent."""
    matches = re.findall(r"ANSWER:\s*\[?([A-Z])\]?", response, re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def load_items(name: str) -> List[Dict[str, object]]:
    from datasets import load_dataset

    config = DATASETS[name]
    dataset = load_dataset(f"{HF_NAMESPACE}/{name}", split=config["split"])
    if config["limit"] and len(dataset) > config["limit"]:
        dataset = dataset.shuffle(seed=SUBSET_SEED).select(range(config["limit"])).flatten_indices()
    items = []
    for index, row in enumerate(dataset):
        question = row["question"]
        if config["context"] and row.get("context"):
            question = f"{row['context']}\n{question}"
        items.append({
            "item": index,
            "prompt": build_prompt(question, row["choices"]),
            "gold": answer_character(int(row["answer_index"])),
        })
    return items


# --------------------------------------------------------------------------- #
# Stage 1: trace
# --------------------------------------------------------------------------- #

def sample_positions(n_tokens: int, n_positions: int = N_POSITIONS) -> np.ndarray:
    """Eq. 6 over generated-token indices, stretched to `n_positions` samples."""
    return np.array([int(k * (n_tokens - 1) / (n_positions - 1)) for k in range(n_positions)])


def token_signals(model, prompt_ids, gen_ids) -> Dict[str, np.ndarray]:
    """Attention entropy and sparsity at every generated token, per layer.

    One forward pass over prompt + generation.  The query at generated token t
    sees exactly the context it had while decoding, so no prefix replay is
    needed.  Returns (n_layers, G) arrays.
    """
    import torch

    ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to(model.device)
    start = prompt_ids.shape[0]
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True, output_hidden_states=True)

    entropy, sparsity = [], []
    for layer in LAYERS:
        hidden = out.hidden_states[layer][0, start:].float()
        sparsity.append((hidden.abs() < SPARSITY_THRESHOLD).float().mean(-1).cpu().numpy())
        # Head-averaged attention of each generated query (Eq. 1), renormalised
        # (Eq. 2); masked future positions are zero and add nothing to Eq. 3.
        p = out.attentions[layer - 1][0, :, start:, :].float().mean(0)
        p = p / (p.sum(-1, keepdim=True) + DELTA)
        entropy.append((-(p * torch.log(p + DELTA)).sum(-1)).cpu().numpy())
    del out
    return {"attention_entropy": np.stack(entropy), "activation_sparsity": np.stack(sparsity)}


def run_trace(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_items(args.dataset)[args.shard::args.num_shards]
    if args.limit:
        items = items[:args.limit]
    out_path = os.path.join(args.out_dir, f"{args.dataset}__shard{args.shard:02d}of{args.num_shards:02d}.npz")
    os.makedirs(args.out_dir, exist_ok=True)

    snapshot = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    torch.manual_seed(args.seed + args.shard)

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        for it in items
    ]

    records, trajectories = [], {m: [] for m in METRICS}
    for b in tqdm(range(0, len(items), args.batch_size), desc=f"{args.dataset}[{args.shard}]"):
        batch = rendered[b:b + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=tokenizer.pad_token_id,
            )
        width = enc.input_ids.shape[1]
        # SDPA samples fast but returns no attention weights; trace under eager.
        model.set_attn_implementation("eager")
        for j, item in enumerate(items[b:b + args.batch_size]):
            prompt_ids = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            gen = generated[j, width:].cpu()
            # Keep the answer tokens, drop the stop token and any padding.
            stop = [k for k, t in enumerate(gen.tolist()) if t in stop_ids]
            truncated = not stop
            gen = gen[: stop[0]] if stop else gen
            response = tokenizer.decode(gen, skip_special_tokens=True)
            predicted = extract_predicted_answer(response)
            record = {
                "dataset": args.dataset, "item": item["item"], "gold": item["gold"],
                "predicted": predicted, "is_correct": predicted == item["gold"],
                "parsed": bool(predicted), "truncated": truncated,
                "n_tokens": int(gen.shape[0]), "response": response,
            }
            records.append(record)
            if gen.shape[0] < 2:
                for m in METRICS:
                    trajectories[m].append(np.full((len(LAYERS), N_POSITIONS), np.nan))
                continue
            signals = token_signals(model, prompt_ids, gen)
            positions = sample_positions(gen.shape[0])
            for m in METRICS:
                trajectories[m].append(signals[m][:, positions])
        model.set_attn_implementation("sdpa")
        torch.cuda.empty_cache()

    np.savez(out_path, **{m: np.stack(v) for m, v in trajectories.items()},
             is_correct=np.array([r["is_correct"] for r in records]),
             parsed=np.array([r["parsed"] for r in records]),
             truncated=np.array([r["truncated"] for r in records]),
             n_tokens=np.array([r["n_tokens"] for r in records]),
             item=np.array([r["item"] for r in records]))
    with open(out_path[:-len(".npz")] + ".jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}: {len(records)} items, "
          f"accuracy {np.mean([r['is_correct'] for r in records]):.3f}, "
          f"unparsed {sum(not r['parsed'] for r in records)}, "
          f"truncated {sum(r['truncated'] for r in records)}")


# --------------------------------------------------------------------------- #
# Stage 2: analysis
# --------------------------------------------------------------------------- #

def load_runs(out_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Concatenate the shards of every dataset, keeping analysable generations.

    A generation is analysed when an answer letter was parsed and it stopped
    before the token limit, so "incorrect" never just means "cut off".
    """
    runs = {}
    for name in DATASETS:
        shards = sorted(glob(os.path.join(out_dir, f"{name}__shard*.npz")))
        if not shards:
            continue
        parts = [dict(np.load(p)) for p in shards]
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        keep = merged["parsed"] & ~merged["truncated"] & (merged["n_tokens"] >= 2)
        merged["n_total"] = np.array(len(keep))
        merged["n_dropped"] = np.array(int((~keep).sum()))
        runs[name] = {k: (v[keep] if v.ndim and v.shape[0] == keep.shape[0] else v)
                      for k, v in merged.items()}
    return runs


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    return q


def length_residual(spread: np.ndarray, n_tokens: np.ndarray) -> np.ndarray:
    """Within-trajectory SD with the linear effect of log generation length removed."""
    X = np.column_stack([np.ones_like(spread), np.log(n_tokens)])
    beta, *_ = np.linalg.lstsq(X, spread, rcond=None)
    return spread - X @ beta


def run_analyse(args: argparse.Namespace) -> None:
    from scipy import stats

    runs = load_runs(args.out_dir)
    rows = []
    for name, run in runs.items():
        correct = run["is_correct"].astype(bool)
        n_tokens = run["n_tokens"].astype(float)
        print(f"\n=== {DATASETS[name]['label']}: {correct.size} analysed "
              f"({int(correct.sum())} correct, {int((~correct).sum())} incorrect; "
              f"{int(run['n_dropped'])} of {int(run['n_total'])} dropped as unparsed or truncated)")
        print(f"    median tokens: correct {np.median(n_tokens[correct]):.0f}, "
              f"incorrect {np.median(n_tokens[~correct]):.0f}")
        for metric in METRICS:
            for control in ("none", "length"):
                pvalues, packed = [], []
                for index, depth in enumerate(LAYER_NAMES):
                    spread = run[metric][:, index, :].std(axis=1)
                    if control == "length":
                        spread = length_residual(spread, n_tokens)
                    u, p = stats.mannwhitneyu(spread[~correct], spread[correct], alternative="greater")
                    rho = stats.spearmanr(run[metric][:, index, :].std(axis=1), n_tokens).statistic
                    pvalues.append(p)
                    packed.append((depth, u, p, rho))
                for (depth, u, p, rho), q in zip(packed, benjamini_hochberg(np.array(pvalues))):
                    delta = 2 * u / ((~correct).sum() * correct.sum()) - 1
                    rows.append({
                        "dataset": DATASETS[name]["label"], "metric": metric, "depth": depth,
                        "length_control": control, "n_correct": int(correct.sum()),
                        "n_incorrect": int((~correct).sum()), "p": f"{p:.4g}", "q": f"{q:.4g}",
                        "cliffs_delta": f"{delta:.3f}", "spearman_sd_vs_length": f"{rho:.3f}",
                    })
                    print(f"    {metric:20s} {depth:13s} control={control:6s} "
                          f"q={q:.4f} delta={delta:+.3f}  rho(SD,len)={rho:+.3f}")

    path = f"{args.stem}_variability.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


# --------------------------------------------------------------------------- #
# Stage 3: plotting
# --------------------------------------------------------------------------- #

LAYER_COLOURS = ["#0072B2", "#D55E00", "#009E73"]   # Okabe-Ito
GROUP_COLOURS = {"Correct": "#0072B2", "Incorrect": "#D55E00"}


def _rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "lines.linewidth": 1.1,
    })


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {stem}.{ext}")


# The submitted Figure 2 style (colour-blind-safe redraw): each depth carries a
# colour, a line style and a marker; rows share a y-range.
DEPTH_COLOURS = ["#0072B2", "#D55E00", "#CC79A7"]   # blue, vermillion, reddish purple
DEPTH_LINESTYLES = ["-", "--", "-."]
DEPTH_MARKERS = ["o", "s", "^"]
MARKER_EVERY_N_POINTS = 20
BAND_ALPHA = 0.14
GROUP_TITLES = ["Correct Predictions", "Incorrect Predictions"]
X_AXIS_LABEL = "Aligned Generated Token Index"
Y_LIMIT_PADDING_FRACTION = 0.06


def plot_trajectories(run: Dict[str, np.ndarray], label: str, stem: str) -> None:
    """Figure 2 layout for one benchmark, drawn exactly like the paper's Figure 2."""
    import matplotlib.pyplot as plt

    correct = run["is_correct"].astype(bool)
    masks = [correct, ~correct]
    plt.rcParams.update({"legend.fontsize": 8, "lines.linewidth": 1.2})
    fig, axes = plt.subplots(
        2, 2, figsize=(7.16, 4.05), sharex=True, sharey="row",
        gridspec_kw={"left": 0.085, "right": 0.985, "bottom": 0.115, "top": 0.86,
                     "wspace": 0.12, "hspace": 0.22},
    )
    for row, (metric, metric_title) in enumerate(METRICS.items()):
        stats = [(run[metric][m].mean(0), run[metric][m].std(0)) for m in masks]
        lo = min(float(np.min(mean - sd)) for mean, sd in stats)
        hi = max(float(np.max(mean + sd)) for mean, sd in stats)
        pad = Y_LIMIT_PADDING_FRACTION * (hi - lo)
        for col, (mean, sd) in enumerate(stats):
            ax = axes[row, col]
            x = np.arange(mean.shape[1])
            for depth, name in enumerate(LAYER_NAMES):
                ax.fill_between(x, mean[depth] - sd[depth], mean[depth] + sd[depth],
                                color=DEPTH_COLOURS[depth], alpha=BAND_ALPHA, linewidth=0)
                ax.plot(x, mean[depth], color=DEPTH_COLOURS[depth],
                        linestyle=DEPTH_LINESTYLES[depth], marker=DEPTH_MARKERS[depth],
                        markevery=MARKER_EVERY_N_POINTS, markersize=3.5,
                        markerfacecolor="white", markeredgewidth=0.7, label=name)
            if row == 0:
                ax.set_title(GROUP_TITLES[col], pad=5)
            if col == 0:
                ax.set_ylabel(metric_title)
            if row == 1:
                ax.set_xlabel(X_AXIS_LABEL)
            ax.set_xlim(0, mean.shape[1] - 1)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
            _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(LAYER_NAMES), frameon=False, handlelength=2.6, columnspacing=1.6)
    print(f"{label}: correct {int(correct.sum())}, incorrect {int((~correct).sum())}")
    _save(fig, stem)
    plt.close(fig)


def plot_variability(runs: Dict[str, Dict[str, np.ndarray]], stem: str) -> None:
    """Within-trajectory SD of attention entropy, correct vs incorrect.

    Benchmarks as columns, depths along the x-axis; boxes span the
    interquartile range with the median marked.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(7.16, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for ax, (name, run) in zip(axes, runs.items()):
        correct = run["is_correct"].astype(bool)
        for index, depth in enumerate(LAYER_NAMES):
            spread = run["attention_entropy"][:, index, :].std(axis=1)
            for k, (group, mask) in enumerate({"Correct": correct, "Incorrect": ~correct}.items()):
                ax.boxplot(spread[mask], positions=[index + (k - 0.5) * 0.36], widths=0.3,
                           patch_artist=True, showfliers=False,
                           boxprops={"facecolor": GROUP_COLOURS[group], "alpha": 0.85, "linewidth": 0.6},
                           medianprops={"color": "white", "linewidth": 1.0},
                           whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
        ax.set_xticks(range(len(LAYER_NAMES)))
        ax.set_xticklabels([d.split()[0] for d in LAYER_NAMES])
        ax.set_title(f"{DATASETS[name]['label']} ({DATASETS[name]['type']})", pad=4)
        _style(ax)
    axes[0].set_ylabel("Within-trajectory SD\nof attention entropy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in GROUP_COLOURS.values()]
    fig.legend(handles, list(GROUP_COLOURS), loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False)
    _save(fig, stem)
    plt.close(fig)


def run_plot(args: argparse.Namespace) -> None:
    _rc()
    runs = load_runs(args.out_dir)
    for name, run in runs.items():
        plot_trajectories(run, DATASETS[name]["label"], f"{args.stem}_{name}")
    plot_variability(runs, f"{args.stem}_variability")


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    common.add_argument("--stem", default=DEFAULT_STEM)

    p = sub.add_parser("trace", parents=[common], help="sample and trace (GPU)")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="first N items of the shard (smoke test)")
    p.set_defaults(func=run_trace)

    p = sub.add_parser("analyse", parents=[common], help="variability tests (CPU)")
    p.set_defaults(func=run_analyse)

    p = sub.add_parser("plot", parents=[common], help="draw the figures")
    p.set_defaults(func=run_plot)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
