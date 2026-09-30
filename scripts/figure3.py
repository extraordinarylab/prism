"""Figure 3 of PRISM: Tracing Knowledge and Reasoning in LLMs.

Layer-wise linear-probe accuracy under the multiple-choice setting.  Four
panels, one per diagnostic benchmark (BioMixQA, SciQ, LogiQA, MathQA), each
showing the probe accuracy of three models against Transformer layer.

The pipeline has three stages:

  extract  one forward pass per question with the Figure 10 template of its
           task type (no generation), keeping the hidden state of every layer
           at the final user-content token.  Requires a GPU.
  probe    train a layer-wise logistic-regression probe on the extracted
           states under 5-fold stratified cross-validation.  CPU.
  plot     draw the four panels from the probe table.

Usage
-----
    python scripts/figure3.py extract --model qwen      # needs a GPU
    python scripts/figure3.py probe
    python scripts/figure3.py plot [--published assets/figure3_published.csv]

Method (paper Section 3.3, "Layer-Wise Probing").  Each item is rendered with
the multiple-choice template of Figure 10 -- (a) for the knowledge-centric
BioMixQA and SciQ, (b) for the reasoning-centric LogiQA and MathQA -- wrapped in
the chat template with the system prompt "You are a helpful assistant." and
`add_generation_prompt=True`.  One forward pass is run and the hidden state at
the final input position before answer generation, i.e. the last token of the
user content, is extracted from every layer.  At each layer a logistic
regression is trained to predict the index of the correct option, and the mean
held-out accuracy over 5 stratified folds is reported.

Subset.  Full test split, capped at 1000 items by `shuffle(seed=42)`, the rule
of the original probing code.  It reproduces the published layer-0 majority
baselines of BioMixQA (306 items) and SciQ; for LogiQA and MathQA the subset
behind the published curves could not be recovered, so their curves are a
re-measurement under the documented rule rather than an exact replay.

Hidden states include the embedding output as layer 0, so Qwen2.5-7B (28
Transformer layers) yields 29 points and the two 8B-class models 33.
"""

import argparse
import csv
import json
import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SYSTEM_PROMPT = "You are a helpful assistant."

MODELS: Dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "olmo": "allenai/Olmo-3-7B-Instruct",
}
MODEL_LABELS: Dict[str, str] = {
    "llama": "Llama-3.1-8B-Instruct",
    "qwen": "Qwen2.5-7B-Instruct",
    "olmo": "Olmo-3-7B-Instruct",
}

# The four diagnostic benchmarks, in panel order.  `context` prepends the
# passage to the question; `type` selects the Figure 10 template.
DATASETS: Dict[str, Dict[str, object]] = {
    "biomix-qa": {"split": "test", "context": False, "type": "knowledge", "label": "BioMixQA"},
    "sciq": {"split": "test", "context": False, "type": "knowledge", "label": "SciQ"},
    "logiqa": {"split": "test", "context": True, "type": "reasoning", "label": "LogiQA"},
    "math-qa": {"split": "test", "context": False, "type": "reasoning", "label": "MathQA"},
}
HF_NAMESPACE = "extraordinarylab"

# Figure 10: the reasoning template adds an instruction prefix and closes the
# answer cue with an empty <think> block and an opening <final_answer> tag.
REASONING_PREFIX = (
    "Please enclose your thinking process in <think></think> and the final "
    "answer in <final_answer></final_answer>."
)
CUE_KNOWLEDGE = "Respond with one letter. The answer is _"
CUE_REASONING = "Respond with one letter. The answer is _ <think></think><final_answer>"

MAX_ITEMS = 1000
SUBSET_SEED = 42
N_FOLDS = 5
PROBE_SEED = 42

DEFAULT_STATES_DIR = "/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/prism/figure3_states"
DEFAULT_STEM = "assets/figure3"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def answer_character(index: int) -> str:
    """Map an option index to its letter: 0 -> 'A', 1 -> 'B', ..."""
    if index < 26:
        return chr(ord("A") + index)
    return str(index - 25)


def answer_options(choices: Sequence[str]) -> str:
    """Render choices as "A) choice 1\\nB) choice 2\\n..."."""
    return "\n".join(f"{answer_character(i)}) {c}" for i, c in enumerate(choices))


def build_prompt(question: str, choices: Sequence[str], task_type: str) -> str:
    """Render one item with the Figure 10 template of its task type."""
    body = f"Question: {question}\nChoices:\n{answer_options(choices)}"
    if task_type == "knowledge":
        return f"{body}\n\n{CUE_KNOWLEDGE}"
    if task_type == "reasoning":
        return f"{REASONING_PREFIX}\n\n{body}\n\n{CUE_REASONING}"
    raise ValueError(f"unknown task type {task_type!r}")


def load_items(name: str) -> Tuple[List[str], List[List[str]], np.ndarray]:
    """Load one benchmark under the subset rule (cap 1000, shuffle seed 42)."""
    from datasets import load_dataset

    config = DATASETS[name]
    dataset = load_dataset(f"{HF_NAMESPACE}/{name}", split=config["split"])
    if len(dataset) > MAX_ITEMS:
        dataset = dataset.shuffle(seed=SUBSET_SEED).select(range(MAX_ITEMS)).flatten_indices()

    questions = []
    for row in dataset:
        question = row["question"]
        if config["context"] and row.get("context"):
            question = f"{row['context']}\n{question}"
        questions.append(question)
    choices = [list(c) for c in dataset["choices"]]
    labels = np.asarray(dataset["answer_index"], dtype=np.int64)
    return questions, choices, labels


# --------------------------------------------------------------------------- #
# Stage 1: extraction
# --------------------------------------------------------------------------- #

def states_path(states_dir: str, model: str, dataset: str) -> str:
    return os.path.join(states_dir, f"{model}__{dataset}.npz")


def probe_position(tokenizer, rendered: str, user_text: str) -> Tuple[Dict, int]:
    """Tokenise a chat-wrapped prompt and locate the last user-content token.

    The position is found from the character offsets rather than a fixed
    offset from the end, so it stays correct whatever scaffolding a model's
    chat template appends after the user turn.
    """
    end = rendered.rfind(user_text)
    if end < 0:
        raise RuntimeError("user content not found verbatim in the chat template")
    end += len(user_text)

    enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                    return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].numpy()
    inside = np.nonzero((offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0]))[0]
    return enc, int(inside[-1])


def run_extract(args: argparse.Namespace) -> None:
    """Extract the per-layer probe features of every dataset for one model."""
    import torch
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.states_dir, exist_ok=True)
    # Resolve the local snapshot so compute nodes never need the network.
    snapshot = snapshot_download(MODELS[args.model], local_files_only=True)
    print(f"Loading {MODELS[args.model]} from {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    for name in args.datasets:
        path = states_path(args.states_dir, args.model, name)
        if os.path.exists(path) and not args.overwrite:
            print(f"skip {path} (exists)")
            continue

        questions, choices, labels = load_items(name)
        if args.limit:
            questions, choices, labels = questions[:args.limit], choices[:args.limit], labels[:args.limit]
        task_type = DATASETS[name]["type"]
        n_layers = model.config.num_hidden_layers + 1  # embeddings + blocks
        states = np.zeros((len(questions), n_layers, model.config.hidden_size), dtype=np.float16)
        offsets_from_end = np.zeros(len(questions), dtype=np.int32)

        for i in tqdm(range(len(questions)), desc=f"{args.model}/{name}"):
            user_text = build_prompt(questions[i], choices[i], task_type)
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc, pos = probe_position(tokenizer, rendered, user_text)
            offsets_from_end[i] = enc["input_ids"].shape[1] - pos
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for layer, hidden in enumerate(out.hidden_states):
                states[i, layer] = hidden[0, pos].float().cpu().numpy()

        np.savez(path, states=states, labels=labels, offsets_from_end=offsets_from_end)
        meta = {
            "model": MODELS[args.model], "snapshot": snapshot, "dataset": name,
            "task_type": task_type, "n_items": len(questions), "n_layers": n_layers,
            "probe_offset_from_end": sorted(set(offsets_from_end.tolist())),
            "majority_baseline": float(np.bincount(labels).max() / len(labels)),
            "example_prompt": rendered,
        }
        with open(path[:-len(".npz")] + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {path}  (probe offset from end: {meta['probe_offset_from_end']}, "
              f"majority {meta['majority_baseline']:.4f})")


# --------------------------------------------------------------------------- #
# Stage 2: probing
# --------------------------------------------------------------------------- #

def probe_layers(states: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Mean held-out accuracy of a per-layer logistic-regression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                 .split(np.zeros(len(labels)), labels))
    accuracy = np.zeros(states.shape[1])
    for layer in range(states.shape[1]):
        X = states[:, layer].astype(np.float32)
        scores = []
        for train, val in folds:
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X[train], labels[train])
            scores.append(np.mean(clf.predict(X[val]) == labels[val]))
        accuracy[layer] = np.mean(scores)
    return accuracy


def run_probe(args: argparse.Namespace) -> None:
    from joblib import Parallel, delayed

    runs = sorted(glob(os.path.join(args.states_dir, "*__*.npz")))
    if not runs:
        raise SystemExit(f"no extracted states under {args.states_dir}; run `extract` first")

    def one(path: str) -> List[Dict[str, object]]:
        model, name = os.path.basename(path)[:-len(".npz")].split("__")
        with np.load(path) as z:
            accuracy = probe_layers(z["states"], z["labels"], args.seed)
        print(f"{model}/{name}: peak {accuracy.max():.4f} at layer {accuracy.argmax()}", flush=True)
        return [{"dataset": DATASETS[name]["label"], "model": MODEL_LABELS[model],
                 "layer": layer, "accuracy": f"{acc:.6f}"}
                for layer, acc in enumerate(accuracy)]

    rows = [r for block in Parallel(n_jobs=args.jobs)(delayed(one)(p) for p in runs) for r in block]
    with open(f"{args.stem}_probe_accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.stem}_probe_accuracy.csv ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# Stage 3: plotting
# --------------------------------------------------------------------------- #

PANEL_DATASETS = [c["label"] for c in DATASETS.values()]

# Shared y-range for every panel, so accuracies compare across benchmarks.
Y_LIMITS = (0.2, 1.0)

# Colour-blind-safe Okabe-Ito triple shared with Figure 2; line style and
# marker repeat the identity so the models stay separable in greyscale.
MODEL_STYLES = {
    "Llama-3.1-8B-Instruct": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Qwen2.5-7B-Instruct": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "Olmo-3-7B-Instruct": {"color": "#CC79A7", "linestyle": "-.", "marker": "^"},
}


def plot_figure3(table, stem: str, published=None) -> None:
    """Four panels of probe accuracy against layer, one line per model.

    With `published`, the curves of the submitted manuscript are drawn
    underneath in the same colour at low opacity for comparison.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "lines.linewidth": 1.2,
    })
    fig, axes = plt.subplots(1, len(PANEL_DATASETS), figsize=(7.16, 2.35), gridspec_kw={
        "left": 0.075, "right": 0.992, "bottom": 0.25, "top": 0.77, "wspace": 0.26})

    for ax, dataset in zip(axes, PANEL_DATASETS):
        for model, style in MODEL_STYLES.items():
            if published is not None:
                ref = published[(published.dataset == dataset) & (published.model == model)]
                ax.plot(ref.layer, ref.accuracy, color=style["color"], alpha=0.3, linewidth=0.9)
            series = table[(table.dataset == dataset) & (table.model == model)].sort_values("layer")
            if series.empty:
                continue
            ax.plot(series.layer, series.accuracy, label=model, markevery=4, markersize=3.2,
                    markerfacecolor="white", markeredgewidth=0.65, **style)

        ax.set_title(dataset, pad=5)
        ax.set_xlabel("Layer")
        ax.set_xlim(0, 32)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(*Y_LIMITS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel("Linear-Probe Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(MODEL_STYLES), frameon=False, handlelength=2.4, columnspacing=1.4)

    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {stem}.{ext}")
    plt.close(fig)


def run_plot(args: argparse.Namespace) -> None:
    import pandas as pd

    table = pd.read_csv(f"{args.stem}_probe_accuracy.csv")
    published = pd.read_csv(args.published) if args.published else None
    plot_figure3(table, args.stem, published)


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--states-dir", default=DEFAULT_STATES_DIR)
    common.add_argument("--stem", default=DEFAULT_STEM, help="output stem")

    p = sub.add_parser("extract", parents=[common], help="hidden states (GPU)")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=run_extract)

    p = sub.add_parser("probe", parents=[common], help="layer-wise probes (CPU)")
    p.add_argument("--seed", type=int, default=PROBE_SEED)
    p.add_argument("--jobs", type=int, default=4, help="(model, dataset) cells in parallel")
    p.set_defaults(func=run_probe)

    p = sub.add_parser("plot", parents=[common], help="draw the figure")
    p.add_argument("--published", help="overlay the manuscript's curves from this CSV")
    p.set_defaults(func=run_plot)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
