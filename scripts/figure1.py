"""Figure 1 of PRISM: Tracing Knowledge and Reasoning in LLMs.

Comparison of internal activation features for Qwen2.5-7B-Instruct across all 24
benchmarks.  Two heatmaps (attention entropy, activation sparsity) plot a
metric's value for each Transformer layer (y-axis) on a given dataset (x-axis).

The pipeline has three stages:

  extract    one forward pass per question (no generation), reducing each
             dataset to a per-layer attention-entropy and activation-sparsity
             profile.  Requires a GPU and the HF datasets.
  aggregate  stack the per-dataset profiles into two (n_layers, n_datasets)
             matrices.
  plot       order datasets by the first principal component of the activation
             sparsity matrix and draw the two heatmaps.

Usage
-----
    # full replication (needs a GPU)
    python prism/figure1.py all

    # redraw from cached per-dataset features
    python prism/figure1.py plot

Method (paper Section 3.2).  For each benchmark we process all available
questions with Qwen2.5-7B-Instruct *without generating an answer* and read the
attention distribution and layer-output hidden state at the final input-token
position.

Attention entropy, averaging the attention paid by the final query token over
the M heads of layer l and renormalising:

    a_ij^(l)  = (1/M) sum_h a_ij^(l,h)                                  (1)
    p_ij^(l)  = a_ij^(l) / (sum_k a_ik^(l) + delta)                     (2)
    H_l       = -(1/N) sum_i sum_j p_ij^(l) log(p_ij^(l) + delta)       (3)

Activation sparsity, the fraction of near-zero dimensions of the layer-output
hidden state u^(l) (hidden size d):

    S_l = (1/N) sum_i (1/d) sum_r 1[ |u_ir^(l)| < 0.01 ]                (4)

giving the dataset-level static feature vector F_d = [H_1..H_L, S_1..S_L]  (5).
"""

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

# Threshold below which a hidden dimension counts as "near zero" (Eq. 4).
SPARSITY_THRESHOLD = 0.01

# Numerical floor used inside the entropy computation (Eqs. 2-3).
DELTA = 1e-10

# Reference model of the static profile (Section 3.2).
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Questions sampled per benchmark.
DEFAULT_NUM_EXAMPLES = 100

DEFAULT_CACHE_DIR = "activation_analysis/question_and_choices"
DEFAULT_OUTPUT_STEM = "assets/figure1"

# The 24 benchmarks, in extraction order.  `has_choices` appends the answer
# options to the prompt; `has_context` prepends the passage.
DATASETS: Dict[str, Dict[str, object]] = {
    "biomix-qa": {"split": "test", "has_choices": True, "has_context": False},
    "sciq": {"split": "test", "has_choices": True, "has_context": False},
    "music-trivia": {"split": "test", "has_choices": True, "has_context": False},
    "qasc": {"split": "validation", "has_choices": True, "has_context": False},
    "medmcqa": {"split": "validation", "has_choices": True, "has_context": False},
    "halu-eval": {"split": "test", "has_choices": False, "has_context": False},
    "truthful-qa": {"split": "validation", "has_choices": True, "has_context": False},
    "mri-mcqa": {"split": "test", "has_choices": True, "has_context": False},
    "arc": {"split": "test", "has_choices": True, "has_context": False},
    "piqa": {"split": "validation", "has_choices": True, "has_context": False},
    "race": {"split": "test", "has_choices": True, "has_context": True},
    "commonsense-qa": {"split": "validation", "has_choices": True, "has_context": False},
    "mmlu-pro": {"split": "test", "has_choices": True, "has_context": False},
    "math-500": {"split": "test", "has_choices": False, "has_context": False},
    "winogrande": {"split": "validation", "has_choices": True, "has_context": False},
    "math-qa": {"split": "test", "has_choices": True, "has_context": False},
    "gsm8k": {"split": "test", "has_choices": False, "has_context": False},
    "gpqa-diamond": {"split": "test", "has_choices": True, "has_context": False},
    "siqa": {"split": "validation", "has_choices": True, "has_context": True},
    "aime": {"split": "test", "has_choices": False, "has_context": False},
    "amc": {"split": "test", "has_choices": False, "has_context": False},
    "coin-flip": {"split": "test", "has_choices": False, "has_context": False},
    "pubmed-qa": {"split": "test", "has_choices": False, "has_context": True},
    "logiqa": {"split": "test", "has_choices": True, "has_context": True},
}

HF_NAMESPACE = "extraordinarylab"


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


def format_example(question: str, choices: Optional[Sequence[str]] = None) -> str:
    """Format one item as the raw prompt fed to the model (no chat template)."""
    if choices:
        return f"{question}\n{answer_options(choices)}"
    return question


def build_prompt(example: Dict, config: Dict[str, object]) -> str:
    """Extract question / context / choices from a raw dataset row."""
    question = example.get("question", "")
    if not question and "context" in example:
        question = example["context"]

    if config["has_context"] and "context" in example:
        question = f"{example['context']}\n{question}"

    choices = example.get("choices") if config["has_choices"] else None
    return format_example(question, choices)


# --------------------------------------------------------------------------- #
# Stage 1: extraction
# --------------------------------------------------------------------------- #

def layerwise_features(model, tokenizer, prompt: str, device) -> Dict[str, np.ndarray]:
    """Per-layer attention entropy and activation sparsity for a single prompt.

    Both are read at the final input-token position, so the model is only run
    forward over the question -- no answer is generated.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # hidden_states holds the embedding output at index 0, so the number of
    # Transformer layers is one less than its length.
    n_layers = len(outputs.hidden_states) - 1
    last_token = inputs.input_ids.shape[1] - 1

    entropy = np.zeros(n_layers)
    sparsity = np.zeros(n_layers)

    for layer in range(n_layers):
        # Layer-output hidden state at the final input token (Eq. 4).
        hidden = outputs.hidden_states[layer + 1][0, last_token].float().cpu().numpy()
        sparsity[layer] = np.mean(np.abs(hidden) < SPARSITY_THRESHOLD)

        # Attention from the final query token, averaged over heads (Eq. 1)...
        attn = outputs.attentions[layer][0, :, last_token, :].float().cpu().numpy()
        p = attn.mean(axis=0)

        # ...renormalised (Eq. 2) and reduced to its entropy (Eq. 3).
        total = p.sum()
        if total > 0:
            p = p / total
        nz = p > 0
        entropy[layer] = -np.sum(p[nz] * np.log(p[nz])) if np.any(nz) else 0.0

    return {"attention_entropy": entropy, "activation_sparsity": sparsity}


def extract_dataset(
    model,
    tokenizer,
    name: str,
    config: Dict[str, object],
    device,
    num_examples: Optional[int],
) -> Dict[str, np.ndarray]:
    """Average the layer-wise features over the questions of one benchmark."""
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset(f"{HF_NAMESPACE}/{name}", split=config["split"])
    if num_examples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(num_examples, len(dataset))))
    print(f"  {name}: {len(dataset)} examples")

    totals: Optional[Dict[str, np.ndarray]] = None
    n_ok = 0

    for i, example in enumerate(tqdm(dataset, desc=name, leave=False)):
        try:
            features = layerwise_features(
                model, tokenizer, build_prompt(example, config), device
            )
        except Exception as exc:  # a single malformed row must not kill the run
            print(f"  error on example {i} of {name}: {exc}")
            continue

        if totals is None:
            totals = {k: np.zeros_like(v) for k, v in features.items()}
        for k, v in features.items():
            totals[k] += v
        n_ok += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if totals is None:
        raise RuntimeError(f"no example of {name} could be processed")
    return {k: v / n_ok for k, v in totals.items()}


def run_extract(
    cache_dir: str,
    model_name: str,
    num_examples: Optional[int],
    overwrite: bool,
    only: Optional[Sequence[str]] = None,
) -> None:
    """Populate `cache_dir` with one `<dataset>_features.npy` per benchmark."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    selected = list(only) if only else list(DATASETS)
    unknown = [n for n in selected if n not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")

    todo = [
        name
        for name in selected
        if overwrite or not os.path.exists(feature_path(cache_dir, name))
    ]
    if not todo:
        print("All datasets already extracted.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # required for output_attentions
    )
    model.eval()

    for name in todo:
        print(f"=== {name} ===")
        features = extract_dataset(
            model, tokenizer, name, DATASETS[name], device, num_examples
        )
        np.save(feature_path(cache_dir, name), features)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage 2: aggregation
# --------------------------------------------------------------------------- #

def feature_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, f"{name}_features.npy")


def ensure_parent(path: str) -> None:
    """Create the directory a path will be written into, if any."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def aggregate(cache_dir: str) -> Dict[str, np.ndarray]:
    """Stack per-dataset profiles into (n_layers, n_datasets) matrices.

    Columns follow the order of `DATASETS`.
    """
    missing = [n for n in DATASETS if not os.path.exists(feature_path(cache_dir, n))]
    if missing:
        raise FileNotFoundError(
            f"missing features for {', '.join(missing)} in {cache_dir}; "
            "run the `extract` stage first"
        )

    per_dataset = [
        np.load(feature_path(cache_dir, n), allow_pickle=True).item() for n in DATASETS
    ]
    return {
        metric: np.stack([f[metric] for f in per_dataset], axis=1)
        for metric in ("attention_entropy", "activation_sparsity")
    }


# --------------------------------------------------------------------------- #
# Stage 3: plotting
# --------------------------------------------------------------------------- #

def pc1_order(activation_sparsity: np.ndarray) -> np.ndarray:
    """Dataset order given by the first PC of the activation sparsity matrix.

    Each dataset is one sample whose features are its per-layer sparsities, so
    the (n_layers, n_datasets) matrix is transposed before the fit.  Ordering by
    PC1 yields the smooth progression in sparsity seen in the figure.
    """
    from sklearn.decomposition import PCA

    scores = PCA(n_components=1).fit_transform(activation_sparsity.T).flatten()
    return np.argsort(scores)


def plot_figure1(matrices: Dict[str, np.ndarray], output_stem: str) -> None:
    """Draw the two heatmaps and write `<output_stem>.pdf` and `.png`."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = pc1_order(matrices["activation_sparsity"])
    names = [list(DATASETS)[i] for i in order]
    n_layers = matrices["attention_entropy"].shape[0]

    panels = [
        ("attention_entropy", "Attention Entropy"),
        ("activation_sparsity", "Activation Sparsity"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (metric, title) in zip(axes, panels):
        sns.heatmap(
            matrices[metric][:, order],
            annot=False,
            cmap="viridis",
            xticklabels=names,
            yticklabels=[f"Layer {i}" for i in range(n_layers)],
            ax=ax,
        )
        ax.set_title(title, fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=18)
        ax.set_ylabel("Layer", fontsize=14)

    plt.tight_layout()

    ensure_parent(output_stem)
    for ext in ("pdf", "png"):
        path = f"{output_stem}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage",
        nargs="?",
        default="plot",
        choices=["extract", "plot", "all"],
        help="extract features, plot from cached features (default), or both",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_STEM, help="output stem")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="questions per benchmark; -1 for all",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract cached datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help="restrict extraction to these benchmarks (plotting always needs all 24)",
    )
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="also dump the aggregated matrices next to the figure",
    )
    args = parser.parse_args(argv)

    if args.stage in ("extract", "all"):
        run_extract(
            args.cache_dir,
            args.model,
            None if args.num_examples < 0 else args.num_examples,
            args.overwrite,
            args.datasets,
        )

    if args.stage in ("plot", "all"):
        matrices = aggregate(args.cache_dir)
        if args.save_matrices:
            ensure_parent(args.output)
            for metric, matrix in matrices.items():
                np.save(f"{args.output}_{metric}.npy", matrix)
        plot_figure1(matrices, args.output)


if __name__ == "__main__":
    main()
