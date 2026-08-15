"""Figure 2 of PRISM: Tracing Knowledge and Reasoning in LLMs.

Decoding-time trajectory analysis on SciQ using Qwen2.5-7B-Instruct.  Token-wise
attention entropy and activation sparsity are traced during decoding for
representative lower/middle/upper depths (layers 3, 14 and 22), shown separately
for correct (left) and incorrect (right) predictions.  Shaded regions denote
variability across examples.

The pipeline has two stages:

  trace   generate an answer for each SciQ question with the retrieve-then-answer
          prompt, score it, then replay the generation prefix by prefix and
          record the two metrics at each of the three depths.  Requires a GPU.
  plot    average the per-example trajectories within the correct and incorrect
          groups and draw the 2x2 comparison figure.

Usage
-----
    python prism/figure2.py all --model Qwen/Qwen2.5-7B-Instruct   # needs a GPU
    python prism/figure2.py plot                                   # redraw only

Everything a run produces -- figure, per-generation trajectories, per-example
table and run manifest -- shares one path stem, `assets/figure2` by default.

Method (paper Section 3.2, "Decoding-Time Trajectories").  We trace the same
signals as the static profile over all available SciQ examples.  Generation uses
temperature 0.7, top-p 0.8 and a maximum of 512 new tokens.  To compare
responses of different lengths, P uniformly spaced positions are selected from a
generation of length G:

    i_k = floor( k (G - 1) / (P - 1) ),   k in {0, ..., P - 1}          (6)

The figure caption length-normalises, giving every generation the same P = 101
positions (--alignment stretch, the default); Eq. 6 as written instead sets
P = min(G, 100), leaving ragged trajectories (--alignment truncate).  See the
README -- the choice visibly changes the figure.

At each selected position the generation prefix is fed back through the model
and, at the final token, we read the same attention entropy (Eqs. 1-3) and
activation sparsity (Eq. 4) used for Figure 1.

Note on exactness.  Generation is stochastic (temperature 0.7) and the reasoning
traces behind the published Figure 2 were not cached, so a re-run reproduces the
figure's structure and conclusions rather than its exact curves.  Pass --seed to
make your own runs repeatable; traces are cached to JSONL so replotting and
re-tracing reuse the same generations.
"""

import argparse
import csv
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Representative lower / middle / upper depths (1-indexed Transformer layers).
DEFAULT_LAYERS = [3, 14, 22]
LAYER_NAMES = ["Lower Layer", "Middle Layer", "Upper Layer"]

# Metrics traced during decoding, in plot order (one figure row each).
METRIC_TITLES = {
    "attention_entropy": "Attention Entropy",
    "activation_sparsity": "Activation Sparsity",
}

SPARSITY_THRESHOLD = 0.01

# Uniformly spaced positions sampled per generation (P in Eq. 6).  The figure
# caption specifies 101, giving an x-axis of 0..100.
N_POSITIONS = 101

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = "extraordinarylab/sciq"
DEFAULT_NUM_EXAMPLES = 100
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8

# Figure, trajectories, per-example table and run manifest all share this stem,
# so a run's outputs and the data behind them stay together.
DEFAULT_STEM = "assets/figure2"

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The structured retrieve-then-answer prompt (paper Figure 9).
INSTRUCTION = (
    "Identify key facts and technical principles related to the following "
    "question. Based only on the knowledge retrieved above, analyze the "
    "following question. The last line of your response should be of the "
    "following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is "
    "one of {letters}."
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green


# --------------------------------------------------------------------------- #
# Prompt construction and answer scoring
# --------------------------------------------------------------------------- #

def answer_character(index: int) -> str:
    """Map an option index to its letter: 0 -> 'A', 1 -> 'B', ..."""
    if index < 26:
        return chr(ord("A") + index)
    return str(index - 25)


def answer_options(choices: Sequence[str]) -> str:
    """Render choices as "A) choice 1\\nB) choice 2\\n..."."""
    return "\n".join(f"{answer_character(i)}) {c}" for i, c in enumerate(choices))


def format_letter_choices(choices: Sequence[str]) -> str:
    """Render the admissible answer letters as "A,B,C,D"."""
    return ",".join(answer_character(i) for i in range(len(choices)))


def format_example(question: str, choices: Optional[Sequence[str]] = None) -> str:
    """Build the retrieve-then-answer prompt for one question."""
    letters = format_letter_choices(choices) if choices else ""
    instruction = INSTRUCTION.format(letters=letters)
    if choices:
        return f"{instruction}\n{question}\n{answer_options(choices)}"
    return f"{instruction}\n{question}"


def build_question(example: Dict) -> Tuple[str, Optional[List[str]], str]:
    """Pull question text, choices and gold answer out of a raw dataset row."""
    question = example.get("question", "")
    if "context" in example:
        question = f"{example['context']}\n{question}"
    return question, example.get("choices"), example.get("answer", "")


def extract_predicted_answer(response: str) -> str:
    """Read the letter out of the trailing 'ANSWER: X' line, '' if absent."""
    match = re.search(r"ANSWER:\s*([A-Z0-9])\s*$", response, re.MULTILINE | re.IGNORECASE)
    return match.group(1).upper() if match else ""


def correct_answer_letter(answer: str, choices: Optional[Sequence[str]]) -> str:
    """Normalise the gold answer to a letter, accepting letters or option text."""
    if not choices:
        return ""
    letters = [answer_character(i) for i in range(len(choices))]
    if len(answer) == 1 and answer.upper() in letters:
        return answer.upper()
    for i, choice in enumerate(choices):
        if choice == answer:
            return answer_character(i)
    return ""


def sample_positions(
    n_tokens: int,
    n_positions: int = N_POSITIONS,
    alignment: str = "stretch",
) -> List[int]:
    """Uniformly spaced prefix positions over a generation of `n_tokens` words.

    With `alignment="stretch"` every generation contributes exactly
    `n_positions` samples, so the x-axis becomes relative progress through the
    generation and all generations start and end at the same x.  This is the
    length normalisation the Figure 2 caption describes; short generations
    simply repeat prefix indices.

    With `alignment="truncate"` a generation shorter than `n_positions` yields
    only `n_tokens` samples, i.e. P = min(G, n_positions) as written in Eq. 6.
    The x-axis is then an absolute prefix index and trajectories are ragged.
    """
    if n_tokens <= 0:
        return []
    positions = min(n_tokens, n_positions) if alignment == "truncate" else n_positions
    if positions <= 1:
        return [0]
    return [int(k * (n_tokens - 1) / (positions - 1)) for k in range(positions)]


# --------------------------------------------------------------------------- #
# Stage 1: generation and tracing
# --------------------------------------------------------------------------- #

def prefix_metrics(model, tokenizer, text: str, layers: Sequence[int]) -> Dict[str, List[float]]:
    """Attention entropy and activation sparsity at the final token of `text`.

    `layers` are 1-indexed Transformer layers: layer l corresponds to
    `hidden_states[l]` (index 0 being the embedding output) and to
    `attentions[l - 1]`.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    entropies, sparsities = [], []
    for layer in layers:
        hidden = outputs.hidden_states[layer][0, -1].cpu().float().numpy()
        sparsities.append(float(np.mean(np.abs(hidden) < SPARSITY_THRESHOLD)))

        attn = outputs.attentions[layer - 1][0, :, -1, :].cpu().float().numpy()
        p = attn.mean(axis=0)
        total = p.sum()
        if total > 0:
            p = p / total
        nz = p > 0
        entropies.append(float(-np.sum(p[nz] * np.log(p[nz]))) if np.any(nz) else 0.0)

    return {"attention_entropy": entropies, "activation_sparsity": sparsities}


def generate_response(
    model, tokenizer, prompt: str, args: argparse.Namespace
) -> str:
    """Sample one answer with the chat template applied."""
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    new_tokens = generated[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_trace_cache(path: str) -> Dict[str, Dict]:
    """Read cached generations keyed by example id."""
    cache: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[record["example_id"]] = record
    return cache


def run_trace(args: argparse.Namespace) -> None:
    """Generate answers, replay their prefixes, and cache the trajectories."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    traces_path = f"{args.stem}_reasoning_traces.jsonl"
    ensure_parent(traces_path)
    cache = load_trace_cache(traces_path)

    splits = load_dataset(args.dataset)
    split = "test" if "test" in splits else "validation"
    dataset = splits[split].shuffle(seed=42)
    if args.num_examples > 0:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    model.set_attn_implementation("eager")  # required for output_attentions
    model.eval()

    # trajectories[metric] collects one (n_layers, n_positions) array per example;
    # `examples` keeps the per-example bookkeeping that goes with it.
    trajectories: Dict[str, List[np.ndarray]] = {m: [] for m in METRIC_TITLES}
    examples: List[Dict] = []

    for example_idx, example in enumerate(tqdm(dataset, desc="examples")):
        example_id = str(example.get("id", example_idx))
        question, choices, answer = build_question(example)
        prompt = format_example(question, choices)

        record = cache.get(example_id)
        if record is None:
            response = generate_response(model, tokenizer, prompt, args)
            record = {
                "example_id": example_id,
                "question": question,
                "choices": list(choices) if choices else None,
                "answer": answer,
                "reasoning_trace": response,
                "predicted_answer": extract_predicted_answer(response),
            }
            with open(traces_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            cache[example_id] = record

        response = record["reasoning_trace"]
        if not response.split():
            print(f"skipping example {example_id}: empty generation")
            continue

        predicted = record["predicted_answer"]
        gold = correct_answer_letter(answer, choices)

        # Prefixes are cut on whitespace tokens and fed back as raw text (no chat
        # template), matching how the trajectory was originally traced.
        words = response.split()
        positions = sample_positions(len(words), args.positions, args.alignment)

        examples.append(
            {
                "example_id": example_id,
                "question": question,
                "choices": list(choices) if choices else None,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "is_correct": bool(predicted) and predicted == gold,
                "generation_words": len(words),
                "source_index": positions,
            }
        )
        per_metric = {
            m: np.full((len(args.layers), len(positions)), np.nan) for m in METRIC_TITLES
        }

        # Stretching a short generation to a fixed width repeats prefix indices;
        # each distinct prefix only needs one forward pass.
        computed: Dict[int, Dict[str, List[float]]] = {}
        for slot, token_idx in enumerate(positions):
            if token_idx not in computed:
                prefix = " ".join(words[: token_idx + 1])
                computed[token_idx] = prefix_metrics(
                    model, tokenizer, f"{prompt}\n{prefix}", args.layers
                )
            for metric, per_layer in computed[token_idx].items():
                per_metric[metric][:, slot] = per_layer

        for metric, array in per_metric.items():
            trajectories[metric].append(array)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_run(args, trajectories, examples)


def save_run(
    args: argparse.Namespace,
    trajectories: Dict[str, List[np.ndarray]],
    examples: List[Dict],
) -> None:
    """Write the per-example trajectories, their metadata, and a run manifest.

    Five files are produced under the output stem:

      _trajectories.npz    (n_examples, n_layers, n_positions) arrays plus the
                           per-example bookkeeping needed to interpret them
      _trajectories.csv    the same values in tidy long form, one row per
                           (example, layer, position)
      _examples.csv        one row per example: identity, scoring, length
      _run.json            model, decoding and sampling settings, group counts
    """
    stem = args.stem
    ensure_parent(f"{stem}_run.json")

    n_examples = len(examples)
    n_layers = len(args.layers)
    width = max(a.shape[1] for a in trajectories["attention_entropy"])

    # Ragged only under --alignment truncate; NaN marks positions a short
    # generation never reached.
    arrays = {}
    for metric, per_example in trajectories.items():
        padded = np.full((n_examples, n_layers, width), np.nan)
        for i, a in enumerate(per_example):
            padded[i, :, : a.shape[1]] = a
        arrays[metric] = padded

    source_index = np.full((n_examples, width), -1, dtype=int)
    for i, ex in enumerate(examples):
        source_index[i, : len(ex["source_index"])] = ex["source_index"]

    is_correct = np.asarray([ex["is_correct"] for ex in examples], dtype=bool)
    generation_words = np.asarray([ex["generation_words"] for ex in examples])
    n_positions = np.asarray([len(ex["source_index"]) for ex in examples])
    example_ids = np.asarray([ex["example_id"] for ex in examples])

    npz_path = f"{stem}_trajectories.npz"
    np.savez(
        npz_path,
        example_ids=example_ids,
        is_correct=is_correct,
        layers=np.asarray(args.layers),
        layer_names=np.asarray(LAYER_NAMES[:n_layers]),
        generation_words=generation_words,
        n_positions=n_positions,
        source_index=source_index,
        **arrays,
    )
    print(f"wrote {npz_path}")

    # Tidy long form: one row per (example, layer, position).
    rows_path = f"{stem}_trajectories.csv"
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "is_correct",
                "layer",
                "layer_role",
                "position",
                "position_normalised",
                "source_word_index",
                "generation_words",
                "attention_entropy",
                "activation_sparsity",
            ]
        )
        for i, ex in enumerate(examples):
            denom = max(len(ex["source_index"]) - 1, 1)
            span = max(ex["generation_words"] - 1, 1)
            for layer_slot, layer in enumerate(args.layers):
                for pos, word_idx in enumerate(ex["source_index"]):
                    writer.writerow(
                        [
                            ex["example_id"],
                            int(ex["is_correct"]),
                            layer,
                            LAYER_NAMES[layer_slot],
                            pos,
                            # Progress through the generation, 0 at the first
                            # sampled prefix and 1 at the last.
                            round(word_idx / span, 6),
                            word_idx,
                            ex["generation_words"],
                            round(float(arrays["attention_entropy"][i, layer_slot, pos]), 6),
                            round(float(arrays["activation_sparsity"][i, layer_slot, pos]), 6),
                        ]
                    )
    print(f"wrote {rows_path} ({n_examples * n_layers * width} rows)")

    examples_path = f"{stem}_examples.csv"
    with open(examples_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_id",
                "question",
                "choices",
                "gold_answer",
                "predicted_answer",
                "is_correct",
                "generation_words",
                "n_positions",
            ]
        )
        for ex in examples:
            writer.writerow(
                [
                    ex["example_id"],
                    ex["question"],
                    " | ".join(ex["choices"]) if ex["choices"] else "",
                    ex["gold_answer"],
                    ex["predicted_answer"],
                    int(ex["is_correct"]),
                    ex["generation_words"],
                    len(ex["source_index"]),
                ]
            )
    print(f"wrote {examples_path}")

    n_correct = int(is_correct.sum())
    n_incorrect = n_examples - n_correct
    manifest = {
        "run_id": args.run_id,
        "model": args.model,
        "dataset": args.dataset,
        "num_examples_requested": args.num_examples,
        "num_examples_traced": n_examples,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": round(n_correct / n_examples, 4) if n_examples else None,
        "layers": list(args.layers),
        "layer_names": LAYER_NAMES[:n_layers],
        "alignment": args.alignment,
        "n_positions": args.positions,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "seed": args.seed,
            "system_prompt": args.system_prompt,
        },
        "generation_words": {
            "min": int(generation_words.min()),
            "median": float(np.median(generation_words)),
            "max": int(generation_words.max()),
        },
        "sparsity_threshold": SPARSITY_THRESHOLD,
        "shaded_region": (
            "mean +/- 1 population standard deviation (numpy std, ddof=0) across "
            "examples within the group, computed independently at each position "
            "and layer; not a standard error or confidence interval"
        ),
        "prefix_replay": (
            "generation prefixes are cut on whitespace words and appended to the "
            "raw prompt without the chat template; metrics are read at the final "
            "token of that text"
        ),
    }
    manifest_path = f"{stem}_run.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")
    print(f"correct: {n_correct}, incorrect: {n_incorrect}")


# --------------------------------------------------------------------------- #
# Stage 2: plotting
# --------------------------------------------------------------------------- #

def group_stats(
    padded: np.ndarray, padding: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation across examples at each position.

    `padded` is (n_examples, n_layers, n_positions) with NaN beyond the end of
    each example's trajectory.  With `padding="zeros"` those NaNs become zeros
    before averaging, reproducing the original implementation; with
    `padding="nan"` they are excluded, so short generations do not drag the tail
    of the curve toward zero.
    """
    if padding == "zeros":
        filled = np.nan_to_num(padded, nan=0.0)
        return filled.mean(axis=0), filled.std(axis=0)
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def shared_ylim(
    stats: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], metric: str
) -> Tuple[float, float]:
    """y-range covering the mean +/- std bands of both groups, plus 10% buffer."""
    lo, hi = np.inf, -np.inf
    for group in stats.values():
        mean, std = group[metric]
        lo = min(lo, float(np.min(mean - std)))
        hi = max(hi, float(np.max(mean + std)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    buffer = (hi - lo) * 0.1
    return max(0.0, lo - buffer), hi + buffer


def plot_figure2(npz_path: str, output_stem: str, padding: str) -> None:
    """Draw the 2x2 correct/incorrect trajectory comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = np.load(npz_path)
    is_correct = data["is_correct"]
    n_correct = int(is_correct.sum())
    n_incorrect = int((~is_correct).sum())
    if n_correct == 0 or n_incorrect == 0:
        raise ValueError("need both correct and incorrect examples to compare")
    print(f"correct: {n_correct}, incorrect: {n_incorrect}")

    groups = {"Correct": is_correct, "Incorrect": ~is_correct}
    stats = {
        name: {m: group_stats(data[m][mask], padding) for m in METRIC_TITLES}
        for name, mask in groups.items()
    }

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))

    for row, (metric, metric_title) in enumerate(METRIC_TITLES.items()):
        ylim = shared_ylim(stats, metric)
        for col, group in enumerate(groups):
            ax = axes[row, col]
            mean, std = stats[group][metric]
            x = np.arange(mean.shape[1])
            for layer_idx, layer_name in enumerate(LAYER_NAMES):
                ax.plot(
                    x,
                    mean[layer_idx],
                    label=layer_name,
                    color=COLORS[layer_idx],
                    linewidth=2,
                )
                ax.fill_between(
                    x,
                    mean[layer_idx] - std[layer_idx],
                    mean[layer_idx] + std[layer_idx],
                    alpha=0.3,
                    color=COLORS[layer_idx],
                )
            ax.set_title(f"{group} Predictions: {metric_title}", fontsize=22)
            ax.set_xlabel("Generated Token Index", fontsize=18)
            ax.set_ylabel(f"{metric_title} Value", fontsize=18)
            ax.legend(loc="upper right", fontsize=18)
            ax.grid(True)
            ax.set_ylim(ylim)

    plt.tight_layout()
    ensure_parent(output_stem)
    for ext in ("pdf", "png"):
        path = f"{output_stem}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def ensure_parent(path: str) -> None:
    """Create the directory a path will be written into, if any."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage",
        nargs="?",
        default="plot",
        choices=["trace", "plot", "all"],
        help="trace decoding trajectories, plot cached ones (default), or both",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="path stem shared by the figure and every data file it is built "
        "from (default assets/figure2)",
    )
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=None, help="sampling seed")
    parser.add_argument(
        "--run-id",
        default=None,
        help="identifier recorded in the run manifest (default: UTC timestamp)",
    )
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--layers",
        type=int,
        nargs=3,
        default=DEFAULT_LAYERS,
        metavar=("LOWER", "MIDDLE", "UPPER"),
        help="1-indexed Transformer layers to trace",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=N_POSITIONS,
        help="positions sampled per generation (default 101, per the caption)",
    )
    parser.add_argument(
        "--alignment",
        choices=["stretch", "truncate"],
        default="stretch",
        help="'stretch' gives every generation the same number of positions, so "
        "the x-axis is relative progress (the caption's length normalisation); "
        "'truncate' takes P = min(G, positions) as written in Eq. 6, leaving "
        "ragged trajectories",
    )
    parser.add_argument(
        "--padding",
        choices=["zeros", "nan"],
        default="zeros",
        help="only applies to --alignment truncate: how to treat positions past "
        "the end of a short generation when averaging ('zeros' reproduces the "
        "original code, 'nan' excludes them)",
    )
    args = parser.parse_args(argv)
    if args.run_id is None:
        args.run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    if args.stage in ("trace", "all"):
        run_trace(args)

    if args.stage in ("plot", "all"):
        plot_figure2(f"{args.stem}_trajectories.npz", args.stem, args.padding)


if __name__ == "__main__":
    main()
