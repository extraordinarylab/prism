"""Figures 6 and 7 of PRISM: Tracing Knowledge and Reasoning in LLMs.

Robustness to NOTA ("none of the above") perturbations.  Figure 6 covers MathQA
and Figure 7 covers SciQ; they are the same experiment on two benchmarks, so one
script builds both.

Each figure is a 2x2 grid of dumbbell plots.  Rows are the prompting condition
(Standard, then CoT); columns are the perturbation (NOTA-as-answer, then
NOTA-as-distractor).  Every model contributes one horizontal segment running
from its Base accuracy (hollow marker) to its accuracy under the perturbation
(solid marker), so the segment length is the accuracy drop.

The pipeline has two stages:

  run     build the three option sets per question, prompt every model under
          Standard and CoT, and record one row per (model, dataset, prompting,
          condition, item).  Requires a GPU; uses vLLM for batched generation.
  plot    aggregate those rows into the two figures and write the exports.

Usage
-----
    python prism/figure6.py run --model Qwen2.5-7B-Instruct=/path/to/model
    python prism/figure6.py plot

Design (paper Section RQ2.3).  To separate *parametric shortcuts* -- answering
by matching a memorised option -- from *dynamic reasoning*, two perturbations
are applied to a zero-shot multiple-choice item:

  base             the original options, unchanged.
  nota_answer      the ground-truth option is REMOVED and "None of the above" is
                   appended, so NOTA becomes the correct choice.  A model that
                   relies on recognising the gold string has nothing to match.
  nota_distractor  "None of the above" is appended as an ADDITIONAL incorrect
                   option; the correct answer is unchanged.  This tests whether
                   merely enlarging the option set disrupts behaviour.

The item identifier is the benchmark row index and is stable across all three
conditions and both prompting styles, so drops can be estimated pairwise.

Note on the paper's prompts.  The paper does not print the prompt used for the
NOTA stress test (Figure 10 covers representation extraction, Figure 9 the
trajectory analysis).  The templates below follow the paper's described setup --
zero-shot, multiple choice, Standard versus CoT -- and are recorded verbatim in
the run manifest so the exact wording behind these numbers is auditable.
"""

import argparse
import csv
import datetime
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Panel/legend order, matching the published figures.
MODELS = [
    "Llama3.1",
    "Qwen2.5",
    "Olmo3",
    "Gemma3-4B",
    "Gemma3-12B",
    "Gemma3-27B",
]

MODEL_REPOS = {
    "Llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "Olmo3": "allenai/Olmo-3-7B-Instruct",
    "Gemma3-4B": "google/gemma-3-4b-it",
    "Gemma3-12B": "google/gemma-3-12b-it",
    "Gemma3-27B": "google/gemma-3-27b-it",
}

# Figure 6 is MathQA, Figure 7 is SciQ.
DATASETS = {"math-qa": "figure6", "sciq": "figure7"}

CONDITIONS = ["base", "nota_answer", "nota_distractor"]
PERTURBATIONS = ["nota_answer", "nota_distractor"]
PROMPTINGS = ["standard", "cot"]

CONDITION_TITLES = {
    "nota_answer": "Base vs NOTA-as-answer",
    "nota_distractor": "Base vs NOTA-as-distractor",
}
PROMPTING_TITLES = {"standard": "Standard", "cot": "CoT"}

NOTA_TEXT = "None of the above"

# Options already phrased as a none-of-the-above escape hatch.  MathQA ships
# several; injecting another NOTA on top of one changes what the perturbation
# means, so these items are flagged rather than silently mixed in.
NATIVE_NOTA = re.compile(
    r"^\s*(none of (these|the above|them)|not given|no correct answer|"
    r"cannot be determined|data inadequate)\s*\.?\s*$",
    re.IGNORECASE,
)

STANDARD_TEMPLATE = (
    "Answer the following multiple-choice question.\n\n"
    "Question: {question}\n"
    "Choices:\n{choices}\n\n"
    "Respond with one letter only. The answer is"
)

COT_TEMPLATE = (
    "Answer the following multiple-choice question.\n\n"
    "Question: {question}\n"
    "Choices:\n{choices}\n\n"
    "Think step by step, then give your final answer. The last line of your "
    "response must be of the form 'ANSWER: X' where X is one of {letters}."
)

DEFAULT_STEM = "assets/figure6"
FIGURE7_STEM = "assets/figure7"
DEFAULT_NUM_EXAMPLES = 500
DEFAULT_MAX_TOKENS = {"standard": 8, "cot": 512}
DEFAULT_TEMPERATURE = 0.0

# Dumbbell colours, one per model, in MODELS order.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]


# --------------------------------------------------------------------------- #
# Item construction
# --------------------------------------------------------------------------- #

def answer_character(index: int) -> str:
    """Map an option index to its letter: 0 -> 'A', 1 -> 'B', ..."""
    if index < 26:
        return chr(ord("A") + index)
    return str(index - 25)


def render_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{answer_character(i)}) {c}" for i, c in enumerate(choices))


def letters_of(choices: Sequence[str]) -> str:
    return ",".join(answer_character(i) for i in range(len(choices)))


def build_conditions(
    choices: Sequence[str], gold_index: int
) -> Dict[str, Tuple[List[str], int]]:
    """The three option sets for one item, with the gold index in each.

    `nota_answer` drops the gold option, which shifts the letters of everything
    after it, and appends NOTA as the new correct answer.  `nota_distractor`
    keeps the option set intact and appends NOTA as one more wrong answer, so
    the gold index is unchanged.
    """
    base = list(choices)

    without_gold = [c for i, c in enumerate(base) if i != gold_index]
    nota_answer = without_gold + [NOTA_TEXT]

    nota_distractor = base + [NOTA_TEXT]

    return {
        "base": (base, gold_index),
        "nota_answer": (nota_answer, len(nota_answer) - 1),
        "nota_distractor": (nota_distractor, gold_index),
    }


def build_prompt(question: str, choices: Sequence[str], prompting: str) -> str:
    template = STANDARD_TEMPLATE if prompting == "standard" else COT_TEMPLATE
    return template.format(
        question=question,
        choices=render_choices(choices),
        letters=letters_of(choices),
    )


def parse_prediction(text: str, n_choices: int, prompting: str) -> str:
    """Recover the predicted option letter, '' if none could be read.

    Tried in order of how explicit the signal is.  The final fallback is
    case-sensitive under CoT: a bare lowercase "a" is far more likely to be the
    English article inside a reasoning trace than a vote for option A, whereas a
    Standard response is only a few tokens long and often just the letter.
    """
    valid = {answer_character(i) for i in range(n_choices)}

    def first_valid(candidates: Iterable[str]) -> str:
        for candidate in candidates:
            if candidate.upper() in valid:
                return candidate.upper()
        return ""

    # 1. An explicit 'ANSWER: X' line; take the last, which is the conclusion.
    found = first_valid(
        reversed(re.findall(r"ANSWER\s*:\s*\(?\*{0,2}([A-Za-z0-9])", text, re.IGNORECASE))
    )
    if found:
        return found

    # 2. Prose that names the answer.
    found = first_valid(
        reversed(
            re.findall(
                r"answer\s+(?:is|would be|:)\s*\(?\*{0,2}([A-Za-z0-9])\b",
                text,
                re.IGNORECASE,
            )
        )
    )
    if found:
        return found

    # 3. Option-style markup: "(C)" or a line starting "C)".
    for a, b in re.findall(r"\(([A-Za-z0-9])\)|^\s*([A-Za-z0-9])\)", text, re.M):
        found = first_valid([a or b])
        if found:
            return found

    # 4. A standalone letter.
    pattern = r"\b([A-Z0-9])\b" if prompting == "cot" else r"\b([A-Za-z0-9])\b"
    return first_valid(re.findall(pattern, text))


# --------------------------------------------------------------------------- #
# Stage: run
# --------------------------------------------------------------------------- #

def load_items(dataset_name: str, num_examples: int) -> List[Dict]:
    """Questions, options and gold index for one benchmark."""
    from datasets import load_dataset

    splits = load_dataset(f"extraordinarylab/{dataset_name}")
    split = "test" if "test" in splits else "validation"
    dataset = splits[split]
    n = min(num_examples, len(dataset)) if num_examples > 0 else len(dataset)

    items = []
    for i in range(n):
        row = dataset[i]
        choices = list(row["choices"])
        gold = int(row["answer_index"])
        if not 0 <= gold < len(choices):
            continue
        items.append(
            {
                "item_id": str(i),
                "question": row["question"],
                "choices": choices,
                "gold_index": gold,
                # MathQA in particular already offers a none-of-the-above style
                # escape hatch on some items; recorded so it can be excluded.
                "has_native_nota": any(NATIVE_NOTA.match(c) for c in choices),
            }
        )
    return items


def run_model(args: argparse.Namespace, model_name: str, model_path: str) -> List[Dict]:
    """Every (dataset, prompting, condition, item) prediction for one model."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        # Gemma3-it ships a vision tower.  Every prompt here is text, and
        # vLLM's dummy-image profiling run crashes on it, so declare zero
        # images rather than carry the encoder.
        limit_mm_per_prompt={"image": 0},
        # torch.compile needs a usable nvcc, which is not guaranteed on a
        # cluster node; these are single-pass batch jobs, so skip compilation.
        compilation_config={"level": 0},
    )

    rows: List[Dict] = []
    for dataset_name in args.datasets:
        items = load_items(dataset_name, args.num_examples)
        print(f"  {dataset_name}: {len(items)} items")

        for prompting in args.promptings:
            # One batch per (dataset, prompting) covering all three conditions,
            # so vLLM schedules them together.
            prompts, meta = [], []
            for item in items:
                for condition, (choices, gold) in build_conditions(
                    item["choices"], item["gold_index"]
                ).items():
                    text = build_prompt(item["question"], choices, prompting)
                    messages = [{"role": "user", "content": text}]
                    prompts.append(
                        tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
                    meta.append((item, condition, choices, gold))

            sampling = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens[prompting],
                seed=args.seed,
            )
            outputs = llm.generate(prompts, sampling)

            for (item, condition, choices, gold), output in zip(meta, outputs):
                text = output.outputs[0].text
                predicted = parse_prediction(text, len(choices), prompting)
                gold_letter = answer_character(gold)
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "prompting": prompting,
                        "condition": condition,
                        "item_id": item["item_id"],
                        "n_choices": len(choices),
                        "gold_option": gold_letter,
                        "predicted_option": predicted,
                        "is_correct": int(bool(predicted) and predicted == gold_letter),
                        "parsed": int(bool(predicted)),
                        "has_native_nota": int(item["has_native_nota"]),
                        "choices": " | ".join(choices),
                        "response": text.strip().replace("\n", " ")[: args.response_chars],
                    }
                )
            print(f"    {prompting}: {len(outputs)} generations")

    return rows


def run_stage(args: argparse.Namespace) -> None:
    """Generate predictions and append them to the per-item table."""
    requested = args.model_path or {m: MODEL_REPOS[m] for m in args.models}

    rows: List[Dict] = []
    for model_name, model_path in requested.items():
        print(f"=== {model_name} ({model_path}) ===")
        rows.extend(run_model(args, model_name, model_path))

    append_predictions(f"{args.stem}_predictions.csv", rows, args.overwrite)
    write_manifest(args, requested)


PREDICTION_FIELDS = [
    "model",
    "dataset",
    "prompting",
    "condition",
    "item_id",
    "n_choices",
    "gold_option",
    "predicted_option",
    "is_correct",
    "parsed",
    "has_native_nota",
    "choices",
    "response",
]


def append_predictions(path: str, rows: Iterable[Dict], overwrite: bool) -> None:
    """Write rows, replacing any existing rows for the same models."""
    ensure_parent(path)
    rows = list(rows)
    incoming_models = {r["model"] for r in rows}

    existing: List[Dict] = []
    if os.path.exists(path) and not overwrite:
        with open(path, newline="") as f:
            existing = [
                r for r in csv.DictReader(f) if r["model"] not in incoming_models
            ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    print(f"wrote {path} ({len(existing) + len(rows)} rows)")


def write_manifest(args: argparse.Namespace, requested: Dict[str, str]) -> None:
    path = f"{args.stem}_run.json"
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)

    manifest = {
        "run_id": args.run_id,
        "datasets": {"math-qa": "Figure 6", "sciq": "Figure 7"},
        "conditions": {
            "base": "original options, unchanged",
            "nota_answer": (
                "ground-truth option removed and 'None of the above' appended, "
                "which becomes the correct choice"
            ),
            "nota_distractor": (
                "'None of the above' appended as an additional incorrect "
                "option; the correct answer is unchanged"
            ),
        },
        "promptings": list(args.promptings),
        "item_id": (
            "benchmark row index; stable across all three conditions and both "
            "prompting styles, so drops can be estimated pairwise"
        ),
        "num_examples_per_dataset": args.num_examples,
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "prompt_templates": {
            "standard": STANDARD_TEMPLATE,
            "cot": COT_TEMPLATE,
            "note": (
                "the paper does not print the NOTA stress-test prompt; these "
                "follow its described zero-shot multiple-choice setup"
            ),
        },
        "answer_parsing": (
            "CoT reads the last 'ANSWER: X' naming a valid option, else the "
            "first standalone valid option letter; Standard reads the first "
            "standalone valid option letter. Unparseable responses are scored "
            "incorrect and flagged with parsed=0"
        ),
        "has_native_nota": (
            "the item already offered a none-of-the-above style option before "
            "perturbation (common in MathQA); exclude these for a clean read"
        ),
        "uncertainty": (
            "aggregate_*.csv reports paired drops with a percentile bootstrap "
            "over items and McNemar counts, both computed on items answered "
            "under both conditions"
        ),
    }
    existing_models = existing.get("models", {}) if isinstance(existing, dict) else {}
    manifest["models"] = {**existing_models, **requested}

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def load_predictions(path: str, exclude_native_nota: bool) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the `run` stage first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if exclude_native_nota:
        rows = [r for r in rows if r["has_native_nota"] != "1"]
    return rows


def index_predictions(
    rows: Sequence[Dict],
) -> Dict[Tuple[str, str, str, str], Dict[str, Tuple[int, int]]]:
    """(model, dataset, prompting, condition) -> {item_id: (correct, parsed)}."""
    index: Dict[Tuple[str, str, str, str], Dict[str, Tuple[int, int]]] = {}
    for r in rows:
        k = (r["model"], r["dataset"], r["prompting"], r["condition"])
        index.setdefault(k, {})[r["item_id"]] = (int(r["is_correct"]), int(r["parsed"]))
    return index


def paired_drop(
    base: Dict[str, int], perturbed: Dict[str, int], n_boot: int, seed: int
) -> Dict[str, float]:
    """Accuracy under each condition and the paired drop, with uncertainty.

    Only items answered under both conditions enter the comparison, so the two
    accuracies and their difference refer to exactly the same items.  The
    interval is a percentile bootstrap resampling those items; `b` and `c` are
    the McNemar discordant counts (base right / perturbed wrong, and the
    reverse).
    """
    shared = sorted(set(base) & set(perturbed))
    if not shared:
        return {}
    b_vec = np.array([base[i][0] for i in shared])
    p_vec = np.array([perturbed[i][0] for i in shared])
    b_parsed = np.array([base[i][1] for i in shared])
    p_parsed = np.array([perturbed[i][1] for i in shared])
    diff = b_vec - p_vec

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    boot = diff[idx].mean(axis=1)

    return {
        "n_items": len(shared),
        "base_accuracy": float(b_vec.mean()),
        "perturbed_accuracy": float(p_vec.mean()),
        "drop": float(diff.mean()),
        "drop_ci_low": float(np.percentile(boot, 2.5)),
        "drop_ci_high": float(np.percentile(boot, 97.5)),
        "mcnemar_b": int(((b_vec == 1) & (p_vec == 0)).sum()),
        "mcnemar_c": int(((b_vec == 0) & (p_vec == 1)).sum()),
        # A response with no readable option letter is scored incorrect, so a
        # low parse rate inflates the apparent drop.  Reported so it is visible.
        "base_parse_rate": float(b_parsed.mean()),
        "perturbed_parse_rate": float(p_parsed.mean()),
    }


def aggregate(rows: Sequence[Dict], args) -> List[Dict]:
    """Paired accuracy drops for every model x dataset x prompting x perturbation."""
    index = index_predictions(rows)
    models = [m for m in args.models if any(k[0] == m for k in index)]

    out: List[Dict] = []
    for model in models:
        for dataset in args.datasets:
            for prompting in args.promptings:
                base = index.get((model, dataset, prompting, "base"))
                if not base:
                    continue
                for condition in PERTURBATIONS:
                    perturbed = index.get((model, dataset, prompting, condition))
                    if not perturbed:
                        continue
                    stats = paired_drop(base, perturbed, args.n_bootstrap, args.seed)
                    if stats:
                        out.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "prompting": prompting,
                                "condition": condition,
                                **stats,
                            }
                        )
    return out


# --------------------------------------------------------------------------- #
# Stage: plot
# --------------------------------------------------------------------------- #

def plot_dataset(summary: Sequence[Dict], dataset: str, stem: str, args) -> None:
    """The 2x2 dumbbell grid for one benchmark."""
    import matplotlib.pyplot as plt

    models = [m for m in args.models if any(s["model"] == m for s in summary)]
    if not models:
        print(f"no results for {dataset}; skipping {stem}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    lookup = {(s["prompting"], s["condition"], s["model"]): s for s in summary}

    for row, prompting in enumerate(args.promptings):
        for col, condition in enumerate(PERTURBATIONS):
            ax = axes[row, col]
            for i, model in enumerate(models):
                s = lookup.get((prompting, condition, model))
                if s is None:
                    continue
                y = len(models) - 1 - i
                color = COLORS[i % len(COLORS)]
                ax.plot(
                    [s["perturbed_accuracy"], s["base_accuracy"]],
                    [y, y],
                    color=color,
                    linewidth=2.5,
                    zorder=1,
                )
                ax.scatter(
                    s["base_accuracy"], y, facecolors="none", edgecolors=color,
                    s=70, linewidths=2, zorder=2,
                )
                ax.scatter(
                    s["perturbed_accuracy"], y, color=color, s=70, zorder=2,
                )

            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(list(reversed(models)), fontsize=11)
            ax.set_title(
                f"{PROMPTING_TITLES[prompting]}: {CONDITION_TITLES[condition]}",
                fontsize=13,
            )
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if row == len(args.promptings) - 1:
                ax.set_xlabel("Accuracy", fontsize=12)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor="black", markersize=9, label="Base (original options)",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color="black", markersize=9,
            label="Perturbed condition",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    ensure_parent(stem)
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def export_aggregate(summary: Sequence[Dict], path: str) -> None:
    ensure_parent(path)
    fields = [
        "model",
        "dataset",
        "prompting",
        "condition",
        "n_items",
        "base_accuracy",
        "perturbed_accuracy",
        "drop",
        "drop_ci_low",
        "drop_ci_high",
        "mcnemar_b",
        "mcnemar_c",
        "base_parse_rate",
        "perturbed_parse_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summary:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()})
    print(f"wrote {path} ({len(summary)} rows)")


def plot_stage(args: argparse.Namespace) -> None:
    rows = load_predictions(f"{args.stem}_predictions.csv", args.exclude_native_nota)
    summary = aggregate(rows, args)
    if not summary:
        raise SystemExit("no complete base/perturbed pairs found in the predictions")

    export_aggregate(summary, f"{args.stem}_aggregate.csv")
    stems = {"math-qa": args.stem, "sciq": args.figure7_stem}
    for dataset in DATASETS:
        subset = [s for s in summary if s["dataset"] == dataset]
        plot_dataset(subset, dataset, stems[dataset], args)


# --------------------------------------------------------------------------- #

def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_model_path(values: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--model expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        mapping[name] = os.path.expanduser(path)
    return mapping


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="?", default="plot", choices=["run", "plot"])
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="stem for Figure 6 (MathQA) and for every data file",
    )
    parser.add_argument(
        "--figure7-stem", default=FIGURE7_STEM, help="stem for Figure 7 (SciQ)"
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--promptings", nargs="+", default=PROMPTINGS)
    parser.add_argument(
        "--model",
        dest="model_path_args",
        nargs="+",
        metavar="NAME=PATH",
        help="models to run, e.g. Gemma3-4B=/scratch/gemma-3-4b-it",
    )
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens-standard", type=int, default=DEFAULT_MAX_TOKENS["standard"])
    parser.add_argument("--max-tokens-cot", type=int, default=DEFAULT_MAX_TOKENS["cot"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--response-chars", type=int, default=600)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the whole predictions file instead of only this run's models",
    )
    parser.add_argument(
        "--exclude-native-nota",
        action="store_true",
        help="drop items whose original options already contained a "
        "none-of-the-above style choice",
    )
    args = parser.parse_args(argv)

    args.model_path = parse_model_path(args.model_path_args)
    args.max_tokens = {
        "standard": args.max_tokens_standard,
        "cot": args.max_tokens_cot,
    }
    if args.run_id is None:
        args.run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    if args.stage == "run":
        run_stage(args)
    else:
        plot_stage(args)


if __name__ == "__main__":
    main()
