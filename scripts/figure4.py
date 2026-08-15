"""Figures 4 and 8 of PRISM: Tracing Knowledge and Reasoning in LLMs.

Both figures are two views of one measurement, so one script builds both.

  Figure 4  Semantic divergence between ground-truth and distractor
            representations: layer-wise cosine similarity between the hidden
            states of a prompt appended with the correct answer versus with an
            incorrect one, one panel per model, one line per benchmark.
  Figure 8  The across-benchmark standard deviation of those same layer means,
            i.e. how far apart the four benchmarks sit at each layer.

The pipeline has three stages:

  encode    run the models over the four diagnostic benchmarks, measure the
            per-question correct-vs-distractor cosine similarity at every layer,
            and measure the random-pair baseline of Eq. 8.  Requires a GPU.
  ingest    build the same store from the repository's cached
            `similarity_results/*.json` instead.  These hold the per-question
            cosine similarities behind the published figures but no baseline, so
            the Eq. 8 adjustment is left empty.
  plot      draw Figure 4 and Figure 8 and export the per-question tables.

Usage
-----
    python prism/figure4.py ingest && python prism/figure4.py plot   # no GPU
    python prism/figure4.py encode --model-path ...                  # needs a GPU

Method (paper Section 3.3, "Representation Similarity").  For each question q we
append either the correct answer a+ or an incorrect distractor a-:

    x+ = [q; a+],      x- = [q; a-]                                  (7)

extract their final-token hidden states at each layer, and compute an adjusted
cosine similarity

    sim_adj(h+, h-) = cos(h+, h-) - E_{i,j}[ cos(h_i, h_j) ]         (8)

where the expectation is over randomly paired completions from the same
benchmark.  Lower values mean the correct and incorrect completions are held
further apart in representation space.

Two adjustments.  Eq. 8 subtracts the random-pair baseline above.  The published
figures instead subtract, per layer, the mean of the four benchmarks' layer
means for that model -- a cross-benchmark centring, which is what the y-axis
label "Normalized Cosine Similarity" refers to.  Both are computed and exported;
`--adjustment` selects which one the figures use, defaulting to the published
one.  See the README.
"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Panel order in both figures.
MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Olmo-3-7B-Instruct",
]

# Line order in Figure 4; also the four benchmarks Figure 8 takes the std over.
# Two reasoning-centric (logiqa, math-qa) and two knowledge-centric (biomix-qa,
# sciq), matching the balanced diagnostic subset of Section 3.3.
BENCHMARKS = ["logiqa", "math-qa", "biomix-qa", "sciq"]

HF_NAMESPACE = "extraordinarylab"

# Hugging Face repo ids used by the `encode` stage; override with --model-path.
MODEL_REPOS = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Olmo-3-7B-Instruct": "allenai/Olmo-3-7B-Instruct",
}

LEGACY_RESULTS_DIR = "similarity_results"
DEFAULT_STEM = "assets/figure4"
FIGURE8_STEM = "assets/figure8"

# Questions per benchmark.
DEFAULT_NUM_EXAMPLES = 100

SYSTEM_PROMPT = "You are a helpful assistant."

# With the chat template applied and `add_generation_prompt=True`, the sequence
# ends with the assistant generation prompt.  The published measurements read the
# hidden state this many positions back from the end, landing on the last token
# of the appended answer rather than on the generation prompt.
TEMPLATE_OFFSET = 6

MARKERS = ["o", "s", "D", "^"]
LINESTYLES = ["-", "--", "-.", ":"]


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
# One npz holds every model/benchmark pair under composite keys, because layer
# counts differ across models (29 for Qwen2.5-7B, 33 for the 8B models).

def key(model: str, benchmark: str, field: str) -> str:
    return f"{model}|{benchmark}|{field}"


def save_store(stem: str, store: Dict[str, Dict[str, Dict]], manifest: Dict) -> None:
    """Write the per-question measurements plus a manifest."""
    ensure_parent(f"{stem}_similarity.npz")
    arrays = {}
    for model, per_benchmark in store.items():
        for benchmark, rec in per_benchmark.items():
            arrays[key(model, benchmark, "cosine")] = np.asarray(rec["cosine"])
            arrays[key(model, benchmark, "question_ids")] = np.asarray(
                rec["question_ids"], dtype=object
            ).astype(str)
            for field in ("questions", "correct_answers", "distractor_answers"):
                arrays[key(model, benchmark, field)] = np.asarray(
                    rec[field], dtype=object
                ).astype(str)
            for field in ("baseline_mean", "baseline_std", "published_cosine"):
                if rec.get(field) is not None:
                    arrays[key(model, benchmark, field)] = np.asarray(rec[field])

    path = f"{stem}_similarity.npz"
    np.savez(path, **arrays)
    print(f"wrote {path}")

    manifest_path = f"{stem}_run.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


def load_store(stem: str) -> Tuple[Dict[str, Dict[str, Dict]], Dict]:
    """Read back what `save_store` wrote."""
    path = f"{stem}_similarity.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run the `ingest` stage (cached results, no GPU) "
            "or the `encode` stage first"
        )
    data = np.load(path, allow_pickle=False)
    with open(f"{stem}_run.json") as f:
        manifest = json.load(f)

    store: Dict[str, Dict[str, Dict]] = {}
    for model in manifest["models"]:
        store[model] = {}
        for benchmark in manifest["benchmarks"]:
            rec = {
                "cosine": data[key(model, benchmark, "cosine")],
                "question_ids": data[key(model, benchmark, "question_ids")],
                "questions": data[key(model, benchmark, "questions")],
                "correct_answers": data[key(model, benchmark, "correct_answers")],
                "distractor_answers": data[key(model, benchmark, "distractor_answers")],
            }
            for field in ("baseline_mean", "baseline_std", "published_cosine"):
                k = key(model, benchmark, field)
                rec[field] = data[k] if k in data.files else None
            store[model][benchmark] = rec
    return store, manifest


# --------------------------------------------------------------------------- #
# Stage: ingest cached results
# --------------------------------------------------------------------------- #

def run_ingest(args: argparse.Namespace) -> None:
    """Build the store from the repository's cached per-question JSONs.

    These are the measurements behind the published figures.  They record the
    correct-distractor pairing and the per-layer cosine similarity, but not the
    hidden states, so the Eq. 8 random-pair baseline cannot be recovered from
    them and is left unset.
    """
    store: Dict[str, Dict[str, Dict]] = {}
    for model in args.models:
        store[model] = {}
        for benchmark in args.benchmarks:
            path = os.path.join(
                args.legacy_dir, model, HF_NAMESPACE, f"{benchmark}.json"
            )
            with open(path) as f:
                rows = json.load(f)
            cosine = np.asarray([r["cosine_similarity"] for r in rows])
            store[model][benchmark] = {
                "cosine": cosine,
                # For an ingest store the measured and published values are the
                # same numbers; keeping both makes the column meaning uniform.
                "published_cosine": cosine,
                "question_ids": [str(r["id"]) for r in rows],
                "questions": [r["question"] for r in rows],
                "correct_answers": [r["correct_answer"] for r in rows],
                "distractor_answers": [r["incorrect_answer"] for r in rows],
                "baseline_mean": None,
                "baseline_std": None,
            }
            print(f"  {model}/{benchmark}: {store[model][benchmark]['cosine'].shape}")

    save_store(args.stem, store, build_manifest(args, store, source="cached-json"))


# --------------------------------------------------------------------------- #
# Stage: encode
# --------------------------------------------------------------------------- #

def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    batch_size: int,
    use_template: bool,
    template_offset: int,
) -> np.ndarray:
    """Final-token hidden state of every text, at every layer.

    Returns an array of shape (n_layers, n_texts, hidden_size).
    """
    import torch

    per_layer: List[List] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        if use_template:
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
        else:
            rendered = batch

        inputs = tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Left padding puts every sequence's end at the same index, so a fixed
        # negative offset selects the same relative position for the batch.
        pos = -template_offset if use_template else -1
        states = [hs[:, pos, :].float().cpu().numpy() for hs in outputs.hidden_states]
        if not per_layer:
            per_layer = [[] for _ in states]
        for layer, s in enumerate(states):
            per_layer[layer].append(s)

    return np.stack([np.concatenate(chunks, axis=0) for chunks in per_layer])


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (n, d) matrices."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_n * b_n, axis=1)


def random_pair_baseline(
    pool: np.ndarray, n_pairs: Optional[int], rng: np.random.Generator
) -> Tuple[float, float, int]:
    """E[cos(h_i, h_j)] over distinct pairs drawn from one benchmark's pool.

    `pool` is (n_completions, hidden_size).  With `n_pairs=None` the expectation
    is exact -- the mean over all distinct unordered pairs -- which is the limit
    of the random pairing in Eq. 8 and avoids any sampling noise.  Passing a
    number samples that many random pairs instead.
    """
    normed = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
    n = normed.shape[0]

    if n_pairs is None:
        gram = normed @ normed.T
        iu = np.triu_indices(n, k=1)
        values = gram[iu]
    else:
        i = rng.integers(0, n, size=n_pairs)
        j = rng.integers(0, n, size=n_pairs)
        distinct = i != j
        i, j = i[distinct], j[distinct]
        values = np.sum(normed[i] * normed[j], axis=1)

    return float(values.mean()), float(values.std()), int(values.size)


def run_encode(args: argparse.Namespace) -> None:
    """Measure per-question similarity and the Eq. 8 baseline from the models."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reuse the published correct/distractor pairing where it exists, so the new
    # measurement is comparable question by question.
    pairing = load_pairing(args) if args.reuse_pairing else {}

    store: Dict[str, Dict[str, Dict]] = {}
    for model_name in args.models:
        path = args.model_path.get(model_name, MODEL_REPOS[model_name])
        print(f"=== {model_name} ({path}) ===")
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype="auto", device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        store[model_name] = {}
        for benchmark in args.benchmarks:
            rec = encode_benchmark(
                model, tokenizer, model_name, benchmark, args, pairing, load_dataset
            )
            store[model_name][benchmark] = rec
            print(
                f"  {benchmark}: cosine {rec['cosine'].shape}, "
                f"baseline over {rec['n_pairs']} pairs"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_store(args.stem, store, build_manifest(args, store, source="encode"))


def encode_benchmark(
    model, tokenizer, model_name: str, benchmark: str, args, pairing, load_dataset
) -> Dict:
    """Per-question cosine similarity and random-pair baseline for one benchmark."""
    splits = load_dataset(f"{HF_NAMESPACE}/{benchmark}")
    split = "test" if "test" in splits else "validation"
    dataset = splits[split]
    n = min(args.num_examples, len(dataset)) if args.num_examples > 0 else len(dataset)
    dataset = dataset[:n]

    rng = np.random.default_rng(args.seed)
    questions, correct, distractor, ids = [], [], [], []
    for i in range(n):
        choices = dataset["choices"][i]
        gold = dataset["answer_index"][i]
        questions.append(dataset["question"][i])
        correct.append(choices[gold])
        cached = pairing.get((model_name, benchmark, str(i)))
        if cached is not None:
            distractor.append(cached)
        else:
            wrong = [j for j in range(len(choices)) if j != gold]
            distractor.append(choices[int(rng.choice(wrong))])
        ids.append(str(i))

    # x+ = [q; a+] and x- = [q; a-]  (Eq. 7)
    plus = [f"{q} {a}" for q, a in zip(questions, correct)]
    minus = [f"{q} {a}" for q, a in zip(questions, distractor)]

    h_plus = encode_texts(
        plus, tokenizer, model, args.batch_size, args.use_template, args.template_offset
    )
    h_minus = encode_texts(
        minus, tokenizer, model, args.batch_size, args.use_template, args.template_offset
    )

    n_layers = h_plus.shape[0]
    cosine = np.zeros((n, n_layers))
    baseline_mean = np.zeros(n_layers)
    baseline_std = np.zeros(n_layers)
    n_pairs = 0
    for layer in range(n_layers):
        cosine[:, layer] = cosine_rows(h_plus[layer], h_minus[layer])
        pool = pool_for_baseline(h_plus[layer], h_minus[layer], args.baseline_pool)
        baseline_mean[layer], baseline_std[layer], n_pairs = random_pair_baseline(
            pool, args.n_random_pairs, np.random.default_rng(args.seed + layer)
        )

    return {
        "cosine": cosine,
        # The values behind the published figures, when the cached results are
        # available, so both measurements sit side by side in the exports.
        "published_cosine": published_cosine(args, model_name, benchmark, cosine.shape),
        "question_ids": ids,
        "questions": questions,
        "correct_answers": correct,
        "distractor_answers": distractor,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "n_pairs": n_pairs,
    }


def published_cosine(
    args: argparse.Namespace, model: str, benchmark: str, shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Per-question cosine similarities from the cached results, if they match."""
    path = os.path.join(args.legacy_dir, model, HF_NAMESPACE, f"{benchmark}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    values = np.asarray([r["cosine_similarity"] for r in rows])
    if values.shape != shape:
        print(f"  cached {model}/{benchmark} has shape {values.shape}, expected {shape}")
        return None
    return values


def pool_for_baseline(h_plus: np.ndarray, h_minus: np.ndarray, pool: str) -> np.ndarray:
    """Completions the random-pair expectation is taken over."""
    if pool == "correct":
        return h_plus
    if pool == "distractor":
        return h_minus
    return np.concatenate([h_plus, h_minus], axis=0)


def load_pairing(args: argparse.Namespace) -> Dict[Tuple[str, str, str], str]:
    """Correct/distractor pairing recorded in the cached results, if present."""
    pairing: Dict[Tuple[str, str, str], str] = {}
    for model in args.models:
        for benchmark in args.benchmarks:
            path = os.path.join(
                args.legacy_dir, model, HF_NAMESPACE, f"{benchmark}.json"
            )
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for row in json.load(f):
                    pairing[(model, benchmark, str(row["id"]))] = row["incorrect_answer"]
    if pairing:
        print(f"reusing {len(pairing)} cached correct/distractor pairings")
    return pairing


# --------------------------------------------------------------------------- #
# Adjustments
# --------------------------------------------------------------------------- #

def values_of(rec: Dict, source: str) -> np.ndarray:
    """Per-question cosine matrix, from the chosen measurement.

    `published` is the cached measurement behind the paper's figures;
    `measured` is whatever the `encode` stage produced in this repository.  An
    ingest store holds the same numbers under both names.
    """
    if source == "published" and rec.get("published_cosine") is not None:
        return rec["published_cosine"]
    return rec["cosine"]


def cross_benchmark_offset(store_model: Dict[str, Dict], source: str = "published") -> np.ndarray:
    """Per-layer mean of the benchmarks' layer means, for one model.

    This is what the published figures subtract: a centring across the four
    benchmarks, so each line shows how far that benchmark sits from the
    model's average at that layer.
    """
    layer_means = [values_of(rec, source).mean(axis=0) for rec in store_model.values()]
    return np.mean(np.stack(layer_means), axis=0)


def adjusted_curves(
    store_model: Dict[str, Dict], adjustment: str, source: str = "published"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-benchmark (mean, std) curves under the chosen adjustment."""
    offset = cross_benchmark_offset(store_model, source)
    curves = {}
    for benchmark, rec in store_model.items():
        values = values_of(rec, source)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        if adjustment == "random-pair":
            if rec.get("baseline_mean") is None:
                raise ValueError(
                    "the random-pair baseline is absent from this store; it "
                    "requires the `encode` stage (the cached JSONs do not "
                    "record hidden states)"
                )
            mean = mean - rec["baseline_mean"]
        elif adjustment == "cross-benchmark":
            mean = mean - offset
        curves[benchmark] = (mean, std)
    return curves


# --------------------------------------------------------------------------- #
# Stage: plot
# --------------------------------------------------------------------------- #

def plot_figure4(store: Dict[str, Dict[str, Dict]], args) -> None:
    """Layer-wise correct-vs-distractor similarity, one panel per model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)

    for ax, model in zip(axes, args.models):
        curves = adjusted_curves(store[model], args.adjustment, args.curve_source)
        for (benchmark, (mean, std)), marker, ls in zip(
            curves.items(), MARKERS, LINESTYLES
        ):
            layers = np.arange(len(mean))
            ax.plot(
                layers,
                mean,
                label=benchmark,
                marker=marker,
                markersize=4,
                linewidth=2,
                linestyle=ls,
            )
            ax.fill_between(layers, mean - std, mean + std, alpha=0.15)

        ax.set_title(model, fontsize=22)
        ax.set_xlabel("Layer", fontsize=20)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False, fontsize=14, loc="lower right")

    axes[0].set_ylabel(ylabel_for(args.adjustment), fontsize=20)
    save_figure(fig, args.stem)


def plot_figure8(store: Dict[str, Dict[str, Dict]], args) -> None:
    """Across-benchmark standard deviation of the layer means, per model.

    Subtracting a per-layer constant cannot change a standard deviation taken
    across benchmarks at that layer, so this curve is identical under the
    cross-benchmark adjustment and under no adjustment at all.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)

    for ax, model in zip(axes, args.models):
        curves = adjusted_curves(store[model], args.adjustment, args.curve_source)
        means = np.stack([mean for mean, _ in curves.values()])
        separation = means.std(axis=0)

        ax.plot(np.arange(means.shape[1]), separation, marker="o", markersize=4, linewidth=2)
        ax.set_title(model, fontsize=22)
        ax.set_xlabel("Layer", fontsize=20)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Normalized Across-dataset std", fontsize=20)
    save_figure(fig, args.figure8_stem)


def ylabel_for(adjustment: str) -> str:
    return {
        "cross-benchmark": "Normalized Cosine Similarity",
        "random-pair": "Adjusted Cosine Similarity",
        "none": "Cosine Similarity",
    }[adjustment]


def save_figure(fig, stem: str) -> None:
    import matplotlib.pyplot as plt

    ensure_parent(stem)
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

def export_tables(store: Dict[str, Dict[str, Dict]], args) -> None:
    """Per-question values, the pairing behind them, and the baselines."""
    values_path = f"{args.stem}_similarity.csv"
    pairs_path = f"{args.stem}_pairs.csv"
    baselines_path = f"{args.stem}_baselines.csv"
    ensure_parent(values_path)

    n_rows = 0
    with open(values_path, "w", newline="") as vf, open(pairs_path, "w", newline="") as pf:
        values = csv.writer(vf)
        values.writerow(
            [
                "model",
                "benchmark",
                "question_id",
                "layer",
                "cosine_similarity",
                "cosine_similarity_published",
                "random_pair_baseline",
                "adjusted_cosine_random_pair",
                "cross_benchmark_offset",
                "adjusted_cosine_cross_benchmark",
            ]
        )
        pairs = csv.writer(pf)
        pairs.writerow(
            [
                "model",
                "benchmark",
                "question_id",
                "question",
                "correct_answer",
                "distractor_answer",
            ]
        )

        for model in args.models:
            offset = cross_benchmark_offset(store[model], args.curve_source)
            for benchmark in args.benchmarks:
                rec = store[model][benchmark]
                cosine = rec["cosine"]
                published = rec.get("published_cosine")
                baseline = rec.get("baseline_mean")
                n_questions, n_layers = cosine.shape

                for i in range(n_questions):
                    pairs.writerow(
                        [
                            model,
                            benchmark,
                            rec["question_ids"][i],
                            rec["questions"][i],
                            rec["correct_answers"][i],
                            rec["distractor_answers"][i],
                        ]
                    )
                    for layer in range(n_layers):
                        c = float(cosine[i, layer])
                        p = float(published[i, layer]) if published is not None else None
                        b = float(baseline[layer]) if baseline is not None else None
                        values.writerow(
                            [
                                model,
                                benchmark,
                                rec["question_ids"][i],
                                layer,
                                round(c, 6),
                                "" if p is None else round(p, 6),
                                "" if b is None else round(b, 6),
                                "" if b is None else round(c - b, 6),
                                round(float(offset[layer]), 6),
                                round(c - float(offset[layer]), 6),
                            ]
                        )
                        n_rows += 1
    print(f"wrote {values_path} ({n_rows} rows)")
    print(f"wrote {pairs_path}")

    with open(baselines_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "benchmark",
                "layer",
                "n_questions",
                "mean_cosine_similarity",
                "std_cosine_similarity",
                "random_pair_baseline_mean",
                "random_pair_baseline_std",
                "cross_benchmark_offset",
                "adjusted_mean_random_pair",
                "adjusted_mean_cross_benchmark",
            ]
        )
        for model in args.models:
            offset = cross_benchmark_offset(store[model], args.curve_source)
            for benchmark in args.benchmarks:
                rec = store[model][benchmark]
                cosine = values_of(rec, args.curve_source)
                baseline = rec.get("baseline_mean")
                baseline_std = rec.get("baseline_std")
                mean = cosine.mean(axis=0)
                std = cosine.std(axis=0)
                for layer in range(cosine.shape[1]):
                    b = float(baseline[layer]) if baseline is not None else None
                    bs = float(baseline_std[layer]) if baseline_std is not None else None
                    w.writerow(
                        [
                            model,
                            benchmark,
                            layer,
                            cosine.shape[0],
                            round(float(mean[layer]), 6),
                            round(float(std[layer]), 6),
                            "" if b is None else round(b, 6),
                            "" if bs is None else round(bs, 6),
                            round(float(offset[layer]), 6),
                            "" if b is None else round(float(mean[layer]) - b, 6),
                            round(float(mean[layer] - offset[layer]), 6),
                        ]
                    )
    print(f"wrote {baselines_path}")


# --------------------------------------------------------------------------- #

def build_manifest(args, store: Dict[str, Dict[str, Dict]], source: str) -> Dict:
    """Provenance for whatever produced the store."""
    has_baseline = any(
        rec.get("baseline_mean") is not None
        for per_benchmark in store.values()
        for rec in per_benchmark.values()
    )
    manifest = {
        "run_id": args.run_id,
        "source": source,
        "models": list(args.models),
        "benchmarks": list(args.benchmarks),
        "n_layers": {
            model: int(next(iter(per_benchmark.values()))["cosine"].shape[1])
            for model, per_benchmark in store.items()
        },
        "n_questions_per_benchmark": args.num_examples,
        "has_random_pair_baseline": has_baseline,
        "eq7_completions": "x+ = '<question> <correct answer>', x- = '<question> <distractor>'",
        "eq8_adjustment": (
            "sim_adj = cos(h+, h-) - E[cos(h_i, h_j)] over distinct pairs of "
            "completions from the same benchmark"
        ),
        "cross_benchmark_adjustment": (
            "cos(h+, h-) minus, per layer, the mean over the four benchmarks of "
            "their layer-mean cosine for that model; this is what the published "
            "figures plot"
        ),
        "shaded_region": (
            "mean +/- 1 population standard deviation (numpy std, ddof=0) of the "
            "per-question cosine similarity across questions, at each layer; not "
            "a standard error or confidence interval"
        ),
        "figure8": (
            "population standard deviation across the four benchmarks' layer "
            "means at each layer; invariant to the cross-benchmark adjustment"
        ),
    }
    if source == "encode":
        manifest["encoding"] = {
            "use_template": args.use_template,
            "system_prompt": SYSTEM_PROMPT if args.use_template else None,
            "template_offset": args.template_offset,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "baseline_pool": args.baseline_pool,
            "n_random_pairs": args.n_random_pairs or "exact (all distinct pairs)",
            "reuse_pairing": args.reuse_pairing,
            "model_paths": {m: args.model_path.get(m, MODEL_REPOS[m]) for m in args.models},
        }
    else:
        manifest["cached_results_dir"] = args.legacy_dir
        manifest["note"] = (
            "cached per-question cosine similarities behind the published "
            "figures; hidden states were not retained, so the Eq. 8 random-pair "
            "baseline is absent -- run the `encode` stage to measure it"
        )
    return manifest


def ensure_parent(path: str) -> None:
    """Create the directory a path will be written into, if any."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_model_path(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated --model-path NAME=PATH arguments."""
    mapping: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--model-path expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        mapping[name] = os.path.expanduser(path)
    return mapping


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage",
        nargs="?",
        default="plot",
        choices=["encode", "ingest", "plot"],
        help="measure from the models, build from cached results, or draw "
        "Figures 4 and 8 from the store (default)",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Figure 4 / data stem")
    parser.add_argument("--figure8-stem", default=FIGURE8_STEM)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--legacy-dir", default=LEGACY_RESULTS_DIR)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument(
        "--adjustment",
        choices=["cross-benchmark", "random-pair", "none"],
        default="cross-benchmark",
        help="'cross-benchmark' reproduces the published figures (default); "
        "'random-pair' applies Eq. 8, which needs the `encode` stage",
    )
    parser.add_argument(
        "--curve-source",
        choices=["published", "measured"],
        default="published",
        help="which per-question measurement the figures and the aggregate "
        "columns use: the cached values behind the paper (default) or this "
        "repository's `encode` run",
    )
    parser.add_argument(
        "--model-path",
        nargs="+",
        metavar="NAME=PATH",
        help="local path for a model, e.g. Qwen2.5-7B-Instruct=/scratch/qwen",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--no-template",
        dest="use_template",
        action="store_false",
        help="encode the raw text instead of applying the chat template",
    )
    parser.add_argument("--template-offset", type=int, default=TEMPLATE_OFFSET)
    parser.add_argument(
        "--baseline-pool",
        choices=["all", "correct", "distractor"],
        default="all",
        help="completions the Eq. 8 expectation is taken over (default: both)",
    )
    parser.add_argument(
        "--n-random-pairs",
        type=int,
        default=None,
        help="sample this many random pairs instead of the exact expectation "
        "over all distinct pairs",
    )
    parser.add_argument(
        "--no-reuse-pairing",
        dest="reuse_pairing",
        action="store_false",
        help="sample fresh distractors instead of reusing the cached pairing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    args.model_path = parse_model_path(args.model_path)
    if args.run_id is None:
        args.run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

    if args.stage == "ingest":
        run_ingest(args)
    elif args.stage == "encode":
        run_encode(args)
    else:
        store, manifest = load_store(args.stem)
        args.models = [m for m in args.models if m in store]
        plot_figure4(store, args)
        plot_figure8(store, args)
        export_tables(store, args)


if __name__ == "__main__":
    main()
