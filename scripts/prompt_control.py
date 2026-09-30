"""Prompt-structure control for the probing analysis of PRISM.

Addresses TACL mandatory revision 1: the published probing analysis (Figure 3)
renders knowledge-centric and reasoning-centric benchmarks with *different*
prompt templates (Figure 10a vs 10b).  The probe reads the hidden state at the
final user-content token, and that token is not the same thing in the two
templates -- `_` under 10a, the closing `>` of `<final_answer>` under 10b --
so the published knowledge/reasoning contrast is confounded with template.

This script crosses the factor out: every benchmark is run under every
template, so the task-type effect and the template effect can be separated.

  T_K   Figure 10a, the knowledge template
  T_R   Figure 10b, the reasoning template (a 32-token <think>/<final_answer>
        scaffold around the same question)
  T_N   a neutral control: question and options only, no answer cue at all,
        matching the plain prompt Figure 1 uses for all 24 benchmarks
  T_A   a minimal cue, `Answer:`, with no reasoning scaffold

T_N and T_A differ in what the probe ends up reading.  Under T_K, T_R and T_A
the final user-content token is the same string for every item (`_`, the `>`
closing `<final_answer>`, and `:`), so a layer-0 probe can do no better than
the majority class.  Under T_N it is the last token of the last answer option,
which varies per item and carries lexical information from layer 0 onward.
T_A is therefore the control that isolates the reasoning scaffold while
holding the probe position constant; T_N is the one comparable to Figure 1.

Three measurements per (model, dataset, template):

  probe accuracy   logistic regression over the correct option index, per
                   layer, 5-fold stratified CV -- the Figure 3 metric
  attention        of the final user-content token: Shannon entropy per layer,
                   and how its mass splits over the prompt's segments
                   (sink / system / instruction / question / choices / cue)
  representation   the raw hidden states, kept so the same items can be
                   compared across templates with linear CKA

Stages
------
    extract    one forward pass per item, no generation.  GPU.
    probe      layer-wise probes over the extracted states.  CPU.
    attention  aggregate the per-item attention readings.  CPU.
    cka        template-vs-template representation similarity.  CPU.

Usage
-----
    python scripts/prompt_control.py extract --model qwen --dataset sciq \
        --template T_K --out-dir /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/prism/prompt_control
    python scripts/prompt_control.py probe --out-dir ...

Subset rule.  Full test split, capped at 1000 items by `shuffle(seed=42)`.
Under this rule BioMixQA (306) and SciQ (1000) reproduce the published
layer-0 majority baselines exactly (0.232027, 0.253002); LogiQA (651) and
MathQA (1000) do not, and the subset behind the published reasoning curves
could not be recovered -- no probing code or cached probe table survives in
either repository.  See `docs` in the accompanying notes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant."

# Tokens of chat scaffolding that follow the user content once
# `add_generation_prompt=True` has been applied.  Verified to be 5 for all
# three models (Qwen/OLMo `<|im_end|>\n<|im_start|>assistant\n`, Llama
# `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`), so the final
# user-content token sits at offset -6 throughout.
TEMPLATE_OFFSET = 6

MODELS: Dict[str, str] = {
    "qwen": "models--Qwen--Qwen2.5-7B-Instruct",
    "llama": "models--meta-llama--Llama-3.1-8B-Instruct",
    "olmo": "models--allenai--Olmo-3-7B-Instruct",
}

MODEL_LABELS: Dict[str, str] = {
    "qwen": "Qwen2.5-7B-Instruct",
    "llama": "Llama-3.1-8B-Instruct",
    "olmo": "Olmo-3-7B-Instruct",
}

# The four diagnostic benchmarks of Section 3.3, with the task type the paper
# assigns them.  `context` prepends a passage to the question.
DATASETS: Dict[str, Dict[str, object]] = {
    "biomix-qa": {"hf": "extraordinarylab/biomix-qa", "split": "test",
                  "context": False, "type": "knowledge", "label": "BioMixQA"},
    "sciq": {"hf": "extraordinarylab/sciq", "split": "test",
             "context": False, "type": "knowledge", "label": "SciQ"},
    "logiqa": {"hf": "extraordinarylab/logiqa", "split": "test",
               "context": True, "type": "reasoning", "label": "LogiQA"},
    "math-qa": {"hf": "extraordinarylab/math-qa", "split": "test",
                "context": False, "type": "reasoning", "label": "MathQA"},
}

TEMPLATES = ("T_K", "T_R", "T_N", "T_A")

TEMPLATE_LABELS: Dict[str, str] = {
    "T_K": "Knowledge template (Fig. 10a)",
    "T_R": "Reasoning template (Fig. 10b)",
    "T_N": "Neutral control (no cue)",
    "T_A": "Minimal cue control",
}

# The instruction prefix and answer cue that distinguish the templates.
REASONING_PREFIX = (
    "Please enclose your thinking process in <think></think> and the final "
    "answer in <final_answer></final_answer>."
)
CUE_K = "Respond with one letter. The answer is _"
CUE_R = "Respond with one letter. The answer is _ <think></think><final_answer>"
CUE_A = "Answer:"

# Attention-mass segments, in the order they are reported.
SEGMENTS = ("sink", "system", "instruction", "question", "choices", "cue", "other")

MAX_ITEMS = 1000
SUBSET_SEED = 42
N_FOLDS = 5
DELTA = 1e-10


def hub_snapshot(repo_dir: str) -> str:
    """Resolve a local Hugging Face hub snapshot directory.

    Loading by repo id under `HF_HUB_OFFLINE=1` trips a network call in
    transformers 4.57 (`_patch_mistral_regex`), so we hand it a path instead.
    """
    from huggingface_hub import constants
    base = constants.HF_HUB_CACHE
    hits = sorted(glob(os.path.join(base, repo_dir, "snapshots", "*")))
    if not hits:
        raise FileNotFoundError(f"no local snapshot for {repo_dir} under {base}")
    return hits[-1]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A rendered prompt plus the character spans of its segments."""

    text: str                      # the user-message content
    spans: Dict[str, Tuple[int, int]]   # segment -> (char start, char end)


def option_letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index - 25)


def render_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{option_letter(i)}) {c}" for i, c in enumerate(choices))


def build_prompt(question: str, choices: Sequence[str], template: str) -> Prompt:
    """Render one item under one template, recording segment spans.

    The spans are what the attention decomposition is computed over, so they
    are tracked while the string is assembled rather than searched for
    afterwards -- question text can contain anything, including the literal
    option markers.
    """
    parts: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    def push(chunk: str, name: Optional[str] = None) -> None:
        nonlocal cursor
        start = cursor
        parts.append(chunk)
        cursor += len(chunk)
        if name is not None:
            spans[name] = (start, cursor)

    if template == "T_R":
        push(REASONING_PREFIX, "instruction")
        push("\n\n")

    push("Question: ")
    push(question, "question")
    push("\nChoices:\n")
    push(render_choices(choices), "choices")

    if template == "T_K":
        push("\n\n")
        push(CUE_K, "cue")
    elif template == "T_R":
        push("\n\n")
        push(CUE_R, "cue")
    elif template == "T_A":
        push("\n\n")
        push(CUE_A, "cue")
    elif template != "T_N":
        raise ValueError(f"unknown template {template!r}")

    return Prompt("".join(parts), spans)


def chat_wrap(tokenizer, prompt: Prompt) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Apply the chat template, shifting the segment spans to match."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt.text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    offset = rendered.rfind(prompt.text)
    if offset < 0:
        raise RuntimeError("user content not found verbatim in the chat template")

    spans = {k: (s + offset, e + offset) for k, (s, e) in prompt.spans.items()}

    sys_at = rendered.find(SYSTEM_PROMPT)
    if sys_at >= 0:
        spans["system"] = (sys_at, sys_at + len(SYSTEM_PROMPT))

    return rendered, spans


def segment_ids(offsets: np.ndarray, spans: Dict[str, Tuple[int, int]],
                n_tokens: int) -> np.ndarray:
    """Label every token with the segment it falls in.

    Returns an int array of indices into SEGMENTS.  Token 0 is always `sink`:
    the first position absorbs a large, content-independent share of attention
    in every model here, and folding it into `system` would swamp that
    segment.
    """
    order = {name: i for i, name in enumerate(SEGMENTS)}
    labels = np.full(n_tokens, order["other"], dtype=np.int8)

    for name in ("system", "instruction", "question", "choices", "cue"):
        span = spans.get(name)
        if span is None:
            continue
        start, end = span
        # A token belongs to a segment when its character span overlaps it.
        hit = (offsets[:, 0] < end) & (offsets[:, 1] > start)
        labels[hit] = order[name]

    labels[0] = order["sink"]
    return labels


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_items(dataset: str) -> Tuple[List[str], List[List[str]], np.ndarray]:
    """Load one benchmark under the documented subset rule.

    Cached: each dataset is reused across the four templates of a run, and
    re-reading and re-shuffling MathQA for every cell costs more than the
    forward passes do.  Callers slice rather than mutate the result.
    """
    from datasets import load_dataset

    cfg = DATASETS[dataset]
    ds = load_dataset(cfg["hf"], split=cfg["split"])
    if len(ds) > MAX_ITEMS:
        ds = ds.shuffle(seed=SUBSET_SEED).select(range(MAX_ITEMS)).flatten_indices()

    questions: List[str] = []
    for row in ds:
        q = row["question"]
        if cfg["context"] and row.get("context"):
            q = f"{row['context']}\n{q}"
        questions.append(q)

    choices = [list(c) for c in ds["choices"]]
    labels = np.asarray(ds["answer_index"], dtype=np.int64)
    return questions, choices, labels


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------

def run_extract(args: argparse.Namespace) -> None:
    """Extract every requested (dataset, template) cell on one model load.

    `--dataset all` / `--template all` expand to the full set.  Loading a 7-8B
    checkpoint costs more than a cell's worth of forward passes, and on a
    contended cluster one long allocation schedules far more easily than a
    fan-out of short ones, so the loop lives inside the job rather than in the
    array index.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    datasets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    templates = list(TEMPLATES) if args.template == "all" else [args.template]

    snapshot = hub_snapshot(MODELS[args.model])
    print(f"[extract] model     {MODEL_LABELS[args.model]}", flush=True)
    print(f"[extract] snapshot  {snapshot}", flush=True)
    print(f"[extract] cells     {len(datasets)} datasets x {len(templates)} templates",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",   # output_attentions is a no-op under SDPA
    )
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for dataset in datasets:
        for template in templates:
            stem = f"{args.model}__{dataset}__{template}"
            if args.skip_existing and os.path.exists(os.path.join(args.out_dir, f"{stem}.npz")):
                print(f"[extract] skip {stem} (already present)", flush=True)
                continue
            _extract_cell(args, model, tokenizer, snapshot, dataset, template)


def _extract_cell(args, model, tokenizer, snapshot: str,
                  dataset: str, template: str) -> None:
    import torch

    print(f"[extract] --- {args.model} / {dataset} / {template}", flush=True)

    questions, choices, labels = load_items(dataset)
    if args.limit:
        questions, choices, labels = questions[:args.limit], choices[:args.limit], labels[:args.limit]
    n_items = len(questions)

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"[extract] {n_items} items, {n_layers} layers, hidden {hidden_size}", flush=True)

    # hidden_states has n_layers + 1 entries (embeddings first).
    states = np.zeros((n_items, n_layers + 1, hidden_size), dtype=np.float16)
    entropy = np.zeros((n_items, n_layers), dtype=np.float32)
    mass = np.zeros((n_items, n_layers, len(SEGMENTS)), dtype=np.float32)
    lengths = np.zeros(n_items, dtype=np.int32)
    seg_counts = np.zeros((n_items, len(SEGMENTS)), dtype=np.int32)

    for i in range(n_items):
        prompt = build_prompt(questions[i], choices[i], template)
        rendered, spans = chat_wrap(tokenizer, prompt)

        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].numpy()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_tokens = offsets.shape[0]
        lengths[i] = n_tokens

        # The probe position: the last token of the user content, i.e. the
        # position the published Figure 3 reads.
        pos = n_tokens - TEMPLATE_OFFSET
        if pos <= 0:
            raise RuntimeError(f"prompt too short at item {i}: {n_tokens} tokens")

        labels_seg = segment_ids(offsets, spans, n_tokens)
        for s in range(len(SEGMENTS)):
            seg_counts[i, s] = int((labels_seg[:pos + 1] == s).sum())

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, output_attentions=True)

        for l, hs in enumerate(out.hidden_states):
            states[i, l] = hs[0, pos].float().cpu().numpy().astype(np.float16)

        for l, attn in enumerate(out.attentions):
            # attn: (1, heads, seq, seq).  Row `pos` is causally masked beyond
            # `pos`, so the distribution already lives on 0..pos.
            row = attn[0, :, pos, :pos + 1].float().mean(dim=0)
            row = row / (row.sum() + DELTA)
            row_np = row.cpu().numpy()
            entropy[i, l] = float(-(row_np * np.log(row_np + DELTA)).sum())
            for s in range(len(SEGMENTS)):
                mass[i, l, s] = float(row_np[labels_seg[:pos + 1] == s].sum())

        del out
        if (i + 1) % 100 == 0:
            print(f"[extract]   {i + 1}/{n_items}", flush=True)
            torch.cuda.empty_cache()

    stem = f"{args.model}__{dataset}__{template}"
    path = os.path.join(args.out_dir, f"{stem}.npz")
    # Uncompressed: the payload is float16 hidden states, which deflate barely
    # at all, and the CPU time costs more than the disk does.
    np.savez(
        path,
        states=states, entropy=entropy, mass=mass, labels=labels,
        lengths=lengths, seg_counts=seg_counts,
        segments=np.array(SEGMENTS),
    )

    meta = {
        "model": args.model, "model_label": MODEL_LABELS[args.model],
        "snapshot": snapshot, "dataset": dataset,
        "dataset_label": DATASETS[dataset]["label"],
        "task_type": DATASETS[dataset]["type"],
        "template": template, "n_items": int(n_items),
        "n_layers": int(n_layers), "hidden_size": int(hidden_size),
        "template_offset": TEMPLATE_OFFSET,
        "subset": {"split": DATASETS[dataset]["split"],
                   "cap": MAX_ITEMS, "seed": SUBSET_SEED},
        "median_prompt_tokens": int(np.median(lengths)),
        "majority_baseline": float(np.bincount(labels).max() / len(labels)),
    }
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[extract] wrote {path}", flush=True)
    print(f"[extract] median prompt length {meta['median_prompt_tokens']} tokens", flush=True)
    print(f"[extract] majority baseline    {meta['majority_baseline']:.6f}", flush=True)


# ---------------------------------------------------------------------------
# Stage: probe
# ---------------------------------------------------------------------------

def probe_one(states: np.ndarray, labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Layer-wise probe accuracy: multi-class logistic regression, 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    n_layers = states.shape[1]
    out = np.zeros(n_layers, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    for layer in range(n_layers):
        X = states[:, layer, :].astype(np.float32)
        scores = []
        for train, val in folds:
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X[train], labels[train])
            scores.append(accuracy_score(labels[val], clf.predict(X[val])))
        out[layer] = float(np.mean(scores))
        print(f"    layer {layer:3d}  {out[layer]:.4f}", flush=True)
    return out


def _probe_cell(path: str, parts_dir: str, seed: int) -> str:
    """Probe one extracted cell and write its rows to `parts_dir/<stem>.csv`."""
    import csv

    stem = os.path.basename(path)[:-len(".npz")]
    model, dataset, template = stem.split("__")
    print(f"[probe] {stem}", flush=True)
    with np.load(path, allow_pickle=True) as z:
        states, labels = z["states"], z["labels"]
    acc = probe_one(states, labels, seed=seed)
    rows = [{
        "model": MODEL_LABELS[model], "model_key": model,
        "dataset": DATASETS[dataset]["label"], "dataset_key": dataset,
        "task_type": DATASETS[dataset]["type"],
        "template": template, "layer": layer, "accuracy": f"{value:.6f}",
    } for layer, value in enumerate(acc)]
    dest = os.path.join(parts_dir, f"{stem}.csv")
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return dest


def run_probe(args: argparse.Namespace) -> None:
    """Probe the extracted cells in parallel, then merge the per-cell tables.

    `--only` restricts the run to stems with the given prefixes so the cells
    can be spread over several jobs; `--merge-only` just concatenates the
    per-cell tables already written into `probe_accuracy.csv`.
    """
    import csv
    from joblib import Parallel, delayed

    parts_dir = os.path.join(args.out_dir, "probe_parts")
    os.makedirs(parts_dir, exist_ok=True)

    if not args.merge_only:
        runs = sorted(glob(os.path.join(args.out_dir, "*__*__*.npz")))
        if args.only:
            runs = [p for p in runs if os.path.basename(p).startswith(tuple(args.only))]
        if not runs:
            raise SystemExit(f"no extracted runs under {args.out_dir}")
        Parallel(n_jobs=args.jobs)(
            delayed(_probe_cell)(p, parts_dir, args.seed) for p in runs)
        if args.only:
            return

    rows = []
    for part in sorted(glob(os.path.join(parts_dir, "*.csv"))):
        rows.extend(_read_csv(part))
    if not rows:
        raise SystemExit(f"no per-cell probe tables under {parts_dir}")

    dest = os.path.join(args.out_dir, "probe_accuracy.csv")
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[probe] wrote {dest}  ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Stage: attention
# ---------------------------------------------------------------------------

def run_attention(args: argparse.Namespace) -> None:
    import csv

    runs = sorted(glob(os.path.join(args.out_dir, "*__*__*.npz")))
    if not runs:
        raise SystemExit(f"no extracted runs under {args.out_dir}")

    rows = []
    for path in runs:
        stem = os.path.basename(path)[:-len(".npz")]
        model, dataset, template = stem.split("__")
        with np.load(path, allow_pickle=True) as z:
            entropy, mass, lengths = z["entropy"], z["mass"], z["lengths"]
        n_layers = entropy.shape[1]
        for layer in range(n_layers):
            row = {
                "model": MODEL_LABELS[model], "model_key": model,
                "dataset": DATASETS[dataset]["label"], "dataset_key": dataset,
                "task_type": DATASETS[dataset]["type"],
                "template": template, "layer": layer,
                "entropy_mean": f"{entropy[:, layer].mean():.6f}",
                "entropy_std": f"{entropy[:, layer].std():.6f}",
                "prompt_tokens_mean": f"{lengths.mean():.2f}",
            }
            for s, name in enumerate(SEGMENTS):
                row[f"mass_{name}"] = f"{mass[:, layer, s].mean():.6f}"
            rows.append(row)

    dest = os.path.join(args.out_dir, "attention_profile.csv")
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[attention] wrote {dest}  ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Stage: cka
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two item x feature matrices over the same items."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    nx = np.linalg.norm(X.T @ X, ord="fro")
    ny = np.linalg.norm(Y.T @ Y, ord="fro")
    denom = nx * ny
    return float(hsic / denom) if denom > 0 else float("nan")


def run_cka(args: argparse.Namespace) -> None:
    import csv
    from itertools import combinations

    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            loaded = {}
            for template in TEMPLATES:
                path = os.path.join(args.out_dir, f"{model}__{dataset}__{template}.npz")
                if os.path.exists(path):
                    with np.load(path, allow_pickle=True) as z:
                        loaded[template] = z["states"]
            for a, b in combinations(sorted(loaded), 2):
                Xa, Xb = loaded[a], loaded[b]
                if Xa.shape != Xb.shape:
                    print(f"[cka] shape mismatch {model}/{dataset} {a} vs {b}", flush=True)
                    continue
                # Layer 0 is skipped: under the cued templates the probe token is
                # the same string for every item, so its embedding has no
                # variance and CKA is undefined there.
                for layer in range(1, Xa.shape[1]):
                    rows.append({
                        "model": MODEL_LABELS[model], "model_key": model,
                        "dataset": DATASETS[dataset]["label"], "dataset_key": dataset,
                        "task_type": DATASETS[dataset]["type"],
                        "pair": f"{a}_vs_{b}", "layer": layer,
                        "cka": f"{linear_cka(Xa[:, layer, :], Xb[:, layer, :]):.6f}",
                    })
            if loaded:
                print(f"[cka] {model}/{dataset}: {sorted(loaded)}", flush=True)

    if not rows:
        raise SystemExit(f"no extracted runs under {args.out_dir}")
    dest = os.path.join(args.out_dir, "template_cka.csv")
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cka] wrote {dest}  ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, str]]:
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_report(args: argparse.Namespace) -> None:
    """Separate the task-type effect from the template effect.

    The design has one observation per (model, dataset, template) cell, so the
    summary stays at the level the design supports: a per-cell probe gain, and
    the knowledge-minus-reasoning gap it implies under each template.  With
    three models there is no room for an omnibus test to say anything a
    per-model table does not.
    """
    probe_path = os.path.join(args.out_dir, "probe_accuracy.csv")
    if not os.path.exists(probe_path):
        raise SystemExit(f"run the probe stage first ({probe_path} missing)")
    rows = _read_csv(probe_path)

    baseline: Dict[str, float] = {}
    for dataset in DATASETS:
        hits = glob(os.path.join(args.out_dir, f"*__{dataset}__*.json"))
        if hits:
            with open(hits[0]) as fh:
                baseline[DATASETS[dataset]["label"]] = json.load(fh)["majority_baseline"]

    # (model, dataset, template) -> accuracy by layer
    curves: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["template"])
        curves.setdefault(key, {})[int(r["layer"])] = float(r["accuracy"])

    print("\n=== probe gain over the majority baseline "
          "(peak across layers; late = mean of the final quarter)\n")
    header = f"{'model':24s} {'dataset':10s} {'type':9s} {'tmpl':5s} " \
             f"{'base':>7s} {'peak':>7s} {'gain':>7s} {'late':>7s} {'lategain':>8s}"
    print(header)
    print("-" * len(header))

    gains: Dict[Tuple[str, str, str], float] = {}
    for key in sorted(curves):
        model, dataset, template = key
        acc = curves[key]
        layers = sorted(acc)
        values = np.array([acc[l] for l in layers])
        tail = values[int(0.75 * len(values)):]
        base = baseline.get(dataset, float("nan"))
        peak, late = float(values.max()), float(tail.mean())
        gains[key] = peak - base
        task = "knowledge" if dataset in ("BioMixQA", "SciQ") else "reasoning"
        print(f"{model:24s} {dataset:10s} {task:9s} {template:5s} "
              f"{base:7.4f} {peak:7.4f} {peak - base:7.4f} {late:7.4f} {late - base:8.4f}")

    print("\n=== knowledge-minus-reasoning gap in peak probe gain, by template\n")
    print(f"{'model':24s} {'template':6s} {'knowledge':>10s} {'reasoning':>10s} {'gap':>8s}")
    print("-" * 62)
    for model in sorted({k[0] for k in gains}):
        for template in TEMPLATES:
            k = [gains[key] for key in gains
                 if key[0] == model and key[2] == template and key[1] in ("BioMixQA", "SciQ")]
            r = [gains[key] for key in gains
                 if key[0] == model and key[2] == template and key[1] in ("LogiQA", "MathQA")]
            if not k or not r:
                continue
            print(f"{model:24s} {template:6s} {np.mean(k):10.4f} "
                  f"{np.mean(r):10.4f} {np.mean(k) - np.mean(r):8.4f}")

    attn_path = os.path.join(args.out_dir, "attention_profile.csv")
    if os.path.exists(attn_path):
        print("\n=== attention mass at the probe position, averaged over layers\n")
        arows = _read_csv(attn_path)
        acc: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
        for r in arows:
            acc.setdefault((r["model"], r["dataset"], r["template"]), []).append(r)
        head = f"{'model':24s} {'dataset':10s} {'tmpl':5s} {'tokens':>7s} {'entropy':>8s} " + \
               " ".join(f"{s:>9s}" for s in SEGMENTS)
        print(head)
        print("-" * len(head))
        for key in sorted(acc):
            group = acc[key]
            ent = np.mean([float(r["entropy_mean"]) for r in group])
            ntok = float(group[0]["prompt_tokens_mean"])
            masses = [np.mean([float(r[f"mass_{s}"]) for r in group]) for s in SEGMENTS]
            print(f"{key[0]:24s} {key[1]:10s} {key[2]:5s} {ntok:7.1f} {ent:8.4f} " +
                  " ".join(f"{m:9.4f}" for m in masses))

    cka_path = os.path.join(args.out_dir, "template_cka.csv")
    if os.path.exists(cka_path):
        print("\n=== linear CKA between templates over the same items "
              "(mean across layers)\n")
        crows = _read_csv(cka_path)
        agg: Dict[Tuple[str, str, str], List[float]] = {}
        for r in crows:
            agg.setdefault((r["model"], r["dataset"], r["pair"]), []).append(float(r["cka"]))
        print(f"{'model':24s} {'dataset':10s} {'pair':16s} {'mean CKA':>9s} {'min':>7s}")
        print("-" * 70)
        for key in sorted(agg):
            v = np.array(agg[key])
            v = v[np.isfinite(v)]
            print(f"{key[0]:24s} {key[1]:10s} {key[2]:16s} {v.mean():9.4f} {v.min():7.4f}")


# ---------------------------------------------------------------------------
# Stage: plot
# ---------------------------------------------------------------------------

# Okabe-Ito, yellow-free; line style repeats the template for greyscale.
TEMPLATE_STYLES: Dict[str, Dict[str, str]] = {
    "T_K": {"color": "#0072B2", "linestyle": "-"},
    "T_R": {"color": "#D55E00", "linestyle": "--"},
    "T_N": {"color": "#009E73", "linestyle": ":"},
    "T_A": {"color": "#CC79A7", "linestyle": "-."},
}
PLOT_Y_LIMITS = (0.2, 1.0)


def run_plot(args: argparse.Namespace) -> None:
    """Probe accuracy per template: models as rows, benchmarks as columns.

    With `--published`, the submitted Figure 3 curve of each panel (drawn
    under Figure 10a for knowledge-centric, 10b for reasoning-centric
    benchmarks) is overlaid in grey.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.read_csv(os.path.join(args.out_dir, "probe_accuracy.csv"))
    published = pd.read_csv(args.published) if args.published else None

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "lines.linewidth": 1.1,
    })
    models = list(MODELS)
    datasets = list(DATASETS)
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(7.16, 5.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.12})

    for r, model in enumerate(models):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            cell = table[(table.model_key == model) & (table.dataset_key == dataset)]
            if published is not None:
                ref = published[(published.model == MODEL_LABELS[model]) &
                                (published.dataset == DATASETS[dataset]["label"])]
                ax.plot(ref.layer, ref.accuracy, color="0.6", linewidth=2.2, alpha=0.5,
                        label="Submitted Fig. 3")
            for template, style in TEMPLATE_STYLES.items():
                series = cell[cell.template == template].sort_values("layer")
                ax.plot(series.layer, series.accuracy, label=TEMPLATE_LABELS[template], **style)
            if r == 0:
                ax.set_title(DATASETS[dataset]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nProbe accuracy")
            if r == len(models) - 1:
                ax.set_xlabel("Layer")
            ax.set_xlim(0, 32)
            ax.set_ylim(*PLOT_Y_LIMITS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, handlelength=2.4)
    for ext in ("pdf", "png"):
        path = f"{args.stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {path}")
    plt.close(fig)


def run_plot_fig3(args: argparse.Namespace) -> None:
    """One Figure-3-style figure per template, drawn by `figure3.plot_figure3`.

    Same layout, colours, y-range and submitted-curve overlay as
    `assets/figure3.png`, so each template can be set beside it directly.
    """
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from figure3 import plot_figure3

    table = pd.read_csv(os.path.join(args.out_dir, "probe_accuracy.csv"))
    published = pd.read_csv(args.published) if args.published else None
    for template in TEMPLATES:
        cell = table[table.template == template][["dataset", "model", "layer", "accuracy"]]
        plot_figure3(cell, f"{args.stem}_{template}", published)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def _save(fig, stem: str) -> None:
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[plot] wrote {stem}.{ext}")


# Okabe-Ito colours for the attention segments; `other` (chat scaffolding
# outside the named spans) is a light neutral so the content segments stand out.
SEGMENT_COLOURS: Dict[str, str] = {
    "sink": "#000000", "system": "#56B4E9", "instruction": "#E69F00",
    "question": "#009E73", "choices": "#0072B2", "cue": "#D55E00", "other": "#DDDDDD",
}


def run_plot_attention(args: argparse.Namespace) -> None:
    """Where the probe token attends, and how diffusely.

    `<stem>_mass`: stacked attention mass per segment, averaged over layers,
    for every (dataset, template) cell, one panel per model.
    `<stem>_entropy`: layer-wise attention entropy, models as rows and
    benchmarks as columns, one line per template.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.read_csv(os.path.join(args.out_dir, "attention_profile.csv"))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 7.5})

    # --- stacked segment mass ------------------------------------------------
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.16, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, model in zip(axes, MODELS):
        cell = table[table.model_key == model]
        x, ticks, tick_labels = 0.0, [], []
        for dataset in DATASETS:
            for template in TEMPLATES:
                rows = cell[(cell.dataset_key == dataset) & (cell.template == template)]
                bottom = 0.0
                for seg in SEGMENTS:
                    value = rows[f"mass_{seg}"].mean()
                    ax.bar(x, value, bottom=bottom, width=0.8, color=SEGMENT_COLOURS[seg],
                           edgecolor="white", linewidth=0.3,
                           label=seg if (dataset, template) == (list(DATASETS)[0], TEMPLATES[0]) else None)
                    bottom += value
                ticks.append(x)
                tick_labels.append(template.replace("T_", ""))
                x += 1
            ax.text(x - 2.5, 1.03, DATASETS[dataset]["label"], ha="center", fontsize=6.5,
                    transform=ax.get_xaxis_transform())
            x += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 1)
        ax.set_title(MODEL_LABELS[model], pad=14)
        _style_axes(ax)
    axes[0].set_ylabel("Attention mass (mean over layers)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(SEGMENTS), frameon=False)
    _save(fig, f"{args.stem}_mass")
    plt.close(fig)

    # --- layer-wise entropy --------------------------------------------------
    fig, axes = plt.subplots(len(MODELS), len(DATASETS), figsize=(7.16, 5.2),
                             sharex=True, sharey="row", gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            cell = table[(table.model_key == model) & (table.dataset_key == dataset)]
            for template, style in TEMPLATE_STYLES.items():
                series = cell[cell.template == template].sort_values("layer")
                ax.plot(series.layer, series.entropy_mean, label=TEMPLATE_LABELS[template], **style)
            if r == 0:
                ax.set_title(DATASETS[dataset]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nAttention entropy")
            if r == len(MODELS) - 1:
                ax.set_xlabel("Layer")
            ax.set_xlim(0, 32)
            _style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, handlelength=2.4)
    _save(fig, f"{args.stem}_entropy")
    plt.close(fig)


CKA_PAIR_STYLES: Dict[str, Dict[str, str]] = {
    "T_K_vs_T_R": {"color": "#0072B2", "linestyle": "-"},
    "T_A_vs_T_K": {"color": "#56B4E9", "linestyle": "--"},
    "T_A_vs_T_R": {"color": "#D55E00", "linestyle": "-"},
    "T_K_vs_T_N": {"color": "#009E73", "linestyle": ":"},
    "T_N_vs_T_R": {"color": "#E69F00", "linestyle": ":"},
    "T_A_vs_T_N": {"color": "#CC79A7", "linestyle": "-."},
}


def run_plot_cka(args: argparse.Namespace) -> None:
    """Layer-wise linear CKA between templates over the same items."""
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.read_csv(os.path.join(args.out_dir, "template_cka.csv"))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(len(MODELS), len(DATASETS), figsize=(7.16, 5.2),
                             sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.12})
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            cell = table[(table.model_key == model) & (table.dataset_key == dataset)]
            for pair, style in CKA_PAIR_STYLES.items():
                series = cell[cell.pair == pair].sort_values("layer")
                a, b = pair.split("_vs_")
                ax.plot(series.layer, series.cka, label=f"{a} vs {b}", linewidth=1.1, **style)
            if r == 0:
                ax.set_title(DATASETS[dataset]["label"], pad=4)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nLinear CKA")
            if r == len(MODELS) - 1:
                ax.set_xlabel("Layer")
            ax.set_xlim(0, 32)
            ax.set_ylim(0, 1)
            _style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, handlelength=2.4)
    _save(fig, args.stem)
    plt.close(fig)


TASK_COLOURS = {"knowledge": "#0072B2", "reasoning": "#D55E00"}
SUMMARY_TEMPLATES = ("T_K", "T_R", "T_A", "T_N")   # paper templates first
SUMMARY_LABELS = {"T_K": "Fig. 10a\n(knowl.)", "T_R": "Fig. 10b\n(reason.)",
                  "T_A": "Minimal\ncue", "T_N": "No\ncue"}
MODEL_MARKERS = {"llama": "o", "qwen": "s", "olmo": "^"}


def run_plot_summary(args: argparse.Namespace) -> None:
    """One figure for revision 1: does prompt structure explain the contrast?

    Every benchmark is run under every template, and each panel compares
    knowledge-centric with reasoning-centric benchmarks under the same
    template.  (a) late-layer probe gain over the majority baseline -- what
    the final-token representation encodes; (b) share of the final token's
    attention on the template itself (instruction + answer cue); (c) its
    attention entropy.  Small points are (model, benchmark) cells, the large
    marker their mean.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    probe = pd.read_csv(os.path.join(args.out_dir, "probe_accuracy.csv"))
    attn = pd.read_csv(os.path.join(args.out_dir, "attention_profile.csv"))

    baseline = {}
    for dataset in DATASETS:
        with open(glob(os.path.join(args.out_dir, f"*__{dataset}__*.json"))[0]) as fh:
            baseline[dataset] = json.load(fh)["majority_baseline"]

    # Late-layer gain: mean accuracy over the final quarter of layers.
    cells = []
    for (model, dataset, template), g in probe.groupby(["model_key", "dataset_key", "template"]):
        g = g.sort_values("layer")
        tail = g.accuracy.to_numpy()[int(0.75 * len(g)):]
        a = attn[(attn.model_key == model) & (attn.dataset_key == dataset) & (attn.template == template)]
        cells.append({
            "model": model, "dataset": dataset, "template": template,
            "task": DATASETS[dataset]["type"],
            "gain": tail.mean() - baseline[dataset],
            "scaffold": (a.mass_instruction + a.mass_cue).mean(),
            "entropy": a.entropy_mean.mean(),
        })
    cells = pd.DataFrame(cells)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35),
                             gridspec_kw={"left": 0.07, "right": 0.995, "bottom": 0.25,
                                          "top": 0.8, "wspace": 0.3})
    panels = [
        ("gain", "(a) Answer recoverability", "Late-layer probe gain\nover majority"),
        ("scaffold", "(b) Attention on template", "Attention mass on\ninstruction + cue"),
        ("entropy", "(c) Attention dispersion", "Attention entropy"),
    ]
    offsets = {"knowledge": -0.17, "reasoning": 0.17}
    rng = np.random.default_rng(0)
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for i, template in enumerate(SUMMARY_TEMPLATES):
            for task, dx in offsets.items():
                group = cells[(cells.template == template) & (cells.task == task)]
                for _, row in group.iterrows():
                    ax.scatter(i + dx + rng.uniform(-0.05, 0.05), row[metric], s=9,
                               marker=MODEL_MARKERS[row["model"]], facecolor="white",
                               edgecolor=TASK_COLOURS[task], linewidth=0.6, zorder=2)
                ax.scatter(i + dx, group[metric].mean(), s=46, marker="D",
                           color=TASK_COLOURS[task], zorder=3,
                           label=f"{task.capitalize()}-centric" if i == 0 else None)
        ax.set_xticks(range(len(SUMMARY_TEMPLATES)))
        ax.set_xticklabels([SUMMARY_LABELS[t] for t in SUMMARY_TEMPLATES])
        ax.set_xlim(-0.5, len(SUMMARY_TEMPLATES) - 0.5)
        ax.set_title(title, pad=4)
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[0].axhline(0, color="0.5", linewidth=0.6, linestyle="--", zorder=1)

    task_handles, task_labels = axes[0].get_legend_handles_labels()
    model_handles = [plt.Line2D([], [], linestyle="none", marker=MODEL_MARKERS[m],
                                markerfacecolor="white", markeredgecolor="0.3",
                                markersize=4, label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(task_handles + model_handles, task_labels + [h.get_label() for h in model_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    _save(fig, args.stem)
    plt.close(fig)
    cells.to_csv(f"{args.stem}_cells.csv", index=False)
    print(cells.groupby(["template", "task"])[["gain", "scaffold", "entropy"]].mean().round(3))


SWAP_TEMPLATES = {"T_K": ("Knowledge template", "#0072B2"),
                  "T_R": ("Reasoning template", "#D55E00")}


def run_plot_swap(args: argparse.Namespace) -> None:
    """Template swap: every benchmark under both Figure 10 templates.

    (a) peak probe accuracy, (b) the final token's attention mass on the
    question and options -- the content both templates share, so unlike the
    instruction or cue it is comparable across them.  Bars are the mean over the three models,
    points the individual models; dashes in (a) mark the majority baseline.
    If the Figure 3 contrast came from the template, the bars in (a) would
    follow the template colour; they follow the benchmark instead.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    probe = pd.read_csv(os.path.join(args.out_dir, "probe_accuracy.csv"))
    attn = pd.read_csv(os.path.join(args.out_dir, "attention_profile.csv"))
    baseline = {}
    for dataset in DATASETS:
        with open(glob(os.path.join(args.out_dir, f"*__{dataset}__*.json"))[0]) as fh:
            baseline[dataset] = json.load(fh)["majority_baseline"]

    peak = probe.groupby(["model_key", "dataset_key", "template"]).accuracy.max()
    content = (attn.assign(s=attn.mass_question + attn.mass_choices)
               .groupby(["model_key", "dataset_key", "template"]).s.mean())

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4),
                             gridspec_kw={"left": 0.075, "right": 0.995, "bottom": 0.2,
                                          "top": 0.8, "wspace": 0.22})
    width = 0.36
    panels = [(peak, "(a) Is the answer linearly recoverable?", "Peak probe accuracy", (0.0, 1.0)),
              (content, "(b) How much does the final token attend to the task?",
               "Attention on question + options", (0.0, 0.4))]
    for ax, (values, title, ylabel, ylim) in zip(axes, panels):
        for i, dataset in enumerate(DATASETS):
            for j, (template, (label, colour)) in enumerate(SWAP_TEMPLATES.items()):
                x = i + (j - 0.5) * width
                per_model = [values[(m, dataset, template)] for m in MODELS]
                ax.bar(x, np.mean(per_model), width=width * 0.92, color=colour,
                       label=label if i == 0 else None)
                for m, v in zip(MODELS, per_model):
                    ax.scatter(x, v, s=10, marker=MODEL_MARKERS[m], facecolor="white",
                               edgecolor="0.15", linewidth=0.6, zorder=3)
            if values is peak:
                ax.hlines(baseline[dataset], i - width, i + width, colors="0.2",
                          linestyles="--", linewidth=0.8, zorder=4)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([f"{c['label']}\n({c['type']})" for c in DATASETS.values()])
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=4)
        ax.set_ylabel(ylabel)
        ax.axvline(1.5, color="0.75", linewidth=0.6)
        _style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.2", linestyle="--", linewidth=0.8))
    labels.append("Majority baseline")
    handles += [plt.Line2D([], [], linestyle="none", marker=MODEL_MARKERS[m], markerfacecolor="white",
                           markeredgecolor="0.15", markersize=4) for m in MODELS]
    labels += [MODEL_LABELS[m] for m in MODELS]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=6,
               frameon=False, handletextpad=0.4, columnspacing=1.1)
    _save(fig, args.stem)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    # `--out-dir` is shared by every stage and accepted on either side of the
    # subcommand, so the sbatch wrapper can append it to the command line.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir",
                        default="/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/prism/prompt_control")

    p = sub.add_parser("extract", help="one forward pass per item (GPU)", parents=[common])
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    p.add_argument("--template", required=True, choices=list(TEMPLATES) + ["all"])
    p.add_argument("--limit", type=int, default=0, help="smoke-test on the first N items")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave cells that already have an .npz alone (resume a killed job)")
    p.set_defaults(func=run_extract)

    p = sub.add_parser("probe", help="layer-wise probes over extracted states", parents=[common])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1, help="cells probed in parallel")
    p.add_argument("--only", nargs="+", metavar="PREFIX",
                   help="probe only stems starting with these, e.g. qwen__sciq")
    p.add_argument("--merge-only", action="store_true",
                   help="only merge probe_parts/*.csv into probe_accuracy.csv")
    p.set_defaults(func=run_probe)

    p = sub.add_parser("attention", help="aggregate attention readings", parents=[common])
    p.set_defaults(func=run_attention)

    p = sub.add_parser("cka", help="template-vs-template representation similarity", parents=[common])
    p.set_defaults(func=run_cka)

    p = sub.add_parser("report", help="separate the task-type and template effects", parents=[common])
    p.set_defaults(func=run_report)

    p = sub.add_parser("plot", help="probe accuracy per template", parents=[common])
    p.add_argument("--stem", default="assets/prompt_control_probe")
    p.add_argument("--published", default="assets/figure3_published.csv",
                   help="overlay the submitted Figure 3 curves ('' to disable)")
    p.set_defaults(func=run_plot)

    p = sub.add_parser("plot-fig3", help="Figure-3-style figure per template", parents=[common])
    p.add_argument("--stem", default="assets/figure3")
    p.add_argument("--published", default="",
                   help="overlay the submitted Figure 3 curves from this CSV")
    p.set_defaults(func=run_plot_fig3)

    p = sub.add_parser("plot-attention", help="attention mass and entropy", parents=[common])
    p.add_argument("--stem", default="assets/prompt_control_attention")
    p.set_defaults(func=run_plot_attention)

    p = sub.add_parser("plot-cka", help="template-vs-template CKA", parents=[common])
    p.add_argument("--stem", default="assets/prompt_control_cka")
    p.set_defaults(func=run_plot_cka)

    p = sub.add_parser("plot-summary", help="one-figure summary for revision 1", parents=[common])
    p.add_argument("--stem", default="assets/prompt_control_summary")
    p.set_defaults(func=run_plot_summary)

    p = sub.add_parser("plot-swap", help="template-swap figure for revision 1", parents=[common])
    p.add_argument("--stem", default="assets/prompt_control_swap")
    p.set_defaults(func=run_plot_swap)

    args = parser.parse_args(argv)
    args.out_dir = os.path.expanduser(args.out_dir)
    args.func(args)


if __name__ == "__main__":
    main()
