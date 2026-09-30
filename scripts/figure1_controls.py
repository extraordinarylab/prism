"""Length and token-composition controls for the Figure 1 static profile.

Addresses TACL mandatory revision 3 (Reviewer A, W2 and general comment 3):
could sequence length or the share of symbolic tokens explain the lower-layer
activation sparsity of Figure 1?  The static profile reads the layer-output
hidden state at the final input token, so a third candidate is checked as
well: the identity of that final token.

Every item of Figure 1 (Qwen2.5-7B-Instruct, raw prompt without a chat
template, 100 items per benchmark drawn with shuffle(seed=42)) is re-run one
at a time, keeping per item what Figure 1 only kept as a benchmark mean:

  n_tokens         prompt length in tokens
  symbolic_ratio   share of prompt tokens carrying a digit or a math symbol
  final_class      the final token: digit / symbol / punctuation / word
  sparsity         activation sparsity (Eq. 4) at every layer

under two conditions:

  original   the Figure 1 prompt, unchanged
  suffix     the same prompt followed by "\\nAnswer:", so that every item of
             every benchmark ends on the same token

Three checks follow (`analyse`):

  (a) association  across the 24 benchmarks, Spearman correlation of
                   lower-layer sparsity with mean length and symbolic ratio,
                   plus an item-level regression
  (b) length       lower-layer sparsity recomputed on items inside a common
                   length band, compared with the original benchmark order
  (c) final token  the suffix condition, compared with the original order

Usage
-----
    python scripts/figure1_controls.py extract          # GPU
    python scripts/figure1_controls.py analyse
    python scripts/figure1_controls.py plot
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure1  # noqa: E402  (prompt construction and the benchmark list)

MODEL = figure1.DEFAULT_MODEL
NUM_EXAMPLES = figure1.DEFAULT_NUM_EXAMPLES
SUFFIX = "\nAnswer:"
CONDITIONS = ("original", "suffix")

# Figure 1 labels the first Transformer block "Layer 0"; lower layers = 0-3.
LOWER_LAYERS = slice(0, 4)

# Benchmark groups of the appendix taxonomy.
KNOWLEDGE = {"biomix-qa", "medmcqa", "mri-mcqa", "sciq", "qasc",
             "music-trivia", "truthful-qa", "halu-eval"}

MATH_SYMBOLS = set("+-*/=^_<>\\$%{}|()[]")
DEFAULT_OUT_DIR = "/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/prism/figure1_controls"
DEFAULT_STEM = "assets/figure1_controls"


# --------------------------------------------------------------------------- #
# Token composition
# --------------------------------------------------------------------------- #

def is_symbolic(piece: str) -> bool:
    """A token counts as symbolic when it carries a digit or a math symbol."""
    return any(ch.isdigit() or ch in MATH_SYMBOLS for ch in piece)


def token_class(piece: str) -> str:
    stripped = piece.strip()
    if not stripped:
        return "whitespace"
    if any(ch.isdigit() for ch in stripped):
        return "digit"
    if any(ch in MATH_SYMBOLS for ch in stripped):
        return "symbol"
    if re.fullmatch(r"[^\w\s]+", stripped):
        return "punctuation"
    return "word"


# --------------------------------------------------------------------------- #
# Stage 1: extraction
# --------------------------------------------------------------------------- #

def run_extract(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    snapshot = snapshot_download(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    # Same numerics as Figure 1: float16, eager attention.
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.float16, device_map="cuda", attn_implementation="eager")
    model.eval()

    names = args.datasets or list(figure1.DATASETS)
    for name in names:
        config = figure1.DATASETS[name]
        dataset = load_dataset(f"{figure1.HF_NAMESPACE}/{name}", split=config["split"])
        dataset = dataset.shuffle(seed=42).select(range(min(NUM_EXAMPLES, len(dataset))))
        record: Dict[str, List] = {f"{c}_{k}": [] for c in CONDITIONS
                                   for k in ("sparsity", "n_tokens", "symbolic_ratio", "final_class", "final_token")}
        for example in tqdm(dataset, desc=name):
            prompt = figure1.build_prompt(example, config)
            for condition in CONDITIONS:
                text = prompt + SUFFIX if condition == "suffix" else prompt
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                ids = inputs.input_ids[0].tolist()
                pieces = [tokenizer.decode([t]) for t in ids]
                hidden = torch.stack([h[0, -1] for h in out.hidden_states[1:]]).float()
                record[f"{condition}_sparsity"].append(
                    (hidden.abs() < figure1.SPARSITY_THRESHOLD).float().mean(-1).cpu().numpy())
                record[f"{condition}_n_tokens"].append(len(ids))
                record[f"{condition}_symbolic_ratio"].append(float(np.mean([is_symbolic(p) for p in pieces])))
                record[f"{condition}_final_class"].append(token_class(pieces[-1]))
                record[f"{condition}_final_token"].append(pieces[-1])
        np.savez(os.path.join(args.out_dir, f"{name}.npz"),
                 **{k: np.asarray(v) for k, v in record.items()})
        print(f"{name}: {len(dataset)} items, median {np.median(record['original_n_tokens']):.0f} tokens, "
              f"lower-layer sparsity {np.mean(record['original_sparsity'], 0)[LOWER_LAYERS].mean():.4f}",
              flush=True)


# --------------------------------------------------------------------------- #
# Stage 2: analysis
# --------------------------------------------------------------------------- #

def load_all(out_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    return {n: dict(np.load(os.path.join(out_dir, f"{n}.npz")))
            for n in figure1.DATASETS if os.path.exists(os.path.join(out_dir, f"{n}.npz"))}


def lower(sparsity: np.ndarray) -> np.ndarray:
    """Per-item mean activation sparsity over the lower layers."""
    return sparsity[:, LOWER_LAYERS].mean(1)


def length_band(runs: Dict[str, Dict[str, np.ndarray]], min_items: int,
                max_ratio: float = 2.5) -> tuple:
    """The widest-coverage token-length band shared by the benchmarks.

    Scans candidate bands no wider than `max_ratio` (hi / lo) and keeps the one
    retaining the most benchmarks with at least `min_items` items inside it
    (ties broken by total items kept).
    """
    lengths = {n: r["original_n_tokens"] for n, r in runs.items()}
    edges = sorted({int(x) for v in lengths.values() for x in np.percentile(v, [10, 25, 50, 75, 90])})
    best = None
    for lo in edges:
        for hi in edges:
            if not lo * 1.5 <= hi <= lo * max_ratio:
                continue
            kept = {n: int(((v >= lo) & (v <= hi)).sum()) for n, v in lengths.items()}
            ok = [n for n, k in kept.items() if k >= min_items]
            score = (len(ok), sum(kept[n] for n in ok), -(hi - lo))
            if best is None or score > best[0]:
                best = (score, lo, hi, ok)
    return best[1], best[2], best[3]


def run_analyse(args: argparse.Namespace) -> None:
    from scipy import stats

    runs = load_all(args.out_dir)
    names = list(runs)
    summary = {"datasets": names}

    # Reproduction check against the cached Figure 1 matrix.
    cached = np.load("assets/figure1_activation_sparsity.npy")
    order = [list(figure1.DATASETS).index(n) for n in names]
    mine = np.stack([runs[n]["original_sparsity"].mean(0) for n in names], axis=1)
    summary["reproduction_max_abs_diff"] = float(np.abs(mine - cached[:, order]).max())
    print(f"reproduction vs assets/figure1_activation_sparsity.npy: max |diff| = "
          f"{summary['reproduction_max_abs_diff']:.2e}")

    base = np.array([lower(runs[n]["original_sparsity"]).mean() for n in names])
    mean_len = np.array([runs[n]["original_n_tokens"].mean() for n in names])
    mean_sym = np.array([runs[n]["original_symbolic_ratio"].mean() for n in names])

    # (a) association across benchmarks.
    for label, x in (("log length", np.log(mean_len)), ("symbolic ratio", mean_sym)):
        r = stats.spearmanr(base, x)
        summary[f"a_spearman_{label}"] = [float(r.statistic), float(r.pvalue)]
        print(f"(a) benchmarks: Spearman(lower sparsity, {label}) = {r.statistic:+.2f} (p={r.pvalue:.3g})")

    # (a) item level: how much of the item variance do length, symbolic ratio
    # and final-token class explain, next to benchmark identity?
    y = np.concatenate([lower(runs[n]["original_sparsity"]) for n in names])
    ln = np.concatenate([np.log(runs[n]["original_n_tokens"]) for n in names])
    sym = np.concatenate([runs[n]["original_symbolic_ratio"] for n in names])
    cls = np.concatenate([runs[n]["original_final_class"] for n in names])
    bench = np.concatenate([[n] * len(runs[n]["original_n_tokens"]) for n in names])

    def r2(*blocks) -> float:
        X = np.column_stack([np.ones_like(y)] + list(blocks))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    def dummies(labels):
        levels = sorted(set(labels))[1:]
        return np.column_stack([(labels == l).astype(float) for l in levels])

    fits = {
        "length + symbolic": r2(ln, sym),
        "final-token class": r2(dummies(cls)),
        "length + symbolic + final-token class": r2(ln, sym, dummies(cls)),
        "benchmark identity": r2(dummies(bench)),
    }
    summary["a_item_r2"] = fits
    for k, v in fits.items():
        print(f"(a) items: R^2 of lower sparsity on {k:40s} {v:.3f}")
    classes = sorted(set(cls))
    summary["a_final_class_mean"] = {c: [float(y[cls == c].mean()), int((cls == c).sum())] for c in classes}
    for c in classes:
        print(f"    final token '{c}': mean lower sparsity {y[cls == c].mean():.4f} (n={int((cls == c).sum())})")

    # (b) common length band.
    lo, hi, kept = length_band(runs, args.min_items)
    band = {}
    for n in kept:
        m = (runs[n]["original_n_tokens"] >= lo) & (runs[n]["original_n_tokens"] <= hi)
        band[n] = float(lower(runs[n]["original_sparsity"][m]).mean())
    idx = [names.index(n) for n in kept]
    r = stats.spearmanr(base[idx], [band[n] for n in kept])
    summary["b_band"] = {"tokens": [lo, hi], "benchmarks": kept, "values": band,
                         "spearman_with_original": [float(r.statistic), float(r.pvalue)]}
    print(f"(b) length band {lo}-{hi} tokens keeps {len(kept)} benchmarks (>= {args.min_items} items); "
          f"Spearman with original order {r.statistic:+.2f} (p={r.pvalue:.3g})")

    # (b') length adjustment without a band: within each benchmark, regress
    # item sparsity on log length and read the fit at the pooled median length.
    ref = np.log(np.median(np.concatenate([runs[n]["original_n_tokens"] for n in names])))
    adjusted = []
    for n in names:
        yy, xx = lower(runs[n]["original_sparsity"]), np.log(runs[n]["original_n_tokens"])
        slope, intercept = np.polyfit(xx, yy, 1) if xx.std() > 0 else (0.0, yy.mean())
        adjusted.append(intercept + slope * ref)
    adjusted = np.array(adjusted)
    r = stats.spearmanr(base, adjusted)
    summary["b_adjusted"] = {"reference_tokens": float(np.exp(ref)),
                             "values": dict(zip(names, map(float, adjusted))),
                             "spearman_with_original": [float(r.statistic), float(r.pvalue)]}
    print(f"(b') length-adjusted to {np.exp(ref):.0f} tokens: Spearman with original order "
          f"{r.statistic:+.2f} (p={r.pvalue:.3g})")

    # (c) identical final token.
    suffix = np.array([lower(runs[n]["suffix_sparsity"]).mean() for n in names])
    r = stats.spearmanr(base, suffix)
    summary["c_suffix"] = {"values": dict(zip(names, map(float, suffix))),
                           "spearman_with_original": [float(r.statistic), float(r.pvalue)],
                           "spread_original": float(base.std()), "spread_suffix": float(suffix.std())}
    print(f"(c) fixed final token: Spearman with original order {r.statistic:+.2f} (p={r.pvalue:.3g}); "
          f"across-benchmark SD {base.std():.4f} -> {suffix.std():.4f}")
    by_depth = {}
    for label, sl in (("layers 0-3", slice(0, 4)), ("layers 4-13", slice(4, 14)), ("layers 14-27", slice(14, 28))):
        o = np.array([runs[n]["original_sparsity"][:, sl].mean() for n in names])
        x = np.array([runs[n]["suffix_sparsity"][:, sl].mean() for n in names])
        by_depth[label] = {"sd_original": float(o.std()), "sd_suffix": float(x.std()),
                           "spearman": float(stats.spearmanr(o, x).statistic)}
        print(f"    {label}: across-benchmark SD {o.std():.4f} -> {x.std():.4f}, "
              f"Spearman {by_depth[label]['spearman']:+.2f}")
    summary["c_by_depth"] = by_depth

    # Knowledge vs reasoning under each condition (benchmark means).
    group = np.array([n in KNOWLEDGE for n in names])
    for label, v in (("original", base), ("suffix", suffix)):
        u = stats.mannwhitneyu(v[~group], v[group], alternative="two-sided")
        summary[f"group_{label}"] = {"knowledge_mean": float(v[group].mean()),
                                     "reasoning_mean": float(v[~group].mean()), "p": float(u.pvalue)}
        print(f"    {label}: knowledge {v[group].mean():.4f} vs reasoning {v[~group].mean():.4f} "
              f"(Mann-Whitney p={u.pvalue:.3g})")

    summary["benchmarks"] = {n: {"lower_sparsity": float(b), "mean_tokens": float(l),
                                 "symbolic_ratio": float(s), "suffix_lower_sparsity": float(x)}
                             for n, b, l, s, x in zip(names, base, mean_len, mean_sym, suffix)}
    with open(f"{args.stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.stem}_summary.json")


# --------------------------------------------------------------------------- #
# Stage 3: plotting
# --------------------------------------------------------------------------- #

# A light, square frame so in-panel legends read as a unit without dominating.
LEGEND_STYLE = {"frameon": True, "fancybox": False, "edgecolor": "0.85", "framealpha": 1.0,
                "handletextpad": 0.3, "borderaxespad": 0.5, "title_fontsize": 7}

GROUP_STYLE = {True: ("Knowledge-centric", "#0072B2", "o"), False: ("Reasoning-centric", "#D55E00", "s")}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def run_plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    with open(f"{args.stem}_summary.json") as f:
        summary = json.load(f)
    bench = summary["benchmarks"]
    names = list(bench)
    knowledge = np.array([n in KNOWLEDGE for n in names])
    base = np.array([bench[n]["lower_sparsity"] for n in names])

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4),
                             gridspec_kw={"left": 0.075, "right": 0.995, "bottom": 0.2,
                                          "top": 0.9, "wspace": 0.32})

    # Left, middle: benchmark means against length and symbolic-token ratio.
    for ax, key, xlabel, logx in ((axes[0], "mean_tokens", "Mean prompt length (tokens)", True),
                                  (axes[1], "symbolic_ratio", "Symbolic-token ratio", False)):
        x = np.array([bench[n][key] for n in names])
        for g, (label, colour, marker) in GROUP_STYLE.items():
            m = knowledge == g
            ax.scatter(x[m], base[m], s=14, color=colour, marker=marker, label=label, zorder=3)
        if logx:
            ax.set_xscale("log")
        rho = summary[f"a_spearman_{'log length' if logx else 'symbolic ratio'}"][0]
        ax.set_title("Prompt length" if logx else "Symbolic tokens", pad=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Lower-layer sparsity")
        # Headroom above the data keeps the in-panel legend clear of the points.
        ax.set_ylim(base.min() - 0.005, base.max() + 0.045)
        ax.legend(loc="upper right", title=f"Spearman ρ = {rho:+.2f}", **LEGEND_STYLE
                  ).get_frame().set_linewidth(0.5)
        _style(ax)

    # Right: benchmark means under the two controls against the original.
    adjusted = summary["b_adjusted"]["values"]
    suffix = summary["c_suffix"]["values"]
    ax = axes[2]
    # Every benchmark lies above 0.03, so a zero-based axis leaves the
    # lower-right corner free for a two-line legend.
    lim = [0, max(base.max(), max(suffix.values()), max(adjusted.values())) * 1.06]
    ax.plot(lim, lim, color="0.6", linewidth=0.6, linestyle="--", zorder=1)
    ax.scatter(base, [adjusted[n] for n in names], s=14,
               facecolor="white", edgecolor="#009E73", linewidth=0.8, marker="D", zorder=3,
               label=f"Length-adjusted, ρ = {summary['b_adjusted']['spearman_with_original'][0]:+.2f}")
    ax.scatter(base, [suffix[n] for n in names], s=14, color="#CC79A7", marker="^", zorder=3,
               label=f"Same final token, ρ = {summary['c_suffix']['spearman_with_original'][0]:+.2f}")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Lower-layer sparsity (original)")
    ax.set_ylabel("Lower-layer sparsity (control)")
    ax.set_title("Controls", pad=4)
    ax.legend(loc="lower right", **LEGEND_STYLE
              ).get_frame().set_linewidth(0.5)
    _style(ax)

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.stem}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {args.stem}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    common.add_argument("--stem", default=DEFAULT_STEM)

    p = sub.add_parser("extract", parents=[common], help="per-item features (GPU)")
    p.add_argument("--datasets", nargs="+", choices=list(figure1.DATASETS))
    p.set_defaults(func=run_extract)

    p = sub.add_parser("analyse", parents=[common], help="checks (a)-(c)")
    p.add_argument("--min-items", type=int, default=20,
                   help="items a benchmark needs inside the length band to be kept")
    p.set_defaults(func=run_analyse)

    p = sub.add_parser("plot", parents=[common], help="draw the summary figure")
    p.set_defaults(func=run_plot)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
