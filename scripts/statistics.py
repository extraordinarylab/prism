"""Candidate statistical tests for PRISM, run over the figures' own data.

The manuscript is descriptive throughout.  This script computes the tests a
reader (or reviewer) would most naturally run against each claim, so the ones
worth reporting can be chosen on evidence rather than guessed at, and so the
numbers quoted in the paper can be regenerated.

Usage
-----
    python scripts/statistics.py            # -> assets/statistics_summary.csv

Design notes.  Every test here is non-parametric: the samples are small, the
distributions are not normal, and the paired designs are already available in
the exported data.  Where a design is paired it is tested as paired -- Wilcoxon
signed-rank across models for the CoT effect, exact McNemar across items for the
NOTA perturbations -- because a paired test on paired data is both more powerful
and more honest than treating the two conditions as independent.  Families of
tests are corrected with Benjamini-Hochberg, and every test is reported with an
effect size, since with these sample sizes a p-value alone says little.

The unit of analysis is stated for each test and is the thing to watch: a
question-level test over benchmarks that were not sampled at random is
pseudo-replicated with respect to any claim about benchmarks in general.
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_OUT = "assets/statistics_summary.csv"

# The Appendix A.1.1 taxonomy, over the 24 benchmarks of Figure 1.
KNOWLEDGE_BENCHMARKS = {
    "biomix-qa", "medmcqa", "mri-mcqa", "pubmed-qa",
    "sciq", "music-trivia", "truthful-qa", "halu-eval",
}
FIGURE4_KNOWLEDGE = ["sciq", "biomix-qa"]
FIGURE4_REASONING = ["logiqa", "math-qa"]
FIGURE4_MODELS = ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Olmo-3-7B-Instruct"]


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values, in the input order."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    return q


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test."""
    ranks = stats.rankdata(np.abs(differences))
    total = ranks.sum()
    if total == 0:
        return 0.0
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / total)


def cliffs_delta(u_statistic: float, n1: int, n2: int) -> float:
    """Cliff's delta from a Mann-Whitney U."""
    return float(2 * u_statistic / (n1 * n2) - 1)


def cot_effect(rows: List[Dict]) -> None:
    """Figure 5: is the CoT shift reliable across models?  Unit = model."""
    table = pd.read_csv("assets/figure5_scores.csv").pivot(
        index="model", columns="prompting"
    )
    for dimension in ("knowledge", "reasoning"):
        standard = table[(f"{dimension}_raw", "standard")]
        cot = table[(f"{dimension}_raw", "cot")]
        delta = (cot - standard).to_numpy()
        statistic, pvalue = stats.wilcoxon(cot, standard)
        rows.append(
            {
                "family": "figure5_cot_effect",
                "test": "Wilcoxon signed-rank (paired, two-sided)",
                "unit": "model",
                "n": delta.size,
                "comparison": f"{dimension}: CoT vs Standard",
                "estimate": float(np.median(delta)),
                "estimate_label": "median paired difference",
                "effect_size": rank_biserial(delta),
                "effect_size_label": "matched-pairs rank-biserial",
                "statistic": float(statistic),
                "p": float(pvalue),
                "detail": f"{int((delta > 0).sum())}/{delta.size} models improve",
            }
        )


def nota_perturbations(rows: List[Dict]) -> None:
    """Figures 6-7: does each perturbation move accuracy?  Unit = item (paired)."""
    aggregate = pd.read_csv("assets/figure6_aggregate.csv")
    records = []
    for _, row in aggregate.iterrows():
        b, c = int(row["mcnemar_b"]), int(row["mcnemar_c"])
        # Exact McNemar: among discordant pairs, is the split away from 50/50?
        pvalue = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        records.append(
            {
                "dataset": row["dataset"],
                "prompting": row["prompting"],
                "condition": row["condition"],
                "model": row["model"],
                "delta": row["drop"],
                "b": b,
                "c": c,
                "p": pvalue,
            }
        )
    frame = pd.DataFrame(records)
    frame["q"] = benjamini_hochberg(frame["p"].to_numpy())

    for condition, subset in frame.groupby("condition"):
        rows.append(
            {
                "family": "figure6_7_nota",
                "test": "exact McNemar (paired), Benjamini-Hochberg over 48 tests",
                "unit": "item",
                "n": int(len(subset)),
                "comparison": f"{condition}: perturbed vs base",
                "estimate": float(subset["delta"].median()),
                "estimate_label": "median accuracy drop",
                "effect_size": float((subset["q"] < 0.05).mean()),
                "effect_size_label": "fraction of model x dataset x prompting cells with q<0.05",
                "statistic": np.nan,
                "p": np.nan,
                "detail": (
                    f"{int((subset['q'] < 0.05).sum())}/{len(subset)} cells significant; "
                    f"drops {subset['delta'].min():+.3f} to {subset['delta'].max():+.3f}"
                ),
            }
        )
    frame.to_csv("assets/statistics_nota_mcnemar.csv", index=False, float_format="%.6g")
    print("wrote assets/statistics_nota_mcnemar.csv")


def figure1_group_difference(rows: List[Dict]) -> None:
    """Figure 1: do the two benchmark groups separate?  Unit = benchmark."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figure1

    names = list(figure1.DATASETS)
    knowledge = [i for i, n in enumerate(names) if n in KNOWLEDGE_BENCHMARKS]
    reasoning = [i for i, n in enumerate(names) if n not in KNOWLEDGE_BENCHMARKS]

    for metric in ("attention_entropy", "activation_sparsity"):
        matrix = np.load(f"assets/figure1_{metric}.npy")
        pvalues = [
            stats.mannwhitneyu(
                matrix[layer, knowledge], matrix[layer, reasoning], alternative="two-sided"
            ).pvalue
            for layer in range(matrix.shape[0])
        ]
        qvalues = benjamini_hochberg(np.array(pvalues))
        best = int(np.argmin(qvalues))
        rows.append(
            {
                "family": "figure1_group_difference",
                "test": "Mann-Whitney U per layer, BH over layers",
                "unit": "benchmark",
                "n": len(knowledge) + len(reasoning),
                "comparison": f"{metric}: knowledge vs reasoning benchmarks",
                "estimate": float(
                    np.median(matrix[best, knowledge]) - np.median(matrix[best, reasoning])
                ),
                "estimate_label": f"median difference at strongest layer ({best})",
                "effect_size": float((qvalues < 0.05).sum()),
                "effect_size_label": "layers significant at q<0.05",
                "statistic": np.nan,
                "p": float(np.min(pvalues)),
                "detail": f"{int((qvalues < 0.05).sum())}/{matrix.shape[0]} layers; min q={qvalues.min():.3f}",
            }
        )


def representation_gap(rows: List[Dict], n_bootstrap: int, seed: int) -> None:
    """Figure 4: how far apart are the two benchmark groups?  Unit = question."""
    store = np.load("assets/figure4_similarity.npz")
    rng = np.random.default_rng(seed)

    for model in FIGURE4_MODELS:
        n_layers = store[f"{model}|sciq|published_cosine"].shape[1]
        half = n_layers // 2

        def adjusted(benchmarks):
            """Per-question Eq. 8 similarity, averaged over mid-to-late layers."""
            out = []
            for benchmark in benchmarks:
                cosine = store[f"{model}|{benchmark}|published_cosine"][:, half:].mean(axis=1)
                baseline = store[f"{model}|{benchmark}|baseline_mean"][half:].mean()
                out.append(cosine - baseline)
            return np.concatenate(out)

        knowledge, reasoning = adjusted(FIGURE4_KNOWLEDGE), adjusted(FIGURE4_REASONING)
        u, pvalue = stats.mannwhitneyu(knowledge, reasoning, alternative="less")
        boot = [
            rng.choice(reasoning, reasoning.size).mean() - rng.choice(knowledge, knowledge.size).mean()
            for _ in range(n_bootstrap)
        ]
        rows.append(
            {
                "family": "figure4_representation_gap",
                "test": "Mann-Whitney U (one-sided) + percentile bootstrap CI",
                "unit": "question (nested in 2 benchmarks per group -- see README)",
                "n": knowledge.size + reasoning.size,
                "comparison": f"{model}: reasoning minus knowledge, mid-to-late layers",
                "estimate": float(reasoning.mean() - knowledge.mean()),
                "estimate_label": "mean gap",
                "effect_size": cliffs_delta(u, knowledge.size, reasoning.size),
                "effect_size_label": "Cliff's delta",
                "statistic": float(u),
                "p": float(pvalue),
                "detail": f"95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]",
            }
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows: List[Dict] = []
    cot_effect(rows)
    nota_perturbations(rows)
    # The Figure 2 trajectory test (Table tab:trajectory_variability) is run by
    # `python scripts/figure2_aligned.py analyse`.
    figure1_group_difference(rows)
    representation_gap(rows, args.n_bootstrap, args.seed)

    frame = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.6g")
    print(f"wrote {args.out} ({len(frame)} tests)")

    for family, subset in frame.groupby("family", sort=False):
        print(f"\n{family}")
        for _, row in subset.iterrows():
            p = "" if np.isnan(row["p"]) else f"  p={row['p']:.3g}"
            print(
                f"  {row['comparison']:58s} {row['estimate']:+.4f}"
                f"  {row['effect_size_label'][:28]}={row['effect_size']:+.3g}{p}"
            )
            print(f"      {row['detail']}")


if __name__ == "__main__":
    main()
