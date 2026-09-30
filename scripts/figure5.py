"""Figure 5 of PRISM: Tracing Knowledge and Reasoning in LLMs.

The knowledge-reasoning capability landscape.  Each model contributes one point
under Standard prompting (hollow marker) and one under CoT (solid marker), with
an arrow between them showing the CoT effect.  Axes are the min-max normalised
knowledge and reasoning scores of Section 3.4.

The pipeline has two stages:

  extract   parse the per-benchmark accuracies of Tables 3-6 out of the paper
            PDF into one long table.  These are the s_{m,p,d} of Section 3.4 and
            the only complete record of them in this repository.
  plot      average them into knowledge and reasoning scores, normalise, and
            draw the landscape.

Usage
-----
    python scripts/figure5.py extract      # -> assets/figure5_benchmark_scores.csv
    python scripts/figure5.py plot         # -> assets/figure5.{pdf,png} + tables

Method (paper Section 3.4).  Let s_{m,p,d} be the accuracy of model m under
prompting condition p on benchmark d.  Knowledge and reasoning scores are
unweighted benchmark-level averages,

    K_{m,p} = (1/|D_K|) sum_{d in D_K} s_{m,p,d}                        (9)
    R_{m,p} = (1/|D_R|) sum_{d in D_R} s_{m,p,d}                       (10)

and for either dimension X in {K, R},

    X_min = min_{m,p} X_{m,p}                                          (11)
    X_max = max_{m,p} X_{m,p}                                          (12)
    X~_{m,p} = (X_{m,p} - X_min) / (X_max - X_min)                     (13)

The minimum and maximum run over models *and* prompting conditions jointly, so
Standard and CoT share one scale and the arrow between them is meaningful.

What D_K and D_R actually contain.  The reported tables carry 10 knowledge
columns and 20 reasoning columns, but only 24 distinct datasets: HaluEval is
reported as three columns (D/Q/S), AIME as four (2022-2025), and AI2-ARC as two
(Easy/Challenge).  Averaging the columns -- which is what reproduces the
published figure, and is what `--average column`, the default, does -- therefore
weights HaluEval 3x, AIME 4x and AI2-ARC 2x relative to the other datasets in
their dimension.  `--average dataset` collapses each of those to a single score
first, giving a genuinely per-dataset unweighted mean over 8 knowledge and 16
reasoning datasets.  See the README.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_PDF = "TACL_2026_Tracing_Knowledge_and_Reasoning_in_LLM.pdf"
DEFAULT_STEM = "assets/figure5"

# Model rows of Tables 3-6, in table order, mapped to the figure's labels.
MODEL_LABELS = {
    "Llama3-8B-Instruct": "Llama3-8B",
    "Llama3.1-8B-Instruct": "Llama3.1-8B",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen3-8B (No-Think)": "Qwen3-NonThink-8B",
    "Qwen3-8B (Think)": "Qwen3-Think-8B",
    "Gemma-7B-Instruct": "Gemma-7B",
    "Gemma-2-2B-Instruct": "Gemma-2-2B",
    "Gemma-2-9B-Instruct": "Gemma-2-9B",
    "Gemma-3-4B-Instruct": "Gemma-3-4B",
    "Gemma-3-12B-Instruct": "Gemma-3-12B",
    "Gemma-3-27B-Instruct": "Gemma-3-27B",
    "GLM-4-9B-Instruct": "GLM-4-9B",
    "Olmo-3-7B-Instruct": "Olmo-3-7B",
    "GPT-OSS-20B": "GPT-OSS-20B",
    "Qwen3-30B-A3B": "Qwen3-30B-A3B",
}

# Each block of Tables 3-6: which table it belongs to, how many numbers a model
# row carries, the column names in order, and the dimension.
TABLE_BLOCKS = [
    (3, 14, ["MRI-MCQA", "BioMixQA", "SciQ", "QASC", "MusicTrivia", "TruthfulQA", "MedMCQA"], "knowledge"),
    (3, 6, ["HaluEval-D", "HaluEval-Q", "HaluEval-S"], "knowledge"),
    (4, 16, ["GSM8K", "MathQA", "MATH500", "AMC", "AIME22", "AIME23", "AIME24", "AIME25"], "reasoning"),
    (5, 10, ["CSQA", "SiQA", "PiQA", "Winogrande", "LogiQA"], "reasoning"),
    (6, 14, ["ARC-Easy", "ARC-Chal", "RACE", "PubMedQA", "MMLU-Pro", "GPQA-Dia", "CoinFlip"], "reasoning"),
]

# Columns that are subsets of one dataset in Table 1.
COLUMN_TO_DATASET = {
    "HaluEval-D": "HaluEval", "HaluEval-Q": "HaluEval", "HaluEval-S": "HaluEval",
    "AIME22": "AIME", "AIME23": "AIME", "AIME24": "AIME", "AIME25": "AIME",
    "ARC-Easy": "AI2-ARC", "ARC-Chal": "AI2-ARC",
}

PROMPTINGS = ["standard", "cot"]

# The knowledge-centric group as Appendix A.1.1 lists it.  This differs from
# where the results tables put two datasets: A.1.1 calls QASC reasoning-centric
# while Table 3 scores it under knowledge, and calls PubMedQA knowledge-centric
# while Table 6 scores it under reasoning.  The tables are what reproduce the
# published figure; --taxonomy appendix applies A.1.1 instead.
APPENDIX_KNOWLEDGE = {
    "BioMixQA", "MedMCQA", "MRI-MCQA", "PubMedQA", "SciQ", "MusicTrivia",
    "TruthfulQA", "HaluEval",
}

# One colour per model family, six broad families rather than the finer
# variants: a colour identifies a lineage, and scale or thinking mode is read
# from the label.
FAMILIES = {
    "Llama3-8B": "llama", "Llama3.1-8B": "llama",
    "Qwen2.5-3B": "qwen", "Qwen2.5-7B": "qwen", "Qwen2.5-14B": "qwen",
    "Qwen3-NonThink-8B": "qwen", "Qwen3-Think-8B": "qwen", "Qwen3-30B-A3B": "qwen",
    "Olmo-3-7B": "olmo",
    "Gemma-7B": "gemma", "Gemma-2-2B": "gemma", "Gemma-2-9B": "gemma",
    "Gemma-3-4B": "gemma", "Gemma-3-12B": "gemma", "Gemma-3-27B": "gemma",
    "GLM-4-9B": "glm",
    "GPT-OSS-20B": "gpt_oss",
}

FAMILY_ORDER = ["llama", "qwen", "olmo", "gemma", "glm", "gpt_oss"]
FAMILY_DISPLAY = {
    "llama": "Llama", "qwen": "Qwen", "olmo": "OLMo",
    "gemma": "Gemma", "glm": "GLM", "gpt_oss": "GPT-OSS",
}

# Six families need all six usable Okabe-Ito hues (yellow is unreadable as a
# marker on white), so the palette is forced and only the assignment is free.
# Three pairs stay confusable under simulated colour-vision deficiency --
# reddish purple/orange, blue/bluish green, sky blue/bluish green -- so the
# assignment below was chosen to maximise, over every family pair, the product
# of their colours' worst-case deltaE and how close those families actually sit
# on the plot.  The near-coincident families get the most separable colours:
# OLMo and Qwen almost touch and are blue against vermillion (deltaE 86).
FAMILY_COLOURS = {
    "llama": "#56B4E9",     # sky blue
    "qwen": "#D55E00",      # vermillion
    "olmo": "#0072B2",      # blue
    "gemma": "#CC79A7",     # reddish purple
    "glm": "#009E73",       # bluish green
    "gpt_oss": "#E69F00",   # orange
}

FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 4.65
SUBPLOT_LEFT = 0.085
SUBPLOT_RIGHT = 0.985
SUBPLOT_BOTTOM = 0.095
SUBPLOT_TOP = 0.90

# Midpoint of the normalised range, so a model's quadrant says whether it is
# above or below the middle of the field on each axis.
REFERENCE_LINE_VALUE = 0.5
AXIS_LIMIT = (-0.04, 1.04)
AXIS_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

STANDARD_MARKER_SIZE = 30
COT_MARKER_SIZE = 32
ARROW_LINE_WIDTH = 1.0
ARROW_MUTATION_SCALE = 8
ARROW_SHRINK_POINTS = 4
LABEL_FONT_SIZE = 6.5
LEGEND_ANCHOR = (0.5, 0.995)

# --- Automatic label placement -------------------------------------------- #
# Seventeen labels in one panel collide if placed blindly.  Each label is
# anchored to its own arrow -- at the CoT point first, since that is the point a
# reader associates with the model -- and only slides to the arrow midpoint or
# the standard point when no direction around the CoT point is clear enough.  A
# model's own markers and arrow never count as obstacles for its own label.
LABEL_ANCHOR_FRACTIONS = [1.0, 0.5, 0.0]
LABEL_ANGLES_DEGREES = [0, 45, 90, 135, 180, 225, 270, 315]
LABEL_OFFSET_RADIUS_POINTS = 7.5
ACCEPTABLE_OVERLAP_PENALTY = 1
LABEL_CHAR_WIDTH_POINTS = LABEL_FONT_SIZE * 0.52
LABEL_TEXT_HEIGHT_POINTS = LABEL_FONT_SIZE * 1.25
MARKER_CLEARANCE_POINTS = 3.0
ARROW_CLEARANCE_POINTS = 2.5
LABEL_CLEARANCE_POINTS = 1.5
CROWDING_RADIUS = 0.08


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #

def apply_taxonomy(df: pd.DataFrame, taxonomy: str) -> pd.DataFrame:
    """Which dimension each benchmark counts toward."""
    if taxonomy == "tables":
        return df
    out = df.copy()
    out["dimension"] = np.where(
        out["dataset"].isin(APPENDIX_KNOWLEDGE), "knowledge", "reasoning"
    )
    return out


def dimension_scores(df: pd.DataFrame, dimension: str, average: str) -> pd.DataFrame:
    """Eq. 9 / Eq. 10 for one dimension, as a model x prompting frame."""
    sub = df[df.dimension == dimension]
    if average == "dataset":
        # Collapse each dataset's reported subsets before averaging, so every
        # dataset carries equal weight.
        sub = sub.groupby(["model", "prompting", "dataset"], as_index=False)["score"].mean()
    return sub.groupby(["model", "prompting"])["score"].mean().unstack()


def minmax_normalise(scores: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Eqs. 11-13 — bounds taken over models and prompting conditions jointly."""
    stacked = pd.concat([scores[p] for p in scores.columns])
    lo, hi = float(stacked.min()), float(stacked.max())
    if hi == lo:
        raise ValueError("cannot normalise a dimension with zero range")
    return (scores - lo) / (hi - lo), lo, hi


def build_scores(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    df = apply_taxonomy(df, args.taxonomy)
    raw = {d: dimension_scores(df, d, args.average) for d in ("knowledge", "reasoning")}
    norm, bounds = {}, {}
    for dim, scores in raw.items():
        norm[dim], lo, hi = minmax_normalise(scores)
        bounds[dim] = {"min": round(lo, 6), "max": round(hi, 6)}

    rows = []
    for model in raw["knowledge"].index:
        for prompting in PROMPTINGS:
            rows.append(
                {
                    "model": model,
                    "family": FAMILIES[model],
                    "prompting": prompting,
                    "knowledge_raw": raw["knowledge"].loc[model, prompting],
                    "reasoning_raw": raw["reasoning"].loc[model, prompting],
                    "knowledge_normalised": norm["knowledge"].loc[model, prompting],
                    "reasoning_normalised": norm["reasoning"].loc[model, prompting],
                }
            )
    table = pd.DataFrame(rows)

    unit = "benchmark" if args.average == "column" else "dataset"
    counts = {
        d: int(df[df.dimension == d][unit].nunique()) for d in ("knowledge", "reasoning")
    }
    manifest = {
        "run_id": args.run_id,
        "average": args.average,
        "taxonomy": args.taxonomy,
        "averaging_unit": unit,
        "D_K_size": counts["knowledge"],
        "D_R_size": counts["reasoning"],
        "n_models": int(table.model.nunique()),
        "promptings": PROMPTINGS,
        "normalisation_bounds": bounds,
        "equations": {
            "9_10": "unweighted mean of the accuracies in each dimension",
            "11_13": (
                "min and max over models and prompting conditions jointly, "
                "separately per dimension, then min-max scaled to [0, 1]"
            ),
        },
        "column_vs_dataset": (
            "the reported tables carry 10 knowledge and 20 reasoning columns "
            "over 24 distinct datasets; HaluEval is 3 columns, AIME 4 and "
            "AI2-ARC 2. --average column (default) reproduces the published "
            "figure and weights those datasets 3x/4x/2x; --average dataset "
            "collapses them first for an equal-weight per-dataset mean"
        ),
        "taxonomy_note": (
            "the results tables score QASC under knowledge and PubMedQA under "
            "reasoning; Appendix A.1.1 assigns them the other way round. "
            "--taxonomy tables (default) reproduces the published figure"
        ),
        "source": f"Tables 3-6 of {args.pdf}",
    }
    return table, manifest


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

import math


def data_units_per_point(args) -> Tuple[float, float]:
    """Conversion from typographic points to data units for this figure."""
    width_points = args.width * (SUBPLOT_RIGHT - SUBPLOT_LEFT) * 72.0
    height_points = args.height * (SUBPLOT_TOP - SUBPLOT_BOTTOM) * 72.0
    span = AXIS_LIMIT[1] - AXIS_LIMIT[0]
    return span / width_points, span / height_points


def alignment_for_angle(angle: int) -> Tuple[str, str]:
    """Alignment so the text reads outward from its anchor."""
    vertical = "bottom" if angle in (45, 90, 135) else "top" if angle in (225, 270, 315) else "center"
    horizontal = "right" if angle in (135, 180, 225) else "left" if angle in (315, 0, 45) else "center"
    return horizontal, vertical


def label_box(anchor, offset, alignment, text, per_point) -> Tuple[float, float, float, float]:
    """The data-space box a label would occupy."""
    (x_per_point, y_per_point) = per_point
    horizontal, vertical = alignment
    cx = anchor[0] + offset[0] * x_per_point
    cy = anchor[1] + offset[1] * y_per_point
    width = len(text) * LABEL_CHAR_WIDTH_POINTS * x_per_point
    height = LABEL_TEXT_HEIGHT_POINTS * y_per_point

    if horizontal == "left":
        x0, x1 = cx, cx + width
    elif horizontal == "right":
        x0, x1 = cx - width, cx
    else:
        x0, x1 = cx - width / 2, cx + width / 2
    if vertical == "bottom":
        y0, y1 = cy, cy + height
    elif vertical == "top":
        y0, y1 = cy - height, cy
    else:
        y0, y1 = cy - height / 2, cy + height / 2
    return (x0, x1, y0, y1)


def point_in_box(x, y, box, pad_x, pad_y) -> bool:
    return box[0] - pad_x <= x <= box[1] + pad_x and box[2] - pad_y <= y <= box[3] + pad_y


def boxes_overlap(a, b, pad_x, pad_y) -> bool:
    if a[1] + pad_x < b[0] or b[1] + pad_x < a[0]:
        return False
    return not (a[3] + pad_y < b[2] or b[3] + pad_y < a[2])


def segment_crosses_box(p0, p1, box, pad_x, pad_y, samples: int = 12) -> bool:
    """Whether the straight arrow from p0 to p1 passes through box.

    Sampled rather than solved: every arrow here is short and straight, so a
    dozen samples is indistinguishable from exact at this scale.
    """
    for step in range(samples + 1):
        t = step / samples
        if point_in_box(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, box, pad_x, pad_y):
            return True
    return False


def overlap_penalty(box, points, segments, placed, per_point) -> int:
    """How much a candidate label box collides with everything else.

    Another label counts triple: two overlapping labels are the most confusing
    outcome for a reader, worse than a label grazing a marker.
    """
    x_per_point, y_per_point = per_point
    penalty = 0
    for x, y in points:
        if point_in_box(x, y, box, MARKER_CLEARANCE_POINTS * x_per_point, MARKER_CLEARANCE_POINTS * y_per_point):
            penalty += 1
    for p0, p1 in segments:
        if segment_crosses_box(p0, p1, box, ARROW_CLEARANCE_POINTS * x_per_point, ARROW_CLEARANCE_POINTS * y_per_point):
            penalty += 1
    for other in placed:
        if boxes_overlap(box, other, LABEL_CLEARANCE_POINTS * x_per_point, LABEL_CLEARANCE_POINTS * y_per_point):
            penalty += 3
    return penalty


def choose_label_layout(model, standard, cot, points, segments, placed, per_point):
    """Anchor, offset and alignment for one model's label."""
    fallback = None
    for fraction in LABEL_ANCHOR_FRACTIONS:
        anchor = (
            standard[0] + (cot[0] - standard[0]) * fraction,
            standard[1] + (cot[1] - standard[1]) * fraction,
        )
        best = None
        for angle in LABEL_ANGLES_DEGREES:
            offset = (
                LABEL_OFFSET_RADIUS_POINTS * math.cos(math.radians(angle)),
                LABEL_OFFSET_RADIUS_POINTS * math.sin(math.radians(angle)),
            )
            alignment = alignment_for_angle(angle)
            box = label_box(anchor, offset, alignment, model, per_point)
            penalty = overlap_penalty(box, points, segments, placed, per_point)
            if best is None or penalty < best[0]:
                best = (penalty, (anchor, offset, alignment, box))
        if fallback is None or best[0] < fallback[0]:
            fallback = best
        if best[0] <= ACCEPTABLE_OVERLAP_PENALTY:
            return best[1]
    return fallback[1]


def plot_landscape(table: pd.DataFrame, args: argparse.Namespace) -> None:
    """One arrow per model, from its Standard point to its CoT point."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.lines import Line2D

    suffix = "raw" if args.raw else "normalised"
    xcol, ycol = f"knowledge_{suffix}", f"reasoning_{suffix}"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
        }
    )

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.subplots_adjust(
        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP,
    )

    ax.axvline(REFERENCE_LINE_VALUE, color="#9a9a9a", linestyle=(0, (4, 3)), linewidth=0.9, zorder=0)
    ax.axhline(REFERENCE_LINE_VALUE, color="#9a9a9a", linestyle=(0, (4, 3)), linewidth=0.9, zorder=0)

    wide = table.pivot(index="model", columns="prompting")
    models = []
    for model in wide.index:
        colour = FAMILY_COLOURS[FAMILIES[model]]
        standard = (wide.loc[model, (xcol, "standard")], wide.loc[model, (ycol, "standard")])
        cot = (wide.loc[model, (xcol, "cot")], wide.loc[model, (ycol, "cot")])
        models.append((model, colour, standard, cot))

        ax.annotate(
            "", xy=cot, xytext=standard,
            arrowprops={
                "arrowstyle": "-|>", "color": colour,
                "linewidth": ARROW_LINE_WIDTH, "alpha": 0.75,
                "mutation_scale": ARROW_MUTATION_SCALE,
                "shrinkA": ARROW_SHRINK_POINTS, "shrinkB": ARROW_SHRINK_POINTS,
            },
            zorder=1,
        )
        ax.scatter(*standard, s=STANDARD_MARKER_SIZE, facecolor="white",
                   edgecolor=colour, linewidth=1.25, zorder=3)
        ax.scatter(*cot, s=COT_MARKER_SIZE, facecolor=colour,
                   edgecolor="white", linewidth=0.55, zorder=4)

    # Most crowded models label first, so the hardest cases get first pick of
    # clear space.
    crowding = []
    for i, (_, _, standard, cot) in enumerate(models):
        count = 0
        for j, (_, _, other_std, other_cot) in enumerate(models):
            if i == j:
                continue
            for mine in (standard, cot):
                for theirs in (other_std, other_cot):
                    if math.hypot(mine[0] - theirs[0], mine[1] - theirs[1]) < CROWDING_RADIUS:
                        count += 1
        crowding.append((count, i))
    crowding.sort(reverse=True)

    per_point = data_units_per_point(args)
    placed = []
    for _, i in crowding:
        model, _, standard, cot = models[i]
        points, segments = [], []
        for j, (_, _, other_std, other_cot) in enumerate(models):
            if i == j:
                continue
            points.extend([other_std, other_cot])
            segments.append((other_std, other_cot))

        anchor, offset, alignment, box = choose_label_layout(
            model, standard, cot, points, segments, placed, per_point
        )
        placed.append(box)
        ax.annotate(
            model, anchor, xytext=offset, textcoords="offset points",
            fontsize=LABEL_FONT_SIZE, ha=alignment[0], va=alignment[1], zorder=5,
        )

    prefix = "" if args.raw else "Normalized "
    ax.set_xlabel(f"{prefix}Knowledge Score")
    ax.set_ylabel(f"{prefix}Reasoning Score")
    ax.set_xlim(*AXIS_LIMIT)
    ax.set_ylim(*AXIS_LIMIT)
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)

    class HandlerArrow(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            return [mpatches.FancyArrowPatch(
                posA=(0, height / 2), posB=(width, height / 2), arrowstyle="-|>",
                color="#444444", lw=1.0, mutation_scale=7, transform=trans)]

    present = [f for f in FAMILY_ORDER if f in set(table["model"].map(FAMILIES))]
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=FAMILY_COLOURS[f], markeredgecolor=FAMILY_COLOURS[f],
               label=FAMILY_DISPLAY[f])
        for f in present
    ]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor="white", markeredgecolor="#444444", label="Standard"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor="#444444", markeredgecolor="white", label="CoT"),
        mpatches.FancyArrowPatch((0, 0), (1, 1), label="CoT Effect"),
    ]
    fig.legend(
        handles=handles,
        handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
        loc="upper center", bbox_to_anchor=LEGEND_ANCHOR,
        ncol=len(handles), frameon=False,
        columnspacing=1.0, handletextpad=0.4, handlelength=1.8,
    )

    ensure_parent(args.stem)
    for ext in ("pdf", "png"):
        path = f"{args.stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="?", default="plot", choices=["extract", "plot", "all"])
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument(
        "--average",
        choices=["column", "dataset"],
        default="column",
        help="average the reported table columns (default, reproduces the "
        "published figure) or collapse HaluEval/AIME/AI2-ARC subsets first",
    )
    parser.add_argument(
        "--taxonomy",
        choices=["tables", "appendix"],
        default="tables",
        help="assign benchmarks to dimensions as the results tables do "
        "(default, reproduces the published figure) or as Appendix A.1.1 does",
    )
    parser.add_argument("--width", type=float, default=FIGURE_WIDTH_IN)
    parser.add_argument("--height", type=float, default=FIGURE_HEIGHT_IN)
    parser.add_argument("--raw", action="store_true",
                        help="plot the unnormalised scores of Eqs. 9-10")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    if args.run_id is None:
        args.run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    scores_path = f"{args.stem}_benchmark_scores.csv"
    if args.stage in ("extract", "all") or not os.path.exists(scores_path):
        df = run_extract(args)
    else:
        df = pd.read_csv(scores_path)
    if args.stage == "extract":
        return

    table, manifest = build_scores(df, args)
    out = f"{args.stem}_scores.csv"
    table.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({len(table)} rows)")
    with open(f"{args.stem}_run.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.stem}_run.json")
    plot_landscape(table, args)


if __name__ == "__main__":
    main()
