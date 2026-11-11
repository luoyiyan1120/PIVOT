"""Plot layer-wise cosine similarity trends from a CSV file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layer-wise cosine similarity trends.")
    parser.add_argument("--input-csv", required=True, help="CSV produced by compute_layer_similarity_llava.py.")
    parser.add_argument("--output-png", required=True, help="Path to save the figure.")
    parser.add_argument("--highlight-start", type=int, default=None, help="Optional highlighted layer range start.")
    parser.add_argument("--highlight-end", type=int, default=None, help="Optional highlighted layer range end.")
    parser.add_argument("--title", default="Layer-wise Cosine Similarity Trend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    layer_columns = [column for column in df.columns if column.startswith("layer_")]
    layer_columns.sort(key=lambda column: int(column.split("_")[1]))
    if not layer_columns:
        raise ValueError("No layer_* columns found in the input CSV.")

    df[layer_columns] = df[layer_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=layer_columns)

    mean_values = df[layer_columns].mean()
    std_values = df[layer_columns].std()
    x_indices = [int(column.split("_")[1]) for column in layer_columns]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    line_color = "#1f77b4"
    fill_color = "#a6cee3"
    highlight_color = "#ff7f0e"

    plt.plot(
        x_indices,
        mean_values,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=2,
        color=line_color,
        label="Avg. Cosine Similarity",
    )
    plt.fill_between(
        x_indices,
        mean_values - std_values,
        mean_values + std_values,
        color=fill_color,
        alpha=0.4,
        label="Standard Deviation",
    )

    if args.highlight_start is not None and args.highlight_end is not None:
        plt.axvspan(
            args.highlight_start,
            args.highlight_end,
            color=highlight_color,
            alpha=0.15,
            label=f"Selected MIP layers (L{args.highlight_start}-L{args.highlight_end})",
        )
        plt.axvline(x=args.highlight_start, color=highlight_color, linestyle="--", alpha=0.6)
        plt.axvline(x=args.highlight_end, color=highlight_color, linestyle="--", alpha=0.6)

    min_val = mean_values.min()
    min_idx = int(mean_values.idxmin().split("_")[1])
    plt.annotate(
        "Max Divergence",
        xy=(min_idx, min_val),
        xytext=(min_idx, min_val + 0.05),
        arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1.5, "headwidth": 8},
        horizontalalignment="center",
        fontsize=10,
        fontweight="bold",
    )

    plt.title(args.title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Transformer Layer Index", fontsize=12)
    plt.ylabel("Cosine Similarity (Original vs. Legitimized)", fontsize=12)
    plt.xticks(np.arange(min(x_indices), max(x_indices) + 1, 2))
    plt.ylim(max(0, min_val - 0.1), 1.01)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    print(f"Saved figure to {output_png}")


if __name__ == "__main__":
    main()
