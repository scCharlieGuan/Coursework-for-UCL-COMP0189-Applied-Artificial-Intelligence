"""Plotting helpers for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_topk_barh(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    output_path: str | Path,
    top_k: int = 20,
) -> None:
    """Save a horizontal bar chart for the top-k rows of a dataframe."""
    plot_df = df.head(top_k).sort_values(value_col, ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
