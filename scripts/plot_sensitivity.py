# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

"""
Plot test accuracy against perturbation level for the top modulation configs.

The configuration names below are produced by generate_config_name and are the
three configurations carried into the sensitivity analysis.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_CSV = Path(
    os.environ.get("CCRCC_RESULTS_CSV", Path.cwd() / "mlflow_results.csv")
)
FIGURES_DIR = Path(os.environ.get("CCRCC_FIGURES", Path.cwd() / "figures"))
OUTPUT_DIR = FIGURES_DIR / "sensitivity"

CONFIG_NAMES = [
    "Mix_CoEm_a085_b30_s15_o050_w100125160200100",
    "Mix_CoEm_a065_b25_s20_o030_w100125155185105",
    "Mix_CoEm_a070_b25_s20_o035_w100120150185100",
]


def plot_config(df, config_name, output_dir):
    subset = df[df["config_name"].str.contains(config_name, na=False)]
    if subset.empty:
        print(f"No rows for {config_name}, skipping")
        return

    subset = subset.sort_values("perturbation_level")

    plt.figure(figsize=(8, 5))
    plt.plot(
        subset["perturbation_level"],
        subset["test_accuracy"],
        marker="o",
        label="Test Accuracy",
    )
    plt.plot(
        subset["perturbation_level"],
        subset["test_balanced_accuracy"],
        marker="s",
        label="Test Balanced Accuracy",
    )
    plt.xlabel("Perturbation Level")
    plt.ylabel("Accuracy")
    plt.title(config_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_dir / f"{config_name}.svg", format="svg")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in CONFIG_NAMES:
        plot_config(df, name, OUTPUT_DIR)

    print(f"Plots written to {OUTPUT_DIR}")
