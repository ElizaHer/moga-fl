from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.ga.pareto import non_dominated_set

# 全局字体配置，确保中文正常显示
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _compute_pareto_indices(df: pd.DataFrame) -> List[int]:
    metrics: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        metrics.append(
            {
                "acc": float(row["acc"]),
                "time": float(row["time"]),
                "fairness": float(row["fairness"]),
                "energy": float(row["energy"]),
            }
        )
    return non_dominated_set(metrics)


def _plot_scatter(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    exp_name: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    ax.scatter(df[x_col], df[y_col], s=20, alpha=0.6, color="tab:gray", label="所有候选解")
    if not pareto_df.empty:
        ax.scatter(
            pareto_df[x_col],
            pareto_df[y_col],
            s=40,
            alpha=0.9,
            color="tab:orange",
            label="近似 Pareto 前沿",
        )

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_title(f"{exp_name}: {x_label} vs {y_label}")

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.25)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_name}_{x_col}_vs_{y_col}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = os.path.join("outputs", "ga_experiments")
    plot_dir = os.path.join("outputs", "ga_plots")

    if not os.path.isdir(base_dir):
        print(f"No GA experiments found at {base_dir}, nothing to plot.")
        return

    for exp_name in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        csv_path = os.path.join(exp_path, "pareto_candidates.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        required_cols = {"acc", "time", "fairness", "energy"}
        if not required_cols.issubset(df.columns):
            print(f"[WARN] {csv_path} missing required columns, skip.")
            continue

        if df.empty:
            print(f"[WARN] {csv_path} is empty, skip.")
            continue

        pareto_idx = _compute_pareto_indices(df)
        pareto_df = df.iloc[pareto_idx] if pareto_idx else df.iloc[[]]

        _plot_scatter(df, pareto_df, exp_name, "time", "acc", "时间", "准确率", plot_dir)
        _plot_scatter(df, pareto_df, exp_name, "energy", "acc", "能耗", "准确率", plot_dir)
        _plot_scatter(df, pareto_df, exp_name, "energy", "fairness", "能耗", "公平性", plot_dir)

        print(f"[INFO] Plots generated for experiment: {exp_name}")


if __name__ == "__main__":
    main()
