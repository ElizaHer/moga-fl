from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})(?=\.csv$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hybrid vs baselines from latest metrics CSVs")
    parser.add_argument("--metrics-root", type=str, default="outputs/hybrid_metrics")
    parser.add_argument("--out-dir", type=str, default="outputs/hybrid_plots")
    parser.add_argument(
        "--strategies",
        type=str,
        default="hybrid_opt,sync,async,bridge_free,bandwidth_first,energy_first",
        help="Comma-separated strategy folder names",
    )
    return parser.parse_args()


def _csv_timestamp(name: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def latest_csv_in_dir(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not files:
        return None

    dated: List[Tuple[datetime, str]] = []
    undated: List[str] = []
    for name in files:
        ts = _csv_timestamp(name)
        if ts is None:
            undated.append(name)
        else:
            dated.append((ts, name))

    if dated:
        dated.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(folder, dated[0][1])

    undated.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
    return os.path.join(folder, undated[0])


def load_latest_metrics(metrics_root: str, strategies: List[str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        folder = os.path.join(metrics_root, strategy)
        csv_path = latest_csv_in_dir(folder)
        if csv_path is None:
            print(f"[WARN] No CSV found for strategy: {strategy} ({folder})")
            continue
        df = pd.read_csv(csv_path)
        if "round" not in df.columns:
            print(f"[WARN] Missing 'round' column in {csv_path}, skip")
            continue
        data[strategy] = df.sort_values("round")
        print(f"[INFO] {strategy}: {csv_path}")
    return data


def plot_metric(
    data: Dict[str, pd.DataFrame],
    metric_col: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for strategy, df in data.items():
        if metric_col not in df.columns:
            print(f"[WARN] '{metric_col}' not in {strategy}, skip this line")
            continue
        plt.plot(df["round"], df[metric_col], marker="o", label=strategy)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_latest_metrics(args.metrics_root, strategies)
    if not data:
        raise RuntimeError("No valid metrics CSV found. Please run strategies first.")

    metric_specs = [
        ("accuracy", "Accuracy", "acc_comparison.png"),
        ("loss", "Loss", "loss_comparison.png"),
        ("energy", "Energy", "energy_comparison.png"),
        ("est_upload_time", "Total Upload Time", "upload_time_comparison.png"),
        ("jain", "Fairness (Jain)", "fairness_comparison.png"),
    ]
    for col, ylabel, filename in metric_specs:
        plot_metric(data, col, ylabel, os.path.join(args.out_dir, filename))


if __name__ == "__main__":
    main()
