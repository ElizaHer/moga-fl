from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hybrid experiment CSVs")
    parser.add_argument("--runs-root", type=str, default="outputs/hybrid_runs")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis")
    return parser.parse_args()


def latest_run_dir(runs_root: str) -> Optional[str]:
    if not os.path.isdir(runs_root):
        return None
    candidates = [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(runs_root, candidates[0])


def summarize_csv(path: str) -> dict:
    df = pd.read_csv(path)
    if "round" not in df.columns:
        return {}
    df = df.sort_values("round")
    last = df.iloc[-1]
    return {
        "final_accuracy": float(last.get("accuracy", 0.0)),
        "final_loss": float(last.get("loss", 0.0)),
        "total_energy": float(df.get("energy", pd.Series([0.0])).sum()),
        "total_upload_time": float(df.get("est_upload_time", pd.Series([0.0])).sum()),
        "mean_jain": float(df.get("jain", pd.Series([0.0])).mean()),
        "min_jain": float(df.get("jain", pd.Series([0.0])).min()),
        "rounds": int(df["round"].max()),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.strip() or latest_run_dir(args.runs_root)
    if not run_dir:
        raise RuntimeError("No run directory found.")

    run_tag = os.path.basename(run_dir.rstrip("/\\"))
    out_dir = os.path.join(args.out_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(run_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=["strategy", "csv_path"])
        for root, _, files in os.walk(run_dir):
            for f in files:
                if f.lower().endswith(".csv") and f != "manifest.csv":
                    manifest.loc[len(manifest)] = [os.path.basename(root), os.path.join(root, f)]

    rows = []
    for _, row in manifest.iterrows():
        csv_path = row.get("csv_path")
        if not isinstance(csv_path, str) or not os.path.exists(csv_path):
            continue
        summary = summarize_csv(csv_path)
        if not summary:
            continue
        record = {**row.to_dict(), **summary}
        rows.append(record)

    if not rows:
        raise RuntimeError("No valid CSV found to summarize.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary_runs.csv"), index=False)

    group_cols = [c for c in ["strategy", "mode", "alpha", "algorithm"] if c in df.columns]
    if not group_cols:
        group_cols = ["strategy"]

    agg = df.groupby(group_cols).agg(
        final_accuracy_mean=("final_accuracy", "mean"),
        final_accuracy_std=("final_accuracy", "std"),
        final_loss_mean=("final_loss", "mean"),
        final_loss_std=("final_loss", "std"),
        total_energy_mean=("total_energy", "mean"),
        total_energy_std=("total_energy", "std"),
        total_upload_time_mean=("total_upload_time", "mean"),
        total_upload_time_std=("total_upload_time", "std"),
        mean_jain_mean=("mean_jain", "mean"),
        mean_jain_std=("mean_jain", "std"),
    ).reset_index()

    agg.to_csv(os.path.join(out_dir, "summary_by_strategy.csv"), index=False)
    print(f"Wrote: {os.path.join(out_dir, 'summary_runs.csv')}")
    print(f"Wrote: {os.path.join(out_dir, 'summary_by_strategy.csv')}")


if __name__ == "__main__":
    main()
