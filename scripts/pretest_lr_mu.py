from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import List

import pandas as pd


METRICS_RE = re.compile(r"Metrics CSV:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretest lr/fedprox_mu on wsn+hybrid+inv_true")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--config", default="src/configs/strategies/hybrid_opt.yaml")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--wireless-model", default="wsn", choices=["wsn", "simulated"])
    p.add_argument("--simulated-mode", default="jitter", choices=["good", "bad", "jitter"])
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lrs", default="0.0005,0.001,0.002")
    p.add_argument("--mus", default="0.01,0.05,0.1")
    return p.parse_args()


def parse_metrics_path(stdout_text: str) -> Path | None:
    lines = stdout_text.splitlines()
    for line in reversed(lines):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def run_one(args: argparse.Namespace, lr: float, mu: float, out_dir: Path) -> dict:
    cmd: List[str] = [
        args.python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        "hybrid_opt",
        "--config",
        args.config,
        "--wireless-model",
        args.wireless_model,
        "--num-rounds",
        str(args.num_rounds),
        "--alpha",
        str(args.alpha),
        "--algorithm",
        "fedprox",
        "--fedprox-mu",
        str(mu),
        "--lr",
        str(lr),
        "--seed",
        str(args.seed),
    ]
    if args.wireless_model == "simulated":
        cmd.extend(["--simulated-mode", args.simulated_mode])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) if not env.get("PYTHONPATH") else f"{Path.cwd()}:{env['PYTHONPATH']}"
    proc = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, env=env)
    log_path = out_dir / f"pretest_lr{lr}_mu{mu}.log"
    err_path = out_dir / f"pretest_lr{lr}_mu{mu}.err"
    log_path.write_text(proc.stdout, encoding="utf-8")
    err_path.write_text(proc.stderr, encoding="utf-8")

    metrics_path = parse_metrics_path(proc.stdout)
    rec = {
        "lr": lr,
        "fedprox_mu": mu,
        "exit_code": proc.returncode,
        "metrics_csv": "",
        "round": 0,
        "accuracy": 0.0,
        "loss": 0.0,
        "energy_sum": 0.0,
        "upload_time_sum": 0.0,
        "mean_jain": 0.0,
    }
    if proc.returncode == 0 and metrics_path is not None:
        csv_path = (Path.cwd() / metrics_path).resolve()
        if csv_path.exists():
            df = pd.read_csv(csv_path).sort_values("round")
            last = df.iloc[-1]
            rec["metrics_csv"] = str(csv_path)
            rec["round"] = int(last["round"])
            rec["accuracy"] = float(last.get("accuracy", 0.0))
            rec["loss"] = float(last.get("loss", 0.0))
            rec["energy_sum"] = float(df.get("energy", pd.Series([0.0])).sum())
            rec["upload_time_sum"] = float(df.get("est_upload_time", pd.Series([0.0])).sum())
            rec["mean_jain"] = float(df.get("jain", pd.Series([0.0])).mean())
    return rec


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    mus = [float(x.strip()) for x in args.mus.split(",") if x.strip()]

    rows = []
    for lr in lrs:
        for mu in mus:
            print(f"[RUN] lr={lr}, mu={mu}")
            rows.append(run_one(args, lr, mu, out_dir))

    csv_out = out_dir / "pretest_lr_mu_summary.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["exit_code"] == 0 and r["round"] > 0]
    if not ok:
        raise RuntimeError("No successful pretest run found.")
    ok.sort(key=lambda r: (r["accuracy"], -r["loss"]), reverse=True)
    best = ok[0]
    print(f"[BEST] lr={best['lr']} mu={best['fedprox_mu']} acc={best['accuracy']:.6f} loss={best['loss']:.6f}")
    print(f"[OUT] {csv_out}")


if __name__ == "__main__":
    main()
