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
    p = argparse.ArgumentParser(description="Contrast fedavg vs fedprox under jitter for hybrid inv_true")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--config", default="src/configs/strategies/hybrid_opt.yaml")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", default="0.5,0.3")
    p.add_argument("--mus", default="0.01,0.02,0.05,0.1")
    return p.parse_args()


def metrics_from_stdout(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def run_once(
    python_exe: str,
    config: str,
    rounds: int,
    alpha: float,
    seed: int,
    algorithm: str,
    mu: float,
    out_dir: Path,
    tag: str,
) -> dict:
    cmd: List[str] = [
        python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        "hybrid_opt",
        "--config",
        config,
        "--wireless-model",
        "simulated",
        "--simulated-mode",
        "jitter",
        "--num-rounds",
        str(rounds),
        "--alpha",
        str(alpha),
        "--algorithm",
        algorithm,
        "--seed",
        str(seed),
    ]
    if algorithm == "fedprox":
        cmd.extend(["--fedprox-mu", str(mu)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) if not env.get("PYTHONPATH") else f"{Path.cwd()}:{env['PYTHONPATH']}"
    p = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, env=env)
    (out_dir / f"{tag}.log").write_text(p.stdout, encoding="utf-8")
    (out_dir / f"{tag}.err").write_text(p.stderr, encoding="utf-8")

    rec = {
        "tag": tag,
        "alpha": alpha,
        "algorithm": algorithm,
        "fedprox_mu": mu,
        "exit_code": p.returncode,
        "metrics_csv": "",
        "round": 0,
        "accuracy": 0.0,
        "loss": 0.0,
        "energy_sum": 0.0,
        "upload_time_sum": 0.0,
        "mean_jain": 0.0,
    }
    mp = metrics_from_stdout(p.stdout)
    if p.returncode == 0 and mp is not None:
        ap = (Path.cwd() / mp).resolve()
        if ap.exists():
            df = pd.read_csv(ap).sort_values("round")
            last = df.iloc[-1]
            rec["metrics_csv"] = str(ap)
            rec["round"] = int(last["round"])
            rec["accuracy"] = float(last.get("accuracy", 0.0))
            rec["loss"] = float(last.get("loss", 0.0))
            rec["energy_sum"] = float(df.get("energy", pd.Series([0.0])).sum())
            rec["upload_time_sum"] = float(df.get("est_upload_time", pd.Series([0.0])).sum())
            rec["mean_jain"] = float(df.get("jain", pd.Series([0.0])).mean())
    return rec


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    mus = [float(x.strip()) for x in args.mus.split(",") if x.strip()]

    rows = []
    for a in alphas:
        tag = f"alpha{a}_fedavg".replace(".", "p")
        rows.append(run_once(args.python_exe, args.config, args.rounds, a, args.seed, "fedavg", 0.0, out_dir, tag))
        for mu in mus:
            tag = f"alpha{a}_fedprox_mu{mu}".replace(".", "p")
            rows.append(run_once(args.python_exe, args.config, args.rounds, a, args.seed, "fedprox", mu, out_dir, tag))

    csv_path = out_dir / "contrast_algo_mu.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["exit_code"] == 0 and r["round"] > 0]
    ok.sort(key=lambda r: (r["accuracy"], -r["loss"]), reverse=True)
    if ok:
        best = ok[0]
        print(
            f"[BEST] alpha={best['alpha']} algo={best['algorithm']} mu={best['fedprox_mu']} "
            f"acc={best['accuracy']:.6f} loss={best['loss']:.6f}"
        )
    print(f"[OUT] {csv_path}")


if __name__ == "__main__":
    main()
