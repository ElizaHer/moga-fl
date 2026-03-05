from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


METRICS_RE = re.compile(r"Metrics CSV:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted tuning for jitter+hybrid+inv_true")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--base-config", default="src/configs/strategies/hybrid_opt.yaml")
    p.add_argument("--run-tag", default="")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--fedprox-mu", type=float, default=0.05)
    p.add_argument("--short-rounds", type=int, default=30)
    p.add_argument("--final-rounds", type=int, default=60)
    return p.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def apply_common(cfg: Dict, lr: float, mu: float) -> Dict:
    cfg.setdefault("fl", {})["lr"] = float(lr)
    cfg.setdefault("algorithm", {})["name"] = "fedprox"
    cfg["algorithm"]["fedprox_mu"] = float(mu)
    cfg.setdefault("wireless", {})["wireless_model"] = "simulated"
    cfg["wireless"]["simulated_mode"] = "jitter"
    cfg.setdefault("controller", {}).setdefault("bridge_invariants", {})["enable"] = True
    cfg.setdefault("fedbuff", {})["async_agg_interval"] = 2
    return cfg


def apply_profile(cfg: Dict, name: str) -> Dict:
    c = cfg
    c.setdefault("controller", {})
    c["controller"].setdefault("gate_thresholds", {})
    c["controller"].setdefault("gate_weights", {})
    c["controller"].setdefault("bridge_invariants", {})
    c.setdefault("fedbuff", {})

    if name == "p1_async_sensitive_relaxed_inv":
        c["controller"]["semi_sync_wait_ratio"] = 0.72
        c["controller"]["gate_thresholds"]["to_async"] = 0.54
        c["controller"]["gate_thresholds"]["to_semi_sync"] = 0.44
        c["controller"]["hysteresis_margin"] = 0.02
        c["controller"]["bridge_rounds"] = 2
        c["controller"]["min_rounds_between_switch"] = 2
        c["controller"]["gate_weights"]["per"] = 0.65
        c["controller"]["gate_weights"]["fairness"] = 0.20
        c["controller"]["gate_weights"]["energy"] = 0.15
        c["controller"]["bridge_invariants"]["energy_budget_round"] = 160.0
        c["controller"]["bridge_invariants"]["upload_time_budget_round"] = 3.2
        c["controller"]["bridge_invariants"]["thresholds"] = {"th1": 0.18, "th2": 0.35, "th3": 0.55}
        c["controller"]["bridge_invariants"]["downweight_factor"] = 0.7
        c["controller"]["bridge_invariants"]["rate_limit_factor"] = 0.9
        c["fedbuff"]["buffer_size"] = 10
        c["fedbuff"]["min_updates_to_aggregate"] = 4
        c["fedbuff"]["staleness_alpha"] = 1.2
        c["fedbuff"]["max_staleness"] = 8
    elif name == "p2_balanced_bridge":
        c["controller"]["semi_sync_wait_ratio"] = 0.78
        c["controller"]["gate_thresholds"]["to_async"] = 0.58
        c["controller"]["gate_thresholds"]["to_semi_sync"] = 0.42
        c["controller"]["hysteresis_margin"] = 0.03
        c["controller"]["bridge_rounds"] = 4
        c["controller"]["min_rounds_between_switch"] = 4
        c["controller"]["gate_weights"]["per"] = 0.55
        c["controller"]["gate_weights"]["fairness"] = 0.25
        c["controller"]["gate_weights"]["energy"] = 0.20
        c["controller"]["bridge_invariants"]["energy_budget_round"] = 130.0
        c["controller"]["bridge_invariants"]["upload_time_budget_round"] = 2.6
        c["controller"]["bridge_invariants"]["thresholds"] = {"th1": 0.12, "th2": 0.28, "th3": 0.48}
        c["controller"]["bridge_invariants"]["downweight_factor"] = 0.55
        c["controller"]["bridge_invariants"]["rate_limit_factor"] = 0.85
        c["fedbuff"]["buffer_size"] = 8
        c["fedbuff"]["min_updates_to_aggregate"] = 4
        c["fedbuff"]["staleness_alpha"] = 1.0
        c["fedbuff"]["max_staleness"] = 7
    elif name == "p3_stability_fairness":
        c["controller"]["semi_sync_wait_ratio"] = 0.85
        c["controller"]["gate_thresholds"]["to_async"] = 0.63
        c["controller"]["gate_thresholds"]["to_semi_sync"] = 0.39
        c["controller"]["hysteresis_margin"] = 0.05
        c["controller"]["bridge_rounds"] = 5
        c["controller"]["min_rounds_between_switch"] = 5
        c["controller"]["gate_weights"]["per"] = 0.45
        c["controller"]["gate_weights"]["fairness"] = 0.35
        c["controller"]["gate_weights"]["energy"] = 0.20
        c["controller"]["bridge_invariants"]["energy_budget_round"] = 120.0
        c["controller"]["bridge_invariants"]["upload_time_budget_round"] = 2.5
        c["controller"]["bridge_invariants"]["thresholds"] = {"th1": 0.10, "th2": 0.25, "th3": 0.45}
        c["controller"]["bridge_invariants"]["downweight_factor"] = 0.5
        c["controller"]["bridge_invariants"]["rate_limit_factor"] = 0.8
        c["fedbuff"]["buffer_size"] = 12
        c["fedbuff"]["min_updates_to_aggregate"] = 5
        c["fedbuff"]["staleness_alpha"] = 1.4
        c["fedbuff"]["max_staleness"] = 6
    else:
        raise ValueError(f"unknown profile: {name}")
    return c


def parse_metrics(stdout_text: str) -> Path | None:
    for line in reversed(stdout_text.splitlines()):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def run_job(
    python_exe: str,
    cfg_path: Path,
    rounds: int,
    alpha: float,
    seed: int,
    out_dir: Path,
    tag: str,
) -> Dict:
    cmd = [
        python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        "hybrid_opt",
        "--config",
        str(cfg_path),
        "--wireless-model",
        "simulated",
        "--simulated-mode",
        "jitter",
        "--num-rounds",
        str(rounds),
        "--alpha",
        str(alpha),
        "--algorithm",
        "fedprox",
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) if not env.get("PYTHONPATH") else f"{Path.cwd()}:{env['PYTHONPATH']}"
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), env=env)
    (out_dir / f"{tag}.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{tag}.err").write_text(proc.stderr, encoding="utf-8")

    rec = {
        "tag": tag,
        "exit_code": proc.returncode,
        "metrics_csv": "",
        "round": 0,
        "accuracy": 0.0,
        "loss": 0.0,
        "energy_sum": 0.0,
        "upload_time_sum": 0.0,
        "mean_jain": 0.0,
    }
    p = parse_metrics(proc.stdout)
    if proc.returncode == 0 and p is not None:
        csv_path = (Path.cwd() / p).resolve()
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
    run_tag = args.run_tag or now_tag()
    out_dir = Path("outputs/fl_comp") / run_tag / "C_tune_hybrid_jitter_invtrue"
    cfg_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base = read_yaml(Path(args.base_config))
    base = apply_common(base, args.lr, args.fedprox_mu)

    profiles = [
        "p1_async_sensitive_relaxed_inv",
        "p2_balanced_bridge",
        "p3_stability_fairness",
    ]

    short_rows: List[Dict] = []
    for p in profiles:
        cfg = apply_profile(copy.deepcopy(base), p)
        cfg_path = cfg_dir / f"{p}.yaml"
        write_yaml(cfg_path, cfg)
        rec = run_job(args.python_exe, cfg_path, args.short_rounds, args.alpha, args.seed, out_dir, f"{p}_short")
        rec["profile"] = p
        short_rows.append(rec)
        print(f"[SHORT] {p}: acc={rec['accuracy']:.6f} loss={rec['loss']:.6f} code={rec['exit_code']}")

    short_ok = [r for r in short_rows if r["exit_code"] == 0 and r["round"] > 0]
    if not short_ok:
        raise RuntimeError("No successful short profile in tuning.")
    short_ok.sort(key=lambda r: (r["accuracy"], -r["loss"]), reverse=True)
    best_profile = short_ok[0]["profile"]
    print(f"[BEST_SHORT] {best_profile}")

    best_cfg = cfg_dir / f"{best_profile}.yaml"
    final_rec = run_job(
        args.python_exe,
        best_cfg,
        args.final_rounds,
        args.alpha,
        args.seed,
        out_dir,
        f"{best_profile}_final",
    )
    final_rec["profile"] = best_profile
    print(
        f"[FINAL] {best_profile}: acc={final_rec['accuracy']:.6f} "
        f"loss={final_rec['loss']:.6f} rounds={final_rec['round']}"
    )

    summary_path = out_dir / "tune_summary.csv"
    fields = list(short_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(short_rows + [final_rec])
    print(f"[OUT] {summary_path}")


if __name__ == "__main__":
    main()
