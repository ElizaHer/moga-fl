from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


METRICS_RE = re.compile(r"Metrics CSV:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run additional sync/async/bridge_free jitter matrix and compare with prior hybrid/bandwidth_first.")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--run-tag", default="")
    p.add_argument("--num-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--jitter-lr", type=float, default=0.0005)
    p.add_argument("--base-energy", type=float, default=6666.6667)
    p.add_argument("--prior-summary", default="/root/autodl-tmp/moga-fl/outputs/fl_comp/20260307_stage17_matrixonly/stage17_followup/analysis/env_matrix_200_summary.csv")
    p.add_argument("--job-timeout-sec", type=int, default=0)
    return p.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_csv(path: Path, header: List[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv(path: Path, row: List[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def parse_metrics_path(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def read_last_metrics(csv_path: Path) -> Dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8"))) if csv_path.exists() else []
    if not rows:
        return {"round": 0.0, "accuracy": 0.0, "loss": 0.0, "num_exhausted": 0.0}
    last = rows[-1]
    ex = str(last.get("exhausted_clients", "")).strip()
    ex_cnt = 0.0 if not ex else float(len([x for x in ex.split("|") if x.strip()]))
    return {
        "round": float(last.get("round", 0.0)),
        "accuracy": float(last.get("accuracy", 0.0)),
        "loss": float(last.get("loss", 0.0)),
        "num_exhausted": ex_cnt,
    }


def run_one(
    *,
    project_root: Path,
    python_exe: str,
    out_dir: Path,
    tag: str,
    strategy: str,
    config_path: Path,
    rounds: int,
    alpha: float,
    algorithm: str,
    lr: float,
    mu: float,
    seed: int,
    timeout_sec: int,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{tag}.log"
    err_path = out_dir / f"{tag}.err"
    cmd = [
        python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        strategy,
        "--config",
        str(config_path),
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
        "--lr",
        str(lr),
    ]
    if algorithm == "fedprox":
        cmd.extend(["--fedprox-mu", str(mu)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) if not env.get("PYTHONPATH") else f"{project_root}:{env['PYTHONPATH']}"
    with log_path.open("w", encoding="utf-8") as logf, err_path.open("w", encoding="utf-8") as errf:
        proc = subprocess.Popen(cmd, cwd=project_root, stdout=logf, stderr=errf, env=env)
        try:
            proc.wait(timeout=timeout_sec if timeout_sec > 0 else None)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

    metrics_rel = parse_metrics_path(log_path)
    status = "failed"
    metrics_abs = ""
    copied_csv = ""
    if metrics_rel is not None:
        abs_path = (project_root / metrics_rel).resolve()
        metrics_abs = str(abs_path)
        if abs_path.exists():
            copied = out_dir / f"{tag}_{abs_path.name}"
            shutil.copy2(abs_path, copied)
            copied_csv = str(copied.resolve())
            status = "success"
    return {
        "status": status,
        "log": str(log_path.resolve()),
        "err": str(err_path.resolve()),
        "metrics_csv": metrics_abs,
        "copied_csv": copied_csv,
        "note": f"exit={proc.returncode}; timeout={timed_out}",
    }


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or now_tag()
    project_root = Path.cwd().resolve()
    root = project_root / "outputs" / "fl_comp" / run_tag / "stage17_additional"
    cfg_dir = root / "configs"
    analysis_dir = root / "analysis"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ledger = root / "attempt_ledger.csv"
    ensure_csv(ledger, ["timestamp", "tag", "status", "log", "err", "metrics_csv", "copied_csv", "note"])
    add_csv = analysis_dir / "additional_sync_async_summary.csv"
    ensure_csv(add_csv, ["tag", "env", "strategy", "status", "final_round", "final_acc", "final_loss", "num_exhausted", "copied_csv"])
    cmp_csv = analysis_dir / "comparison_with_hybrid_bandwidth.csv"
    ensure_csv(cmp_csv, ["env", "strategy", "acc", "loss", "num_exhausted", "delta_vs_hybrid", "delta_vs_bandwidth_first", "source"])

    cases: List[Tuple[str, int]] = [("good", 20), ("good", 100), ("bad", 20), ("bad", 100)]
    strategy_algo = [("sync", "fedprox"), ("async", "fedprox"), ("bridge_free", "fedprox")]

    for start_state, period in cases:
        for strategy, algo in strategy_algo:
            cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
            cfg.setdefault("fl", {})["num_rounds"] = int(args.num_rounds)
            cfg["fl"]["lr"] = float(args.jitter_lr)
            cfg.setdefault("wireless", {})["wireless_model"] = "simulated"
            cfg["wireless"]["simulated_mode"] = "jitter"
            cfg["wireless"]["jitter_start_state"] = start_state
            cfg["wireless"]["jitter_period_rounds"] = int(period)
            cfg.setdefault("algorithm", {})["name"] = algo
            cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu)
            cfg.setdefault("energy", {})["initial_client_energy"] = float(args.base_energy)
            cfg["energy"]["client_initial_energies"] = None
            cfg_path = cfg_dir / f"cfg_{strategy}_{start_state}_p{period}.yaml"
            write_yaml(cfg_path, cfg)
            env_key = f"jitter_{start_state}_p{period}"
            tag = f"add_{strategy}_{env_key}"
            res = run_one(
                project_root=project_root,
                python_exe=args.python_exe,
                out_dir=root / "runs",
                tag=tag,
                strategy=strategy,
                config_path=cfg_path,
                rounds=args.num_rounds,
                alpha=args.alpha,
                algorithm=algo,
                lr=args.jitter_lr,
                mu=args.fedprox_mu,
                seed=args.seed,
                timeout_sec=args.job_timeout_sec,
            )
            append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
            if res["status"] == "success":
                m = read_last_metrics(Path(res["copied_csv"]))
                append_csv(add_csv, [tag, env_key, strategy, "success", f"{m['round']:.1f}", f"{m['accuracy']:.6f}", f"{m['loss']:.6f}", f"{m['num_exhausted']:.1f}", res["copied_csv"]])
            else:
                append_csv(add_csv, [tag, env_key, strategy, "failed", "0", "0", "0", "0", ""])

    # comparison with prior hybrid + bandwidth_first
    prior_rows = list(csv.DictReader(Path(args.prior_summary).open("r", encoding="utf-8")))
    prior_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in prior_rows:
        prior_map[(r["env"], r["strategy"])] = r

    current_rows = list(csv.DictReader(add_csv.open("r", encoding="utf-8")))
    for r in current_rows:
        if r["status"] != "success":
            continue
        env = r["env"]
        hybrid = prior_map.get((env, "hybrid_opt"))
        bw = prior_map.get((env, "bandwidth_first"))
        acc = float(r["final_acc"])
        delta_h = ""
        delta_b = ""
        if hybrid and hybrid.get("status") == "success":
            delta_h = f"{acc - float(hybrid['final_acc']):.6f}"
        if bw and bw.get("status") == "success":
            delta_b = f"{acc - float(bw['final_acc']):.6f}"
        append_csv(cmp_csv, [env, r["strategy"], f"{acc:.6f}", r["final_loss"], r["num_exhausted"], delta_h, delta_b, "additional"])

    # also write prior hybrid + bw rows for easy side-by-side
    env_keys = sorted({r["env"] for r in current_rows})
    for env in env_keys:
        for s in ["hybrid_opt", "bandwidth_first"]:
            pr = prior_map.get((env, s))
            if not pr:
                continue
            append_csv(cmp_csv, [env, s, pr["final_acc"], pr["final_loss"], pr["num_exhausted"], "0.000000" if s == "hybrid_opt" else "", "0.000000" if s == "bandwidth_first" else "", "prior"])

    summary = analysis_dir / "stage17_additional_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"run_tag={run_tag}\n")
        f.write(f"additional_csv={add_csv}\n")
        f.write(f"comparison_csv={cmp_csv}\n")
    print(f"[DONE] root={root}")


if __name__ == "__main__":
    main()

