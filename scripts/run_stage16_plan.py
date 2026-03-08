from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


METRICS_RE = re.compile(r"Metrics CSV:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TODO section 16 experiments (A/B/C stages).")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--run-tag", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--wsn-lr", type=float, default=0.0005)
    p.add_argument("--jitter-lr", type=float, default=0.0005)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--base-energy", type=float, default=2000.0)
    p.add_argument("--max-rounds", type=int, default=340)
    p.add_argument("--round-step", type=int, default=40)
    p.add_argument("--start-rounds", type=int, default=100)
    p.add_argument("--job-timeout-sec", type=int, default=0)
    p.add_argument("--skip-stage-a", action="store_true")
    p.add_argument("--fixed-rounds", type=int, default=0)
    p.add_argument("--fixed-energy", type=float, default=0.0)
    p.add_argument("--run-stage-b", action="store_true")
    p.add_argument("--run-stage-c", action="store_true")
    return p.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def parse_metrics_path(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def read_metrics_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_last_metrics(csv_path: Path) -> Dict[str, float]:
    rows = read_metrics_rows(csv_path)
    if not rows:
        return {"round": 0.0, "accuracy": 0.0, "loss": 0.0, "num_exhausted": 0.0}
    last = rows[-1]
    exhausted = str(last.get("exhausted_clients", "")).strip()
    ex_cnt = 0.0 if not exhausted else float(len([x for x in exhausted.split("|") if x.strip()]))
    return {
        "round": float(last.get("round", 0.0)),
        "accuracy": float(last.get("accuracy", 0.0)),
        "loss": float(last.get("loss", 0.0)),
        "num_exhausted": ex_cnt,
    }


def convergence_stats(csv_path: Path) -> Dict[str, float]:
    rows = read_metrics_rows(csv_path)
    if len(rows) < 25:
        return {"slope": 1.0, "std": 1.0, "delta_last5": -1.0, "converged": 0.0}
    acc = np.asarray([float(r.get("accuracy", 0.0)) for r in rows], dtype=np.float64)
    x = np.arange(len(acc), dtype=np.float64)
    tail = acc[-20:]
    x_tail = x[-20:]
    slope, _ = np.polyfit(x_tail, tail, 1)
    std = float(np.std(tail))
    m_last = float(np.mean(acc[-5:]))
    m_prev = float(np.mean(acc[-10:-5]))
    delta = m_last - m_prev
    converged = (abs(float(slope)) < 8e-4) and (std < 0.006) and (delta >= -0.002)
    return {"slope": float(slope), "std": std, "delta_last5": delta, "converged": 1.0 if converged else 0.0}


def run_one(
    *,
    project_root: Path,
    python_exe: str,
    out_dir: Path,
    tag: str,
    strategy: str,
    config: Path,
    wireless_model: str,
    simulated_mode: str,
    rounds: int,
    alpha: float,
    algorithm: str,
    lr: float,
    mu: float,
    seed: int,
    timeout_sec: int,
) -> Dict[str, str]:
    job_dir = out_dir / tag
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / f"{tag}.log"
    err_path = job_dir / f"{tag}.err"
    cmd = [
        python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        strategy,
        "--config",
        str(config),
        "--wireless-model",
        wireless_model,
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
    if simulated_mode:
        cmd.extend(["--simulated-mode", simulated_mode])
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
    metrics_abs = ""
    copied_csv = ""
    status = "failed"
    if metrics_rel is not None:
        mpath = (project_root / metrics_rel).resolve()
        metrics_abs = str(mpath)
        if mpath.exists():
            copied = job_dir / f"{tag}_{mpath.name}"
            shutil.copy2(mpath, copied)
            copied_csv = str(copied.resolve())
            status = "success"
    return {
        "tag": tag,
        "status": status,
        "log": str(log_path.resolve()),
        "err": str(err_path.resolve()),
        "metrics_csv": metrics_abs,
        "copied_csv": copied_csv,
        "note": f"exit={proc.returncode}; timeout={timed_out}",
    }


def append_csv(path: Path, row: List[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def ensure_csv(path: Path, header: List[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def scale_energy_vector(vec: List[float], ratio: float) -> List[float]:
    return [float(v) * ratio for v in vec]


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or now_tag()
    project_root = Path.cwd().resolve()
    root = project_root / "outputs" / "fl_comp" / run_tag / "stage16_plan"
    cfg_dir = root / "configs"
    analysis_dir = root / "analysis"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ledger = root / "attempt_ledger.csv"
    ensure_csv(ledger, ["timestamp", "stage", "tag", "status", "log", "err", "metrics_csv", "copied_csv", "note"])
    conv_csv = analysis_dir / "wsn_convergence.csv"
    ensure_csv(conv_csv, ["num_rounds", "initial_energy", "final_round", "final_acc", "final_loss", "slope", "std", "delta_last5", "converged", "copied_csv"])
    grid_csv = analysis_dir / "jitter_grid_summary.csv"
    ensure_csv(grid_csv, ["tag", "start_state", "period", "inv", "status", "final_round", "final_acc", "final_loss", "num_exhausted", "copied_csv"])
    drop_csv = analysis_dir / "dropout_stress_summary.csv"
    ensure_csv(drop_csv, ["tag", "env", "strategy", "status", "final_round", "final_acc", "final_loss", "num_exhausted", "copied_csv"])

    base_cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
    num_clients = int(base_cfg.get("fl", {}).get("num_clients", 10))

    run_stage_b = args.run_stage_b or (not args.run_stage_b and not args.run_stage_c)
    run_stage_c = args.run_stage_c or (not args.run_stage_b and not args.run_stage_c)

    converged_rounds = args.fixed_rounds if args.fixed_rounds > 0 else args.start_rounds
    converged_energy = args.fixed_energy if args.fixed_energy > 0 else args.base_energy * (converged_rounds / 60.0)
    converged_csv = ""

    # Stage A: WSN convergence sweep (hybrid invTrue)
    if not args.skip_stage_a:
        for rounds in range(args.start_rounds, args.max_rounds + 1, args.round_step):
            ratio = rounds / 60.0
            init_energy = args.base_energy * ratio
            cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
            cfg.setdefault("fl", {})["num_rounds"] = int(rounds)
            cfg["fl"]["lr"] = float(args.wsn_lr)
            cfg.setdefault("wireless", {})["wireless_model"] = "wsn"
            cfg["wireless"]["simulated_mode"] = "good"
            cfg.setdefault("algorithm", {})["name"] = "fedprox"
            cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu)
            cfg.setdefault("energy", {})["initial_client_energy"] = float(init_energy)
            cfg["energy"]["client_initial_energies"] = None
            cfg.setdefault("controller", {}).setdefault("bridge_invariants", {})["enable"] = True
            path = cfg_dir / f"stageA_wsn_hybrid_invTrue_r{rounds}.yaml"
            write_yaml(path, cfg)

            tag = f"A_wsn_hybrid_invTrue_r{rounds}"
            res = run_one(
                project_root=project_root,
                python_exe=args.python_exe,
                out_dir=root / "stageA",
                tag=tag,
                strategy="hybrid_opt",
                config=path,
                wireless_model="wsn",
                simulated_mode="",
                rounds=rounds,
                alpha=args.alpha,
                algorithm="fedprox",
                lr=args.wsn_lr,
                mu=args.fedprox_mu,
                seed=args.seed,
                timeout_sec=args.job_timeout_sec,
            )
            append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "A", tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
            if res["status"] != "success":
                append_csv(conv_csv, [str(rounds), f"{init_energy:.4f}", "0", "0", "0", "1", "1", "-1", "0", ""])
                continue
            m = read_last_metrics(Path(res["copied_csv"]))
            c = convergence_stats(Path(res["copied_csv"]))
            append_csv(
                conv_csv,
                [
                    str(rounds),
                    f"{init_energy:.4f}",
                    f"{m['round']:.1f}",
                    f"{m['accuracy']:.6f}",
                    f"{m['loss']:.6f}",
                    f"{c['slope']:.8f}",
                    f"{c['std']:.8f}",
                    f"{c['delta_last5']:.8f}",
                    str(int(c["converged"])),
                    res["copied_csv"],
                ],
            )
            if int(c["converged"]) == 1:
                converged_rounds = rounds
                converged_energy = init_energy
                converged_csv = res["copied_csv"]
                break
            converged_rounds = rounds
            converged_energy = init_energy
            converged_csv = res["copied_csv"]

    # Stage B: jitter grid
    if run_stage_b:
        periods = sorted(set([20, max(1, int(math.floor(converged_rounds * 0.5))), max(1, int(math.floor(converged_rounds * 0.8)))]))
        for start_state in ["good", "bad"]:
            for period in periods:
                for inv in ["true", "false"]:
                    cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
                    cfg.setdefault("fl", {})["num_rounds"] = int(converged_rounds)
                    cfg["fl"]["lr"] = float(args.jitter_lr)
                    cfg.setdefault("wireless", {})["wireless_model"] = "simulated"
                    cfg["wireless"]["simulated_mode"] = "jitter"
                    cfg["wireless"]["jitter_start_state"] = start_state
                    cfg["wireless"]["jitter_period_rounds"] = int(period)
                    cfg.setdefault("algorithm", {})["name"] = "fedprox"
                    cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu)
                    cfg.setdefault("energy", {})["initial_client_energy"] = float(converged_energy)
                    cfg["energy"]["client_initial_energies"] = None
                    cfg.setdefault("controller", {}).setdefault("bridge_invariants", {})["enable"] = inv == "true"
                    cfg_path = cfg_dir / f"stageB_jitter_{start_state}_p{period}_inv{inv}.yaml"
                    write_yaml(cfg_path, cfg)
                    tag = f"B_jitter_{start_state}_p{period}_inv{inv}"
                    res = run_one(
                        project_root=project_root,
                        python_exe=args.python_exe,
                        out_dir=root / "stageB",
                        tag=tag,
                        strategy="hybrid_opt",
                        config=cfg_path,
                        wireless_model="simulated",
                        simulated_mode="jitter",
                        rounds=converged_rounds,
                        alpha=args.alpha,
                        algorithm="fedprox",
                        lr=args.jitter_lr,
                        mu=args.fedprox_mu,
                        seed=args.seed,
                        timeout_sec=args.job_timeout_sec,
                    )
                    append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "B", tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
                    if res["status"] == "success":
                        m = read_last_metrics(Path(res["copied_csv"]))
                        append_csv(grid_csv, [tag, start_state, str(period), inv, "success", f"{m['round']:.1f}", f"{m['accuracy']:.6f}", f"{m['loss']:.6f}", f"{m['num_exhausted']:.1f}", res["copied_csv"]])
                    else:
                        append_csv(grid_csv, [tag, start_state, str(period), inv, "failed", "0", "0", "0", "0", ""])

    # Stage C: dropout stress comparison
    if run_stage_c:
        base_vec = [2600.0, 2400.0, 2200.0, 2000.0, 1800.0, 1400.0, 1200.0, 1000.0, 900.0, 800.0]
        if len(base_vec) < num_clients:
            base_vec.extend([base_vec[-1]] * (num_clients - len(base_vec)))
        base_vec = base_vec[:num_clients]
        vec = scale_energy_vector(base_vec, converged_rounds / 60.0)
        stress_cases: List[Tuple[str, str, str]] = [
            ("wsn_stress", "wsn", ""),
            ("jitter_stress", "simulated", "jitter"),
        ]
        for env_name, wireless_model, sim_mode in stress_cases:
            for strategy, algo in [("hybrid_opt", "fedprox"), ("energy_first", "fedavg"), ("bandwidth_first", "fedavg")]:
                cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
                cfg.setdefault("fl", {})["num_rounds"] = int(converged_rounds)
                cfg["fl"]["lr"] = float(args.jitter_lr if wireless_model == "simulated" else args.wsn_lr)
                cfg.setdefault("wireless", {})["wireless_model"] = wireless_model
                if wireless_model == "simulated":
                    cfg["wireless"]["simulated_mode"] = "jitter"
                    cfg["wireless"]["jitter_start_state"] = "bad"
                    cfg["wireless"]["jitter_period_rounds"] = 20
                cfg.setdefault("algorithm", {})["name"] = algo
                cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu if algo == "fedprox" else 0.0)
                cfg.setdefault("energy", {})["initial_client_energy"] = float(converged_energy)
                cfg["energy"]["client_initial_energies"] = vec
                cfg["energy"]["min_energy_floor_per_round"] = max(40.0, converged_energy * 0.02)
                cfg.setdefault("scheduler", {}).setdefault("bandwidth_first_penalty", {})["enable"] = strategy == "bandwidth_first"
                if strategy == "bandwidth_first":
                    cfg["scheduler"]["bandwidth_first_penalty"]["streak_threshold"] = 2
                    cfg["scheduler"]["bandwidth_first_penalty"]["energy_multiplier"] = 1.9
                    cfg["scheduler"]["bandwidth_first_penalty"]["cooldown_rounds"] = 2
                cfg_path = cfg_dir / f"stageC_{env_name}_{strategy}.yaml"
                write_yaml(cfg_path, cfg)
                tag = f"C_{env_name}_{strategy}"
                res = run_one(
                    project_root=project_root,
                    python_exe=args.python_exe,
                    out_dir=root / "stageC",
                    tag=tag,
                    strategy=strategy,
                    config=cfg_path,
                    wireless_model=wireless_model,
                    simulated_mode=sim_mode,
                    rounds=converged_rounds,
                    alpha=args.alpha,
                    algorithm=algo,
                    lr=cfg["fl"]["lr"],
                    mu=args.fedprox_mu if algo == "fedprox" else 0.0,
                    seed=args.seed,
                    timeout_sec=args.job_timeout_sec,
                )
                append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "C", tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
                if res["status"] == "success":
                    m = read_last_metrics(Path(res["copied_csv"]))
                    append_csv(drop_csv, [tag, env_name, strategy, "success", f"{m['round']:.1f}", f"{m['accuracy']:.6f}", f"{m['loss']:.6f}", f"{m['num_exhausted']:.1f}", res["copied_csv"]])
                else:
                    append_csv(drop_csv, [tag, env_name, strategy, "failed", "0", "0", "0", "0", ""])

    summary = root / "analysis" / "stage16_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"run_tag={run_tag}\n")
        f.write(f"converged_rounds={converged_rounds}\n")
        f.write(f"converged_energy={converged_energy:.6f}\n")
        f.write(f"converged_csv={converged_csv}\n")
        f.write(f"wsn_convergence_csv={conv_csv}\n")
        f.write(f"jitter_grid_csv={grid_csv}\n")
        f.write(f"dropout_stress_csv={drop_csv}\n")
    print(f"[DONE] stage16 root={root}")
    print(f"[DONE] converged_rounds={converged_rounds}")
    print(f"[DONE] converged_energy={converged_energy:.6f}")


if __name__ == "__main__":
    main()
