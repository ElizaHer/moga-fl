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
    p = argparse.ArgumentParser(description="Run TODO section 17 follow-up experiments.")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--run-tag", default="")
    p.add_argument("--num-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--wsn-lr", type=float, default=0.0005)
    p.add_argument("--jitter-lr", type=float, default=0.0005)
    p.add_argument("--base-energy", type=float, default=6666.6667)
    p.add_argument("--job-timeout-sec", type=int, default=0)
    p.add_argument("--only-matrix", action="store_true")
    return p.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def ensure_csv(path: Path, header: List[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv(path: Path, row: List[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def parse_metrics_path(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        m = METRICS_RE.search(line)
        if m:
            return Path(m.group(1).strip().replace("\\", "/"))
    return None


def read_last_metrics(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {"round": 0.0, "accuracy": 0.0, "loss": 0.0, "num_exhausted": 0.0}
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
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


def scale_vec(vec: List[float], scale: float) -> List[float]:
    return [float(v) * scale for v in vec]


def run_one(
    *,
    project_root: Path,
    python_exe: str,
    out_dir: Path,
    tag: str,
    strategy: str,
    config_path: Path,
    wireless_model: str,
    sim_mode: str,
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
    if sim_mode:
        cmd.extend(["--simulated-mode", sim_mode])
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
    copied_csv = ""
    metrics_abs = ""
    status = "failed"
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


def classify_pressure(stagec_rows: List[Dict[str, str]], num_clients: int) -> str:
    succ = [r for r in stagec_rows if r["status"] == "success"]
    if not succ:
        return "too_strong"
    exhausted_vals = [float(r["num_exhausted"]) for r in succ]
    acc_vals = [float(r["final_acc"]) for r in succ]
    avg_ex = sum(exhausted_vals) / len(exhausted_vals)
    avg_acc = sum(acc_vals) / len(acc_vals)
    if avg_ex < 0.5:
        return "too_weak"
    if avg_ex > max(4.0, 0.7 * num_clients) or avg_acc < 0.45:
        return "too_strong"
    return "controllable"


def hybrid_wins(stagec_rows: List[Dict[str, str]]) -> bool:
    envs = sorted({r["env"] for r in stagec_rows if r["status"] == "success"})
    if not envs:
        return False
    win_cells = 0
    for env in envs:
        rows = [r for r in stagec_rows if r["env"] == env and r["status"] == "success"]
        h = [float(r["final_acc"]) for r in rows if r["strategy"] == "hybrid_opt"]
        o = [float(r["final_acc"]) for r in rows if r["strategy"] != "hybrid_opt"]
        if h and o and h[0] > max(o):
            win_cells += 1
    return win_cells >= max(1, len(envs))


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or now_tag()
    project_root = Path.cwd().resolve()
    root = project_root / "outputs" / "fl_comp" / run_tag / "stage17_followup"
    cfg_dir = root / "configs"
    analysis_dir = root / "analysis"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ledger = root / "attempt_ledger.csv"
    ensure_csv(ledger, ["timestamp", "stage", "tag", "status", "log", "err", "metrics_csv", "copied_csv", "note"])
    stagec_csv = analysis_dir / "dropout_stress_summary.csv"
    ensure_csv(stagec_csv, ["loop_id", "tag", "env", "strategy", "status", "final_round", "final_acc", "final_loss", "num_exhausted", "copied_csv"])
    pressure_csv = analysis_dir / "dropout_pressure_diagnosis.csv"
    ensure_csv(pressure_csv, ["loop_id", "energy_scale", "floor_ratio", "pressure_class", "hybrid_win", "decision"])
    matrix_csv = analysis_dir / "env_matrix_200_summary.csv"
    ensure_csv(matrix_csv, ["tag", "env", "strategy", "status", "final_round", "final_acc", "final_loss", "num_exhausted", "copied_csv"])

    base_cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
    num_clients = int(base_cfg.get("fl", {}).get("num_clients", 10))
    base_vec = [2600.0, 2400.0, 2200.0, 2000.0, 1800.0, 1400.0, 1200.0, 1000.0, 900.0, 800.0]
    if len(base_vec) < num_clients:
        base_vec.extend([base_vec[-1]] * (num_clients - len(base_vec)))
    base_vec = base_vec[:num_clients]

    if not args.only_matrix:
        # 17.1 adaptive Stage-C loop
        profiles = [
            {"scale": 1.00, "floor_ratio": 0.010},
            {"scale": 0.75, "floor_ratio": 0.006},
            {"scale": 0.60, "floor_ratio": 0.004},
            {"scale": 0.50, "floor_ratio": 0.002},
        ]
        final_stagec_rows: List[Dict[str, str]] = []
        for idx, pf in enumerate(profiles, start=1):
            loop_rows: List[Dict[str, str]] = []
            vec = scale_vec(base_vec, pf["scale"] * (args.num_rounds / 60.0))
            init_energy = args.base_energy * pf["scale"]
            min_floor = max(5.0, init_energy * pf["floor_ratio"])
            for env_name, wm, sm in [("wsn_stress", "wsn", ""), ("jitter_stress", "simulated", "jitter")]:
                for strategy, algo in [("hybrid_opt", "fedprox"), ("energy_first", "fedavg"), ("bandwidth_first", "fedavg")]:
                    cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
                    cfg.setdefault("fl", {})["num_rounds"] = int(args.num_rounds)
                    cfg["fl"]["lr"] = float(args.jitter_lr if wm == "simulated" else args.wsn_lr)
                    cfg.setdefault("wireless", {})["wireless_model"] = wm
                    if wm == "simulated":
                        cfg["wireless"]["simulated_mode"] = "jitter"
                        cfg["wireless"]["jitter_start_state"] = "bad"
                        cfg["wireless"]["jitter_period_rounds"] = 20
                    cfg.setdefault("algorithm", {})["name"] = algo
                    cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu if algo == "fedprox" else 0.0)
                    cfg.setdefault("energy", {})["initial_client_energy"] = float(init_energy)
                    cfg["energy"]["client_initial_energies"] = vec
                    cfg["energy"]["min_energy_floor_per_round"] = float(min_floor)
                    cfg.setdefault("scheduler", {}).setdefault("bandwidth_first_penalty", {})["enable"] = strategy == "bandwidth_first"
                    if strategy == "bandwidth_first":
                        cfg["scheduler"]["bandwidth_first_penalty"]["streak_threshold"] = 2
                        cfg["scheduler"]["bandwidth_first_penalty"]["energy_multiplier"] = 1.9
                        cfg["scheduler"]["bandwidth_first_penalty"]["cooldown_rounds"] = 2
                    cfg_path = cfg_dir / f"stageC_loop{idx}_{env_name}_{strategy}.yaml"
                    write_yaml(cfg_path, cfg)
                    tag = f"17C_loop{idx}_{env_name}_{strategy}"
                    res = run_one(
                        project_root=project_root,
                        python_exe=args.python_exe,
                        out_dir=root / "stageC",
                        tag=tag,
                        strategy=strategy,
                        config_path=cfg_path,
                        wireless_model=wm,
                        sim_mode=sm,
                        rounds=args.num_rounds,
                        alpha=args.alpha,
                        algorithm=algo,
                        lr=cfg["fl"]["lr"],
                        mu=args.fedprox_mu if algo == "fedprox" else 0.0,
                        seed=args.seed,
                        timeout_sec=args.job_timeout_sec,
                    )
                    append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "17.1", tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
                    if res["status"] == "success":
                        m = read_last_metrics(Path(res["copied_csv"]))
                        row = {
                            "loop_id": str(idx),
                            "tag": tag,
                            "env": env_name,
                            "strategy": strategy,
                            "status": "success",
                            "final_round": f"{m['round']:.1f}",
                            "final_acc": f"{m['accuracy']:.6f}",
                            "final_loss": f"{m['loss']:.6f}",
                            "num_exhausted": f"{m['num_exhausted']:.1f}",
                            "copied_csv": res["copied_csv"],
                        }
                    else:
                        row = {
                            "loop_id": str(idx),
                            "tag": tag,
                            "env": env_name,
                            "strategy": strategy,
                            "status": "failed",
                            "final_round": "0",
                            "final_acc": "0",
                            "final_loss": "0",
                            "num_exhausted": "0",
                            "copied_csv": "",
                        }
                    loop_rows.append(row)
                    append_csv(
                        stagec_csv,
                        [
                            row["loop_id"],
                            row["tag"],
                            row["env"],
                            row["strategy"],
                            row["status"],
                            row["final_round"],
                            row["final_acc"],
                            row["final_loss"],
                            row["num_exhausted"],
                            row["copied_csv"],
                        ],
                    )
            pclass = classify_pressure(loop_rows, num_clients=num_clients)
            hwin = hybrid_wins(loop_rows)
            decision = "rerun"
            if pclass == "controllable" and not hwin:
                decision = "stop_controllable_hybrid_bad"
                final_stagec_rows = loop_rows
                append_csv(pressure_csv, [str(idx), f"{pf['scale']:.4f}", f"{pf['floor_ratio']:.4f}", pclass, str(int(hwin)), decision])
                break
            if pclass == "controllable" and hwin:
                decision = "stop_controllable_hybrid_good"
                final_stagec_rows = loop_rows
                append_csv(pressure_csv, [str(idx), f"{pf['scale']:.4f}", f"{pf['floor_ratio']:.4f}", pclass, str(int(hwin)), decision])
                break
            append_csv(pressure_csv, [str(idx), f"{pf['scale']:.4f}", f"{pf['floor_ratio']:.4f}", pclass, str(int(hwin)), decision])
            final_stagec_rows = loop_rows

    # 17.2 fixed 200-round env matrix
    envs = [
        ("wsn", "wsn", "", ""),
        ("jitter_good_p20", "simulated", "jitter", ("good", 20)),
        ("jitter_good_p100", "simulated", "jitter", ("good", 100)),
        ("jitter_good_p160", "simulated", "jitter", ("good", 160)),
        ("jitter_bad_p20", "simulated", "jitter", ("bad", 20)),
        ("jitter_bad_p100", "simulated", "jitter", ("bad", 100)),
        ("jitter_bad_p160", "simulated", "jitter", ("bad", 160)),
    ]
    for env_name, wm, sm, extra in envs:
        for strategy, algo in [("hybrid_opt", "fedprox"), ("energy_first", "fedavg"), ("bandwidth_first", "fedavg")]:
            cfg = read_yaml(project_root / "src" / "configs" / "strategies" / "hybrid_opt.yaml")
            cfg.setdefault("fl", {})["num_rounds"] = int(args.num_rounds)
            cfg["fl"]["lr"] = float(args.jitter_lr if wm == "simulated" else args.wsn_lr)
            cfg.setdefault("wireless", {})["wireless_model"] = wm
            if wm == "simulated":
                cfg["wireless"]["simulated_mode"] = "jitter"
                start_state, period = extra  # type: ignore[misc]
                cfg["wireless"]["jitter_start_state"] = start_state
                cfg["wireless"]["jitter_period_rounds"] = int(period)
            cfg.setdefault("algorithm", {})["name"] = algo
            cfg["algorithm"]["fedprox_mu"] = float(args.fedprox_mu if algo == "fedprox" else 0.0)
            cfg.setdefault("energy", {})["initial_client_energy"] = float(args.base_energy)
            cfg["energy"]["client_initial_energies"] = None
            cfg_path = cfg_dir / f"stageM_{env_name}_{strategy}.yaml"
            write_yaml(cfg_path, cfg)
            tag = f"17M_{env_name}_{strategy}"
            res = run_one(
                project_root=project_root,
                python_exe=args.python_exe,
                out_dir=root / "matrix200",
                tag=tag,
                strategy=strategy,
                config_path=cfg_path,
                wireless_model=wm,
                sim_mode=sm,
                rounds=args.num_rounds,
                alpha=args.alpha,
                algorithm=algo,
                lr=cfg["fl"]["lr"],
                mu=args.fedprox_mu if algo == "fedprox" else 0.0,
                seed=args.seed,
                timeout_sec=args.job_timeout_sec,
            )
            append_csv(ledger, [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "17.2", tag, res["status"], res["log"], res["err"], res["metrics_csv"], res["copied_csv"], res["note"]])
            if res["status"] == "success":
                m = read_last_metrics(Path(res["copied_csv"]))
                append_csv(matrix_csv, [tag, env_name, strategy, "success", f"{m['round']:.1f}", f"{m['accuracy']:.6f}", f"{m['loss']:.6f}", f"{m['num_exhausted']:.1f}", res["copied_csv"]])
            else:
                append_csv(matrix_csv, [tag, env_name, strategy, "failed", "0", "0", "0", "0", ""])

    summary = analysis_dir / "stage17_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"run_tag={run_tag}\n")
        f.write(f"num_rounds={args.num_rounds}\n")
        f.write(f"base_energy={args.base_energy}\n")
        f.write(f"dropout_stress_csv={stagec_csv}\n")
        f.write(f"dropout_pressure_csv={pressure_csv}\n")
        f.write(f"env_matrix_200_csv={matrix_csv}\n")
    print(f"[DONE] stage17 root={root}")


if __name__ == "__main__":
    main()
