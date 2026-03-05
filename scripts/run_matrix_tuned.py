from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml


WECHAT_TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjkwMjkwNiwidXVpZCI6ImE5NDViOTgzNWQwOGY4YjUiLCJpc19hZG1pbiI6"
    "ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIs"
    "InVwayI6IiJ9.sw0UVJSDJ8CZ7O6cT9_OQOfWQKi6TwW9JQVB0FGp8SMQueBsN-0mEETOV857BNZBouh2yy-MwJQL7VRE1u0XSg"
)
METRICS_RE = re.compile(r"Metrics CSV:\s*(.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strict hybrid-vs-baselines matrix with retries and ledger.")
    p.add_argument("--run-tag", default="")
    p.add_argument("--run-name", default="B_matrix_tuned")
    p.add_argument("--python-exe", default="python")
    p.add_argument("--num-rounds", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--algorithm", default="fedprox")
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--wsn-lr", type=float, default=0.0005)
    p.add_argument("--wsn-fedprox-mu", type=float, default=0.01)
    p.add_argument("--jitter-lr", type=float, default=0.0005)
    p.add_argument("--jitter-fedprox-mu", type=float, default=0.05)
    p.add_argument("--jitter-start-state", default="bad", choices=["good", "bad"])
    p.add_argument("--jitter-period-rounds", type=int, default=20)
    p.add_argument("--jitter-profile", default="p2_balanced_bridge", choices=["none", "p1_async_sensitive_relaxed_inv", "p2_balanced_bridge", "p3_stability_fairness"])
    p.add_argument("--initial-client-energy", type=float, default=2000.0)
    p.add_argument("--client-initial-energies", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--job-timeout-sec", type=int, default=0)
    p.add_argument("--no-wechat", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def send_wechat(title: str, name: str, content: str, enabled: bool) -> None:
    if not enabled:
        return
    headers = {"Authorization": WECHAT_TOKEN}
    payload = {"title": title, "name": name, "content": content}
    try:
        requests.post("https://www.autodl.com/api/v1/wechat/message/send", json=payload, headers=headers, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] wechat failed: {exc}")


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)


def ensure_ledger(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "tag", "attempt", "status", "log", "err", "metrics_csv", "copied_csv", "note"])


def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "strategy", "wireless_model", "simulated_mode", "alpha", "algorithm", "seed", "fedprox_mu", "csv_path"])


def append_csv(path: Path, row: List[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def next_attempt(job_dir: Path, tag: str) -> int:
    max_attempt = 0
    for p in job_dir.glob(f"run_{tag}_attempt*.log"):
        m = re.search(r"_attempt(\d+)\.log$", p.name)
        if m:
            max_attempt = max(max_attempt, int(m.group(1)))
    for p in job_dir.glob(f"run_{tag}_attempt*.err"):
        m = re.search(r"_attempt(\d+)\.err$", p.name)
        if m:
            max_attempt = max(max_attempt, int(m.group(1)))
    return max_attempt + 1


def parse_metrics_path(log_path: Path) -> Optional[Path]:
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        m = METRICS_RE.search(line)
        if m:
            p = m.group(1).strip().replace("\\", "/")
            return Path(p)
    return None


def apply_jitter_profile(cfg: Dict, profile: str) -> Dict:
    if profile == "none":
        return cfg
    c = cfg
    c.setdefault("controller", {})
    c["controller"].setdefault("gate_thresholds", {})
    c["controller"].setdefault("gate_weights", {})
    c["controller"].setdefault("bridge_invariants", {})
    c.setdefault("fedbuff", {})
    if profile == "p1_async_sensitive_relaxed_inv":
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
    elif profile == "p2_balanced_bridge":
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
    elif profile == "p3_stability_fairness":
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
    return c


def build_configs(
    config_dir: Path,
    wsn_lr: float,
    wsn_mu: float,
    jitter_start_state: str,
    jitter_period_rounds: int,
    jitter_profile: str,
    initial_client_energy: float,
    client_initial_energies: str,
) -> Dict[str, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    base = read_yaml(Path("src/configs/strategies/hybrid_opt.yaml"))

    # Baseline config updates from pretest:
    # best for wsn+hybrid+inv_true pretest: lr=0.0005, fedprox_mu=0.01
    base.setdefault("fl", {})["local_epochs"] = 1
    base["fl"]["lr"] = float(wsn_lr)
    base.setdefault("algorithm", {})["fedprox_mu"] = float(wsn_mu)
    base.setdefault("fedbuff", {})["async_agg_interval"] = 2
    base.setdefault("wireless", {})["jitter_start_state"] = str(jitter_start_state)
    base["wireless"]["jitter_period_rounds"] = int(jitter_period_rounds)
    base = apply_jitter_profile(base, jitter_profile)
    base.setdefault("energy", {})["initial_client_energy"] = float(initial_client_energy)
    if client_initial_energies.strip():
        base["energy"]["client_initial_energies"] = [float(x.strip()) for x in client_initial_energies.split(",") if x.strip()]

    inv_true = config_dir / "hybrid_opt_inv_true.yaml"
    inv_false = config_dir / "hybrid_opt_inv_false.yaml"
    base.setdefault("controller", {}).setdefault("bridge_invariants", {})["enable"] = True
    write_yaml(inv_true, base)
    base["controller"]["bridge_invariants"]["enable"] = False
    write_yaml(inv_false, base)
    return {"inv_true": inv_true, "inv_false": inv_false}


def read_last_metrics(csv_path: Path) -> Dict[str, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return {"round": 0, "accuracy": 0.0, "loss": 0.0}
    last = rows[-1]
    return {
        "round": float(last.get("round", 0.0)),
        "accuracy": float(last.get("accuracy", 0.0)),
        "loss": float(last.get("loss", 0.0)),
    }


def run_attempt(
    job: Dict[str, str],
    attempt: int,
    args: argparse.Namespace,
    project_root: Path,
    ledger_path: Path,
    send_wechat_enabled: bool,
) -> Dict[str, str]:
    tag = job["tag"]
    job_dir = project_root / "outputs/fl_comp" / args.run_tag / args.run_name / tag
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / f"run_{tag}_attempt{attempt}.log"
    err_path = job_dir / f"run_{tag}_attempt{attempt}.err"
    cmd = [
        args.python_exe,
        "-u",
        "src/flower/hybrid_opt_demo.py",
        "--strategy",
        job["strategy"],
        "--config",
        job["config"],
        "--wireless-model",
        job["wireless_model"],
        "--num-rounds",
        str(args.num_rounds),
        "--alpha",
        str(args.alpha),
        "--algorithm",
        job["algorithm"],
        "--seed",
        str(args.seed),
        "--lr",
        str(job["lr"]),
    ]
    if job["algorithm"] == "fedprox":
        cmd.extend(["--fedprox-mu", str(job["fedprox_mu"])])
    if job["simulated_mode"]:
        cmd.extend(["--simulated-mode", job["simulated_mode"]])

    send_wechat(
        title="experiment_start",
        name=tag,
        content=(
            f"tag={tag} strategy={job['strategy']} wm={job['wireless_model']} sm={job['simulated_mode']} "
            f"alpha={args.alpha} algo={job['algorithm']} lr={job['lr']} mu={job['fedprox_mu']} seed={args.seed} rounds={args.num_rounds}"
        ),
        enabled=send_wechat_enabled,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) if not env.get("PYTHONPATH") else f"{project_root}:{env['PYTHONPATH']}"
    with log_path.open("w", encoding="utf-8") as logf, err_path.open("w", encoding="utf-8") as errf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=errf, cwd=project_root, env=env)
        try:
            proc.wait(timeout=args.job_timeout_sec if args.job_timeout_sec > 0 else None)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

    metrics_rel = parse_metrics_path(log_path)
    status = "failed"
    copied_csv = ""
    note = f"exit_code={proc.returncode}; timed_out={timed_out}"
    metrics_abs = ""
    if metrics_rel is not None:
        abs_path = (project_root / metrics_rel).resolve()
        metrics_abs = str(abs_path)
        if abs_path.exists():
            copied_name = f"{tag}_attempt{attempt}_{abs_path.name}"
            copied_path = job_dir / copied_name
            shutil.copy2(abs_path, copied_path)
            copied_csv = str(copied_path.resolve())
            status = "success"
            note += "; success by Metrics CSV line"
        else:
            note += "; Metrics CSV path missing"
    else:
        note += "; missing Metrics CSV line"

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_csv(
        ledger_path,
        [
            ts,
            tag,
            str(attempt),
            status,
            str(log_path.resolve()),
            str(err_path.resolve()),
            metrics_abs,
            copied_csv,
            note,
        ],
    )

    final_round = 0.0
    final_acc = 0.0
    final_loss = 0.0
    if copied_csv:
        m = read_last_metrics(Path(copied_csv))
        final_round = m["round"]
        final_acc = m["accuracy"]
        final_loss = m["loss"]
    send_wechat(
        title="experiment_end",
        name=tag,
        content=(
            f"tag={tag} status={status} rounds={final_round} acc={final_acc:.6f} loss={final_loss:.6f} "
            f"csv={copied_csv}"
        ),
        enabled=send_wechat_enabled,
    )
    return {"status": status, "copied_csv": copied_csv}


def build_jobs(
    inv_true_cfg: Path,
    inv_false_cfg: Path,
    max_attempts: int,
    wsn_lr: float,
    wsn_mu: float,
    jitter_lr: float,
    jitter_mu: float,
) -> List[Dict[str, str]]:
    # TODO text says total 11 jobs while the explicit list is 10.
    # Added hybrid_wsn_invFalse as a disambiguation run to keep the required 11 count.
    jobs = [
        {"tag": "hybrid_wsn_invTrue", "strategy": "hybrid_opt", "wireless_model": "wsn", "simulated_mode": "", "config": str(inv_true_cfg), "algorithm": "fedprox", "lr": str(wsn_lr), "fedprox_mu": str(wsn_mu), "max_attempts": str(max_attempts)},
        {"tag": "bandwidth_first_wsn_invTrue", "strategy": "bandwidth_first", "wireless_model": "wsn", "simulated_mode": "", "config": str(inv_true_cfg), "algorithm": "fedavg", "lr": str(wsn_lr), "fedprox_mu": "0.0", "max_attempts": str(max_attempts)},
        {"tag": "energy_first_wsn_invTrue", "strategy": "energy_first", "wireless_model": "wsn", "simulated_mode": "", "config": str(inv_true_cfg), "algorithm": "fedavg", "lr": str(wsn_lr), "fedprox_mu": "0.0", "max_attempts": str(max_attempts)},
        {"tag": "hybrid_jitter_invFalse", "strategy": "hybrid_opt", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_false_cfg), "algorithm": "fedprox", "lr": str(jitter_lr), "fedprox_mu": str(jitter_mu), "max_attempts": str(max_attempts)},
        {"tag": "hybrid_jitter_invTrue", "strategy": "hybrid_opt", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedprox", "lr": str(jitter_lr), "fedprox_mu": str(jitter_mu), "max_attempts": str(max_attempts)},
        {"tag": "sync_jitter_invTrue", "strategy": "sync", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedprox", "lr": str(jitter_lr), "fedprox_mu": str(jitter_mu), "max_attempts": str(max_attempts)},
        {"tag": "async_jitter_invTrue", "strategy": "async", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedprox", "lr": str(jitter_lr), "fedprox_mu": str(jitter_mu), "max_attempts": str(max_attempts)},
        {"tag": "bridge_free_jitter_invTrue", "strategy": "bridge_free", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedprox", "lr": str(jitter_lr), "fedprox_mu": str(jitter_mu), "max_attempts": str(max_attempts)},
        {"tag": "bandwidth_first_jitter_invTrue", "strategy": "bandwidth_first", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedavg", "lr": str(jitter_lr), "fedprox_mu": "0.0", "max_attempts": str(max_attempts)},
        {"tag": "energy_first_jitter_invTrue", "strategy": "energy_first", "wireless_model": "simulated", "simulated_mode": "jitter", "config": str(inv_true_cfg), "algorithm": "fedavg", "lr": str(jitter_lr), "fedprox_mu": "0.0", "max_attempts": str(max_attempts)},
        {"tag": "hybrid_wsn_invFalse", "strategy": "hybrid_opt", "wireless_model": "wsn", "simulated_mode": "", "config": str(inv_false_cfg), "algorithm": "fedprox", "lr": str(wsn_lr), "fedprox_mu": str(wsn_mu), "max_attempts": str(max_attempts)},
    ]
    return jobs


def latest_success_by_tag(ledger_path: Path, tags: List[str]) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(ledger_path.open("r", encoding="utf-8")))
    rows = [r for r in rows if r.get("status") == "success" and r.get("copied_csv")]
    out: Dict[str, Dict[str, str]] = {}
    for tag in tags:
        cand = [r for r in rows if r.get("tag") == tag]
        if not cand:
            continue
        cand.sort(key=lambda r: r.get("timestamp", ""))
        out[tag] = cand[-1]
    return out


def main() -> None:
    args = parse_args()
    if not args.run_tag:
        args.run_tag = now_tag()
    project_root = Path.cwd().resolve()
    out_root = project_root / "outputs/fl_comp" / args.run_tag
    matrix_dir = out_root / args.run_name
    config_dir = out_root / "configs"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    cfgs = build_configs(
        config_dir,
        args.wsn_lr,
        args.wsn_fedprox_mu,
        args.jitter_start_state,
        args.jitter_period_rounds,
        args.jitter_profile,
        args.initial_client_energy,
        args.client_initial_energies,
    )
    jobs = build_jobs(
        cfgs["inv_true"],
        cfgs["inv_false"],
        args.max_attempts,
        args.wsn_lr,
        args.wsn_fedprox_mu,
        args.jitter_lr,
        args.jitter_fedprox_mu,
    )
    if len(jobs) != 11:
        raise RuntimeError(f"matrix job size mismatch: {len(jobs)}")

    ledger_path = matrix_dir / "attempt_ledger.csv"
    manifest_path = matrix_dir / "matrix_manifest.csv"
    legacy_manifest_path = matrix_dir / "manifest.csv"
    ensure_ledger(ledger_path)
    ensure_manifest(manifest_path)
    ensure_manifest(legacy_manifest_path)

    if args.dry_run:
        print(f"[DRYRUN] run_tag={args.run_tag} matrix_dir={matrix_dir} jobs={len(jobs)}")
        return

    for job in jobs:
        tag = job["tag"]
        job_dir = matrix_dir / tag
        job_dir.mkdir(parents=True, exist_ok=True)
        attempt = next_attempt(job_dir, tag)
        succeeded = False
        copied_csv = ""
        for _ in range(args.max_attempts):
            res = run_attempt(
                job=job,
                attempt=attempt,
                args=args,
                project_root=project_root,
                ledger_path=ledger_path,
                send_wechat_enabled=not args.no_wechat,
            )
            if res["status"] == "success":
                succeeded = True
                copied_csv = res["copied_csv"]
                break
            attempt += 1

        if not succeeded:
            print(f"[FAIL] retries exhausted: {tag}")
            continue

        row = [tag, job["strategy"], job["wireless_model"], job["simulated_mode"], str(args.alpha), job["algorithm"], str(args.seed), str(job["fedprox_mu"]), copied_csv]
        append_csv(manifest_path, row)
        append_csv(legacy_manifest_path, row)

    selected = latest_success_by_tag(ledger_path, [j["tag"] for j in jobs])
    latest_manifest = matrix_dir / "matrix_manifest_latest_success.csv"
    with latest_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "attempt", "copied_csv"])
        for tag in [j["tag"] for j in jobs]:
            pick = selected.get(tag)
            if not pick:
                print(f"[WARN] no successful csv for {tag}")
                continue
            w.writerow([pick["tag"], pick["attempt"], pick["copied_csv"]])
            p = Path(pick["copied_csv"])
            if p.exists():
                m = read_last_metrics(p)
                if m["round"] < 60:
                    print(f"[WARN] final round < 60 tag={tag} csv={p}")

    print(f"[DONE] run_tag={args.run_tag}")
    print(f"[DONE] matrix_dir={matrix_dir}")
    print(f"[DONE] ledger={ledger_path}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
