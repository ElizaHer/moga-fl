from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.configs.strategy_runtime import default_strategy_yaml, load_strategy_yaml
from src.ga.moga_fl import MOGAFLController
from src.ga.moead import MOEAD
from src.ga.nsga3 import NSGA3
from src.ga.sim_runner_flower import make_flower_sim_runner


def _select_best(pop: List[Dict[str, Any]], metrics: List[Dict[str, float]], preference: str) -> int:
    """Select one deployable solution from Pareto candidates by preference."""
    if not pop or not metrics:
        return 0
    pref = str(preference or "time").lower()
    best_idx = 0
    best_score: float | None = None
    for i, m in enumerate(metrics):
        acc = float(m.get("acc", 0.0))
        time = float(m.get("time", 0.0))
        fairness = float(m.get("fairness", 0.0))
        energy = float(m.get("energy", 0.0))
        comm = float(m.get("comm_cost", time))
        if pref == "fairness":
            score = fairness + 0.05 * acc
        elif pref == "energy":
            score = -energy - 0.2 * comm + 0.1 * acc
        else:  # time priority
            score = -time - 0.2 * comm + 0.1 * acc
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="GA optimization for hybrid wireless FL (Flower)")
    parser.add_argument("--config", type=str, default="", help="Path to strategy YAML; default uses src/configs/strategies/<strategy>.yaml")
    parser.add_argument("--strategy", type=str, default="hybrid_opt", help="Strategy name for YAML and Flower strategy factory")
    parser.add_argument("--generations", type=int, default=6, help="GA generations")
    parser.add_argument("--pop", type=int, default=16, help="Population size")
    parser.add_argument(
        "--algo",
        type=str,
        default="nsga3",
        choices=["nsga3", "moead", "moga_fl", "pymoo_nsga3"],
        help="Optimizer backend",
    )
    parser.add_argument("--num-rounds", type=int, default=None, help="Optional override for fl.num_rounds")
    args = parser.parse_args()

    config_path = args.config.strip() or default_strategy_yaml(args.strategy)
    cfg = load_strategy_yaml(config_path)
    cfg = dict(cfg)
    cfg["strategy_name"] = args.strategy
    if args.num_rounds is not None:
        fl = cfg.get("fl", {}) if isinstance(cfg.get("fl", {}), dict) else {}
        fl["num_rounds"] = int(args.num_rounds)
        cfg["fl"] = fl

    low_sim = make_flower_sim_runner(cfg, strategy_name=args.strategy, round_scale=0.5)
    high_sim = make_flower_sim_runner(cfg, strategy_name=args.strategy, round_scale=1.0)

    if args.algo == "nsga3":
        opt = NSGA3(cfg, low_sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)
    elif args.algo == "moead":
        opt = MOEAD(cfg, low_sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)
    elif args.algo == "moga_fl":
        controller = MOGAFLController(cfg, low_fidelity_eval=low_sim, high_fidelity_eval=high_sim, pop_size=args.pop)
        pop, metrics = controller.run(generations=args.generations)
    else:
        from src.ga.pymoo_nsga3 import run_pymoo_nsga3

        pop, metrics = run_pymoo_nsga3(
            cfg,
            strategy_name=args.strategy,
            round_scale=0.5,
            pop_size=args.pop,
            generations=args.generations,
        )

    os.makedirs("outputs/results", exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for p, m in zip(pop, metrics):
        row = {**p, **m}
        rows.append(row)
    df = pd.DataFrame(rows)
    pareto_path = "outputs/results/pareto_candidates.csv"
    df.to_csv(pareto_path, index=False)
    print(f"Saved Pareto candidates to {pareto_path}")

    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    preference = str(eval_cfg.get("preference", "time"))
    best_idx = _select_best(pop, metrics, preference)

    best_params = pop[best_idx] if pop else {}
    deploy_cfg: Dict[str, Any] = copy.deepcopy(cfg)

    scheduler = deploy_cfg.get("scheduler", {}) if isinstance(deploy_cfg.get("scheduler", {}), dict) else {}
    weights = scheduler.get("weights", {}) if isinstance(scheduler.get("weights", {}), dict) else {}
    weights["energy_w"] = float(best_params.get("energy_w", weights.get("energy_w", 0.15)))
    weights["channel_w"] = float(best_params.get("channel_w", weights.get("channel_w", 0.25)))
    weights["data_w"] = float(best_params.get("data_w", weights.get("data_w", 0.25)))
    weights["fair_w"] = float(best_params.get("fair_w", weights.get("fair_w", 0.25)))
    scheduler["weights"] = weights
    if "selection_top_k" in best_params:
        scheduler["selection_top_k"] = int(best_params["selection_top_k"])
    deploy_cfg["scheduler"] = scheduler

    fedbuff = deploy_cfg.get("fedbuff", {}) if isinstance(deploy_cfg.get("fedbuff", {}), dict) else {}
    if "staleness_alpha" in best_params:
        fedbuff["staleness_alpha"] = float(best_params["staleness_alpha"])
    deploy_cfg["fedbuff"] = fedbuff

    controller = deploy_cfg.get("controller", {}) if isinstance(deploy_cfg.get("controller", {}), dict) else {}
    gates = controller.get("gate_thresholds", {}) if isinstance(controller.get("gate_thresholds", {}), dict) else {}
    if "bridge_to_async" in best_params:
        gates["to_async"] = float(best_params["bridge_to_async"])
    if "bridge_to_semi_sync" in best_params:
        gates["to_semi_sync"] = float(best_params["bridge_to_semi_sync"])
    controller["gate_thresholds"] = gates
    deploy_cfg["controller"] = controller

    wireless = deploy_cfg.get("wireless", {}) if isinstance(deploy_cfg.get("wireless", {}), dict) else {}
    if "bandwidth_alloc_factor" in best_params:
        base_bw = float(wireless.get("bandwidth_budget_mb_per_round", 12.0))
        wireless["bandwidth_budget_mb_per_round"] = float(base_bw * float(best_params["bandwidth_alloc_factor"]))
    if "compression_ratio" in best_params:
        wireless["payload_compression_ratio"] = float(best_params["compression_ratio"])
    deploy_cfg["wireless"] = wireless

    best_cfg_path = "outputs/results/best_moga_fl_config.yaml"
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(deploy_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Saved best deployment config to {best_cfg_path}")


if __name__ == "__main__":
    main()
