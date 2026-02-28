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
    """根据偏好从候选解中选出一个“部署用”解。"""
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
        if pref == "fairness":
            score = fairness + 0.05 * acc
        elif pref == "energy":
            score = -energy + 0.1 * acc
        else:  # time 优先
            score = -time + 0.1 * acc
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="GA optimization for hybrid wireless FL (Flower)")
    parser.add_argument("--config", type=str, default="", help="策略 YAML 路径，默认使用 src/configs/strategies/<strategy>.yaml")
    parser.add_argument("--strategy", type=str, default="hybrid_opt", help="策略名称，对应 YAML 与 Flower strategy factory")
    parser.add_argument("--generations", type=int, default=6, help="GA 迭代代数")
    parser.add_argument("--pop", type=int, default=16, help="种群大小")
    parser.add_argument(
        "--algo",
        type=str,
        default="nsga3",
        choices=["nsga3", "moead", "moga_fl", "pymoo_nsga3"],
        help="选择使用的多目标优化算法",
    )
    parser.add_argument("--num-rounds", type=int, default=None, help="可选：覆盖 YAML 中 fl.num_rounds 以加速验证")
    args = parser.parse_args()

    config_path = args.config.strip() or default_strategy_yaml(args.strategy)
    cfg = load_strategy_yaml(config_path)
    cfg = dict(cfg)
    cfg["strategy_name"] = args.strategy
    if args.num_rounds is not None:
        fl = cfg.get("fl", {}) if isinstance(cfg.get("fl", {}), dict) else {}
        fl["num_rounds"] = int(args.num_rounds)
        cfg["fl"] = fl

    # 低保真与高保真评估器：均基于 Flower HybridWirelessStrategy
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
    else:  # pymoo_nsga3
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

    # 根据简单偏好规则选出一个“部署用”解，并导出新的策略 YAML。
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

    best_cfg_path = "outputs/results/best_moga_fl_config.yaml"
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(deploy_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Saved best deployment config to {best_cfg_path}")


if __name__ == "__main__":
    main()
