from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class StrategyController:
    """训练策略控制器：在半同步、异步与桥接态之间动态切换。

    与原工程 `src/training/strategy_controller.py` 等价，用于基于历史丢包率、
    公平性与能耗统计决定当前轮的同步模式以及带宽缩放因子。
    """

    def __init__(self, cfg: Dict[str, Any], num_clients: int) -> None:
        strat = cfg.get("strategy", {})
        self.window_size = strat.get("window_size", 5)
        gates = strat.get("gate_thresholds", {})
        self.to_async = gates.get("to_async", 0.6)
        self.to_semi = gates.get("to_semi_sync", 0.4)

        self.hysteresis_margin = strat.get("hysteresis_margin", 0.05)
        self.bridge_rounds = strat.get("bridge_rounds", 3)
        self.min_rounds_between_switch = strat.get("min_rounds_between_switch", 5)
        w = strat.get("weights", {})
        self.w_per = w.get("per", 0.5)
        self.w_fair = w.get("fairness", 0.3)
        self.w_energy = w.get("energy", 0.2)

        bw_cfg = strat.get("bandwidth_rebalance", {})
        self.bw_low_energy_factor = bw_cfg.get("low_energy_factor", 0.8)
        self.bw_high_energy_factor = bw_cfg.get("high_energy_factor", 1.0)

        self.num_clients = num_clients
        self.history: List[Dict[str, float]] = []
        self.current_mode: str = cfg.get("training", {}).get("sync_mode", "sync")
        self.last_switch_round: int | None = None
        self.bridge_target_mode: str | None = None
        self.bridge_start_round: int | None = None

    # ---------------- 历史统计与门控打分 ----------------
    def register_round_metrics(
        self,
        round_idx: int,
        avg_per: float,
        jain_index: float,
        total_energy: float,
    ) -> None:
        row = {
            "round": round_idx,
            "avg_per": float(avg_per),
            "jain": float(jain_index),
            "energy": float(total_energy),
        }
        self.history.append(row)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def _compute_gate_score(self) -> Tuple[float, Dict[str, float]]:
        if not self.history:
            return 0.0, {"avg_per": 0.0, "jain": 1.0, "energy_norm": 0.0}
        avg_per = float(np.mean([h["avg_per"] for h in self.history]))
        avg_jain = float(np.mean([h["jain"] for h in self.history]))
        avg_energy = float(np.mean([h["energy"] for h in self.history]))
        energy_norm = float(np.clip(avg_energy / 10.0, 0.0, 1.0))
        fairness_deficit = 1.0 - avg_jain

        gate_score = (
            self.w_per * avg_per
            + self.w_fair * fairness_deficit
            + self.w_energy * energy_norm
        )
        return float(gate_score), {
            "avg_per": avg_per,
            "jain": avg_jain,
            "energy_norm": energy_norm,
        }

    # ---------------- 状态机与桥接态 ----------------
    def decide_mode(self, round_idx: int) -> Tuple[str, float, float]:
        """Decide FL sync mode for current round.

        Returns
        -------
        mode: "semi_sync" / "async" / "bridge" / "sync".
        bridge_weight: in [0, 1], only meaningful when mode == "bridge".
        bandwidth_factor: scale factor for bandwidth budget.
        """

        gate_score, stats = self._compute_gate_score()
        energy_norm = stats["energy_norm"]
        bw_factor = self.bw_low_energy_factor + (
            self.bw_high_energy_factor - self.bw_low_energy_factor
        ) * (1.0 - energy_norm)

        # 最少间隔轮数，避免频繁切换
        if self.last_switch_round is not None and (
            round_idx - self.last_switch_round
        ) < self.min_rounds_between_switch:
            mode, w = self._current_mode_and_bridge_weight(round_idx)
            return mode, w, bw_factor

        # 目标模式决策
        target_mode = self.current_mode
        if gate_score >= self.to_async + self.hysteresis_margin:
            print(f"[switch target mode] gate_score={gate_score:.3f} => async")
            target_mode = "async"
        elif gate_score <= self.to_semi - self.hysteresis_margin:
            print(f"[switch target mode] gate_score={gate_score:.3f} => semi_sync")
            target_mode = "semi_sync"

        if target_mode == self.current_mode:
            mode, w = self._current_mode_and_bridge_weight(round_idx)
            return mode, w, bw_factor

        # 触发桥接态
        self.bridge_target_mode = target_mode
        self.bridge_start_round = round_idx
        self.current_mode = "bridge"
        self.last_switch_round = round_idx
        mode, w = self._current_mode_and_bridge_weight(round_idx)
        return mode, w, bw_factor

    def _current_mode_and_bridge_weight(self, round_idx: int) -> Tuple[str, float]:
        if (
            self.current_mode != "bridge"
            or self.bridge_start_round is None
            or self.bridge_target_mode is None
        ):
            return self.current_mode, 0.0
        t = max(0, round_idx - self.bridge_start_round)
        if self.bridge_rounds <= 0:
            w = 1.0
        else:
            w = float(np.clip(t / self.bridge_rounds, 0.0, 1.0))

        if t >= self.bridge_rounds:
            self.current_mode = self.bridge_target_mode
            self.bridge_target_mode = None
            self.bridge_start_round = None
        return "bridge", w
