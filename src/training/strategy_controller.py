from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np


class StrategyController:
    """训练策略控制器：在半同步、异步与桥接态之间动态切换。

    对应问题4：
    - 多指标门控：利用丢包率、能量消耗、公平性等指标做加权打分；
    - 迟滞与防抖：通过滑动窗口与阈值区间，避免频繁来回切换；
    - 带宽与能量预测/再平衡：用简单滑动平均或 EWMA 估计近几轮趋势，
      给出带宽调整建议；
    - 桥接态与混合聚合：在切换阶段返回 "bridge" 状态，以及对应的混合权重；
    - 公平债务结算：依赖外部的公平债务账本与 Jain 指数，优先在长期不公平时
      触发向异步或桥接态迁移。

    设计上刻意保持轻量，方便在低算力环境下运行。
    """

    def __init__(self, cfg: Dict[str, Any], num_clients: int):
        strat = cfg.get('strategy', {})
        self.window_size = strat.get('window_size', 5)
        gates = strat.get('gate_thresholds', {})
        # gate_score 约在 [0, 1] 区间：偏“小”倾向半同步，偏“大”倾向异步
        self.to_async = gates.get('to_async', 0.6)
        self.to_semi = gates.get('to_semi_sync', 0.4)

        self.hysteresis_margin = strat.get('hysteresis_margin', 0.05)
        self.bridge_rounds = strat.get('bridge_rounds', 3)
        self.min_rounds_between_switch = strat.get('min_rounds_between_switch', 5)
        w = strat.get('gate_weights', {})
        # 多指标门控权重
        self.w_per = w.get('per', 0.5)
        self.w_fair = w.get('fairness', 0.3)
        self.w_energy = w.get('energy', 0.2)

        # 带宽再平衡参数
        bw_cfg = strat.get('bandwidth_rebalance', {})
        self.bw_low_energy_factor = bw_cfg.get('low_energy_factor', 0.8)
        self.bw_high_energy_factor = bw_cfg.get('high_energy_factor', 1.0)

        self.num_clients = num_clients
        self.history: List[Dict[str, float]] = []
        self.current_mode: str = cfg['training'].get('sync_mode', 'sync')  # sync / semi_sync / async / bridge
        self.last_switch_round: int | None = None
        self.bridge_target_mode: str | None = None
        self.bridge_start_round: int | None = None

    # --------------------- 历史统计与门控打分 ---------------------
    def register_round_metrics(
        self,
        round_idx: int,
        avg_per: float,
        jain_index: float,
        total_energy: float,
    ) -> None:
        row = {
            'round': round_idx,
            'avg_per': float(avg_per),
            'jain': float(jain_index),
            'energy': float(total_energy),
        }
        self.history.append(row)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def _compute_gate_score(self) -> Tuple[float, Dict[str, float]]:
        if not self.history:
            return 0.0, {'avg_per': 0.0, 'jain': 1.0, 'energy_norm': 0.0}
        avg_per = float(np.mean([h['avg_per'] for h in self.history]))
        avg_jain = float(np.mean([h['jain'] for h in self.history]))
        avg_energy = float(np.mean([h['energy'] for h in self.history]))
        # 将能量粗略归一化到 [0,1]，这里假设 0~10 为典型区间
        energy_norm = float(np.clip(avg_energy / 10.0, 0.0, 1.0))
        # Jain 越小越不公平，因此用 (1-Jain)
        fairness_deficit = 1.0 - avg_jain

        gate_score = (
            self.w_per * avg_per +
            self.w_fair * fairness_deficit +
            self.w_energy * energy_norm
        )
        return float(gate_score), {
            'avg_per': avg_per,
            'jain': avg_jain,
            'energy_norm': energy_norm,
        }

    # --------------------- 状态机与桥接态 ---------------------
    def decide_mode(self, round_idx: int) -> Tuple[str, float, float]:
        """根据历史指标决定当前轮应采用的模式及桥接权重与带宽系数。

        返回
        ------
        mode: "semi_sync" / "async" / "bridge" / "sync"。
        bridge_weight: 若 mode=="bridge"，则给出 [0,1] 之间的混合权重，
            用于在 Aggregator 中混合同步与异步结果；否则忽略。
        bandwidth_factor: 建议对带宽预算做的缩放因子，用于简单能量/带宽再平衡。
        """
        gate_score, stats = self._compute_gate_score()
        # 简单的能量感知带宽调整：能量越高（越偏 1），越接近 high_energy_factor
        energy_norm = stats['energy_norm']
        bw_factor = self.bw_low_energy_factor + (self.bw_high_energy_factor - self.bw_low_energy_factor) * (1.0 - energy_norm)

        # 最少间隔轮数，避免频繁切换
        if self.last_switch_round is not None and (round_idx - self.last_switch_round) < self.min_rounds_between_switch:
            # 若当前正在桥接态，则计算桥接权重；否则保持原模式
            mode, w = self._current_mode_and_bridge_weight(round_idx)
            return mode, w, bw_factor

        # 目标模式决策
        target_mode = self.current_mode
        if gate_score >= self.to_async + self.hysteresis_margin:
            print(f"[switch target mode] gate_score={gate_score:.3f} => async")
            target_mode = 'async'
        elif gate_score <= self.to_semi - self.hysteresis_margin:
            print(f"[switch target mode] gate_score={gate_score:.3f} => semi_sync")
            target_mode = 'semi_sync'

        # 若目标模式与当前模式一致，则维持现状
        if target_mode == self.current_mode:
            mode, w = self._current_mode_and_bridge_weight(round_idx)
            return mode, w, bw_factor

        # 触发桥接态：从当前模式平滑过渡到 target_mode
        self.bridge_target_mode = target_mode
        self.bridge_start_round = round_idx
        self.current_mode = 'bridge'
        self.last_switch_round = round_idx
        mode, w = self._current_mode_and_bridge_weight(round_idx)
        return mode, w, bw_factor

    def _current_mode_and_bridge_weight(self, round_idx: int) -> Tuple[str, float]:
        if self.current_mode != 'bridge' or self.bridge_start_round is None or self.bridge_target_mode is None:
            # 非桥接态：桥接权重无意义，用 0.0 占位
            return self.current_mode, 0.0
        # 线性插值：在 bridge_rounds 内从 0->1 或 1->0
        t = max(0, round_idx - self.bridge_start_round)
        if self.bridge_rounds <= 0:
            w = 1.0
        else:
            w = float(np.clip(t / self.bridge_rounds, 0.0, 1.0))
        # 根据切换方向校正权重含义：
        # - 半同步 -> 异步：权重从 0->1，表示异步结果占比逐渐上升；
        # - 异步   -> 半同步：可以用 1-w 解释为同步占比上升，这里在 Aggregator 内部
        #   只使用 w，具体含义由上层约定。
        # 当过渡结束后，进入目标模式
        if t >= self.bridge_rounds:
            self.current_mode = self.bridge_target_mode
            self.bridge_target_mode = None
            self.bridge_start_round = None
        return 'bridge', w
