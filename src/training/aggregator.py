from typing import Dict, Any, List
import torch
import numpy as np
from .algorithms.fedavg import aggregate_fedavg
from .algorithms.fedbuff import FedBuffBuffer

class Aggregator:
    """聚合器：支持同步 / 半同步 / 异步（FedBuff）及桥接态混合聚合。

    - 对应问题2：中低丢包场景下的半同步聚合（由 Server 控制参与集合，
      此处以 FedAvg 对“按时返回”的客户端做聚合）。
    - 对应问题3：高丢包场景下的异步聚合（FedBuff 风格缓冲＋陈旧度加权）。
    - 对应问题4：桥接态下的混合聚合（同步结果与异步结果按权重线性组合）。
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        fb = cfg['training'].get('fedbuff', {'enabled': False})
        self.mode = cfg['training'].get('sync_mode', 'sync')  # sync / semi_sync / async / bridge
        self.buffer = None
        self.buffer_size = fb.get('buffer_size', 32)
        if fb.get('enabled', False):
            self.buffer = FedBuffBuffer(
                self.buffer_size,
                fb.get('staleness_alpha', 1.0),
                max_staleness=fb.get('max_staleness', None),
            )
        # 异步相关参数
        self.async_min_updates = fb.get('min_updates_to_aggregate', max(1, self.buffer_size // 2))
        self.async_agg_interval = fb.get('async_agg_interval', None)
        self.updates_since_last_agg = 0
        self.last_agg_round: int | None = None

    def aggregate(
        self,
        global_state: Dict[str, Any],
        client_states: List[Dict[str, Any]],
        weights: List[float],
        staleness_list: List[int] | None = None,
        round_idx: int | None = None,
        mode: str | None = None,
        bridge_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """根据模式执行聚合。

        参数
        ------
        global_state: 当前全局模型参数。
        client_states: 本轮收到的客户端更新（只包括“按时返回”的）。
        weights: FedAvg 权重（通常是每客户端样本数）。
        staleness_list: 在异步模式下可用于外部传入陈旧度；若为 None，则
            新进入缓冲的更新默认陈旧度为 0，由缓冲内部随轮 aging。
        round_idx: 全局轮次索引，用于异步模式下的陈旧度 aging 与触发间隔。
        mode: 若为 None，则使用 self.mode；否则临时覆盖（用于策略控制器）。
        bridge_weight: 在 "bridge" 模式下，同步结果与异步结果的插值权重。
        """
        current_mode = mode or self.mode
        print(f"[sync mode] {current_mode}")

        # ---------- 纯同步 / 半同步：直接 FedAvg ----------
        if self.buffer is None or current_mode in ('sync', 'semi_sync'):
            print(f"[(semi) sync agg]")
            if not client_states:
                return global_state
            return aggregate_fedavg(global_state, client_states, weights)

        # ---------- 异步 / 桥接：使用 FedBuff 缓冲 ----------
        # 先让缓冲中的旧条目随着轮数变旧
        if round_idx is not None:
            if self.last_agg_round is None:
                self.last_agg_round = round_idx
            else:
                delta = max(0, round_idx - self.last_agg_round)
                if delta > 0:
                    self.buffer.age_entries(delta)
                    self.last_agg_round = round_idx

        # 将本轮新到达的更新推入缓冲，初始陈旧度为 0 或外部给定
        if client_states:
            if staleness_list is None:
                staleness_list = [0] * len(client_states)
            for sd, st in zip(client_states, staleness_list):
                self.buffer.push(sd, st)
            self.updates_since_last_agg += len(client_states)

        # 判断是否触发一次异步聚合
        should_agg = False
        if len(self.buffer.entries) >= self.buffer_size:
            should_agg = True
        if self.updates_since_last_agg >= self.async_min_updates:
            should_agg = True
        if self.async_agg_interval is not None and round_idx is not None:
            if self.last_agg_round is None or (round_idx - self.last_agg_round) >= self.async_agg_interval:
                should_agg = True

        async_result = global_state
        if should_agg:
            async_result = self.buffer.aggregate(global_state)
            self.updates_since_last_agg = 0
            self.last_agg_round = round_idx

        if current_mode == 'async':
            # 纯异步模式：直接返回异步聚合结果
            print(f"[async agg]")
            return async_result

        # bridge 模式：将“本轮同步 FedAvg 结果”和“异步缓冲结果”做线性组合
        print(f"[bridge agg] bridge weight: {bridge_weight}")
        if not client_states:
            # 没有新的同步更新，只能依赖异步结果
            return async_result

        sync_result = aggregate_fedavg(global_state, client_states, weights)
        w = float(np.clip(bridge_weight, 0.0, 1.0))
        mixed = {}
        for k in sync_result.keys():
            mixed[k] = (1.0 - w) * sync_result[k] + w * async_result[k]
        return mixed
