from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from .utils import eval_model
from .aggregator import Aggregator
from .client import Client
from .compression.quantization import quantize_8bit, dequantize_8bit
from .strategy_controller import StrategyController
from ..scheduling.fairness_ledger import FairnessDebtLedger
from ..scheduling.scorer import ClientScorer
from ..scheduling.selector import TopKSelector
from ..wireless.channel import ChannelSimulator
from ..wireless.bandwidth import BandwidthAllocator
from ..wireless.energy import EnergyEstimator

class Server:
    def __init__(self, model_fn, train_dataset, test_dataset, num_classes: int, cfg: Dict[str, Any], device):
        self.model_fn = model_fn
        self.model = self.model_fn().to(device)
        self.global_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device
        self.num_clients = cfg['clients']['num_clients']
        # Partition indices placeholder (filled externally)
        self.clients = []
        self.aggregator = Aggregator(cfg)
        self.channel = ChannelSimulator(cfg, self.num_clients)
        self.bw_alloc = BandwidthAllocator(cfg)
        self.energy = EnergyEstimator(cfg)
        self.ledger = FairnessDebtLedger(cfg, self.num_clients)
        self.scorer = ClientScorer(cfg, self.num_clients, self.ledger)
        self.selector = TopKSelector(cfg['clients']['selection_top_k'], cfg['clients'].get('sliding_window',5), cfg['clients'].get('hysteresis',0.05))
        # 策略控制器：在半同步 / 异步 / 桥接态间切换（对应问题4）
        self.strategy = StrategyController(cfg, self.num_clients)
        self.training_cfg = cfg.get('training', {})
        self.semi_sync_wait_ratio = self.training_cfg.get('semi_sync_wait_ratio', 0.7)

    def init_clients(self, partitions: Dict[int, List[int]], model_fn):
        self.clients = []
        for cid in range(self.num_clients):
            indices = partitions[cid]
            self.clients.append(Client(cid, model_fn, self.train_dataset, indices, self.cfg, self.device))

    def round(self, r: int, partitions: Dict[int, List[int]]) -> Dict[str, Any]:
        # Channel stats
        stats = self.channel.sample_round()
        avg_per = float(np.mean([stats[cid]['per'] for cid in range(self.num_clients)]))

        # 根据历史情况决定本轮采用的聚合模式与桥接权重、带宽系数
        mode, bridge_w, bw_factor = self.strategy.decide_mode(r)

        # Prepare metrics for scoring
        energy_avail = [1.0 - min(1.0, self.energy.compute_energy(len(partitions[cid]))/10.0) for cid in range(self.num_clients)]
        channel_quality = [1.0 - stats[cid]['per'] for cid in range(self.num_clients)]
        # Data value: rarity of labels approximated by client size (placeholder)
        data_value = [len(partitions[cid]) for cid in range(self.num_clients)]
        bandwidth_cost = [len(partitions[cid])/1000.0 for cid in range(self.num_clients)]
        # Score and select
        scores = self.scorer.score(energy_avail, channel_quality, data_value, bandwidth_cost)
        selected = self.selector.select(scores)
        # Bandwidth allocation（带宽预算按策略建议做简单缩放）
        orig_budget = self.bw_alloc.budget_mb
        self.bw_alloc.budget_mb = orig_budget * float(bw_factor)
        bw_map = self.bw_alloc.allocate_uniform(selected)

        # 估算各客户端发送时间，用于半同步模式的“等待比例”判定
        tx_times = {cid: self.bw_alloc.estimate_tx_time(payload_mb=1.0, allocated_mb=bw_map.get(cid, 0.0)) for cid in selected}
        on_time_clients = list(selected)
        if mode in ('semi_sync', 'bridge') and len(tx_times) > 0:
            # 按发送时间分位数确定“等待阈值”，只接收一部分较快客户端的更新
            t_values = np.array(list(tx_times.values()), dtype=float)
            threshold = float(np.quantile(t_values, self.semi_sync_wait_ratio))
            on_time_clients = [cid for cid in selected if tx_times[cid] <= threshold]

        # Local updates
        client_states = []
        weights = []
        staleness = []
        for cid in selected:
            per = stats[cid]['per']
            # Packet loss: skip update entirely
            if np.random.rand() < per:
                continue
            if cid not in on_time_clients and mode in ('semi_sync', 'bridge'):
                # 半同步/桥接态下，超出等待阈值的客户端被视为“迟到”，本轮不参与同步聚合
                continue
            cl = self.clients[cid]
            sd = cl.local_train(self.global_state)
            # optional compression
            if self.cfg.get('compression', {}).get('quantization_8bit', False):
                q = quantize_8bit(sd)
                sd = dequantize_8bit(q)
            client_states.append(sd)
            weights.append(len(partitions[cid]))
            staleness.append(0)

        # Aggregate
        if client_states or mode in ('async', 'bridge'):
            # 对于异步/桥接模式，即使当前轮没有新更新，也需要调用一次聚合
            self.global_state = self.aggregator.aggregate(
                self.global_state,
                client_states,
                weights,
                staleness_list=staleness,
                round_idx=r,
                mode=mode,
                bridge_weight=bridge_w,
            )
        # Update fairness ledger
        self.ledger.on_round_end(selected)
        # Evaluate
        # Evaluate using a fresh model loaded with global_state
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128)
        model = self.model_fn().to(self.device)
        model.load_state_dict(self.global_state)
        acc = eval_model(model, test_loader, self.device)
        # Costs (rough)
        comm_time = sum(self.bw_alloc.estimate_tx_time(payload_mb=1.0, allocated_mb=bw_map.get(cid, 0.0)) for cid in selected)  # 1MB payload placeholder
        comm_energy = sum(self.energy.comm_energy(self.bw_alloc.estimate_tx_time(1.0, bw_map.get(cid, 0.0))) for cid in selected)
        comp_energy = sum(self.energy.compute_energy(len(partitions[cid])) for cid in selected)

        # 使用本轮指标更新策略控制器（多指标门控与迟滞逻辑依赖于这里的统计）
        self.strategy.register_round_metrics(r, avg_per=avg_per, jain_index=self.jain_index_selection(), total_energy=comm_energy + comp_energy)

        # 恢复带宽预算
        self.bw_alloc.budget_mb = orig_budget

        return {
            'round': r,
            'selected': selected,
            'accuracy': acc,
            'comm_time': comm_time,
            'comm_energy': comm_energy,
            'comp_energy': comp_energy,
            'jain_index': self.jain_index_selection(),
        }

    def jain_index_selection(self) -> float:
        # Compute Jain's index over selection frequency
        freq = np.zeros(self.num_clients)
        for hist in self.selector.history:
            for cid in hist:
                freq[cid] += 1
        if freq.sum() == 0:
            return 0.0
        return float((freq.sum()**2) / (self.num_clients * (freq**2).sum() + 1e-9))

