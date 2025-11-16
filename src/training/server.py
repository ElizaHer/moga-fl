from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from .utils import eval_model
from .aggregator import Aggregator
from .client import Client
from .compression.quantization import quantize_8bit, dequantize_8bit
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

    def init_clients(self, partitions: Dict[int, List[int]], model_fn):
        self.clients = []
        for cid in range(self.num_clients):
            indices = partitions[cid]
            self.clients.append(Client(cid, model_fn, self.train_dataset, indices, self.cfg, self.device))

    def round(self, r: int, partitions: Dict[int, List[int]]) -> Dict[str, Any]:
        # Channel stats
        stats = self.channel.sample_round()
        # Prepare metrics for scoring
        energy_avail = [1.0 - min(1.0, self.energy.compute_energy(len(partitions[cid]))/10.0) for cid in range(self.num_clients)]
        channel_quality = [1.0 - stats[cid]['per'] for cid in range(self.num_clients)]
        # Data value: rarity of labels approximated by client size (placeholder)
        data_value = [len(partitions[cid]) for cid in range(self.num_clients)]
        bandwidth_cost = [len(partitions[cid])/1000.0 for cid in range(self.num_clients)]
        # Score and select
        scores = self.scorer.score(energy_avail, channel_quality, data_value, bandwidth_cost)
        selected = self.selector.select(scores)
        # Bandwidth allocation
        bw_map = self.bw_alloc.allocate_uniform(selected)
        # Local updates
        client_states = []
        weights = []
        staleness = []
        for cid in selected:
            per = stats[cid]['per']
            # Packet loss: skip update
            if np.random.rand() < per:
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
        if client_states:
            self.global_state = self.aggregator.aggregate(self.global_state, client_states, weights, staleness)
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

