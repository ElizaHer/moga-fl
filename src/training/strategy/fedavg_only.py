from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from .hybrid_wireless import HybridWirelessStrategy
from src.utils.train import *


class FedAvgOnlyHybridStrategy(HybridWirelessStrategy):
    """Keep scheduling/wireless/mode control unchanged, but always aggregate with FedAvg."""

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[override]
        del failures
        if self.global_params_cache is None:
            return None, {}

        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)
        scheduled_cids = list(self.last_selected_cids)
        if not scheduled_cids:
            scheduled_cids = [int(res.metrics.get("cid", -1)) for _, res in results]
            scheduled_cids = [cid for cid in scheduled_cids if 0 <= cid < self.cfg.num_clients]

        wireless_stats = self.current_wireless_stats or self.channel.sample_round()
        per_values = [
            float((wireless_stats.get(cid) or {}).get("per", 0.0))
            for cid in scheduled_cids
        ]
        avg_per = float(np.mean(per_values)) if per_values else 0.0

        self._update_bandwidth_budget()
        bw_map = self.bw.allocate_by_stats(wireless_stats, scheduled_cids)
        tx_times: Dict[int, float] = {
            cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0)))
            for cid in scheduled_cids
        }
        print(f"[round {server_round}] bandwidth_allocation_mb={{{', '.join(f'{cid}:{bw_map.get(cid, 0.0):.4f}' for cid in scheduled_cids)}}}")
        print(f"[round {server_round}] upload_time_sec={{{', '.join(f'{cid}:{tx_times.get(cid, 0.0):.4f}' for cid in scheduled_cids)}}}")
        est_upload_time = float(sum(tx_times.values())) if tx_times else 0.0
        threshold = float("inf")
        if scheduled_cids and self.current_mode in ("semi_sync", "bridge"):
            threshold = float(
                np.quantile(
                    np.asarray([tx_times[cid] for cid in scheduled_cids], dtype=np.float64),
                    self.cfg.semi_sync_wait_ratio,
                )
            )

        valid_updates: List[Tuple[List[np.ndarray], float]] = []
        scaffold_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        scaffold_weights: Dict[int, float] = {}
        round_energy = 0.0

        for _, fit_res in results:
            cid = int(fit_res.metrics.get("cid", -1))
            if cid < 0 or cid >= self.cfg.num_clients:
                continue
            num_examples = int(fit_res.num_examples)
            if num_examples <= 0:
                continue

            tx_time = tx_times.get(
                cid,
                self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0))),
            )
            per = float((wireless_stats.get(cid, {})).get("per", 0.0))
            round_energy += self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)

            if np.random.rand() < per:
                continue
            if self.current_mode in ("semi_sync", "bridge") and tx_time > threshold:
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight = float(num_examples)
            valid_updates.append((ndarrays, weight))

            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                payload = fit_res.metrics.get("delta_ci", b"")
                if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
                    delta_ci = unpack_tensor_dict(bytes(payload), torch.device("cpu"))
                    if delta_ci:
                        scaffold_deltas[cid] = delta_ci
                        scaffold_weights[cid] = weight

        merged = self._weighted_avg(valid_updates, global_ndarrays) if valid_updates else global_ndarrays
        self.global_params_cache = ndarrays_to_parameters(merged)

        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None and scaffold_deltas:
            self.scaffold_state.update_global(scaffold_deltas, scaffold_weights)

        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))

        counts = self._compute_selection_counts()
        jain = jain_index(counts)
        self.avg_per_history.append(avg_per)
        self.jain_history.append(jain)
        self.controller.register(avg_per=avg_per, jain=jain, total_energy=round_energy)

        self._log_metrics(
            server_round=server_round,
            mode=self.current_mode,
            accuracy=float(acc),
            loss=float(loss),
            avg_per=avg_per,
            jain=jain,
            energy=float(round_energy),
            est_upload_time=float(est_upload_time),
            bw_factor=float(self.current_bw_factor),
            topk=len(scheduled_cids),
            exhausted_clients=sorted(self.exhausted_clients),
        )
        return self.global_params_cache, {
            "accuracy": float(acc),
            "loss": float(loss),
            "avg_per": float(avg_per),
            "jain": float(jain),
            "mode": self.current_mode,
            "energy": float(round_energy),
            "est_upload_time": float(est_upload_time),
            "bw_factor": float(self.current_bw_factor),
            "topk": float(len(scheduled_cids)),
        }
