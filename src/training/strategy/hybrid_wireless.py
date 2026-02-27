from __future__ import annotations

from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from src.configs.hybrid_cifar import HybridCifarConfig
from src.scheduling.gate import ModeController
from src.training.algorithms.fedbuff import FedBuffState
from src.training.algorithms.scaffold import ScaffoldState
from src.training.models.resnet import build_resnet18
from src.utils.train import *
from src.wireless.channel import ChannelSimulator
from src.wireless.bandwidth import BandwidthAllocator
from src.wireless.energy import EnergyEstimator


# -----------------------------
# 服务器策略：调度 + 半同步/异步/桥接聚合
# -----------------------------


class HybridWirelessStrategy(Strategy):  # type: ignore[misc]
    def __init__(
            self,
            cfg: HybridCifarConfig,
            partition_sizes: List[int],
            testloader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.partition_sizes = partition_sizes
        self.testloader = testloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_for_eval = build_resnet18().to(self.device)

        self.controller = ModeController(cfg, cfg.num_clients)
        self.fedbuff = FedBuffState(
            alpha=cfg.staleness_alpha,
            max_staleness=cfg.max_staleness,
            buffer_size=cfg.fedbuff_buffer_size,
            min_updates=cfg.fedbuff_min_updates_to_aggregate,
            async_agg_interval=cfg.async_agg_interval,
        )

        wireless_cfg = cfg.wireless
        self.channel = ChannelSimulator(wireless_cfg, cfg.num_clients)
        self.bw = BandwidthAllocator(wireless_cfg)
        self.energy = EnergyEstimator(wireless_cfg)

        base_budget = getattr(self.bw, "budget_mb", None)
        self._base_bandwidth_budget: Optional[float] = float(base_budget) if base_budget is not None else None

        self.global_params_cache: Optional[Parameters] = None
        self.scaffold_state: Optional[ScaffoldState] = None
        self.client_energy: Dict[int, float] = {}
        self.exhausted_clients: set[int] = set()

        # 调度和公平性统计
        self.selection_window: Deque[List[int]] = deque(maxlen=self.cfg.fair_window_size)
        self.last_selected_cids: List[int] = []
        self.current_wireless_stats: Optional[Dict[int, Dict[str, float]]] = None

        # 模式与带宽
        self.current_mode: str = "semi_sync"
        self.current_bridge_weight: float = 0.0
        self.current_bw_factor: float = 1.0
        self.last_topk: int = 0

        # 调度评分权重
        strat = cfg.strategy if isinstance(cfg.strategy, dict) else {}
        sched_cfg = strat.get("scheduling", {}) if isinstance(strat.get("scheduling", {}), dict) else {}
        weights = sched_cfg.get("weights", {}) if isinstance(sched_cfg.get("weights", {}), dict) else {}
        self.channel_w = float(weights.get("channel_w", cfg.channel_w))
        self.data_w = float(weights.get("data_w", cfg.data_w))
        self.fair_w = float(weights.get("fair_w", cfg.fair_w))
        self.energy_w = float(weights.get("energy_w", cfg.energy_w))
        self.bwcost_w = float(weights.get("bwcost_w", cfg.bwcost_w))

        # 历史指标
        self.mode_history: List[str] = []
        self.acc_history: List[float] = []
        self.energy_history: List[float] = []
        self.avg_per_history: List[float] = []
        self.jain_history: List[float] = []

        # metrics.csv
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_path = os.path.join("outputs/hybrid_metrics", f"{self.cfg.algorithm}_{current_time}.csv")
        print(f"Result file: {self.metrics_path}")
        self._init_metrics_file()
        self._init_client_energy()

    # --- 工具方法 ---

    def _init_metrics_file(self) -> None:
        os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "round",
                        "mode",
                        "accuracy",
                        "loss",
                        "avg_per",
                        "jain",
                        "energy",
                        "est_upload_time",
                        "bw_factor",
                        "topk",
                        "exhausted_clients",
                    ]
                )

    def _init_client_energy(self) -> None:
        if self.cfg.client_initial_energies is not None:
            energies = list(self.cfg.client_initial_energies)
            if len(energies) != self.cfg.num_clients:
                raise ValueError(
                    f"client_initial_energies length ({len(energies)}) must equal num_clients ({self.cfg.num_clients})"
                )
        else:
            energies = [float(self.cfg.initial_client_energy)] * self.cfg.num_clients
        self.client_energy = {cid: float(max(0.0, e)) for cid, e in enumerate(energies)}
        self.exhausted_clients = {cid for cid, e in self.client_energy.items() if e <= 0.0}

    def _log_metrics(
            self,
            server_round: int,
            mode: str,
            accuracy: float,
            loss: float,
            avg_per: float,
            jain: float,
            energy: float,
            est_upload_time: float,
            bw_factor: float,
            topk: int,
            exhausted_clients: List[int],
    ) -> None:
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(server_round),
                    mode,
                    float(accuracy),
                    float(loss),
                    float(avg_per),
                    float(jain),
                    float(energy),
                    float(est_upload_time),
                    float(bw_factor),
                    int(topk),
                    "|".join(str(cid) for cid in exhausted_clients),
                ]
            )

    def _compute_selection_counts(self) -> np.ndarray:
        counts = np.zeros(self.cfg.num_clients, dtype=np.float64)
        for cids in self.selection_window:
            for cid in cids:
                if 0 <= cid < self.cfg.num_clients:
                    counts[cid] += 1.0
        return counts

    def _compute_fairness_debt(self) -> np.ndarray:
        counts = self._compute_selection_counts()
        if self.selection_window and counts.size > 0:
            mean = float(counts.mean())
            if mean > 0.0:
                debt = (mean - counts) / mean
                debt = np.clip(debt, 0.0, 1.0)
            else:
                debt = np.ones_like(counts)
        else:
            debt = np.ones(self.cfg.num_clients, dtype=np.float64)
        return debt

    def _update_bandwidth_budget(self) -> None:
        if self._base_bandwidth_budget is not None:
            self.bw.budget_mb = self._base_bandwidth_budget * float(self.current_bw_factor)

    def _weighted_avg(
            self, pairs: List[Tuple[List[np.ndarray], float]], fallback: List[np.ndarray]
    ) -> List[np.ndarray]:
        if not pairs:
            return fallback
        total = float(sum(w for _, w in pairs))
        base = [arr.copy() for arr in fallback]
        for i in range(len(base)):
            if np.issubdtype(base[i].dtype, np.floating):
                base[i] = np.zeros_like(base[i], dtype=np.float32)
        for params, weight in pairs:
            scale = weight / max(total, 1e-12)
            for i, arr in enumerate(params):
                if np.issubdtype(base[i].dtype, np.floating):
                    base[i] += scale * arr.astype(np.float32, copy=False)
        return base

    # --- Strategy 接口实现 ---

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:  # type: ignore[override]
        del client_manager
        model = build_resnet18()
        ndarrays = get_parameters(model)
        self.global_params_cache = ndarrays_to_parameters(ndarrays)
        if self.cfg.algorithm.lower() == "scaffold":
            global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.scaffold_state = ScaffoldState(global_state)
        return self.global_params_cache

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ):
        # 1) 门控决策 + 带宽因子
        mode, bridge_w, bw_factor = self.controller.decide(server_round)
        self.current_mode = mode
        self.current_bridge_weight = bridge_w
        self.current_bw_factor = bw_factor
        self.mode_history.append(mode)

        # 2) Top-K 数量
        if self.cfg.selection_top_k > 0:
            top_k = min(self.cfg.selection_top_k, self.cfg.num_clients)
        else:
            top_k = max(1, int(round(self.cfg.num_clients * self.cfg.fraction_fit)))
        self.last_topk = top_k

        # 3) 获取当前可用客户端
        if hasattr(client_manager, "all"):
            all_clients = client_manager.all()
            if isinstance(all_clients, dict):
                available_list = list(all_clients.values())
            else:
                available_list = list(all_clients)
        else:
            sampled_clients = client_manager.sample(
                num_clients=self.cfg.num_clients,
                min_num_clients=self.cfg.num_clients,
            )
            available_list = list(sampled_clients)

        active_cids = [cid for cid in range(self.cfg.num_clients) if cid not in self.exhausted_clients]
        if not active_cids:
            self.last_selected_cids = []
            return []

        cid_to_client: Dict[int, object] = {}
        for cid in active_cids:
            if cid < len(available_list):
                cid_to_client[cid] = available_list[cid]

        if not cid_to_client:
            # 回退为随机采样
            print(f"Warning: No valid cid found for {top_k} clients. Falling back to random sampling.")
            clients = client_manager.sample(
                num_clients=max(1, top_k),
                min_num_clients=max(1, top_k),
            )
            fit_cfg: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "fedprox_mu": float(self.cfg.fedprox_mu),
                "algorithm": str(self.cfg.algorithm),
            }
            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                fit_cfg["scaffold_c_global"] = pack_tensor_dict(self.scaffold_state.c_global)
            fit_ins = FitIns(parameters, fit_cfg)
            return [(client, fit_ins) for client in clients]

        top_k = min(top_k, len(active_cids))
        self.last_topk = top_k

        # 4) 信道抽样 + 带宽预算更新
        self.current_wireless_stats = self.channel.sample_round()
        self._update_bandwidth_budget()

        # 5) 预估各客户端的数据量/能耗
        num_clients = self.cfg.num_clients
        data_sizes = np.asarray(self.partition_sizes, dtype=np.float64)
        if data_sizes.size != num_clients:
            # 防御性处理
            if data_sizes.size < num_clients:
                data_sizes = np.pad(data_sizes, (0, num_clients - data_sizes.size), mode="edge")
            else:
                data_sizes = data_sizes[:num_clients]
        data_norm = data_sizes / max(float(data_sizes.max()), 1e-12)

        fairness_debt = self._compute_fairness_debt()

        # 按所有活跃客户端的信道统计分配带宽，用于近似预测能耗
        bw_map_all = self.bw.allocate_by_stats(self.current_wireless_stats or {}, active_cids)

        channel_score = np.zeros(num_clients, dtype=np.float64)
        energy_arr = np.zeros(num_clients, dtype=np.float64)

        for cid in range(num_clients):
            if cid in self.exhausted_clients:
                continue
            stats = (self.current_wireless_stats or {}).get(cid)
            if stats is None:
                continue
            per = float(stats.get("per", 0.0))
            channel_score[cid] = 1.0 - per
            allocated_mb = float(bw_map_all.get(cid, 0.0))
            tx_time = self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=allocated_mb)
            total_energy = self.energy.comm_energy(tx_time) + self.energy.compute_energy(int(data_sizes[cid]))
            energy_arr[cid] = float(total_energy)

        max_energy = float(energy_arr.max()) if energy_arr.size > 0 else 0.0
        energy_norm = energy_arr / max(max_energy, 1e-12) if max_energy > 0.0 else np.zeros_like(energy_arr)

        # 6) 调度评分：信道 + 数据 + 公平债务 + 能耗
        scores = np.zeros(num_clients, dtype=np.float64)
        for cid in range(num_clients):
            if cid in self.exhausted_clients:
                scores[cid] = -1e9
                continue
            scores[cid] = (
                    self.channel_w * channel_score[cid]
                    + self.data_w * data_norm[cid]
                    + self.fair_w * fairness_debt[cid]
                    + self.energy_w * (1.0 - energy_norm[cid])
            )
            print(f"client {cid} score: {scores[cid]:.4f}, "
                  f"channel: {channel_score[cid]:.4f}, "
                  f"data: {data_norm[cid]:.4f}, "
                  f"fair: {fairness_debt[cid]:.4f}, "
                  f"energy: {1.0 - energy_norm[cid]:.4f}")

        candidates = sorted(cid_to_client.keys())
        if not candidates:
            clients = client_manager.sample(
                num_clients=max(1, top_k),
                min_num_clients=max(1, top_k),
            )
            fit_cfg: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "fedprox_mu": float(self.cfg.fedprox_mu),
                "algorithm": str(self.cfg.algorithm),
            }
            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                fit_cfg["scaffold_c_global"] = pack_tensor_dict(self.scaffold_state.c_global)
            fit_ins = FitIns(parameters, fit_cfg)
            return [(client, fit_ins) for client in clients]

        candidates_sorted = sorted(candidates, key=lambda cid: scores[cid], reverse=True)
        topk = min(len(candidates_sorted), top_k)
        selected_cids = candidates_sorted[:topk]
        self.last_selected_cids = list(selected_cids)

        # 更新滑窗选择历史（供 Jain 和公平债务使用）
        self.selection_window.append(list(selected_cids))

        fit_cfg: Dict[str, Scalar] = {
            "server_round": float(server_round),
            "fedprox_mu": float(self.cfg.fedprox_mu),
            "algorithm": str(self.cfg.algorithm),
        }
        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
            fit_cfg["scaffold_c_global"] = pack_tensor_dict(self.scaffold_state.c_global)
        fit_ins = FitIns(parameters, fit_cfg)
        return [(cid_to_client[cid], fit_ins) for cid in selected_cids]

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[override]
        del failures
        if self.global_params_cache is None:
            return None, {}

        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)

        scheduled_cids = list(self.last_selected_cids)
        if not scheduled_cids:
            # 回退：以本轮实际返回的 cid 作为调度集合
            scheduled_cids = [int(res.metrics.get("cid", -1)) for _, res in results]
            scheduled_cids = [cid for cid in scheduled_cids if 0 <= cid < self.cfg.num_clients]
        scheduled_cids = [cid for cid in scheduled_cids if cid not in self.exhausted_clients]

        print(f"selected cids: {scheduled_cids}")

        wireless_stats = self.current_wireless_stats or self.channel.sample_round()

        # 当前轮的 avg_per
        per_values: List[float] = []
        for cid in scheduled_cids:
            stats = wireless_stats.get(cid)
            if stats is None:
                continue
            per_values.append(float(stats.get("per", 0.0)))
        avg_per = float(np.mean(per_values)) if per_values else 0.0

        # aging 现有 FedBuff 条目
        self.fedbuff.age(server_round)

        # 带宽预算 + tx_time 估计
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
            tx_list = [tx_times[cid] for cid in scheduled_cids]
            threshold = float(
                np.quantile(np.asarray(tx_list, dtype=np.float64), self.cfg.semi_sync_wait_ratio)
            )

        valid_sync_updates: List[Tuple[List[np.ndarray], float]] = []
        scaffold_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        scaffold_weights: Dict[int, float] = {}
        round_energy = 0.0

        for _, fit_res in results:
            cid = int(fit_res.metrics.get("cid", -1))
            if cid < 0 or cid >= self.cfg.num_clients:
                continue
            if cid in self.exhausted_clients:
                continue
            num_examples = int(fit_res.num_examples)
            if num_examples <= 0:
                continue

            tx_time = tx_times.get(
                cid,
                self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0))),
            )
            stats = wireless_stats.get(cid, {})
            per = float(stats.get("per", 0.0))

            # 通信 + 计算能耗均计入，并从客户端剩余能量中扣除
            client_round_energy = self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)
            round_energy += client_round_energy
            remaining = float(self.client_energy.get(cid, 0.0)) - float(client_round_energy)
            self.client_energy[cid] = max(0.0, remaining)
            if remaining <= 0.0:
                self.exhausted_clients.add(cid)
                continue

            # PER 丢包：仅影响是否使用该更新
            if np.random.rand() < per:
                print(f"client {cid} 丢包，跳过更新")
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight = float(num_examples)

            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                payload = fit_res.metrics.get("delta_ci", b"")
                if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
                    delta_ci = unpack_tensor_dict(bytes(payload), torch.device("cpu"))
                    if delta_ci:
                        scaffold_deltas[cid] = delta_ci
                        scaffold_weights[cid] = weight

            # 异步路径：所有未丢包更新进入 FedBuff
            self.fedbuff.push(ndarrays, num_examples)

            # 半同步路径：仅聚合 tx_time 不超过阈值的按时更新
            if self.current_mode in ("semi_sync", "bridge") and tx_time > threshold:
                continue
            valid_sync_updates.append((ndarrays, weight))

        # 同步聚合（FedAvg）
        if valid_sync_updates:
            sync_result = self._weighted_avg(valid_sync_updates, global_ndarrays)
        else:
            sync_result = global_ndarrays

        # 异步聚合（FedBuff）
        async_result = global_ndarrays
        if self.fedbuff.should_aggregate(server_round):
            async_result = self.fedbuff.aggregate(global_ndarrays, server_round)

        # 模式选择与桥接混合
        if self.current_mode == "async":
            merged = async_result
        elif self.current_mode == "semi_sync":
            merged = sync_result
        else:  # bridge
            w = float(np.clip(self.current_bridge_weight, 0.0, 1.0))
            merged = []
            for s, a in zip(sync_result, async_result):
                if np.issubdtype(s.dtype, np.floating):
                    merged.append(
                        (1.0 - w) * s.astype(np.float32, copy=False)
                        + w * a.astype(np.float32, copy=False)
                    )
                else:
                    merged.append(s)

        self.global_params_cache = ndarrays_to_parameters(merged)

        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None and scaffold_deltas:
            self.scaffold_state.update_global(scaffold_deltas, scaffold_weights)

        # 评估全局模型
        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))

        # 基于滑窗选择历史计算 Jain 公平指数
        counts = self._compute_selection_counts()
        jain = jain_index(counts)
        self.avg_per_history.append(avg_per)
        self.jain_history.append(jain)

        # 将本轮指标喂给门控控制器（供下一轮决策）
        self.controller.register(avg_per=avg_per, jain=jain, total_energy=round_energy)

        # 写入 metrics.csv
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

        metrics: Dict[str, Scalar] = {
            "accuracy": float(acc),
            "loss": float(loss),
            "avg_per": float(avg_per),
            "jain": float(jain),
            "mode": self.current_mode,
            "energy": float(round_energy),
            "est_upload_time": float(est_upload_time),
            "bw_factor": float(self.current_bw_factor),
            "topk": float(len(scheduled_cids)),
            "num_exhausted_clients": float(len(self.exhausted_clients)),
        }

        return self.global_params_cache, metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager):  # type: ignore[override]
        del server_round, parameters, client_manager
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):  # type: ignore[override]
        del server_round, results, failures
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):  # type: ignore[override]
        del server_round
        set_parameters(self.model_for_eval, parameters_to_ndarrays(parameters))
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        return float(loss), {"accuracy": float(acc)}

