from __future__ import annotations

import csv
import datetime
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import FitIns, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader

from src.scheduling.gate import ModeController
from src.training.algorithms.fedbuff import FedBuffState
from src.training.algorithms.scaffold import ScaffoldState
from src.training.models.resnet import build_resnet18
from src.utils.train import (
    evaluate_model,
    get_parameters,
    jain_index,
    pack_tensor_dict,
    set_parameters,
    unpack_tensor_dict,
)
from src.wireless.bandwidth import BandwidthAllocator
from src.wireless.channel import ChannelSimulator
from src.wireless.energy import EnergyEstimator


class HybridWirelessStrategy(Strategy):  # type: ignore[misc]
    def __init__(
        self,
        cfg: Dict[str, Any],
        partition_sizes: List[int],
        testloader: DataLoader,
        wsn_wireless_sampler=None,
    ) -> None:
        self.cfg = cfg
        self.partition_sizes = partition_sizes
        self.testloader = testloader
        self.wsn_wireless_sampler = wsn_wireless_sampler

        fl_cfg = cfg.get("fl", {})
        wireless_cfg = cfg.get("wireless", {})
        fedbuff_cfg = cfg.get("fedbuff", {})
        scheduler_cfg = cfg.get("scheduler", {})
        algorithm_cfg = cfg.get("algorithm", {})
        energy_cfg = cfg.get("energy", {})
        controller_cfg = cfg.get("controller", {})

        self.num_clients = int(fl_cfg.get("num_clients", 10))
        self.fraction_fit = float(fl_cfg.get("fraction_fit", 0.5))
        self.selection_top_k = int(scheduler_cfg.get("selection_top_k", 0))
        self.fair_window_size = int(scheduler_cfg.get("fair_window_size", 4))
        self.data_value_weights = (
            scheduler_cfg.get("data_value_weights", {})
            if isinstance(scheduler_cfg.get("data_value_weights", {}), dict)
            else {}
        )
        self.novelty_w = float(self.data_value_weights.get("novelty_w", 0.2))
        self.tail_w = float(self.data_value_weights.get("tail_w", 0.4))
        self.size_w = float(self.data_value_weights.get("size_w", 0.4))
        self.contribution_ema_beta = float(scheduler_cfg.get("contribution_ema_beta", 0.9))
        self.energy_guard_cfg = (
            scheduler_cfg.get("energy_guard", {})
            if isinstance(scheduler_cfg.get("energy_guard", {}), dict)
            else {}
        )
        self.min_reserve_energy_ratio = float(self.energy_guard_cfg.get("min_reserve_energy_ratio", 0.1))
        self.future_energy_factor = float(self.energy_guard_cfg.get("future_energy_factor", 1.2))
        self.survival_gamma = float(self.energy_guard_cfg.get("survival_gamma", 1.5))
        self.selection_guard_cfg = (
            scheduler_cfg.get("selection_guard", {})
            if isinstance(scheduler_cfg.get("selection_guard", {}), dict)
            else {}
        )
        self.max_consecutive_selected = int(self.selection_guard_cfg.get("max_consecutive_selected", 3))
        self.cooldown_rounds = int(self.selection_guard_cfg.get("cooldown_rounds", 2))
        self.wsn_guard_relax_cfg = (
            scheduler_cfg.get("wsn_guard_relax", {})
            if isinstance(scheduler_cfg.get("wsn_guard_relax", {}), dict)
            else {}
        )
        self.algorithm = str(algorithm_cfg.get("name", "fedavg")).lower()
        self.strategy_name = str(cfg.get("strategy_name", "hybrid_opt")).lower()
        self.fedprox_mu = float(algorithm_cfg.get("fedprox_mu", 0.0))
        self.semi_sync_wait_ratio = float(controller_cfg.get("semi_sync_wait_ratio", 0.7))
        self.hybrid_semi_sync_wait_ratio = float(controller_cfg.get("hybrid_semi_sync_wait_ratio", self.semi_sync_wait_ratio))
        self.enable_hybrid_sync_boost = bool(controller_cfg.get("enable_hybrid_sync_boost", True))
        self.mode_policy = str(controller_cfg.get("mode_policy", "hybrid")).lower()
        self.selection_policy = str(scheduler_cfg.get("selection_policy", "hybrid")).lower()
        self.wireless_model = str(wireless_cfg.get("wireless_model", "simulated")).lower()
        self.bridge_inv_cfg = (
            controller_cfg.get("bridge_invariants", {})
            if isinstance(controller_cfg.get("bridge_invariants", {}), dict)
            else {}
        )
        self.bridge_inv_enable = bool(self.bridge_inv_cfg.get("enable", False))
        self.prev_mean_fair_debt: Optional[float] = None
        self.bridge_fail_streak = 0
        self.historical_warmup_rounds = int(scheduler_cfg.get("historical_contribution_warmup_rounds", 0))
        self.bandwidth_first_penalty_cfg = (
            scheduler_cfg.get("bandwidth_first_penalty", {})
            if isinstance(scheduler_cfg.get("bandwidth_first_penalty", {}), dict)
            else {}
        )
        self.bandwidth_first_penalty_enable = bool(self.bandwidth_first_penalty_cfg.get("enable", False))
        self.bandwidth_first_penalty_streak = int(self.bandwidth_first_penalty_cfg.get("streak_threshold", 3))
        self.bandwidth_first_energy_multiplier = float(self.bandwidth_first_penalty_cfg.get("energy_multiplier", 1.0))
        self.bandwidth_first_forced_cooldown_rounds = int(self.bandwidth_first_penalty_cfg.get("cooldown_rounds", 0))

        weights = scheduler_cfg.get("weights", {}) if isinstance(scheduler_cfg.get("weights", {}), dict) else {}
        self.channel_w = float(weights.get("channel_w", 0.25))
        self.data_w = float(weights.get("data_w", 0.25))
        self.fair_w = float(weights.get("fair_w", 0.25))
        self.energy_w = float(weights.get("energy_w", 0.15))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_for_eval = build_resnet18().to(self.device)

        self.controller = ModeController(cfg, self.num_clients)
        self.channel = ChannelSimulator(wireless_cfg, self.num_clients)
        self.bw = BandwidthAllocator(wireless_cfg)
        self.energy = EnergyEstimator(wireless_cfg)
        self.fedbuff = FedBuffState(
            alpha=float(fedbuff_cfg.get("staleness_alpha", 1.0)),
            max_staleness=int(fedbuff_cfg.get("max_staleness", 8)),
            buffer_size=int(fedbuff_cfg.get("buffer_size", 16)),
            min_updates=int(fedbuff_cfg.get("min_updates_to_aggregate", 8)),
            async_agg_interval=int(fedbuff_cfg.get("async_agg_interval", 2)),
        )

        self.global_params_cache: Optional[Parameters] = None
        self.scaffold_state: Optional[ScaffoldState] = None
        self.current_wireless_stats: Optional[Dict[int, Dict[str, float]]] = None
        self.selection_window: Deque[List[int]] = deque(maxlen=self.fair_window_size)

        self.current_mode: str = "semi_sync"
        self.current_bridge_weight: float = 0.0
        self.current_bw_factor: float = 1.0
        self.last_selected_cids: List[int] = []

        self.mode_history: List[str] = []
        self.acc_history: List[float] = []
        self.energy_history: List[float] = []
        self.avg_per_history: List[float] = []
        self.jain_history: List[float] = []

        self.initial_client_energy = float(energy_cfg.get("initial_client_energy", 100.0))
        self.client_initial_energies = energy_cfg.get("client_initial_energies")
        self.min_energy_floor_per_round = float(energy_cfg.get("min_energy_floor_per_round", 0.0))
        self.client_energy: Dict[int, float] = {}
        self.exhausted_clients: set[int] = set()
        self._init_client_energy()
        self.client_contribution_ema: Dict[int, float] = {cid: 0.0 for cid in range(self.num_clients)}
        self.client_selection_streak: Dict[int, int] = {cid: 0 for cid in range(self.num_clients)}
        self.client_cooldown_until: Dict[int, int] = {cid: 0 for cid in range(self.num_clients)}
        self.prev_round_accuracy: float = 0.0

        self._base_bandwidth_budget = float(self.bw.budget_mb)
        self._apply_wsn_guard_relax()

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = os.path.join("outputs", "hybrid_metrics", self.strategy_name)
        self.metrics_path = os.path.join(metrics_dir, f"{self.algorithm}_{current_time}.csv")
        os.makedirs(metrics_dir, exist_ok=True)
        self._init_metrics_file()

    def _init_client_energy(self) -> None:
        if self.client_initial_energies is not None:
            energies = list(self.client_initial_energies)
            if len(energies) != self.num_clients:
                raise ValueError(
                    f"client_initial_energies length ({len(energies)}) must equal num_clients ({self.num_clients})"
                )
        else:
            energies = [self.initial_client_energy] * self.num_clients
        self.client_energy = {cid: float(max(0.0, e)) for cid, e in enumerate(energies)}
        self.exhausted_clients = {cid for cid, e in self.client_energy.items() if e <= 0.0}

    def _init_metrics_file(self) -> None:
        if os.path.exists(self.metrics_path):
            return
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
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
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
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
        counts = np.zeros(self.num_clients, dtype=np.float64)
        for cids in self.selection_window:
            for cid in cids:
                if 0 <= cid < self.num_clients:
                    counts[cid] += 1.0
        return counts

    def _compute_fairness_debt(self) -> np.ndarray:
        counts = self._compute_selection_counts()
        if self.selection_window and counts.size > 0:
            mean = float(counts.mean())
            if mean > 0.0:
                debt = np.clip((mean - counts) / mean, 0.0, 1.0)
            else:
                debt = np.ones_like(counts)
        else:
            debt = np.ones(self.num_clients, dtype=np.float64)
        return debt

    def _update_bandwidth_budget(self) -> None:
        self.bw.budget_mb = self._base_bandwidth_budget * float(self.current_bw_factor)

    def _can_select_by_cooldown(self, cid: int, server_round: int) -> bool:
        return int(self.client_cooldown_until.get(cid, 0)) < int(server_round)

    def _apply_wsn_guard_relax(self) -> None:
        if self.wireless_model != "wsn":
            return
        enable = bool(self.wsn_guard_relax_cfg.get("enable", False))
        if not enable:
            return
        reserve_ratio_mul = float(self.wsn_guard_relax_cfg.get("reserve_ratio_multiplier", 1.0))
        future_factor_mul = float(self.wsn_guard_relax_cfg.get("future_factor_multiplier", 1.0))
        self.min_reserve_energy_ratio = max(0.0, self.min_reserve_energy_ratio * reserve_ratio_mul)
        self.future_energy_factor = max(0.1, self.future_energy_factor * future_factor_mul)
        if "max_consecutive_selected" in self.wsn_guard_relax_cfg:
            self.max_consecutive_selected = int(self.wsn_guard_relax_cfg.get("max_consecutive_selected", self.max_consecutive_selected))
        if "cooldown_rounds" in self.wsn_guard_relax_cfg:
            self.cooldown_rounds = int(self.wsn_guard_relax_cfg.get("cooldown_rounds", self.cooldown_rounds))

    def _effective_semi_sync_wait_ratio(self) -> float:
        ratio = self.semi_sync_wait_ratio
        if self.enable_hybrid_sync_boost and self.current_mode in ("semi_sync", "bridge"):
            ratio = self.hybrid_semi_sync_wait_ratio
        return float(np.clip(ratio, 0.5, 0.99))

    def _is_client_eligible_energy(self, cid: int, expected_round_energy: float) -> bool:
        rem = float(self.client_energy.get(cid, 0.0))
        if rem <= 0.0:
            return False
        init_e = float(self.initial_client_energy) if self.initial_client_energy > 0.0 else 1.0
        if rem / init_e < self.min_reserve_energy_ratio:
            return False
        if rem < self.future_energy_factor * max(expected_round_energy, 1e-12):
            return False
        return True

    def _compute_data_value_scores(self, active_cids: List[int], data_norm: np.ndarray) -> np.ndarray:
        out = np.zeros(self.num_clients, dtype=np.float64)
        counts = self._compute_selection_counts()
        max_count = float(counts.max()) if counts.size > 0 else 0.0
        rarity = np.ones(self.num_clients, dtype=np.float64)
        if max_count > 0.0:
            rarity = 1.0 - counts / max_count
        contrib = np.asarray([float(self.client_contribution_ema.get(cid, 0.0)) for cid in range(self.num_clients)], dtype=np.float64)
        cmin, cmax = float(contrib.min()), float(contrib.max())
        novelty = np.zeros(self.num_clients, dtype=np.float64)
        if cmax > cmin:
            novelty = (contrib - cmin) / (cmax - cmin)
        for cid in active_cids:
            out[cid] = self.novelty_w * novelty[cid] + self.tail_w * rarity[cid] + self.size_w * data_norm[cid]
        return out

    def _compute_historical_contribution_scores(self, active_cids: List[int], server_round: int) -> np.ndarray:
        out = np.zeros(self.num_clients, dtype=np.float64)
        if int(server_round) <= self.historical_warmup_rounds:
            return out
        vals = np.asarray([float(self.client_contribution_ema.get(cid, 0.0)) for cid in range(self.num_clients)], dtype=np.float64)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
        else:
            vals = np.zeros_like(vals)
        for cid in active_cids:
            out[cid] = vals[cid]
        return out

    def _update_selection_streaks(self, selected_cids: List[int], server_round: int) -> None:
        selected = set(selected_cids)
        for cid in range(self.num_clients):
            if cid in selected:
                self.client_selection_streak[cid] = int(self.client_selection_streak.get(cid, 0)) + 1
                if (
                    self.selection_policy == "bandwidth_first"
                    and self.bandwidth_first_penalty_enable
                    and self.bandwidth_first_forced_cooldown_rounds > 0
                    and self.client_selection_streak[cid] >= self.bandwidth_first_penalty_streak
                ):
                    self.client_cooldown_until[cid] = int(server_round) + int(max(1, self.bandwidth_first_forced_cooldown_rounds))
                    self.client_selection_streak[cid] = 0
                    continue
                if self.max_consecutive_selected > 0 and self.client_selection_streak[cid] >= self.max_consecutive_selected:
                    self.client_cooldown_until[cid] = int(server_round) + int(max(1, self.cooldown_rounds))
                    self.client_selection_streak[cid] = 0
            else:
                self.client_selection_streak[cid] = 0

    def _weighted_avg(self, pairs: List[Tuple[List[np.ndarray], float]], fallback: List[np.ndarray]) -> List[np.ndarray]:
        if not pairs:
            return fallback
        total = float(sum(w for _, w in pairs))
        out = [arr.copy() for arr in fallback]
        for i in range(len(out)):
            if np.issubdtype(out[i].dtype, np.floating):
                out[i] = np.zeros_like(out[i], dtype=np.float32)
        for params, weight in pairs:
            scale = weight / max(total, 1e-12)
            for i, arr in enumerate(params):
                if np.issubdtype(out[i].dtype, np.floating):
                    out[i] += scale * arr.astype(np.float32, copy=False)
        return out

    def _bridge_invariant_actions(
        self,
        *,
        round_energy: float,
        est_upload_time: float,
        stale_max: int,
        mean_fair_debt: float,
    ) -> Dict[str, bool]:
        actions = {"downweight": False, "throttle": False, "extend_bridge": False}
        if not self.bridge_inv_enable or self.current_mode != "bridge":
            self.prev_mean_fair_debt = mean_fair_debt
            return actions

        energy_budget = float(self.bridge_inv_cfg.get("energy_budget_round", 120.0))
        upload_budget = float(self.bridge_inv_cfg.get("upload_time_budget_round", 2.5))
        w_cfg = self.bridge_inv_cfg.get("violation_weights", {}) if isinstance(self.bridge_inv_cfg.get("violation_weights", {}), dict) else {}
        t_cfg = self.bridge_inv_cfg.get("thresholds", {}) if isinstance(self.bridge_inv_cfg.get("thresholds", {}), dict) else {}
        w_budget = float(w_cfg.get("budget", 0.5))
        w_stale = float(w_cfg.get("stale", 0.3))
        w_fair = float(w_cfg.get("fair", 0.2))
        th1 = float(t_cfg.get("th1", 0.10))
        th2 = float(t_cfg.get("th2", 0.25))
        th3 = float(t_cfg.get("th3", 0.45))

        v_budget = max(0.0, round_energy / max(1e-12, energy_budget) - 1.0) + max(
            0.0, est_upload_time / max(1e-12, upload_budget) - 1.0
        )
        v_stale = max(0.0, float(stale_max) / max(1.0, float(self.fedbuff.max_staleness)) - 1.0)
        prev = self.prev_mean_fair_debt if self.prev_mean_fair_debt is not None else mean_fair_debt
        trend = str(self.bridge_inv_cfg.get("fairness_debt_trend", "non_increasing")).lower()
        v_fair = max(0.0, mean_fair_debt - prev) if trend == "non_increasing" else 0.0

        v_total = w_budget * v_budget + w_stale * v_stale + w_fair * v_fair
        if v_total >= th1:
            actions["downweight"] = True
        if v_total >= th2:
            actions["throttle"] = True
        if v_total >= th3:
            actions["extend_bridge"] = True

        if any(actions.values()):
            self.bridge_fail_streak += 1
        else:
            self.bridge_fail_streak = 0

        if self.bridge_fail_streak >= int(self.bridge_inv_cfg.get("fail_streak_for_extend", 2)):
            actions["extend_bridge"] = True

        if actions["extend_bridge"]:
            extra = int(self.bridge_inv_cfg.get("bridge_extend_rounds", 1))
            max_bridge = int(self.bridge_inv_cfg.get("max_bridge_rounds", 8))
            self.controller.bridge_rounds = min(max_bridge, int(self.controller.bridge_rounds) + max(1, extra))

        if actions["throttle"]:
            rate_limit = float(self.bridge_inv_cfg.get("rate_limit_factor", 0.8))
            self.current_bw_factor = float(np.clip(self.current_bw_factor * rate_limit, 0.1, 1.0))

        print(
            f"[round {len(self.acc_history)+1}] bridge_invariants: "
            f"actions={actions}, energy={round_energy:.4f}, upload={est_upload_time:.4f}, "
            f"stale_max={stale_max}, mean_fair_debt={mean_fair_debt:.4f}, "
            f"bridge_rounds={self.controller.bridge_rounds}"
        )
        self.prev_mean_fair_debt = mean_fair_debt
        return actions

    def _decide_mode(self, server_round: int) -> Tuple[str, float, float]:
        if self.mode_policy == "semi_sync":
            return "semi_sync", 0.0, self.controller._bandwidth_factor()
        if self.mode_policy == "async":
            return "async", 0.0, self.controller._bandwidth_factor()
        if self.mode_policy == "bridge_free":
            gate = self.controller._gate_score()
            if gate >= self.controller.to_async + self.controller.hysteresis_margin:
                self.current_mode = "async"
            elif gate <= self.controller.to_semi - self.controller.hysteresis_margin:
                self.current_mode = "semi_sync"
            return self.current_mode, 0.0, self.controller._bandwidth_factor()
        return self.controller.decide(server_round)

    def _rank_clients(
        self,
        candidates: List[int],
        channel_score: np.ndarray,
        data_value: np.ndarray,
        hist_contrib_norm: np.ndarray,
        fairness_debt: np.ndarray,
        survival_norm: np.ndarray,
        bw_map_all: Dict[int, float],
    ) -> List[int]:
        if self.selection_policy == "bandwidth_first":
            times = {
                cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map_all.get(cid, 0.0)))
                for cid in candidates
            }
            return sorted(candidates, key=lambda cid: times[cid])
        if self.selection_policy == "energy_first":
            return sorted(candidates, key=lambda cid: survival_norm[cid], reverse=True)

        scores = {}
        for cid in candidates:
            quality = (
                self.channel_w * channel_score[cid]
                + self.data_w * data_value[cid]
                + self.fair_w * fairness_debt[cid]
                + self.energy_w * hist_contrib_norm[cid]
            )
            scores[cid] = quality * (max(0.0, survival_norm[cid]) ** self.survival_gamma)
        return sorted(candidates, key=lambda cid: scores[cid], reverse=True)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:  # type: ignore[override]
        del client_manager
        model = build_resnet18()
        ndarrays = get_parameters(model)
        self.global_params_cache = ndarrays_to_parameters(ndarrays)
        if self.algorithm == "scaffold":
            global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.scaffold_state = ScaffoldState(global_state)
        return self.global_params_cache

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):  # type: ignore[override]
        mode, bridge_w, bw_factor = self._decide_mode(server_round)
        self.current_mode = mode
        self.current_bridge_weight = bridge_w
        self.current_bw_factor = bw_factor
        self.mode_history.append(mode)

        top_k = self.selection_top_k if self.selection_top_k > 0 else max(1, int(round(self.num_clients * self.fraction_fit)))
        top_k = min(top_k, self.num_clients)

        available_clients = list(client_manager.all().values()) if hasattr(client_manager, "all") else []
        active_cids = [cid for cid in range(self.num_clients) if cid not in self.exhausted_clients and cid < len(available_clients)]
        if not active_cids:
            self.last_selected_cids = []
            return []

        if self.wireless_model == "wsn":
            if self.wsn_wireless_sampler is None:
                raise ValueError("wireless_model='wsn' but no WSN wireless sampler provided")
            self.current_wireless_stats = self.wsn_wireless_sampler.sample_round()
        else:
            self.current_wireless_stats = self.channel.sample_round()
        self._update_bandwidth_budget()

        data_sizes = np.asarray(self.partition_sizes, dtype=np.float64)
        if data_sizes.size < self.num_clients:
            data_sizes = np.pad(data_sizes, (0, self.num_clients - data_sizes.size), mode="edge")
        data_sizes = data_sizes[: self.num_clients]
        data_norm = data_sizes / max(float(data_sizes.max()), 1e-12)
        fairness_debt = self._compute_fairness_debt()

        bw_map_all = self.bw.allocate_by_stats(self.current_wireless_stats or {}, active_cids)
        channel_score = np.zeros(self.num_clients, dtype=np.float64)
        expected_energy_arr = np.zeros(self.num_clients, dtype=np.float64)

        for cid in active_cids:
            stats = (self.current_wireless_stats or {}).get(cid, {})
            per = float(stats.get("per", 0.0))
            channel_score[cid] = 1.0 - per
            tx_time = self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map_all.get(cid, 0.0)))
            expected_energy_arr[cid] = self.energy.comm_energy(tx_time) + self.energy.compute_energy(int(data_sizes[cid]))

        gated_cids: List[int] = []
        survival_norm = np.zeros(self.num_clients, dtype=np.float64)
        for cid in active_cids:
            if not self._can_select_by_cooldown(cid, server_round):
                continue
            if not self._is_client_eligible_energy(cid, float(expected_energy_arr[cid])):
                continue
            gated_cids.append(cid)
            rem = float(self.client_energy.get(cid, 0.0))
            survival_norm[cid] = np.clip(rem / max(1e-12, expected_energy_arr[cid]), 0.0, 1.0)

        if not gated_cids:
            self.last_selected_cids = []
            return []

        data_value = self._compute_data_value_scores(gated_cids, data_norm)
        hist_contrib_norm = self._compute_historical_contribution_scores(gated_cids, server_round)
        ranked = self._rank_clients(gated_cids, channel_score, data_value, hist_contrib_norm, fairness_debt, survival_norm, bw_map_all)
        selected_cids = ranked[: min(top_k, len(ranked))]
        self.last_selected_cids = selected_cids
        self.selection_window.append(list(selected_cids))
        self._update_selection_streaks(selected_cids, server_round)

        fit_cfg: Dict[str, Scalar] = {
            "server_round": float(server_round),
            "algorithm": self.algorithm,
            "fedprox_mu": float(self.fedprox_mu),
        }
        if self.algorithm == "scaffold" and self.scaffold_state is not None:
            fit_cfg["scaffold_c_global"] = pack_tensor_dict(self.scaffold_state.c_global)
        fit_ins = FitIns(parameters, fit_cfg)
        return [(available_clients[cid], fit_ins) for cid in selected_cids]

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[override]
        del failures
        if self.global_params_cache is None:
            return None, {}

        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)
        scheduled_cids = [cid for cid in self.last_selected_cids if cid not in self.exhausted_clients]

        if self.current_wireless_stats is not None:
            wireless_stats = self.current_wireless_stats
        elif self.wireless_model == "wsn":
            if self.wsn_wireless_sampler is None:
                raise ValueError("wireless_model='wsn' but no WSN wireless sampler provided")
            wireless_stats = self.wsn_wireless_sampler.sample_round()
        else:
            wireless_stats = self.channel.sample_round()

        per_values = [float((wireless_stats.get(cid) or {}).get("per", 0.0)) for cid in scheduled_cids]
        avg_per = float(np.mean(per_values)) if per_values else 0.0

        self.fedbuff.age(server_round)
        self._update_bandwidth_budget()
        bw_map = self.bw.allocate_by_stats(wireless_stats, scheduled_cids)
        tx_times = {
            cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0)))
            for cid in scheduled_cids
        }
        print(f"[round {server_round}] bandwidth_allocation_mb={{{', '.join(f'{cid}:{bw_map.get(cid, 0.0):.4f}' for cid in scheduled_cids)}}}")
        print(f"[round {server_round}] upload_time_sec={{{', '.join(f'{cid}:{tx_times.get(cid, 0.0):.4f}' for cid in scheduled_cids)}}}")
        est_upload_time = float(sum(tx_times.values())) if tx_times else 0.0

        threshold = float("inf")
        if scheduled_cids and self.current_mode in ("semi_sync", "bridge"):
            tx_list = [tx_times[cid] for cid in scheduled_cids]
            threshold = float(np.quantile(np.asarray(tx_list, dtype=np.float64), self._effective_semi_sync_wait_ratio()))

        candidate_updates: List[Dict[str, object]] = []
        round_energy = 0.0
        for _, fit_res in results:
            cid = int(fit_res.metrics.get("cid", -1))
            if cid < 0 or cid >= self.num_clients or cid in self.exhausted_clients:
                continue

            num_examples = int(fit_res.num_examples)
            if num_examples <= 0:
                continue

            tx_time = tx_times.get(cid, self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0))))
            per = float((wireless_stats.get(cid, {})).get("per", 0.0))
            client_round_energy = self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)
            if (
                self.selection_policy == "bandwidth_first"
                and self.bandwidth_first_penalty_enable
                and self.client_selection_streak.get(cid, 0) >= self.bandwidth_first_penalty_streak
            ):
                client_round_energy *= max(1.0, self.bandwidth_first_energy_multiplier)
            round_energy += client_round_energy
            remaining = float(self.client_energy.get(cid, 0.0)) - float(client_round_energy)
            clamped_remaining = max(self.min_energy_floor_per_round, remaining)
            self.client_energy[cid] = max(0.0, clamped_remaining)
            if clamped_remaining <= 0.0:
                self.exhausted_clients.add(cid)
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight = float(num_examples)
            payload = fit_res.metrics.get("delta_ci", b"")
            candidate_updates.append(
                {
                    "cid": cid,
                    "num_examples": num_examples,
                    "tx_time": tx_time,
                    "per": per,
                    "ndarrays": ndarrays,
                    "weight": weight,
                    "delta_ci_payload": payload,
                }
            )

        stale_max = max([int(s) for _, s, _ in self.fedbuff.entries], default=0)
        mean_fair_debt = float(np.mean(self._compute_fairness_debt()))
        actions = self._bridge_invariant_actions(
            round_energy=round_energy,
            est_upload_time=est_upload_time,
            stale_max=stale_max,
            mean_fair_debt=mean_fair_debt,
        )

        risk_scores: Dict[int, float] = {}
        if candidate_updates:
            max_tx = max(float(u["tx_time"]) for u in candidate_updates)
            for u in candidate_updates:
                cid = int(u["cid"])
                tx_norm = float(u["tx_time"]) / max(1e-12, max_tx)
                per_v = float(u["per"])
                energy_rem = float(self.client_energy.get(cid, 0.0))
                energy_risk = 1.0 - float(np.clip(energy_rem / max(1e-12, self.initial_client_energy), 0.0, 1.0))
                risk_scores[cid] = 0.45 * tx_norm + 0.35 * per_v + 0.20 * energy_risk

        throttle_drop: set[int] = set()
        if actions["throttle"] and candidate_updates:
            ratio = float(self.bridge_inv_cfg.get("throttle_drop_ratio", 0.3))
            drop_n = max(1, int(round(len(candidate_updates) * np.clip(ratio, 0.0, 0.9))))
            ranked = sorted(candidate_updates, key=lambda u: risk_scores.get(int(u["cid"]), 0.0), reverse=True)
            throttle_drop = {int(u["cid"]) for u in ranked[:drop_n]}

        valid_sync_updates: List[Tuple[List[np.ndarray], float]] = []
        contrib_updates: List[Tuple[int, float, float]] = []
        scaffold_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        scaffold_weights: Dict[int, float] = {}
        for u in candidate_updates:
            cid = int(u["cid"])
            if cid in throttle_drop:
                continue
            if np.random.rand() < float(u["per"]):
                if self.current_mode == "semi_sync":
                    continue
                else:
                    # 4 次重传
                    lost_package_flag = 1
                    for _ in range(4):
                        lost_package_flag *= np.random.rand() < float(u["per"])
                    if lost_package_flag:
                        continue

            weight = float(u["weight"])
            if actions["downweight"]:
                factor = float(self.bridge_inv_cfg.get("downweight_factor", 0.5))
                r = float(risk_scores.get(cid, 0.0))
                weight = weight * (1.0 - (1.0 - factor) * np.clip(r, 0.0, 1.0))

            if self.algorithm == "scaffold" and self.scaffold_state is not None:
                payload = u["delta_ci_payload"]
                if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
                    delta_ci = unpack_tensor_dict(bytes(payload), torch.device("cpu"))
                    if delta_ci:
                        scaffold_deltas[cid] = delta_ci
                        scaffold_weights[cid] = weight

            self.fedbuff.push(u["ndarrays"], int(u["num_examples"]))
            if self.current_mode in ("semi_sync", "bridge") and float(u["tx_time"]) > threshold:
                continue
            valid_sync_updates.append((u["ndarrays"], weight))
            contrib_updates.append((cid, weight, float(u["tx_time"])))

        sync_result = self._weighted_avg(valid_sync_updates, global_ndarrays) if valid_sync_updates else global_ndarrays
        async_result = global_ndarrays
        if self.fedbuff.should_aggregate(server_round):
            async_result = self.fedbuff.aggregate(global_ndarrays, server_round)

        if self.current_mode == "async":
            merged = async_result
        elif self.current_mode == "semi_sync":
            merged = sync_result
        else:
            w = float(np.clip(self.current_bridge_weight, 0.0, 1.0))
            merged = []
            for s, a in zip(sync_result, async_result):
                if np.issubdtype(s.dtype, np.floating):
                    merged.append((1.0 - w) * s.astype(np.float32, copy=False) + w * a.astype(np.float32, copy=False))
                else:
                    merged.append(s)

        self.global_params_cache = ndarrays_to_parameters(merged)

        if self.algorithm == "scaffold" and self.scaffold_state is not None and scaffold_deltas:
            self.scaffold_state.update_global(scaffold_deltas, scaffold_weights)

        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))
        round_acc_gain = max(0.0, float(acc) - float(self.prev_round_accuracy))
        total_w = float(sum(w for _, w, _ in contrib_updates))
        for cid, w, tx in contrib_updates:
            share = round_acc_gain * (w / max(1e-12, total_w))
            est_cost = self.energy.comm_energy(tx) + self.energy.compute_energy(int(max(1.0, w)))
            utility = share / max(1e-12, est_cost)
            old = float(self.client_contribution_ema.get(cid, 0.0))
            self.client_contribution_ema[cid] = self.contribution_ema_beta * old + (1.0 - self.contribution_ema_beta) * utility
        self.prev_round_accuracy = float(acc)

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

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):  # type: ignore[override]
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


class SyncStrategy(HybridWirelessStrategy):
    def __init__(self, cfg: Dict[str, Any], partition_sizes: List[int], testloader: DataLoader, wsn_wireless_sampler=None) -> None:
        cfg = dict(cfg)
        controller = dict(cfg.get("controller", {}))
        controller["mode_policy"] = "semi_sync"
        controller["enable_hybrid_sync_boost"] = False
        cfg["controller"] = controller
        algorithm = dict(cfg.get("algorithm", {}))
        algorithm.setdefault("name", "fedavg")
        cfg["algorithm"] = algorithm
        super().__init__(cfg, partition_sizes, testloader, wsn_wireless_sampler)


class AsyncStrategy(HybridWirelessStrategy):
    def __init__(self, cfg: Dict[str, Any], partition_sizes: List[int], testloader: DataLoader, wsn_wireless_sampler=None) -> None:
        cfg = dict(cfg)
        controller = dict(cfg.get("controller", {}))
        controller["mode_policy"] = "async"
        cfg["controller"] = controller
        super().__init__(cfg, partition_sizes, testloader, wsn_wireless_sampler)


class BridgeFreeStrategy(HybridWirelessStrategy):
    def __init__(self, cfg: Dict[str, Any], partition_sizes: List[int], testloader: DataLoader, wsn_wireless_sampler=None) -> None:
        cfg = dict(cfg)
        controller = dict(cfg.get("controller", {}))
        controller["mode_policy"] = "bridge_free"
        cfg["controller"] = controller
        super().__init__(cfg, partition_sizes, testloader, wsn_wireless_sampler)


class BandwidthFirstStrategy(HybridWirelessStrategy):
    def __init__(self, cfg: Dict[str, Any], partition_sizes: List[int], testloader: DataLoader, wsn_wireless_sampler=None) -> None:
        cfg = dict(cfg)
        scheduler = dict(cfg.get("scheduler", {}))
        scheduler["selection_policy"] = "bandwidth_first"
        cfg["scheduler"] = scheduler
        super().__init__(cfg, partition_sizes, testloader, wsn_wireless_sampler)


class EnergyFirstStrategy(HybridWirelessStrategy):
    def __init__(self, cfg: Dict[str, Any], partition_sizes: List[int], testloader: DataLoader, wsn_wireless_sampler=None) -> None:
        cfg = dict(cfg)
        scheduler = dict(cfg.get("scheduler", {}))
        scheduler["selection_policy"] = "energy_first"
        cfg["scheduler"] = scheduler
        super().__init__(cfg, partition_sizes, testloader, wsn_wireless_sampler)
