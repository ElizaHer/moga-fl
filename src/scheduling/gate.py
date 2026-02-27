from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


class ModeController:
    """Windowed mode controller returning (mode, bridge_weight, bandwidth_factor)."""

    def __init__(self, cfg: Dict[str, Any], num_clients: int) -> None:
        self.num_clients = num_clients

        controller = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
        gates = controller.get("gate_thresholds", {}) if isinstance(controller.get("gate_thresholds", {}), dict) else {}
        gate_weights = controller.get("gate_weights", {}) if isinstance(controller.get("gate_weights", {}), dict) else {}
        bw_cfg = controller.get("bandwidth_rebalance", {}) if isinstance(controller.get("bandwidth_rebalance", {}), dict) else {}

        self.window_size = int(controller.get("window_size", 4))
        self.to_async = float(gates.get("to_async", 0.58))
        self.to_semi = float(gates.get("to_semi_sync", 0.42))
        self.hysteresis_margin = float(controller.get("hysteresis_margin", 0.03))
        self.bridge_rounds = int(controller.get("bridge_rounds", 2))
        self.min_rounds_between_switch = int(controller.get("min_rounds_between_switch", 2))

        self.w_per = float(gate_weights.get("per", 0.5))
        self.w_fair = float(gate_weights.get("fairness", 0.3))
        self.w_energy = float(gate_weights.get("energy", 0.2))

        self.bw_low = float(bw_cfg.get("low_energy_factor", 0.8))
        self.bw_high = float(bw_cfg.get("high_energy_factor", 1.0))

        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.current_mode: str = "semi_sync"
        self.bridge_target: Optional[str] = None
        self.bridge_start: int = -1
        self.last_switch_round: int = -10**9

    def register(self, avg_per: float, jain: float, total_energy: float) -> None:
        self.history.append({"avg_per": float(avg_per), "jain": float(jain), "energy": float(total_energy)})

    def _gate_score(self) -> float:
        if not self.history:
            return 0.0
        avg_per = float(np.mean([row["avg_per"] for row in self.history]))
        jain = float(np.mean([row["jain"] for row in self.history]))
        energies = [row["energy"] for row in self.history]
        mean_energy = float(np.mean(energies))
        max_energy = float(max(energies)) if energies else 0.0
        energy_norm = float(np.clip(mean_energy / max_energy, 0.0, 1.0)) if max_energy > 0.0 else 0.0
        fairness_deficit = 1.0 - jain
        score = self.w_per * avg_per + self.w_fair * fairness_deficit + self.w_energy * energy_norm
        print(f"avg_per: {avg_per:.4f}, fairness: {fairness_deficit:.4f}, energy: {energy_norm:.4f}, score: {score:.4f}")
        return float(score)

    def _bandwidth_factor(self) -> float:
        if not self.history:
            return self.bw_high
        energies = [row["energy"] for row in self.history]
        mean_energy = float(np.mean(energies))
        max_energy = float(max(energies)) if energies else 0.0
        energy_norm = float(np.clip(mean_energy / max_energy, 0.0, 1.0)) if max_energy > 0.0 else 0.0
        factor = self.bw_high - (self.bw_high - self.bw_low) * energy_norm
        return float(np.clip(factor, self.bw_low, self.bw_high))

    def decide(self, server_round: int) -> Tuple[str, float, float]:
        bw_factor = self._bandwidth_factor()

        if self.current_mode == "bridge":
            t = max(0, server_round - self.bridge_start)
            w = float(np.clip(t / max(1, self.bridge_rounds), 0.0, 1.0))
            if t >= self.bridge_rounds and self.bridge_target is not None:
                self.current_mode = self.bridge_target
                self.bridge_target = None
                self.bridge_start = -1
            return "bridge", w, bw_factor

        if server_round - self.last_switch_round < self.min_rounds_between_switch:
            return self.current_mode, 0.0, bw_factor

        gate = self._gate_score()
        target = self.current_mode
        if gate >= self.to_async + self.hysteresis_margin:
            target = "async"
        elif gate <= self.to_semi - self.hysteresis_margin:
            target = "semi_sync"

        if target != self.current_mode:
            self.current_mode = "bridge"
            self.bridge_target = target
            self.bridge_start = server_round
            self.last_switch_round = server_round
            return "bridge", 0.0, bw_factor

        return self.current_mode, 0.0, bw_factor
