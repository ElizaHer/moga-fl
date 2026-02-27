from __future__ import annotations

from typing import Any, Dict, List

from torch.utils.data import DataLoader

from .hybrid_wireless import (
    AsyncStrategy,
    BandwidthFirstStrategy,
    BridgeFreeStrategy,
    EnergyFirstStrategy,
    HybridWirelessStrategy,
    SyncStrategy,
)


def build_strategy(
    strategy_name: str,
    cfg: Dict[str, Any],
    partition_sizes: List[int],
    testloader: DataLoader,
    wsn_wireless_sampler=None,
):
    key = strategy_name.lower()
    if key in ("sync", "syncstrategy"):
        return SyncStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
    if key in ("async", "asyncstrategy"):
        return AsyncStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
    if key in ("bridge_free", "bridgefreestrategy"):
        return BridgeFreeStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
    if key in ("bandwidth_first", "bandwidthfirststrategy"):
        return BandwidthFirstStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
    if key in ("energy_first", "energyfirststrategy"):
        return EnergyFirstStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
    return HybridWirelessStrategy(cfg, partition_sizes, testloader, wsn_wireless_sampler=wsn_wireless_sampler)
