import yaml
from dataclasses import dataclass
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Provide minimal defaults to avoid KeyError
    cfg.setdefault('dataset', {})
    cfg['dataset'].setdefault('root', 'data')

    # 模型相关默认配置：支持 small_cnn / resnet18_cifar / resnet18_emnist
    cfg.setdefault('model', {})
    cfg['model'].setdefault('type', 'small_cnn')
    cfg['model'].setdefault('width_factor', 1.0)

    cfg.setdefault('clients', {})
    cfg['clients'].setdefault('num_clients', 10)
    cfg['clients'].setdefault('selection_top_k', max(1, cfg['clients']['num_clients']//2))

    cfg.setdefault('wireless', {})
    cfg['wireless'].setdefault('channel_model', 'rayleigh_siso')
    cfg['wireless'].setdefault('carrier_ghz', 3.5)

    cfg.setdefault('training', {})
    # 聚合与同步模式：sync / semi_sync / async，默认保持向后兼容
    if 'sync_mode' not in cfg['training']:
        # 若旧配置使用了 boolean sync，则据此推断初始模式
        if cfg['training'].get('sync', True):
            cfg['training']['sync_mode'] = 'semi_sync'
        else:
            cfg['training']['sync_mode'] = 'async'
    # 训练算法：fedavg / fedprox / scaffold
    cfg['training'].setdefault('algorithm', 'fedavg')

    cfg.setdefault('compression', {})
    cfg.setdefault('scheduling', {})
    cfg['scheduling'].setdefault('weights', {
        'energy': 0.25, 'channel': 0.25, 'data_value': 0.25, 'fairness_debt': 0.2, 'bandwidth_cost': -0.15
    })
    cfg['scheduling'].setdefault('fairness_ledger', {
        'debt_increase': 0.05, 'repay_rate': 0.1, 'max_debt': 1.0
    })
    # 策略控制器默认参数（用于问题4：策略切换与桥接态）
    strat = cfg.setdefault('strategy', {})
    strat.setdefault('window_size', 5)
    strat.setdefault('gate_thresholds', {'to_async': 0.6, 'to_semi_sync': 0.4})
    strat.setdefault('hysteresis_margin', 0.05)
    strat.setdefault('bridge_rounds', 3)
    strat.setdefault('min_rounds_between_switch', 5)
    strat.setdefault('weights', {'per': 0.5, 'fairness': 0.3, 'energy': 0.2})
    strat.setdefault('bandwidth_rebalance', {'low_energy_factor': 0.8, 'high_energy_factor': 1.0})

    cfg.setdefault('eval', {'rounds': 5, 'test_interval': 1, 'seed': 42, 'save_csv': True, 'save_plots': True, 'preference': 'time'})
    return cfg
