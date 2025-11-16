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
    cfg.setdefault('clients', {})
    cfg['clients'].setdefault('num_clients', 10)
    cfg['clients'].setdefault('selection_top_k', max(1, cfg['clients']['num_clients']//2))
    cfg.setdefault('wireless', {})
    cfg.setdefault('training', {})
    cfg.setdefault('compression', {})
    cfg.setdefault('scheduling', {})
    cfg['scheduling'].setdefault('weights', {
        'energy': 0.25, 'channel': 0.25, 'data_value': 0.25, 'fairness_debt': 0.2, 'bandwidth_cost': -0.15
    })
    cfg['scheduling'].setdefault('fairness_ledger', {
        'debt_increase': 0.05, 'repay_rate': 0.1, 'max_debt': 1.0
    })
    cfg.setdefault('eval', {'rounds': 5, 'test_interval': 1, 'seed': 42, 'save_csv': True, 'save_plots': True, 'preference': 'time'})
    return cfg
