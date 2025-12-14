from typing import Dict, Any
import torch


def quantize_8bit(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    q = {}
    for k, v in state_dict.items():
        t = torch.tensor(v)
        q[k] = (t.float().clamp(-1e3, 1e3) * 127.0 / (t.abs().max().item() + 1e-6)).round().to(torch.int8)
    return q


def dequantize_8bit(qdict: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: naive dequantization assuming scale=1
    dq = {}
    for k, v in qdict.items():
        t = torch.tensor(v).float()
        dq[k] = t
    return dq
