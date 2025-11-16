from typing import Dict, Any
import torch

class ErrorFeedbackMemory:
    def __init__(self):
        self.mem = {}

    def get(self, k):
        return self.mem.get(k, None)

    def set(self, k, v):
        self.mem[k] = v


def topk_sparsify(state_dict: Dict[str, Any], fraction: float, efm: ErrorFeedbackMemory, key: str):
    # Flatten and select top-k entries (by magnitude)
    flat = []
    shapes = {}
    for k, v in state_dict.items():
        t = torch.tensor(v).float().view(-1)
        shapes[k] = v.shape if hasattr(v, 'shape') else None
        flat.append(t)
    vec = torch.cat(flat)
    k = max(1, int(len(vec) * fraction))
    topk_vals, topk_idx = torch.topk(vec.abs(), k)
    mask = torch.zeros_like(vec)
    mask[topk_idx] = 1.0
    sparse_vec = vec * mask
    # Error feedback
    resid = vec - sparse_vec
    efm.set(key, resid)
    # Reconstruct per key (simplified: uniform split)
    out = {}
    offset = 0
    for name, v in state_dict.items():
        t = torch.tensor(v).float().view(-1)
        seg = sparse_vec[offset:offset + t.numel()]
        out[name] = seg.view(*t.shape)
        offset += t.numel()
    return out
