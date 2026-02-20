from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset


def make_loader(dataset, indices: Sequence[int], batch_size: int) -> DataLoader:
    """Create a shuffled DataLoader over a subset of a dataset."""
    subset = Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate classification accuracy of a model on given DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)
