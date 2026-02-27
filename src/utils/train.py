from __future__ import annotations

import argparse
import csv
import datetime
import io
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from flwr.common import NDArrays
from torch.utils.data import DataLoader


def get_parameters(model: nn.Module) -> NDArrays:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return float(avg_loss), float(accuracy)


def global_state_from_ndarrays(model: nn.Module, parameters: NDArrays, device: torch.device) -> Dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    return {
        k: torch.tensor(v, device=device)
        for k, v in zip(keys, parameters)
    }


def pack_tensor_dict(tensors: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in tensors.items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def unpack_tensor_dict(payload: bytes, device: torch.device) -> Dict[str, torch.Tensor]:
    if not payload:
        return {}
    buffer = io.BytesIO(payload)
    obj = torch.load(buffer, map_location=device)
    return {k: v.to(device) for k, v in obj.items()}


def jain_index(values: List[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    return float((x.sum() ** 2) / (len(x) * np.sum(x * x) + 1e-12))
