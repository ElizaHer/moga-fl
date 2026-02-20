from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset


@dataclass
class SyntheticConfig:
    num_samples: int = 2000
    num_clients: int = 10
    num_classes: int = 10
    image_size: Tuple[int, int] = (28, 28)


def make_synthetic_dataset(cfg: SyntheticConfig) -> Tuple[Dataset, Dataset, int]:
    """生成一个可复现实验用的合成分类数据集。

    - 特征：简单的 1×H×W 灰度“图像”；
    - 标签：均匀采样的 num_classes;
    - 训练 / 测试按 8:2 划分。
    """

    rng = np.random.default_rng(0)
    num_train = int(cfg.num_samples * 0.8)
    num_test = cfg.num_samples - num_train

    def _make_part(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = cfg.image_size
        # 随机噪声 + 类别中心偏移，保证任务可学习
        labels = rng.integers(0, cfg.num_classes, size=n, endpoint=False)
        base = rng.normal(0.0, 1.0, size=(n, 1, h, w)).astype("float32")
        centers = rng.normal(0.0, 1.0, size=(cfg.num_classes, 1, 1, 1)).astype("float32")
        images = base + centers[labels]
        return torch.from_numpy(images), torch.from_numpy(labels.astype("int64"))

    x_train, y_train = _make_part(num_train)
    x_test, y_test = _make_part(num_test)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    return train_ds, test_ds, cfg.num_classes


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float) -> Dict[int, List[int]]:
    """基于 Dirichlet 分布的 label-wise non-IID 划分。

    实现思路与常见 FL 基准一致：
    对于每个类别 c，从 Dir(alpha) 采样长度为 num_clients 的向量，
    按比例将该类样本分配给各客户端。
    """

    n_classes = int(labels.max()) + 1
    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}
    rng = np.random.default_rng(0)

    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        # 采样类别在客户端之间的分配比例
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # 按比例切分索引
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())

    # 打乱每个客户端内部顺序
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def build_federated_dataloaders(
    train_dataset: Dataset,
    client_indices: Dict[int, List[int]],
    batch_size: int,
) -> Dict[int, DataLoader]:
    """根据划分结果为每个客户端构造 DataLoader。"""

    loaders: Dict[int, DataLoader] = {}
    for cid, idxs in client_indices.items():
        subset = Subset(train_dataset, idxs)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders
