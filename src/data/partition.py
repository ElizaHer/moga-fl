from typing import Dict, List
import numpy as np
from collections import defaultdict
from fedlab.utils.dataset import CIFAR10Partitioner, MNISTPartitioner


def cifar_iid_partitions(train_dataset, num_clients=10):
    partitioner = CIFAR10Partitioner(
        targets=np.array(train_dataset.targets),  # 传入标签数组
        num_clients=num_clients,
        partition="iid",  # 指定IID分区
        seed=42,  # 固定随机种子，保证结果可复现
        verbose=False
    )

    return partitioner.client_dict  # 返回 {client_id: [sample_indices]}


def cifar_dirichlet_partitions(train_dataset, num_clients=10, alpha=0.5):
    non_iid_partitioner = CIFAR10Partitioner(
        train_dataset.targets,
        num_clients,
        balance=None,
        partition="dirichlet",
        dir_alpha=alpha,
        verbose=False,
        seed=42)
    return non_iid_partitioner.client_dict


def cifar_shards_partitions(train_dataset, num_clients=10):
    non_iid_partitioner = CIFAR10Partitioner(
        targets=np.array(train_dataset.targets),
        num_clients=num_clients,
        balance=None,
        partition="shards",
        num_shards=200,
        seed=42,
        verbose=False
    )

    return non_iid_partitioner.client_dict


def mnist_iid_partitions(train_dataset, num_clients=10):
    partitioner = MNISTPartitioner(
        targets=np.array(train_dataset.targets),  # 传入标签数组
        num_clients=num_clients,
        partition="iid",  # 指定IID分区
        seed=42,  # 固定随机种子，保证结果可复现
        verbose=False
    )

    return partitioner.client_dict  # 返回 {client_id: [sample_indices]}


def mnist_dirichlet_partitions(train_dataset, num_clients=10, alpha=0.5):
    non_iid_partitioner = MNISTPartitioner(
        train_dataset.targets,
        num_clients,
        partition="noniid-labeldir",
        dir_alpha=alpha,
        verbose=False,
        seed=42)
    return non_iid_partitioner.client_dict


def dirichlet_partition(labels: np.ndarray, num_clients: int, num_classes: int, alpha: float) -> Dict[int, List[int]]:
    idx_by_class = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_class[int(y)].append(i)
    client_indices = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        idxs = np.array(idx_by_class[c])
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (proportions / proportions.sum())
        splits = (proportions * len(idxs)).astype(int)
        # Adjust to sum exactly
        diff = len(idxs) - splits.sum()
        for i in range(diff):
            splits[i % num_clients] += 1
        start = 0
        for cl in range(num_clients):
            part = idxs[start:start + splits[cl]].tolist()
            client_indices[cl].extend(part)
            start += splits[cl]
    return client_indices


def label_bias_partition(labels: np.ndarray, num_clients: int, num_classes: int, classes_per_client: int) -> Dict[
    int, List[int]]:
    idx_by_class = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_class[int(y)].append(i)
    client_indices = {i: [] for i in range(num_clients)}
    for cl in range(num_clients):
        chosen = np.random.choice(num_classes, size=classes_per_client, replace=False)
        for c in chosen:
            pool = idx_by_class[c]
            take = max(1, int(len(pool) / num_clients))
            client_indices[cl].extend(pool[:take])
    return client_indices


def quantity_bias_partition(labels: np.ndarray, num_clients: int, sigma: float) -> Dict[int, List[int]]:
    n = len(labels)
    weights = np.random.lognormal(mean=0.0, sigma=sigma, size=num_clients)
    weights = weights / weights.sum()
    splits = (weights * n).astype(int)
    diff = n - splits.sum()
    for i in range(diff):
        splits[i % num_clients] += 1
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    client_indices = {}
    start = 0
    for cl in range(num_clients):
        part = all_indices[start:start + splits[cl]].tolist()
        client_indices[cl] = part
        start += splits[cl]
    return client_indices


def apply_quick_limit(client_indices: Dict[int, List[int]], samples_per_client: int) -> Dict[int, List[int]]:
    return {cl: inds[:samples_per_client] for cl, inds in client_indices.items()}
