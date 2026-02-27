from typing import Tuple, Dict, Any
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset

from src.configs.hybrid_cifar import HybridCifarConfig
from src.data.partition import dirichlet_partition_indices


def cifar_loader(cfg: HybridCifarConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_transform = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
    raw_trainset = CIFAR10(root=cfg.data_dir, train=True, download=False, transform=ToTensor())
    testset = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)

    labels = np.array(raw_trainset.targets)
    partitions = dirichlet_partition_indices(labels, cfg.num_clients, cfg.alpha, cfg.seed)

    trainloaders = [
        DataLoader(
            Subset(trainset, indices),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        for indices in partitions
    ]
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    partition_sizes = [len(idx) for idx in partitions]

    return trainloaders, testloader, partition_sizes


class WSNWirelessSampler:
    """Provide per-round wireless stats from WSN dataset rows."""

    def __init__(self, per_client_rows: Dict[int, np.ndarray], seed: int = 42):
        self.per_client_rows = per_client_rows
        self.rng = np.random.default_rng(seed)

    def sample_round(self) -> Dict[int, Dict[str, float]]:
        stats: Dict[int, Dict[str, float]] = {}
        for cid, rows in self.per_client_rows.items():
            if rows.shape[0] == 0:
                stats[cid] = {"snr_db": 0.0, "snr_lin": 1.0, "per": 0.5}
                continue
            idx = int(self.rng.integers(0, rows.shape[0]))
            snr_db = float(rows[idx, 0])
            per = float(np.clip(rows[idx, 1], 0.0, 1.0))
            stats[cid] = {
                "snr_db": snr_db,
                "snr_lin": float(10.0 ** (snr_db / 10.0)),
                "per": per,
            }
        return stats


def build_wsn_wireless_sampler(cfg: HybridCifarConfig) -> WSNWirelessSampler:
    """Load WSN CSV and build client-wise wireless sampler.

    CSV supports either:
    - direct columns: snr + per
    - or inferred columns: rssi + noise_floor (+ prr or per)
    """
    df = pd.read_csv(cfg.wsn_csv_path)
    cols = {c.lower(): c for c in df.columns}

    def _pick(name: str) -> str | None:
        return cols.get(name.lower())

    snr_col = _pick(cfg.wsn_snr_col)
    if snr_col is None:
        rssi_col = _pick(cfg.wsn_rssi_col)
        noise_col = _pick(cfg.wsn_noise_col)
        if rssi_col is None or noise_col is None:
            raise ValueError(
                f"WSN CSV must contain either '{cfg.wsn_snr_col}' or "
                f"('{cfg.wsn_rssi_col}','{cfg.wsn_noise_col}')"
            )
        df["_snr_db"] = pd.to_numeric(df[rssi_col], errors="coerce") - pd.to_numeric(df[noise_col], errors="coerce")
        snr_col = "_snr_db"

    per_col = _pick(cfg.wsn_per_col)
    if per_col is None:
        prr_col = _pick(cfg.wsn_prr_col)
        if prr_col is None:
            raise ValueError(
                f"WSN CSV must contain either '{cfg.wsn_per_col}' or '{cfg.wsn_prr_col}'"
            )
        df["_per"] = 1.0 - pd.to_numeric(df[prr_col], errors="coerce")
        per_col = "_per"

    arr = df[[snr_col, per_col]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError("WSN CSV produced empty valid rows for snr/per")

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(arr)
    splits = np.array_split(arr, cfg.num_clients)
    per_client_rows = {cid: splits[cid] for cid in range(cfg.num_clients)}
    return WSNWirelessSampler(per_client_rows=per_client_rows, seed=cfg.seed)


class DatasetManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root = cfg['dataset'].get('root', 'data')
        self.name = cfg['dataset']['name']
        self.download = cfg['dataset'].get('download', True)
        self.use_fake_if_unavailable = cfg['dataset'].get('use_fake_if_unavailable', True)
        self.split = cfg['dataset'].get('split', 'balanced')

    def get_transforms(self):
        if self.name == 'cifar10':
            tf_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            tf_test = tf_train
        elif self.name == 'emnist':
            tf_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            tf_test = tf_train
        else:
            tf_train = transforms.ToTensor()
            tf_test = transforms.ToTensor()
        return tf_train, tf_test

    def load(self) -> Tuple[Dataset, Dataset, int]:
        tf_train, tf_test = self.get_transforms()
        num_classes = 10
        try:
            if self.name == 'cifar10':
                train = datasets.CIFAR10(self.root, train=True, download=self.download, transform=tf_train)
                test = datasets.CIFAR10(self.root, train=False, download=self.download, transform=tf_test)
                num_classes = 10
            elif self.name == 'emnist':
                train = datasets.EMNIST(self.root, split=self.split, train=True, download=self.download, transform=tf_train)
                test = datasets.EMNIST(self.root, split=self.split, train=False, download=self.download, transform=tf_test)
                num_classes = 47 if self.split == 'balanced' else 26
            else:
                raise RuntimeError('Unsupported dataset: ' + self.name)
        except Exception as e:
            if self.use_fake_if_unavailable:
                print('[WARN] Dataset unavailable, falling back to FakeData for quick run:', e)
                size = self.cfg['dataset'].get('quick_samples', 4096) or 4096
                num_classes = 10 if self.name == 'cifar10' else (47 if self.split == 'balanced' else 26)
                train = datasets.FakeData(size=size, image_size=(3, 32, 32), num_classes=num_classes, transform=tf_train)
                test = datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=num_classes, transform=tf_test)
            else:
                raise
        return train, test, num_classes
