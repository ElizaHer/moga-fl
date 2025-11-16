import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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
