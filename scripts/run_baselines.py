import argparse
import os
import numpy as np
import torch
from src.configs.config import load_config, merge_defaults
from src.utils.seed import set_seed
from src.data.dataset_loader import DatasetManager
from src.data.partition import dirichlet_partition, label_bias_partition, quantity_bias_partition, apply_quick_limit
from src.training.models.cifar_cnn import SimpleCIFARCNN
from src.training.models.emnist_cnn import SimpleEMNISTCNN
from src.training.models.resnet import build_resnet18
from src.training.models.resnet_emnist import build_resnet18_emnist
from src.training.server import Server
from src.eval.metrics import MetricsRecorder
from src.eval.plot import plot_curves


def build_partitions(train_dataset, cfg, num_classes):
    labels = np.array([y for _, y in train_dataset])
    num_clients = cfg['clients']['num_clients']
    niid = cfg['dataset']['noniid']
    if niid['type'] == 'dirichlet':
        client_indices = dirichlet_partition(labels, num_clients, num_classes, niid.get('alpha', 0.5))
    elif niid['type'] == 'label_bias':
        cpc = niid.get('classes_per_client', niid.get('label_bias_classes_per_client', 2))
        client_indices = label_bias_partition(labels, num_clients, num_classes, cpc)
    else:
        client_indices = quantity_bias_partition(labels, num_clients, niid.get('quantity_bias_sigma', 0.6))
    # quick limit
    spc = cfg['clients'].get('samples_per_client')
    if spc:
        client_indices = apply_quick_limit(client_indices, spc)
    return client_indices


def build_model_fn(cfg, num_classes):
    """根据配置选择模型结构：small_cnn / resnet18_cifar / resnet18_emnist。

    - small_cnn：保持原有 SimpleCIFARCNN / SimpleEMNISTCNN，便于快速实验；
    - resnet18_cifar：CIFAR-10 专用 ResNet-18 变体；
    - resnet18_emnist：EMNIST 专用 ResNet-18 变体。
    """
    model_cfg = cfg.get('model', {})
    mtype = model_cfg.get('type', 'small_cnn')
    width_factor = float(model_cfg.get('width_factor', 1.0))
    dataset_name = cfg['dataset']['name']

    if mtype == 'resnet18_cifar' and dataset_name == 'cifar10':
        return lambda: build_resnet18(num_classes=num_classes, width_factor=width_factor)
    if mtype == 'resnet18_emnist' and dataset_name == 'emnist':
        # EMNIST Balanced 有 47 类，其它 split 可通过 num_classes 传入
        return lambda: build_resnet18_emnist(num_classes=num_classes, width_factor=width_factor)

    # 回退到 small_cnn（保持向后兼容）
    if dataset_name == 'cifar10':
        return SimpleCIFARCNN
    else:
        return lambda: SimpleEMNISTCNN(num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--algos', nargs='*', default=['fedavg'])
    args = parser.parse_args()
    cfg = merge_defaults(load_config(args.config))
    set_seed(cfg['eval']['seed'])

    dm = DatasetManager(cfg)
    train, test, num_classes = dm.load()
    model_fn = build_model_fn(cfg, num_classes)

    partitions = build_partitions(train, cfg, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    server = Server(model_fn, train, test, num_classes, cfg, device)
    server.init_clients(partitions, model_fn)

    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    rec = MetricsRecorder()

    rounds = cfg['eval']['rounds']
    for r in range(rounds):
        row = server.round(r, partitions)
        rec.add(row)
        print(f"Round {r}: acc={row['accuracy']:.4f}, fairness={row['jain_index']:.3f}, comm_energy={row['comm_energy']:.3f}")

    df = rec.to_csv('outputs/results/metrics.csv')
    if cfg['eval'].get('save_plots', True):
        plot_curves(df, 'outputs/plots/run')


if __name__ == '__main__':
    main()
