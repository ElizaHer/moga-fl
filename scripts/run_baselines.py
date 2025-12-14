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
from src.training.server import Server
from src.eval.metrics import MetricsRecorder
from src.eval.plot import plot_curves


def build_partitions(train_dataset, cfg, num_classes):
    labels = np.array([y for _, y in train_dataset])
    num_clients = cfg['clients']['num_clients']
    niid = cfg['dataset']['noniid']
    if niid['type'] == 'dirichlet':
        client_indices = dirichlet_partition(labels, num_clients, num_classes, niid.get('alpha',0.5))
    elif niid['type'] == 'label_bias':
        cpc = niid.get('classes_per_client', niid.get('label_bias_classes_per_client',2))
        client_indices = label_bias_partition(labels, num_clients, num_classes, cpc)
    else:
        client_indices = quantity_bias_partition(labels, num_clients, niid.get('quantity_bias_sigma',0.6))
    # quick limit
    spc = cfg['clients'].get('samples_per_client')
    if spc:
        client_indices = apply_quick_limit(client_indices, spc)
    return client_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--algos', nargs='*', default=['fedavg'])
    args = parser.parse_args()
    cfg = merge_defaults(load_config(args.config))
    set_seed(cfg['eval']['seed'])

    dm = DatasetManager(cfg)
    train, test, num_classes = dm.load()
    model_fn = SimpleCIFARCNN if cfg['dataset']['name']=='cifar10' else (lambda: SimpleEMNISTCNN(num_classes))

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
