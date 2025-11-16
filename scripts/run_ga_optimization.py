import argparse
import os
import numpy as np
import torch
import pandas as pd
from src.configs.config import load_config, merge_defaults
from src.utils.seed import set_seed
from src.data.dataset_loader import DatasetManager
from src.data.partition import dirichlet_partition, label_bias_partition, quantity_bias_partition, apply_quick_limit
from src.training.models.cifar_cnn import SimpleCIFARCNN
from src.training.models.emnist_cnn import SimpleEMNISTCNN
from src.training.server import Server
from src.ga.nsga3 import NSGA3
from src.ga.moead import MOEAD


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
    spc = cfg['clients'].get('samples_per_client')
    if spc:
        client_indices = apply_quick_limit(client_indices, spc)
    return client_indices


def make_sim_runner(cfg, train, test, num_classes, partitions, device):
    def runner(params):
        # apply params into cfg copy
        cfg2 = {**cfg}
        # scheduling weights
        cfg2['scheduling'] = cfg['scheduling'].copy()
        cfg2['scheduling']['weights'] = {
            'energy': params['energy_w'],
            'channel': params['channel_w'],
            'data_value': params['data_w'],
            'fairness_debt': params['fair_w'],
            'bandwidth_cost': params['bwcost_w'],
        }
        cfg2['clients'] = cfg['clients'].copy()
        cfg2['clients']['selection_top_k'] = int(params['selection_top_k'])
        cfg2['clients']['hysteresis'] = float(params['hysteresis'])
        cfg2['training'] = cfg['training'].copy()
        cfg2['training']['fedbuff'] = cfg['training'].get('fedbuff', {}).copy()
        cfg2['training']['fedbuff']['staleness_alpha'] = float(params['staleness_alpha'])
        # short eval rounds
        cfg2['eval'] = cfg['eval'].copy()
        cfg2['eval']['rounds'] = max(3, int(cfg['eval']['rounds']*0.6))
        server = Server(SimpleCIFARCNN if cfg['dataset']['name']=='cifar10' else (lambda: SimpleEMNISTCNN(num_classes)), train, test, num_classes, cfg2, device)
        server.init_clients(partitions, SimpleCIFARCNN if cfg['dataset']['name']=='cifar10' else (lambda: SimpleEMNISTCNN(num_classes)))
        accs=[]; times=[]; fairs=[]; energies=[]
        for r in range(cfg2['eval']['rounds']):
            row = server.round(r, partitions)
            accs.append(row['accuracy']); times.append(row['comm_time']); fairs.append(row['jain_index']); energies.append(row['comm_energy']+row['comp_energy'])
        return {
            'acc': float(np.mean(accs)),
            'time': float(np.mean(times)),
            'fairness': float(np.mean(fairs)),
            'energy': float(np.mean(energies)),
        }
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--generations', type=int, default=6)
    parser.add_argument('--pop', type=int, default=16)
    parser.add_argument('--algo', type=str, default='nsga3', choices=['nsga3','moead'])
    args = parser.parse_args()
    cfg = merge_defaults(load_config(args.config))
    set_seed(cfg['eval']['seed'])
    dm = DatasetManager(cfg)
    train, test, num_classes = dm.load()
    partitions = build_partitions(train, cfg, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim = make_sim_runner(cfg, train, test, num_classes, partitions, device)

    if args.algo == 'nsga3':
        opt = NSGA3(cfg, sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)
    else:
        opt = MOEAD(sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)

    os.makedirs('outputs/results', exist_ok=True)
    rows = []
    for p, m in zip(pop, metrics):
        row = {**p, **m}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('outputs/results/pareto_candidates.csv', index=False)
    print('Saved Pareto candidates to outputs/results/pareto_candidates.csv')

if __name__ == '__main__':
    main()
