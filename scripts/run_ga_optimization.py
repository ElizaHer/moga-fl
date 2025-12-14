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
from src.ga.moga_fl import MOGAFLController


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


def make_sim_runner(cfg, train, test, num_classes, partitions, device, round_scale: float = 0.6):
    """构造一个用于 GA 评估的联邦训练短跑器。

    round_scale 用于控制评估轮数比例：
    - 低保真评估：如 0.4 / 0.6，只跑少量轮数；
    - 高保真评估：如 1.0，跑满配置中的轮数。
    """
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
        # short / full eval rounds
        cfg2['eval'] = cfg['eval'].copy()
        cfg2['eval']['rounds'] = max(3, int(cfg['eval']['rounds'] * round_scale))
        server = Server(
            SimpleCIFARCNN if cfg['dataset']['name'] == 'cifar10' else (lambda: SimpleEMNISTCNN(num_classes)),
            train,
            test,
            num_classes,
            cfg2,
            device,
        )
        server.init_clients(
            partitions,
            SimpleCIFARCNN if cfg['dataset']['name'] == 'cifar10' else (lambda: SimpleEMNISTCNN(num_classes)),
        )
        accs = []
        times = []
        fairs = []
        energies = []
        for r in range(cfg2['eval']['rounds']):
            row = server.round(r, partitions)
            accs.append(row['accuracy'])
            times.append(row['comm_time'])
            fairs.append(row['jain_index'])
            energies.append(row['comm_energy'] + row['comp_energy'])
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
    parser.add_argument('--algo', type=str, default='nsga3', choices=['nsga3', 'moead', 'moga_fl'])
    args = parser.parse_args()
    cfg = merge_defaults(load_config(args.config))
    set_seed(cfg['eval']['seed'])
    dm = DatasetManager(cfg)
    train, test, num_classes = dm.load()
    partitions = build_partitions(train, cfg, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 低保真与高保真评估器：用于多保真 GA
    low_sim = make_sim_runner(cfg, train, test, num_classes, partitions, device, round_scale=0.5)
    high_sim = make_sim_runner(cfg, train, test, num_classes, partitions, device, round_scale=1.0)

    if args.algo == 'nsga3':
        opt = NSGA3(cfg, low_sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)
    elif args.algo == 'moead':
        opt = MOEAD(low_sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)
    else:
        # 统一的 MOGA-FL 控制器：内部组合 NSGA-III + MOEA/D + 岛屿 + 局部搜索
        opt = MOGAFLController(cfg, low_fidelity_eval=low_sim, high_fidelity_eval=high_sim, pop_size=args.pop)
        pop, metrics = opt.run(generations=args.generations)

    os.makedirs('outputs/results', exist_ok=True)
    rows = []
    for p, m in zip(pop, metrics):
        row = {**p, **m}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('outputs/results/pareto_candidates.csv', index=False)
    print('Saved Pareto candidates to outputs/results/pareto_candidates.csv')

    # 从 Pareto 候选中，根据偏好选择一个“部署用”解，并导出配置
    preference = cfg['eval'].get('preference', 'time')  # time / fairness / energy
    best_idx = 0
    best_score = None
    for i, m in enumerate(metrics):
        if preference == 'time':
            score = -m['time'] + 0.1 * m['acc']
        elif preference == 'fairness':
            score = m['fairness'] + 0.05 * m['acc']
        else:  # energy 优先
            score = -m['energy'] + 0.1 * m['acc']
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i

    best_params = pop[best_idx]
    deploy_cfg = cfg.copy()
    deploy_cfg.setdefault('scheduling', {})
    deploy_cfg['scheduling']['weights'] = {
        'energy': best_params['energy_w'],
        'channel': best_params['channel_w'],
        'data_value': best_params['data_w'],
        'fairness_debt': best_params['fair_w'],
        'bandwidth_cost': best_params['bwcost_w'],
    }
    deploy_cfg.setdefault('clients', cfg['clients'].copy())
    deploy_cfg['clients']['selection_top_k'] = int(best_params['selection_top_k'])
    deploy_cfg['clients']['hysteresis'] = float(best_params['hysteresis'])
    deploy_cfg.setdefault('training', cfg['training'].copy())
    fb_cfg = deploy_cfg['training'].get('fedbuff', cfg['training'].get('fedbuff', {}).copy())
    fb_cfg['staleness_alpha'] = float(best_params['staleness_alpha'])
    deploy_cfg['training']['fedbuff'] = fb_cfg

    best_cfg_path = 'outputs/results/best_moga_fl_config.yaml'
    with open(best_cfg_path, 'w') as f:
        import yaml

        yaml.safe_dump(deploy_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Saved best deployment config to {best_cfg_path}")

if __name__ == '__main__':
    main()
