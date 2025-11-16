import numpy as np
import torch
from src.training.models.cifar_cnn import SimpleCIFARCNN
from src.training.server import Server

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=100):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.rand(3,32,32)
        y = torch.randint(0,10,(1,)).item()
        return x, y


def test_server_round():
    cfg = {
        'clients': {'num_clients': 4, 'selection_top_k': 2, 'samples_per_client': 16, 'sliding_window': 3, 'hysteresis': 0.05},
        'wireless': {'block_fading_intensity': 1.0, 'base_snr_db': 10.0, 'per_k': 1.0, 'bandwidth_budget_mb_per_round': 4.0, 'tx_power_watts': 1.0, 'compute_power_watts': 5.0, 'compute_rate_samples_per_sec': 1000},
        'training': {'local_epochs': 1, 'batch_size': 16, 'lr': 0.01, 'momentum': 0.9, 'fedprox_mu': 0.0, 'fedbuff': {'enabled': False}},
        'scheduling': {'weights': {'energy':0.25,'channel':0.25,'data_value':0.25,'fairness_debt':0.2,'bandwidth_cost':-0.15}, 'fairness_ledger': {'debt_increase':0.05,'repay_rate':0.1,'max_debt':1.0}},
        'eval': {'rounds': 1, 'seed': 123}
    }
    train = DummyDataset(200)
    test = DummyDataset(50)
    device = torch.device('cpu')
    server = Server(SimpleCIFARCNN, train, test, 10, cfg, device)
    # partitions
    parts = {i: list(range(i*50,(i+1)*50)) for i in range(4)}
    server.init_clients(parts, SimpleCIFARCNN)
    row = server.round(0, parts)
    assert 'accuracy' in row
