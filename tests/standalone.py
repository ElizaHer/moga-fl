import argparse
import sys
import torch

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.models.mlp import MLP
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST


if __name__ == "__main__":
    # 超参数设置
    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument("--total_clients", type=int, default=100)
    parser.add_argument("--com_round", type=int, default=10)
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()

    model = MLP(784, 10)

    # server
    handler = SyncServerHandler(
        model, args.com_round, args.sample_ratio
    )

    # client
    trainer = SGDSerialClientTrainer(model, args.total_clients, cuda=True)
    dataset = PathologicalMNIST(
        root="../data/",
        path="../data/",
        num_clients=args.total_clients,
    )
    dataset.preprocess()

    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    handler.num_clients = args.total_clients
    handler.setup_dataset(dataset)

    # main
    pipeline = StandalonePipeline(handler, trainer)
    pipeline.main()
