from src.data.partition import *
from torchvision.datasets import CIFAR10, MNIST


cifar10_dataset = CIFAR10(root='../data', train=True, download=False)
mnist_dataset = MNIST(root='../data', train=True, download=False)


def print_partition(train_dataset, client_indices):
    num_clients = len(client_indices)

    print("=== 分区结果 ===")
    print(f"参数：客户端数={num_clients}")

    # 统计每个客户端的类别分布
    for cid in range(num_clients):
        indices = client_indices[cid]
        unique_count = len(set(indices))  # 客户端内无重复
        client_targets = [train_dataset.targets[idx] for idx in indices]
        class_dist = {cls: client_targets.count(cls) for cls in range(10)}
        # 计算占比最高的类别（体现Non-IID）
        max_cls = max(class_dist, key=class_dist.get)
        max_ratio = class_dist[max_cls] / len(indices)

        print(f"客户端{cid}:")
        print(f"  - 样本数: {len(indices)} | 内部唯一样本数: {unique_count}")
        print(f"  - 占比最高类别: 类别{max_cls} ({max_ratio:.2%})")
        print(f"  - 类别分布: {class_dist}\n")

    # 验证样本覆盖度
    all_selected = set()
    for cid in client_indices:
        all_selected.update(client_indices[cid])
    coverage_rate = len(all_selected) / len(train_dataset)

    print(f"整体样本覆盖度: {len(all_selected)}/{len(train_dataset)} = {coverage_rate:.2%}")


def test_cifar_partition(num_clients, alpha):
    print_partition(cifar10_dataset, cifar_iid_partitions(cifar10_dataset, num_clients))
    print_partition(cifar10_dataset, cifar_dirichlet_partitions(cifar10_dataset, num_clients, alpha))
    print_partition(cifar10_dataset, cifar_shards_partitions(cifar10_dataset, num_clients))


def test_mnist_partition(num_clients, alpha):
    print_partition(mnist_dataset, mnist_iid_partitions(mnist_dataset, num_clients))
    print_partition(mnist_dataset, mnist_dirichlet_partitions(mnist_dataset, num_clients, alpha))


if __name__ == '__main__':
    test_cifar_partition(10, 0.5)
    test_mnist_partition(10, 0.5)
