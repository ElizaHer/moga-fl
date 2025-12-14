import numpy as np
from src.data.partition import dirichlet_partition

def test_dirichlet_partition():
    labels = np.array([0,1,2,3,4,5,6,7,8,9]*10)
    parts = dirichlet_partition(labels, num_clients=5, num_classes=10, alpha=0.5)
    assert len(parts)==5
    total = sum(len(v) for v in parts.values())
    assert total == len(labels)
