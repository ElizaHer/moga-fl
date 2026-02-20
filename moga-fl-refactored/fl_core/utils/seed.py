import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Set random seed for Python, NumPy and PyTorch.

    若 seed 为 None，则不做任何操作。
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
