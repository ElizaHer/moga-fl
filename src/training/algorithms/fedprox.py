import torch
from typing import Dict, Any


def apply_fedprox(local_model, global_model, mu: float):
    # Add proximal term during local training via hook (here as placeholder, see client.py for actual usage)
    # This function is intentionally simple; the proximal regularization is applied in client update.
    pass
