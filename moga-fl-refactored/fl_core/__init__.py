"""Core modules for the refactored MOGA-FL implementation.

This package reorganizes the original project around a clearer
module boundary and integrates FedLab 1.3.0 as the federated
learning backend.
"""

from .config import load_config, merge_defaults
