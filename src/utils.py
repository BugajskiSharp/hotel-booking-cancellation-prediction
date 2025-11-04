"""
utils.py
--------
Utility functions for reproducibility and device setup.
"""

import torch
import numpy as np
import random

def set_seed(seed=42):
    """Ensure full reproducibility across torch, numpy, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Return available device ('cuda' or 'cpu')."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
