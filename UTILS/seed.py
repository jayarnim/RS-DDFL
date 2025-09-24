import random
import numpy as np
import torch
from .constants import SEED


def reset(seed: int=SEED):
    print(f"SETTING ALL SEEDS TO {seed}...")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("ALL SEEDS SET")
