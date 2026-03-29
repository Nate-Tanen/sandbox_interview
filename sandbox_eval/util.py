import os
import random

import torch


def debug_print(*args, **kwargs) -> None:
    if os.getenv("DEBUG_PRINT") == "1":
        print(*args, **kwargs)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
