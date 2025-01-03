import os
import random
import numpy as np

import torch
from torch.backends import cudnn


def lock_seed(seed: int=42) -> None:
    """
    Lock seed for reproducibility and reduce training time.

        Considerations:
        Seed:
            - numpy.random.seed()
            - torch.manual_seed()
            - torch.cuda.manual_seed()
            - torch.cuda.manual_seed_all() if multi-GPU
            - torch.backends.cudnn.deterministic = True
            - torch.backends.cudnn.benchmark = False
        Training speed:
            - cuda.allow_tf32 = True
            - cudnn.allow_tf32 = True

    See https://pytorch.org/docs/stable/notes/randomness.html for more information.

    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.enabled = True
        torch.cuda.allow_tf32 = True  # allowing TensorFloat32 for faster training
        cudnn.enabled = False
        torch.set_float32_matmul_precision("high")

    else:
        print("cuda is not available")
