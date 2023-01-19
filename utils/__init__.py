import random
from typing import List, Tuple, Union

import numpy as np
import torch

from conf import project as project_conf


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_cuda(
    x: Union[Tuple, List, torch.Tensor, torch.nn.Module]
) -> Union[Tuple, List, torch.Tensor, torch.nn.Module]:
    if project_conf.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
            x = x.cuda()
        elif isinstance(x, Tuple) or isinstance(x, List):
            x = tuple(t.cuda() for t in x)
    return x


def colorize(string: str, ansii_code: int) -> str:
    return f"\033[{ansii_code}m{string}\033[0m"
