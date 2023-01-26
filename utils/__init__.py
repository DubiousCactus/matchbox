#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import random
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm

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


def colorize(string: str, ansii_code: Union[int, str]) -> str:
    return f"\033[{ansii_code}m{string}\033[0m"


def blink_pbar(i: int, pbar: tqdm.tqdm, n: int) -> None:
    """Blink the progress bar every n iterations.
    Args:
        i (int): current iteration
        pbar (tqdm.tqdm): progress bar
        n (int): blink every n iterations
    """
    if i % n == 0:
        pbar.colour = (
            project_conf.Theme.TRAINING.value
            if pbar.colour == project_conf.Theme.VALIDATION.value
            else project_conf.Theme.VALIDATION.value
        )


def update_pbar_str(pbar: tqdm.tqdm, string: str, color_code: int) -> None:
    """Update the progress bar string.
    Args:
        pbar (tqdm.tqdm): progress bar
        string (str): string to update the progress bar with
        color_code (int): color code for the string
    """
    pbar.set_description_str(colorize(string, color_code))
