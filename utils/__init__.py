#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch import Tensor, nn

from conf import project as project_conf


def load_model_ckpt(load_from: Optional[str], training_mode: bool) -> Optional[str]:
    model_ckpt_path = None
    if load_from is not None:
        if load_from.endswith(".ckpt"):
            model_ckpt_path = to_absolute_path(load_from)
            if not os.path.exists(model_ckpt_path):
                raise ValueError(f"File {model_ckpt_path} does not exist!")
        else:
            run_models = sorted(
                [
                    f
                    for f in os.listdir(to_absolute_path(f"runs/{load_from}/"))
                    if f.endswith(".ckpt")
                    and (not f.startswith("last") if not training_mode else True)
                ]
            )
            if len(run_models) < 1:
                raise ValueError(f"No model found in runs/{load_from}/")
            model_ckpt_path = to_absolute_path(
                os.path.join(
                    "runs",
                    load_from,
                    run_models[-1],
                )
            )
    return model_ckpt_path


def seed_everything(seed: int):
    torch.manual_seed(seed)  # type: ignore
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_cuda_(x: Any) -> Any:
    device = "cpu"
    dtype = x.dtype if isinstance(x, Tensor) else None
    if project_conf.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
    elif project_conf.USE_MPS_IF_AVAILABLE and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32 if dtype is torch.float64 else dtype
    else:
        return x
    if isinstance(x, (Tensor, nn.Module)):
        x = x.to(device, dtype=dtype)
    elif isinstance(x, tuple):
        x = tuple(to_cuda_(t) for t in x)  # type: ignore
    elif isinstance(x, list):
        x = [to_cuda_(t) for t in x]  # type: ignore
    elif isinstance(x, dict):
        x = {key: to_cuda_(value) for key, value in x.items()}  # type: ignore
    return x


def to_cuda(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to move function arguments to cuda if available and if they are
    torch tensors, torch modules or tuples/lists of."""

    def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        args = to_cuda_(args)
        for key, value in kwargs.items():
            kwargs[key] = to_cuda_(value)
        return func(*args, **kwargs)

    return wrapper
