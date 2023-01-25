#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Project-level constants: you would typically configure these once for your project. Avoid putting
experiment-specific constants here.
"""

from enum import Enum


class TerminationBehavior(Enum):
    WAIT_FOR_EPOCH_END = 0
    ABORT_EPOCH = 1


REPRODUCIBLE = True
CKPT_PATH = "ckpt"
USE_CUDA_IF_AVAILABLE = True
SIGINT_BEHAVIOR = TerminationBehavior.WAIT_FOR_EPOCH_END
PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH = True
BEST_N_MODELS_TO_KEEP = 5  # 0 means keep all models
USE_WANDB = True
PROJECT_NAME = "my-python-project"

# Theming
class Theme(Enum):
    TRAINING = "green"
    VALIDATION = "blue"
    TESTING = "cyan"

    def __str__(self):
        return f"{self.value}"

    def __repr__(self):
        return f"{self.value}"


ANSI_COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}
