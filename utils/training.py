#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

import torch


def visualize_model_predictions(model: torch.nn.Module, batch) -> None:
    """
    Visualize model predictions on a dataset.
    """
    raise NotImplementedError
