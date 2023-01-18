#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training code.
"""

import hydra_zen
import torch
from hydra_zen import store, zen
from conf import project as project_conf

from utils import (
    seed_everything,
    to_cuda,
)
import wandb
from conf.experiment import BaseExperiementConfig


from src.base_trainer import BaseTrainer

# TODO: What I want in this
# - [x] Torchmetrics: this takes care of batching the loss and averaging it
# - [x] Saving top 3 best val models (easily configure metric)
# - [x] Training + evaluation loop
# - [x] Wandb integration with predefined logging metrics
# - [x] Automatic instantiation for the optimizer, scheduler, model
# - [ ] The best progress display I can ever get!! (kinda like torchlightning template? But I want
# colour (as in PIP), I want to see my hydra conf, and I want to see a little graph in a curses style in real
# time (look into Rich, Textual, etc.).
# - [x] Interception of SIGKILL, SIGTERM to stop training but save everything: two behaviours (1
# will be the default ofc) -- a) wait for epoch end and validation, b) abort epoch.
# - [ ] Add git hooks for linting, formatting, etc.


"""
The ideas of this template are:
- Keep it DRY
    - Use hydra-zen to configure the experiment
- Raw PyTorch for maximum flexibility and transparency
- Minimal abstraction and opacity
- The sweet spot between a template and a framework
    - The bare minimum boilerplate is taken care of but not hidden away
    - The user is free to do whatever they want, everything is transparent
    - Provide base classes for datasets, models, etc. to make it easier to get started and provide
      good structure for DRY code and easy debugging
    - Provide a good set of defaults for the most common use cases
    - Provide a good set of tools to make it easier to debug and visualize
- Good Python practices enforced with git hooks:
    - Black
    - Isort
    - Autoflake
"""


def launch_experiment(exp: BaseExperiementConfig):
    "============ CUDA ============"
    model: torch.nn.Module = to_cuda(exp.model)  # type: ignore

    "============ Weights & Biases ============"
    if project_conf.USE_WANDB:
        wandb.init( project=project_conf.PROJECT_NAME)

    " ============ Partials instantiation ============ "
    opt = exp.opt(model.parameters())
    train_dataset = exp.train_dataset(tiny=exp.tiny_dataset)
    val_dataset = exp.val_dataset(tiny=exp.tiny_dataset)

    " ============ Reproducibility of data loaders ============ "
    g = None
    if project_conf.REPRODUCIBLE:
        g = torch.Generator()
        g.manual_seed(0)

    train_loader = exp.train_loader(
        train_dataset, generator=g
    )
    val_loader = exp.val_loader(
        val_dataset, generator=g
    )


    " ============ Training ============ "
    BaseTrainer(
        model=model,
        opt=opt,
        train_loader=train_loader,
        val_loader=val_loader,
    ).train(
        epochs=exp.epochs,
        val_every=exp.val_every,
        visualize_every=exp.visualize_every,
    )


if __name__ == "__main__":
    " ============ Hydra-Zen ============ "
    store.add_to_hydra_store()
    zen(
        launch_experiment,
        pre_call=hydra_zen.zen(
            lambda exp: seed_everything(exp.seed) if project_conf.REPRODUCIBLE else lambda: None
        ),
    ).hydra_main(
        config_name="exp/model_a",
        version_base="1.1",  # Hydra base version
    )
