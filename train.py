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

import wandb
from conf import project as project_conf
from conf.experiment import BaseExperiementConfig
from src.base_trainer import BaseTrainer
from utils import seed_everything, to_cuda


def launch_experiment(exp: BaseExperiementConfig):
    "============ CUDA ============"
    model: torch.nn.Module = to_cuda(exp.model)  # type: ignore

    "============ Weights & Biases ============"
    if project_conf.USE_WANDB:
        wandb.init(project=project_conf.PROJECT_NAME)

    " ============ Partials instantiation ============ "
    opt = exp.opt(model.parameters())
    train_dataset = exp.train_dataset(tiny=exp.tiny_dataset)
    val_dataset = exp.val_dataset(tiny=exp.tiny_dataset)

    " ============ Reproducibility of data loaders ============ "
    g = None
    if project_conf.REPRODUCIBLE:
        g = torch.Generator()
        g.manual_seed(0)

    train_loader = exp.train_loader(train_dataset, generator=g)
    val_loader = exp.val_loader(val_dataset, generator=g)

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
    "============ Hydra-Zen ============"
    store.add_to_hydra_store()
    zen(
        launch_experiment,
        pre_call=hydra_zen.zen(
            lambda exp: seed_everything(exp.seed)
            if project_conf.REPRODUCIBLE
            else lambda: None
        ),
    ).hydra_main(
        config_name="exp/model_a",
        version_base="1.1",  # Hydra base version
    )
