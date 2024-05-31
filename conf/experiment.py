#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for the experiments and config groups, using hydra-zen.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import (
    MISSING,
    ZenStore,
    builds,
    make_config,
    make_custom_builds_fn,
    store,
)
from hydra_zen.typing import SupportedPrimitive
from hydra_zen.typing._builds_overloads import PBuilds
from torch.utils.data import DataLoader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from dataset.example import ExampleDataset
from launch_experiment import launch_experiment
from model.example import ExampleModel
from src.base_tester import BaseTester
from src.base_trainer import BaseTrainer
from src.losses.mse import MSELoss

# Set hydra.job.chdir=True using store():
hydra_store = ZenStore(overwrite_ok=True)
hydra_store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")
# We'll generate a unique name for the experiment and use it as the run name
hydra_store(
    HydraConf(
        run=RunDir(
            f"runs/{get_random_name(combo=[ADJECTIVES, NAMES], separator='-', style='lowercase')}"
        )
    ),
    name="config",
    group="hydra",
)
hydra_store.add_to_hydra_store()
pbuilds: PBuilds[SupportedPrimitive] = make_custom_builds_fn(
    zen_partial=True, populate_full_signature=False
)

""" ================== Dataset ================== """


# Dataclasses are a great and simple way to define a base config group with default values.
@dataclass
class ExampleDatasetConf:
    dataset_name: str = "image_dataset"
    dataset_root: str = "data/a"
    tiny: bool = False
    normalize: bool = True
    augment: bool = False
    debug: bool = False
    img_dim: int = ExampleDataset.IMG_SIZE[0]


# Pre-set the group for store's dataset entries
dataset_store = store(group="dataset")
dataset_store(
    pbuilds(ExampleDataset, builds_bases=(ExampleDatasetConf,)), name="image_a"
)

dataset_store(
    pbuilds(
        ExampleDataset,
        builds_bases=(ExampleDatasetConf,),
        dataset_root="data/b",
        img_dim=64,
    ),
    name="image_b",
)
dataset_store(
    pbuilds(
        ExampleDataset,
        builds_bases=(ExampleDatasetConf,),
        tiny=True,
    ),
    name="image_a_tiny",
)

""" ================== Dataloader & sampler ================== """


@dataclass
class SamplerConf:
    batch_size: int = 16
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class DataloaderConf:
    batch_size: int = 16
    drop_last: bool = True
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False


""" ================== Model ================== """
# Pre-set the group for store's model entries
model_store = store(group="model")

# Not that encoder_input_dim depend on dataset.img_dim, so we need to use a partial to set them in
# the launch_experiment function.
model_store(
    pbuilds(
        ExampleModel,
        encoder_dim=128,
        decoder_dim=64,
        latent_dim=32,
        decoder_output_dim=8,
    ),
    name="model_a",
)
model_store(
    pbuilds(
        ExampleModel,
        encoder_dim=256,
        decoder_dim=128,
        latent_dim=64,
        decoder_output_dim=8,
    ),
    name="model_b",
)

""" ================== Losses ================== """
training_loss_store = store(group="training_loss")
training_loss_store(
    pbuilds(
        MSELoss,
        reduction="mean",
    ),
    name="mse",
)


""" ================== Optimizer ================== """


@dataclass
class Optimizer:
    lr: float = 1e-3
    weight_decay: float = 0.0


opt_store = store(group="optimizer")
opt_store(
    pbuilds(
        torch.optim.Adam,
        builds_bases=(Optimizer,),
    ),
    name="adam",
)
opt_store(
    pbuilds(
        torch.optim.SGD,
        builds_bases=(Optimizer,),
    ),
    name="sgd",
)


""" ================== Scheduler ================== """
sched_store = store(group="scheduler")
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.StepLR,
        step_size=100,
        gamma=0.5,
    ),
    name="step",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=0.5,
        patience=10,
    ),
    name="plateau",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ),
    name="cosine",
)

""" ================== Experiment ================== """


@dataclass
class RunConfig:
    epochs: int = 200
    seed: int = 42
    val_every: int = 1
    viz_every: int = 10
    viz_train_every: int = 0
    viz_num_samples: int = 5
    load_from: Optional[str] = None
    training_mode: bool = True


run_store = store(group="run")
run_store(RunConfig, name="default")


trainer_store = store(group="trainer")
trainer_store(pbuilds(BaseTrainer, populate_full_signature=True), name="base")

tester_store = store(group="tester")
tester_store(pbuilds(BaseTester, populate_full_signature=True), name="base")

Experiment = builds(
    launch_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"trainer": "base"},
        {"tester": "base"},
        {"dataset": "image_a"},
        {"model": "model_a"},
        {"optimizer": "adam"},
        {"scheduler": "step"},
        {"run": "default"},
        {"training_loss": "mse"},
    ],
    trainer=MISSING,
    tester=MISSING,
    dataset=MISSING,
    model=MISSING,
    optimizer=MISSING,
    scheduler=MISSING,
    run=MISSING,
    training_loss=MISSING,
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(Experiment, name="base_experiment")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment", package="_global_")
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "model_a"},
            {"override /dataset": "image_a"},
        ],
        # training=dict(epochs=100),
        bases=(Experiment,),
    ),
    name="exp_a",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "model_b"},
            {"override /dataset": "image_b"},
        ],
        bases=(Experiment,),
    ),
    name="exp_b",
)
