#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import hydra_zen
import torch
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra_zen import just
from hydra_zen.typing import Partial
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from torch.utils.data import DataLoader, Dataset

from conf import project as project_conf
from model import TransparentDataParallel
from src.base_tester import BaseTester
from src.base_trainer import BaseTrainer
from utils import load_model_ckpt, to_cuda_

console = Console()


def print_config(run_name: str, exp_conf: str) -> None:
    # Generate a random ANSI code:
    run_color = f"color({hash(run_name) % 255})"
    background_color = f"color({(hash(run_name) + 128) % 255})"
    console.print(
        f"Running {run_name}",
        style=f"bold {run_color} on {background_color}",
        justify="center",
    )
    console.rule()
    console.print(
        Panel(
            Syntax(
                exp_conf, lexer="yaml", dedent=True, word_wrap=False, theme="dracula"
            ),
            title="Experiment configuration",
            expand=False,
        ),
        overflow="ellipsis",
    )


def print_model(model: torch.nn.Module) -> None:
    console.print(
        Panel(
            Group(
                Pretty(model),
                f"Number of parameters: {sum(p.numel() for p in model.parameters())}",
                f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            ),
            title="Model architecture",
            expand=False,
        ),
        overflow="ellipsis",
    )
    console.rule()


def init_wandb(
    run_name: str,
    model: torch.nn.Module,
    exp_conf: str,
    log="gradients",
    log_graph=False,
) -> None:
    if project_conf.USE_WANDB:
        with console.status("Initializing Weights & Biases...", spinner="moon"):
            # exp_conf is a string, so we need to load it back to a dict:
            exp_conf = yaml.safe_load(exp_conf)
            wandb.init(  # type: ignore
                project=project_conf.PROJECT_NAME,
                name=run_name,
                config=exp_conf,
            )
            wandb.watch(model, log=log, log_graph=log_graph)  # type: ignore


def make_datasets(
    training_mode: bool, seed: int, dataset_partial: Partial[Dataset[Any]]
) -> Tuple[Optional[Dataset[Any]], Optional[Dataset[Any]], Optional[Dataset[Any]]]:
    train_dataset: Optional[Dataset[Any]] = None
    val_dataset: Optional[Dataset[Any]] = None
    test_dataset: Optional[Dataset[Any]] = None
    with console.status("Loading datasets...", spinner="monkey"):
        if training_mode:
            train_dataset = dataset_partial(split="train", seed=seed)
            val_dataset = dataset_partial(split="val", seed=seed)
        else:
            test_dataset = dataset_partial(split="test", augment=False, seed=seed)
    return train_dataset, val_dataset, test_dataset


def make_dataloaders(
    data_loader_partial: Partial[DataLoader[Dataset[Any]]],
    train_dataset: Optional[Dataset[Any]],
    val_dataset: Optional[Dataset[Any]],
    test_dataset: Optional[Dataset[Any]],
    training_mode: bool,
    seed: int,
) -> Tuple[
    Optional[DataLoader[Dataset[Any]]],
    Optional[DataLoader[Dataset[Any]]],
    Optional[DataLoader[Dataset[Any]]],
]:
    generator = None
    if project_conf.REPRODUCIBLE:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader_inst: Optional[DataLoader[Any]] = None
    val_loader_inst: Optional[DataLoader[Dataset[Any]]] = None
    test_loader_inst: Optional[DataLoader[Any]] = None
    if training_mode:
        if train_dataset is None or val_dataset is None:
            raise ValueError(
                "train_dataset and val_dataset must be defined in training mode!"
            )
        train_loader_inst = data_loader_partial(train_dataset, generator=generator)
        val_loader_inst = data_loader_partial(
            val_dataset, generator=generator, shuffle=False, drop_last=False
        )
    else:
        if test_dataset is None:
            raise ValueError("test_dataset must be defined in testing mode!")
        test_loader_inst = data_loader_partial(
            test_dataset, generator=generator, shuffle=False, drop_last=False
        )
    return train_loader_inst, val_loader_inst, test_loader_inst


def make_model(
    model_partial: Partial[torch.nn.Module], dataset: Partial[Dataset[Any]]
) -> torch.nn.Module:
    with console.status("Loading model...", spinner="runner"):
        model_inst = model_partial(
            encoder_input_dim=just(dataset).img_dim ** 2  # type: ignore
        )  # Use just() to get the config out of the Zen-Partial

    return model_inst


def parallelize_model(model: torch.nn.Module) -> torch.nn.Module:
    console.print(
        f"[*] Number of GPUs: {torch.cuda.device_count()}",
        style="bold cyan",
    )
    if torch.cuda.device_count() > 1:
        console.print(
            f"-> Using {torch.cuda.device_count()} GPUs!",
            style="bold cyan",
        )
        model = TransparentDataParallel(model)
    return model


def make_optimizer(
    optimizer_partial: Partial[torch.optim.Optimizer], model: torch.nn.Module
) -> torch.optim.Optimizer:
    return optimizer_partial(model.parameters())


def make_scheduler(
    scheduler_partial: Partial[torch.optim.lr_scheduler.LRScheduler],
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = scheduler_partial(
        optimizer
    )  # TODO: less hacky way to set T_max for CosineAnnealingLR?
    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.T_max = epochs
    return scheduler


def make_training_loss(
    training_mode: bool, training_loss_partial: Partial[torch.nn.Module]
):
    training_loss: Optional[torch.nn.Module] = None
    if training_mode:
        training_loss = training_loss_partial()
    return training_loss


def launch_experiment(
    run,  # type: ignore
    data_loader: Partial[DataLoader[Any]],
    optimizer: Partial[torch.optim.Optimizer],
    scheduler: Partial[torch.optim.lr_scheduler.LRScheduler],
    trainer: Partial[BaseTrainer],
    tester: Partial[BaseTester],
    dataset: Partial[Dataset[Any]],
    model: Partial[torch.nn.Module],
    training_loss: Partial[torch.nn.Module],
):
    run_name = os.path.basename(HydraConfig.get().runtime.output_dir)
    exp_conf = hydra_zen.to_yaml(
        dict(
            run_conf=run,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=training_loss,
        )
    )
    print_config(run_name, exp_conf)

    """ ============ Partials instantiation ============ """
    model_inst = make_model(model, dataset)
    print_model(model_inst)
    train_dataset, val_dataset, test_dataset = make_datasets(
        run.training_mode, run.seed, dataset
    )
    opt_inst = make_optimizer(optimizer, model_inst)
    scheduler_inst = make_scheduler(scheduler, opt_inst, run.epochs)
    model_inst = to_cuda_(parallelize_model(model_inst))
    training_loss_inst = to_cuda_(make_training_loss(run.training_mode, training_loss))
    train_loader_inst, val_loader_inst, test_loader_inst = make_dataloaders(
        data_loader,
        train_dataset,
        val_dataset,
        test_dataset,
        run.training_mode,
        run.seed,
    )
    init_wandb(run_name, model_inst, exp_conf)

    """ ============ Training ============ """
    model_ckpt_path = load_model_ckpt(run.load_from, run.training_mode)
    if run.training_mode:
        if training_loss_inst is None:
            raise ValueError("training_loss must be defined in training mode!")
        if val_loader_inst is None or train_loader_inst is None:
            raise ValueError(
                "val_loader and train_loader must be defined in training mode!"
            )
        trainer(
            run_name=run_name,
            model=model_inst,
            opt=opt_inst,
            scheduler=scheduler_inst,
            train_loader=train_loader_inst,
            val_loader=val_loader_inst,
            training_loss=training_loss_inst,
            **asdict(
                run
            ),  # Extra stuff if needed. You can get them from the trainer's __init__ with kwrags.get(key, default_value)
        ).train(
            epochs=run.epochs,
            val_every=run.val_every,
            visualize_every=run.viz_every,
            visualize_train_every=run.viz_train_every,
            visualize_n_samples=run.viz_num_samples,
            model_ckpt_path=model_ckpt_path,
        )
    else:
        if test_loader_inst is None:
            raise ValueError("test_loader must be defined in testing mode!")
        tester(
            run_name=run_name,
            model=model_inst,
            data_loader=test_loader_inst,
            model_ckpt_path=model_ckpt_path,
            training_loss=training_loss_inst,
        ).test(
            visualize_every=run.viz_every,
            **asdict(
                run
            ),  # Extra stuff if needed. You can get them from the trainer's __init__ with kwrags.get(key, default_value)
        )
