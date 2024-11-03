#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import asyncio
import os
from dataclasses import asdict
from time import sleep
from typing import Any

import hydra_zen
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra_zen.typing import Partial
from rich.console import Console, Group
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from torch.utils.data import DataLoader, Dataset

from bootstrap import MatchboxModule
from bootstrap.factories import (
    make_dataloaders,
    make_datasets,
    make_model,
    make_optimizer,
    make_scheduler,
    make_training_loss,
    parallelize_model,
)
from bootstrap.tui.builder_ui import BuilderUI
from bootstrap.tui.training_ui import TrainingUI
from conf import project as project_conf
from src.base_tester import BaseTester
from src.base_trainer import BaseTrainer
from utils import load_model_ckpt, to_cuda_

console = Console()


# ================================= Printing =====================================
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
                "Number of trainable parameters: "
                + f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            ),
            title="Model architecture",
            expand=False,
        ),
        overflow="ellipsis",
    )
    console.rule()


# ==================================================================================


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


def launch_builder(
    run,  # type: ignore
    data_loader: Partial[DataLoader[Any]],
    optimizer: Partial[torch.optim.Optimizer],  # pyright: ignore
    scheduler: Partial[torch.optim.lr_scheduler.LRScheduler],
    trainer: Partial[BaseTrainer],
    tester: Partial[BaseTester],
    dataset: Partial[Dataset[Any]],
    model: Partial[torch.nn.Module],
    training_loss: Partial[torch.nn.Module],
):
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

    async def launch_with_async_gui():
        tui = BuilderUI()
        task = asyncio.create_task(tui.run_async())
        await asyncio.sleep(0.5)  # Wait for the app to start up
        while not tui.is_running:
            await asyncio.sleep(0.01)  # Wait for the app to start up
        dataset_module = MatchboxModule(
            "Dataset",
            dataset,  # TODO: Fix the code reloading, then revert to using the dataset factory
            split="train",
            seed=run.seed,
            progress=None,
            job_id=None,
        )
        model_module = MatchboxModule(
            "Model",
            model,
            encoder_input_dim=hydra_zen.just(dataset).img_dim ** 2,  # type: ignore
        )
        await tui.chain_up(
            [
                dataset_module,
                MatchboxModule(
                    "Dataset test",
                    lambda dataset_obj: dataset_obj.test(),
                    dataset_obj=dataset_module,
                ),
                model_module,
                MatchboxModule(
                    "Model forward",
                    # NOTE: For now we need to call .forward() explicitly, as Matchbox
                    # doesn't yet handle hot code reloading for model() due to Pytorch
                    # wrapping
                    lambda model, dataset: model.forward(dataset[0][0].unsqueeze(0)),
                    model=model_module,
                    dataset=dataset_module,
                ),
                MatchboxModule(
                    "Loss", make_training_loss, run.training_mode, training_loss
                ),
            ]
        )
        tui.run_chain()
        _ = await task

    asyncio.run(launch_with_async_gui())


def launch_experiment(
    run,  # type: ignore
    data_loader: Partial[DataLoader[Any]],
    optimizer: Partial[torch.optim.Optimizer],  # pyright: ignore
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

    # ============ Partials instantiation ============
    model_inst = make_model(model, dataset)
    print_model(model_inst)
    train_dataset, val_dataset, test_dataset = make_datasets(
        run.training_mode, run.seed, dataset
    )
    opt_inst = make_optimizer(optimizer, model_inst)
    scheduler_inst = make_scheduler(scheduler, opt_inst, run.epochs)
    model_inst = to_cuda_(parallelize_model(model_inst))
    training_loss_inst = to_cuda_(make_training_loss(run.training_mode, training_loss))

    # Somehow, the dataloader will crash if it's not forked when using multiprocessing
    # along with Textual.
    mp.set_start_method("fork")
    train_loader_inst, val_loader_inst, test_loader_inst = make_dataloaders(
        data_loader,
        train_dataset,
        val_dataset,
        test_dataset,
        run.training_mode,
        run.seed,
    )
    init_wandb(run_name, model_inst, exp_conf)

    # ============ Training ============
    console.print(
        "Launching TUI...",
        style="bold cyan",
    )
    sleep(1)

    async def launch_with_async_gui():
        tui = TrainingUI(run_name, project_conf.LOG_SCALE_PLOT)
        task = asyncio.create_task(tui.run_async())
        await asyncio.sleep(0.5)  # Wait for the app to start up
        while not tui.is_running:
            await asyncio.sleep(0.01)  # Wait for the app to start up
        model_ckpt_path = load_model_ckpt(run.load_from, run.training_mode)
        common_args = dict(
            run_name=run_name,
            model=model_inst,
            model_ckpt_path=model_ckpt_path,
            training_loss=training_loss_inst,
            tui=tui,
        )
        if run.training_mode:
            print("Training started!")
            if training_loss_inst is None:
                raise ValueError("training_loss must be defined in training mode!")
            if val_loader_inst is None or train_loader_inst is None:
                raise ValueError(
                    "val_loader and train_loader must be defined in training mode!"
                )
            await trainer(
                train_loader=train_loader_inst,
                val_loader=val_loader_inst,
                opt=opt_inst,
                scheduler=scheduler_inst,
                **common_args,
                **asdict(run),
            ).train(
                epochs=run.epochs,
                val_every=run.val_every,
                visualize_every=run.viz_every,
                visualize_train_every=run.viz_train_every,
                visualize_n_samples=run.viz_num_samples,
            )
            print("Training finished!")
        else:
            print("Testing started!")
            if test_loader_inst is None:
                raise ValueError("test_loader must be defined in testing mode!")
            await tester(
                data_loader=test_loader_inst,
                **common_args,
            ).test(
                visualize_every=run.viz_every,
                **asdict(run),
            )
            print("Testing finished!")
        _ = await task

    asyncio.run(launch_with_async_gui())
