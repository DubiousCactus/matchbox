#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
from rich.console import Console
from rich.live import Live

if __name__ == "__main__":
    console = Console()
    status = console.status(
        "[bold cyan]Building experiment configurations...", spinner="monkey"
    )
    with Live(status, console=console):
        from hydra_zen import store, zen

        from bootstrap.launch_experiment import launch_experiment
        from conf import project as project_conf
        from conf.experiment import make_experiment_configs
        from utils import seed_everything

        make_experiment_configs()
    "============ Hydra-Zen ============"
    store.add_to_hydra_store(
        overwrite_ok=True
    )  # Overwrite Hydra's default config to update it
    zen(
        launch_experiment,
        pre_call=[
            lambda cfg: seed_everything(
                cfg.run.seed
            )  # training is the config of the training group, part of the base config
            if project_conf.REPRODUCIBLE
            else lambda: None,
        ],
    ).hydra_main(
        config_name="base_experiment",
        version_base="1.3",  # Hydra base version
    )
