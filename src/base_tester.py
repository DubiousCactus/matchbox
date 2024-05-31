#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base tester class.
"""

import signal
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from conf import project as project_conf
from src.base_trainer import BaseTrainer
from utils import to_cuda, update_pbar_str


class BaseTester(BaseTrainer):
    def __init__(
        self,
        run_name: str,
        data_loader: DataLoader,
        model: torch.nn.Module,
        model_ckpt_path: str,
        **kwargs,
    ) -> None:
        """Base trainer class.
        Args:
            model (torch.nn.Module): Model to train.
            opt (torch.optim.Optimizer): Optimizer to use.
            train_loader (torch.utils.data.DataLoader): Training dataloader.
            val_loader (torch.utils.data.DataLoader): Validation dataloader.
        """
        self._run_name = run_name
        self._model = model
        assert model_ckpt_path is not None, "No model checkpoint path provided."
        self._load_checkpoint(model_ckpt_path, model_only=True)
        self._data_loader = data_loader
        self._running = True
        self._pbar = tqdm(total=len(self._data_loader), desc="Testing")
        signal.signal(signal.SIGINT, self._terminator)

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        pass

    @to_cuda
    def _test_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so this code calls the BaseTrainer._train_val_iteration() method.
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        # x, y = batch  # type: ignore
        # y_hat = self._model(x)
        # TODO: Compute your metrics here!
        return {}

    def test(self, visualize_every: int = 0, **kwargs):
        """Computes the average loss on the test set.
        Args:
            visualize_every (int, optional): Visualize the model predictions every n batches.
            Defaults to 0 (no visualization).
        """
        test_loss, test_metrics = MeanMetric(), defaultdict(MeanMetric)
        self._model.eval()
        self._pbar.reset()
        self._pbar.set_description("Testing")
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TESTING.value]
        " ==================== Training loop for one epoch ==================== "
        for i, batch in enumerate(self._data_loader):
            if not self._running:
                print("[!] Testing aborted.")
                break
            loss, metrics = self._test_iteration(batch)
            test_loss.update(loss.item())
            for k, v in metrics.items():
                test_metrics[k].update(v.item())
            update_pbar_str(
                self._pbar,
                f"Testing [loss={test_loss.compute():.4f}]",
                color_code,
            )
            " ==================== Visualization ==================== "
            if visualize_every > 0 and (i + 1) % visualize_every == 0:
                self._visualize(batch, i)

            self._pbar.update()
        self._pbar.close()
        print("=" * 81)
        print("==" + " " * 31 + " Test results " + " " * 31 + "==")
        print("=" * 81)
        for k, v in test_metrics.items():
            print(f"\t -> {k}: {v.compute().item():.2f}")
        print(f"\t -> Average loss: {test_loss:.4f}")
        print("_" * 81)
