#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base trainer class.
"""

import os
import os.path as osp
import signal
from typing import List, Tuple, Union

import numpy as np
import plotext as plt
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

import wandb
from conf import project as project_conf
from utils import to_cuda
from utils.helpers import BestNModelSaver
from utils.training import visualize_model_predictions


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        opt: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self._model = model
        self._opt = opt
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._epoch = 0
        self._running = True
        self._model_saver = BestNModelSaver(
            project_conf.BEST_N_MODELS_TO_KEEP, self._save_checkpoint
        )
        self._pbar = tqdm(total=len(self._train_loader), desc="Training")
        signal.signal(signal.SIGINT, self._terminator)
        signal.siginterrupt(signal.SIGINT, False)

    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform a single training/validation iteration on a batch and return the loss.
        We want to keep the code DRY and avoid making mistakes, so write this code only once at the
        cost of many function calls!
        """
        # TODO: Implement this
        # x, y = to_cuda(batch)
        # y_hat = self._model(x)
        # loss = my_loss(y_hat, y)
        # return loss
        raise NotImplementedError

    def _train_epoch(self, description: str, epoch: int) -> float:
        epoch_loss = MeanMetric()
        self._pbar.reset()
        self._pbar.set_description(description)
        " ==================== Training loop for one epoch ==================== "
        for batch in self._train_loader:
            if not self._running:
                print("[!] Training stopped.")
                break
            self._opt.zero_grad()
            loss = self._train_val_iteration(
                batch
            )  # User implementation goes here (train.py)
            loss.backward()
            self._opt.step()
            epoch_loss.update(loss.item())
            self._pbar.set_description_str(
                f"\033[93m{description} [loss={epoch_loss.compute().item():.4f} / min_val_loss={self._model_saver.min_val_loss:.4f}]\033[0m"
            )
            self._pbar.update()
        epoch_loss = epoch_loss.compute().item()
        wandb.log({"train_loss": epoch_loss}, step=epoch)
        return epoch_loss

    def _val_epoch(self, description: str, visualize: bool, epoch: int) -> float:
        "==================== Validation loop for one epoch ===================="
        with torch.no_grad():
            val_loss = MeanMetric()
            for i, batch in enumerate(self._val_loader):
                if not self._running:
                    print("[!] Training stopped.")
                    break
                if i % 4 == 0:
                    self._pbar.colour = (
                        "yellow" if self._pbar.colour == "green" else "green"
                    )
                loss = self._train_val_iteration(
                    batch
                )  # User implementation goes here (train.py)
                val_loss.update(loss.item())
                self._pbar.set_description(
                    f"\033[92m{description} [loss={val_loss.compute():.4f}) / min_val_loss={self._model_saver.min_val_loss:.4f}]\033[0m"
                )
                " ==================== Visualization ==================== "
                if visualize:
                    visualize_model_predictions(
                        self._model, batch
                    )  # User implementation goes here (utils/training.py)
            val_loss = val_loss.compute().item()
            wandb.log({"val_loss": val_loss}, step=epoch)
            self._model_saver(epoch, val_loss)
            return val_loss

    def train(
        self,
        epochs: int = 10,
        val_every: int = 1,  # Validate every n epochs
        visualize_every: int = 10,  # Visualize every n validations
    ):
        print(f"[*] Training for {epochs} epochs")
        plt.title("Training losses")
        plt.grid(True, True)
        train_losses, val_losses = [], []
        for epoch in range(self._epoch, epochs):
            self._model.train()
            self._pbar.colour = "yellow"
            train_losses.append(
                self._train_epoch(f"Epoch {epoch}/{epochs}: Training", epoch)
            )
            if epoch % val_every == 0:
                self._model.eval()
                self._pbar.colour = "green"
                val_losses.append(
                    self._val_epoch(
                        f"Epoch {epoch}/{epochs}: Validation",
                        visualize_every > 0 and epoch % visualize_every == 0,
                        epoch,
                    )
                )
            plt.xlim(0, epoch)
            plt.ylim(min(train_losses + val_losses), max(train_losses + val_losses))
            plt.plot(list(range(0, epoch)), train_losses, color="blue")
            plt.plot(list(range(0, epoch)), val_losses, color="green")
            plt.show()

        self._pbar.close()

    def _save_checkpoint(self, val_loss: float, ckpt_path: str, **kwargs) -> None:
        # Check if the checkpoint directory exists
        if not osp.exists(project_conf.CKPT_PATH):
            os.makedirs(project_conf.CKPT_PATH)
        torch.save(
            {
                **{
                    "model_ckpt": self._model.state_dict(),
                    "opt_ckpt": self._opt.state_dict(),
                    "epoch": self._epoch,
                    "val_loss": val_loss,
                },
                **kwargs,
            },
            ckpt_path,
        )

    def _load_checkpoint(self, ckpt_path: str) -> None:
        print(f"[*] Restoring from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        try:
            self._model.load_state_dict(ckpt["model_ckpt"])
        except Exception:
            if project_conf.PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH:
                print(
                    "[!] Partially loading model weights (no full match between model and checkpoint)"
                )
                self._model.load_state_dict(ckpt["model_ckpt"], strict=False)
        self._opt.load_state_dict(ckpt["opt_ckpt"])
        self._epoch = ckpt["epoch"]
        self._min_val_loss = ckpt["val_loss"]

    def _terminator(self, sig, frame):
        if (
            project_conf.SIGINT_BEHAVIOR
            == project_conf.TerminationBehavior.WAIT_FOR_EPOCH_END
        ):
            print("[!] SIGINT received. Waiting for epoch to end.")
            self._running = False
        elif (
            project_conf.SIGINT_BEHAVIOR == project_conf.TerminationBehavior.ABORT_EPOCH
        ):
            print("[!] SIGINT received. Aborting epoch.")
            raise KeyboardInterrupt
