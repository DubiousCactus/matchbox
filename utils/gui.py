#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
The fancy new GUI.
"""

import random
from collections import abc, namedtuple
from datetime import datetime
from functools import partial
from time import sleep
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import plotext as plt
import torch
from rich import box
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor

if __name__ != "__main__":
    from conf import project as project_conf
    from utils.helpers import BestNModelSaver
else:
    project_conf = namedtuple("project_conf", "Theme")(
        Theme=namedtuple("Theme", "TRAINING VALIDATION TESTING")(
            TRAINING=namedtuple("value", "value")("blue"),
            VALIDATION=namedtuple("value", "value")("green"),
            TESTING=namedtuple("value", "value")("cyan"),
        )
    )
    BestNModelSaver = TypeVar("BestNModelSaver")


class PlotextMixin(JupyterMixin):
    def __init__(self, p_make_plot):
        self.decoder = AnsiDecoder()
        self.mk_plot = p_make_plot

    def __rich_console__(self, console, options):
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = self.mk_plot(width=self.width, height=self.height)
        self.rich_canvas = Group(*self.decoder.decode(canvas))
        yield self.rich_canvas


# TODO: Make it a singleton so we can print from anywhere in the code, without passing a reference
# around.
class GUI:
    def __init__(self, run_name: str, plot_log_scale: bool) -> None:
        self._run_name = run_name
        self._plot_log_scale = plot_log_scale
        self._starting_epoch = 0  # TODO:
        self._layout = Layout()
        self._layout.split(
            Layout(name="header", size=2),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=2),
        )
        self._layout["main"].split_row(
            Layout(name="body", ratio=3, minimum_size=60),
            Layout(name="side"),
        )
        self._plot = Panel(
            Padding(
                Text(
                    "Waiting for training curves...",
                    justify="center",
                    style=Style(color="blue", bold=True),
                ),
                pad=30,
                style="on black",
            ),
            title="Training curves",
            expand=True,
        )
        self._layout["body"].update(self._plot)
        self._live = Live(self._layout, screen=True)
        self._console = self._live.console
        self._pbar = Progress(
            SpinnerColumn(spinner_name="monkey"),
            TextColumn(
                "[progress.description]{task.description} \[loss={task.fields[loss]:.3f}]"
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            # expand=True,
        )
        self._main_progress = Panel(  # TODO:
            self._pbar,
            title="Training epoch ?/?",
            expand=True,
        )
        self._layout["footer"].update(self._pbar)
        run_color = f"color({hash(run_name) % 255})"
        background_color = f"color({(hash(run_name) + 128) % 255})"
        self._layout["header"].update(
            Text(
                f"Running {run_name}",
                style=f"bold {run_color} on {background_color}",
                justify="center",
            )
        )
        self._logger = Table.grid(padding=0)
        self._logger.add_column(no_wrap=False)
        self._layout["side"].update(
            Panel(
                self._logger, title="Logs", border_style="bright_red", box=box.ROUNDED
            )
        )
        self.tasks = {
            "training": self._pbar.add_task(
                f"[{project_conf.Theme.TRAINING.value}]Training",
                visible=False,
                loss=torch.inf,
            ),
            "validation": self._pbar.add_task(
                f"[{project_conf.Theme.VALIDATION.value}]Validation",
                visible=False,
                loss=torch.inf,
            ),
            "testing": self._pbar.add_task(
                f"[{project_conf.Theme.TESTING.value}]Testing",
                visible=False,
                loss=torch.inf,
            ),
        }

    @property
    def console(self):
        return self._console

    def open(self) -> None:
        self._live.__enter__()

    def close(self) -> None:
        self._live.__exit__(None, None, None)

    def _track_iterable(self, iterable, task, total) -> Tuple[Iterable, Callable]:
        class LossHook:
            def __init__(self):
                self._loss = None

            def update_loss_hook(self, loss: float):
                self._loss = loss

        class SeqWrapper(abc.Iterator, LossHook):
            def __init__(
                self,
                seq: Sequence,
                len: int,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                super().__init__()
                self._sequence = seq
                self._idx = 0
                self._len = len
                self._update_hook = update_hook
                self._reset_hook = reset_hook

            def __next__(self):
                if self._idx >= self._len:
                    self._reset_hook()
                    raise StopIteration
                item = self._sequence[self._idx]
                self._update_hook(loss=self._loss)
                self._idx += 1
                return item

        class IteratorWrapper(abc.Iterator, LossHook):
            def __init__(
                self,
                iterator: Iterator | DataLoader,
                len: int,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                super().__init__()
                self._iterator = iter(iterator)
                self._len = len
                self._update_hook = update_hook
                self._reset_hook = reset_hook

            def __next__(self):
                try:
                    item = next(self._iterator)
                    self._update_hook(loss=self._loss)
                    return item
                except StopIteration:
                    self._reset_hook()
                    raise StopIteration

        def update_hook(task_id: TaskID, loss: Optional[float] = None):
            self._pbar.advance(task_id)
            if loss is not None:
                self._pbar.tasks[task_id].fields["loss"] = loss
            # TODO: Nice progress panel with overall progress and epoch progress
            # self._main_progress = Panel(self._pbar, title="Training epoch ?/?")
            # self._layout["footer"].update(self._main_progress)
            # self._live.refresh()

        def reset_hook(task_id: TaskID, total: int):
            self._pbar.reset(task_id, total=total, visible=False)

        wrapper = None
        update_p, reset_p = (
            partial(update_hook, task_id=task),
            partial(reset_hook, task, total),
        )
        if isinstance(iterable, abc.Sequence):
            wrapper = SeqWrapper(
                iterable,
                total,
                update_p,
                reset_p,
            )
        elif isinstance(iterable, (abc.Iterator, DataLoader)):
            wrapper = IteratorWrapper(
                iterable,
                total,
                update_p,
                reset_p,
            )
        else:
            raise ValueError(
                f"iterable must be a Sequence or an Iterator, got {type(iterable)}"
            )
        self._pbar.reset(task, total=total, visible=True)
        return wrapper, wrapper.update_loss_hook

    def track_training(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        task = self.tasks["training"]
        return self._track_iterable(iterable, task, total)

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        task = self.tasks["validation"]
        return self._track_iterable(iterable, task, total)

    def track_testing(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        task = self.tasks["testing"]
        return self._track_iterable(iterable, task, total)

    def print_header(self, text: str):
        self._layout["header"].update(text)

    def print(self, text: str | Tensor | Text):
        """
        Print text to the side panel.
        """
        if not isinstance(text, (str, Text)):
            raise NotImplementedError("Only text is supported for now.")

        self._logger.add_row(
            Text(datetime.now().strftime("[%H:%M] "), style="dim cyan"), text
        )

    def _make_plot(
        self,
        width,
        height,
        epoch: int,
        train_losses: List[float],
        val_losses: List[float],
        model_saver: Optional[BestNModelSaver] = None,
    ):
        """Plot the training and validation losses.
        Args:
            epoch (int): Current epoch number.
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.
        Returns:
            None
        """
        if self._plot_log_scale and any(
            loss_val <= 0 for loss_val in train_losses + val_losses
        ):
            raise ValueError(
                "Cannot plot on a log scale if there are non-positive losses."
            )
        plt.clf()
        plt.plotsize(width, height)
        plt.title(f"Training curves for {self._run_name}")
        plt.xlabel("Epoch")
        plt.theme("dark")
        if self._plot_log_scale:
            plt.ylabel("Loss (log scale)")
            plt.yscale("log")
        else:
            plt.ylabel("Loss")
        plt.grid(True, True)

        plt.plot(
            list(range(self._starting_epoch, epoch + 1)),
            train_losses,
            color=project_conf.Theme.TRAINING.value,
            # color="blue",
            label="Training loss",
        )
        plt.plot(
            list(range(self._starting_epoch, epoch + 1)),
            val_losses,
            color=project_conf.Theme.VALIDATION.value,
            # color="green",
            label="Validation loss",
        )
        best_metrics = (
            "["
            + ", ".join(
                [
                    f"{metric_name}={metric_value:.2e} "
                    for metric_name, metric_value in model_saver.best_metrics.items()
                ]
                if model_saver is not None
                else []
            )
            + "]"
        )
        if model_saver is not None:
            plt.scatter(
                [model_saver.min_val_loss_epoch],
                [model_saver.min_val_loss],
                color="red",
                marker="+",
                label=f"Best model {best_metrics}",
                style="inverted",
            )
        return plt.build()

    def plot(
        self,
        epoch: int,
        train_losses: List[float],
        val_losses: List[float],
        model_saver: Optional[BestNModelSaver] = None,
    ) -> None:
        mk_plot = partial(
            self._make_plot,
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            model_saver=model_saver,
        )
        self._plot = Panel(PlotextMixin(mk_plot), title="Training curves")
        self._layout["body"].update(self._plot)
        self._live.refresh()


if __name__ == "__main__":
    mnist = MNIST(root="data", train=False, download=True, transform=to_tensor)
    dataloader = DataLoader(mnist, 32, shuffle=True)
    gui = GUI("test-run", plot_log_scale=False)
    gui.open()  # TODO: Use a context manager, why not??
    try:
        gui.print("Hello, world!")
        gui.print(
            "Veeeeeeeeeeeeeeeeeeeryyyyyyyyyyyyyyyy looooooooooooooooooooooooooooong seeeeeeeeeeeenteeeeeeeeeeeeeeennnnnnnnnce!!!!!!!!!!!!!!!!!"
        )
        pbar, update_progress_loss = gui.track_training(range(10), 10)
        for i, e in enumerate(pbar):
            gui.print(f"[{i}/10]: We can iterate over iterables")
            sleep(0.1)
        train_losses, val_losses = [], []
        pbar, update_progress_loss = gui.track_validation(dataloader, len(dataloader))
        for i, e in enumerate(pbar):
            # gui.print(e)  # TODO: Make this work!
            if i % 10 == 0:
                train_losses.append(random.random())
                val_losses.append(random.random())
                update_progress_loss(random.random())
                gui.plot(epoch=i, train_losses=train_losses, val_losses=val_losses)
                gui.print(
                    f"[{i}/{len(dataloader)}]: We can also iterate over PyTorch dataloaders!"
                )
            sleep(0.01)
        gui.print("Goodbye, world!")
        sleep(1)
    except Exception as e:
        gui.close()
        raise e
    finally:
        gui.close()
