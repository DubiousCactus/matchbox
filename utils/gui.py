
import asyncio
from collections import abc
from datetime import datetime
from enum import Enum
from functools import partial
from itertools import cycle
from random import random
from time import sleep
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.multiprocessing as mp
from rich.console import Group, RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Center
from textual.reactive import var
from textual.widgets import (
    Footer,
    Header,
    Label,
    Placeholder,
    ProgressBar,
    RichLog,
    Static,
)
from textual_plotext import PlotextPlot
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor


class PlotterWidget(PlotextPlot):
    marker: var[str] = var("sd")

    """The type of marker to use for the plot."""

    def __init__(
        self,
        title: str,
        use_log_scale: bool = False,
        *,
        name: str | None = None,
        id: str | None = None,  # pylint:disable=redefined-builtin
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialise the training curves plotter widget.

        Args:
            name: The name of the plotter widget.
            id: The ID of the plotter widget in the DOM.
            classes: The CSS classes of the plotter widget.
            disabled: Whether the plotter widget is disabled or not.
        """
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._title = title
        self._log_scale = use_log_scale
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._start_epoch = 0
        self._epoch = 0

    def on_mount(self) -> None:
        """Plot the data using Plotext."""
        self.plt.title(self._title)
        self.plt.xlabel("Epoch")
        if self._log_scale:
            self.plt.ylabel("Loss (log scale)")
            self.plt.yscale("log")
        else:
            self.plt.ylabel("Loss")
        self.plt.grid(True, True)

    def replot(self) -> None:
        """Redraw the plot."""
        self.plt.clear_data()
        if self._log_scale and (
            self._train_losses[-1] <= 0 or self._val_losses[-1] <= 0
        ):
            raise ValueError(
                "Cannot plot on a log scale if there are non-positive losses."
            )
        if len(self._train_losses) > 0:
            assert len(self._val_losses) == len(self._train_losses)
            self.plt.plot(
                list(range(self._start_epoch, self._epoch + 1)),
                self._train_losses,
                color="blue",  # TODO: Theme
                label="Training loss",
                marker=self.marker,
            )
            self.plt.plot(
                list(range(self._start_epoch, self._epoch + 1)),
                self._val_losses,
                color="green",  # TODO: Theme
                label="Validation loss",
                marker=self.marker,
            )
        self.refresh()

    def set_start_epoch(self, start_epoch: int):
        self._start_epoch = start_epoch

    def update(
        self, epoch: int, train_loss: float, val_loss: Optional[float] = None
    ) -> None:
        """Update the data for the training curves plot.

        Args:
            epoch: (int) The current epoch number.
            train_loss: (float) The last training loss.
            val_loss: (float) The last validation loss.
        """
        self._epoch = epoch
        self._train_losses.append(train_loss)
        self._val_losses.append(
            val_loss if val_loss is not None else self._val_losses[-1]
        )
        self.replot()

    def _watch_marker(self) -> None:
        """React to the marker being changed."""
        self.replot()


# TODO: Also make a Rich renderable for Tensors (using tables?)


class Task(Enum):
    IDLE = -1
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class DatasetProgressBar(Static):
    """A progress bar for PyTorch dataloader iteration."""

    DESCRIPTIONS = {
        Task.IDLE: Text("Waiting for work..."),
        Task.TRAINING: Text("Training: ", style="bold blue"),
        Task.VALIDATION: Text("Validation: ", style="bold green"),
        Task.TESTING: Text("Testing: ", style="bold yellow"),
    }

    def compose(self) -> ComposeResult:
        with Center():
            yield Label(self.DESCRIPTIONS[Task.IDLE], id="progress_label")
            yield ProgressBar()

    def track_iterable(
        self,
        iterable: Iterable | Sequence | Iterator | DataLoader,
        task: Task,
        total: int,
    ) -> Tuple[Iterable, Callable]:
        class LossHook:
            def __init__(self):
                self._loss = None

            def update_loss_hook(
                self, loss: float, min_val_loss: Optional[float] = None
            ) -> None:
                """Update the loss value in the progress bar."""
                # TODO: min_val_loss during validation, val_loss during training. Ideally the
                # second parameter would be super flexible (use a dict then).
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

        def update_hook(loss: Optional[float] = None):
            self.query_one(ProgressBar).advance()
            if loss is not None:
                plabel: Label = self.query_one("#progress_label")  # type: ignore
                plabel.update(self.DESCRIPTIONS[task] + f"[loss={loss:.4f}]")

        def reset_hook(total: int):
            sleep(0.5)
            self.query_one(ProgressBar).update(total=100, progress=0)
            plabel: Label = self.query_one("#progress_label")  # type: ignore
            plabel.update(self.DESCRIPTIONS[Task.IDLE])

        wrapper = None
        update_p, reset_p = (
            partial(update_hook),
            partial(reset_hook, total),
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
        self.query_one(ProgressBar).update(total=total, progress=0)
        plabel: Label = self.query_one("#progress_label")  # type: ignore
        plabel.update(self.DESCRIPTIONS[task])
        return wrapper, wrapper.update_loss_hook


class GUI(App):
    """A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework."""

    TITLE = "Matchbox TUI"
    CSS_PATH = "style.css"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("p", "marker", "Change plotter style"),
        ("ctrl+z", "suspend_progress"),
    ]

    MARKERS = {
        "dot": "Dot",
        "hd": "High Definition",
        "fhd": "Higher Definition",
        "braille": "Braille",
        "sd": "Standard Definition",
    }

    marker: var[str] = var("hd")

    def __init__(self, run_name: str, log_scale: bool) -> None:
        """Initialise the application."""
        super().__init__()
        self._markers = cycle(self.MARKERS.keys())
        self._log_scale = log_scale
        self.run_name = run_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield PlotterWidget(
            title=f"Trainign curves for {self.run_name}",
            use_log_scale=self._log_scale,
            classes="box",
        )
        yield RichLog(
            highlight=True, markup=True, wrap=True, id="logger", classes="box"
        )
        yield DatasetProgressBar()
        yield Placeholder(classes="box")
        yield Footer()

    def on_mount(self):
        self.query_one(PlotterWidget).loading = True

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def watch_marker(self) -> None:
        """React to the marker type being changed."""
        self.sub_title = self.MARKERS[self.marker]
        self.query_one(PlotterWidget).marker = self.marker

    def action_marker(self) -> None:
        """Cycle to the next marker type."""
        self.marker = next(self._markers)

    def print(self, message: Any):
        logger: RichLog = self.query_one(RichLog)
        if isinstance(message, (RenderableType, str)):
            logger.write(
                Group(
                    Text(
                        datetime.now().strftime("[%H:%M] "),
                        style="dim cyan",
                        end="",
                    ),
                    message,
                ),
            )
        else:
            ppable, pp_msg = True, None
            try:
                pp_msg = Pretty(message)
            except Exception:
                ppable = False
            if ppable and pp_msg is not None:
                logger.write(
                    Group(
                        Text(
                            datetime.now().strftime("[%H:%M] "),
                            style="dim cyan",
                            end="",
                        ),
                        Text(str(type(message)) + " ", style="italic blue", end=""),
                        pp_msg,
                    )
                )
            else:
                try:
                    logger.write(
                        Group(
                            Text(
                                datetime.now().strftime("[%H:%M] "),
                                style="dim cyan",
                                end="",
                            ),
                            message,
                        ),
                    )
                except Exception as e:
                    logger.write(
                        Group(
                            Text(
                                datetime.now().strftime("[%H:%M] "),
                                style="dim cyan",
                                end="",
                            ),
                            Text("Logging error: ", style="bold red"),
                            Text(str(e), style="bold red"),
                        )
                    )

    def track_training(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the training process, and a progress bar
        hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TRAINING, total
        )

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the validation process, and a progress bar
        hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.VALIDATION, total
        )

    def track_testing(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the testing process, and a progress bar
        hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TESTING, total
        )

    def plot(
        self, epoch: int, train_loss: float, val_loss: Optional[float] = None
    ) -> None:
        """Plot the training and validation losses for the current epoch."""
        self.query_one(PlotterWidget).loading = False
        self.query_one(PlotterWidget).update(epoch, train_loss, val_loss)

    def set_start_epoch(self, start_epoch: int) -> None:
        """Set the starting epoch for the plotter widget."""
        self.query_one(PlotterWidget).set_start_epoch(start_epoch)


async def run_my_app():
    gui = GUI("test-run", log_scale=False)
    task = asyncio.create_task(gui.run_async())
    while not gui.is_running:
        await asyncio.sleep(0.01)  # Wait for the app to start up
    gui.print("Hello, World!")
    await asyncio.sleep(2)
    gui.print(Text("Let's log some tensors :)", style="bold magenta"))
    await asyncio.sleep(0.5)
    gui.print(torch.rand(2, 4))
    await asyncio.sleep(2)
    gui.print(Text("How about some numpy arrays?!", style="italic green"))
    await asyncio.sleep(1)
    gui.print(np.random.rand(3, 3))
    pbar, update_progress_loss = gui.track_training(range(10), 10)
    for i, e in enumerate(pbar):
        gui.print(f"[{i+1}/10]: We can iterate over iterables")
        gui.print(e)
        await asyncio.sleep(0.1)
    await asyncio.sleep(2)
    mnist = MNIST(root="data", train=False, download=True, transform=to_tensor)
    # Somehow, the dataloader will crash if it's not forked when using multiprocessing along with
    # Textual.
    mp.set_start_method("fork")
    dataloader = DataLoader(mnist, 32, shuffle=True, num_workers=2)
    pbar, update_progress_loss = gui.track_validation(dataloader, len(dataloader))
    for i, batch in enumerate(pbar):
        await asyncio.sleep(0.01)
        if i % 10 == 0:
            gui.print(batch)
            update_progress_loss(random())
            gui.plot(epoch=i, train_loss=random(), val_loss=random())
            gui.print(
                f"[{i+1}/{len(dataloader)}]: We can also iterate over PyTorch dataloaders!"
            )
        if i == 0:
            gui.print(batch)
    gui.print("Goodbye, world!")
    _ = await task


if __name__ == "__main__":
    asyncio.run(run_my_app())
