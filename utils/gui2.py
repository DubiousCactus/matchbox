# Let's use Textual to rewrite the GUI with better features.

import asyncio
from collections import abc
from datetime import datetime
from enum import Enum
from functools import partial
from itertools import cycle
from random import randint, random
from time import sleep
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from rich.console import Group, RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult, RenderResult
from textual.containers import Horizontal
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
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._epoch = 0

    def on_mount(self) -> None:
        """Plot the data using Plotext."""
        self.plt.title(self._title)
        self.plt.xlabel("Epoch")
        self.plt.ylabel("Loss")  # TODO: update
        self.plt.grid(True, True)

    def replot(self) -> None:
        """Redraw the plot."""
        self.plt.clear_data()
        if len(self._train_losses) > 0:
            assert len(self._val_losses) == len(self._train_losses)
            self.plt.plot(
                list(range(0, self._epoch + 1)),  # TODO: start epoch
                self._train_losses,
                # color=project_conf.Theme.TRAINING.value, # TODO:
                color="blue",
                label="Training loss",
                marker=self.marker,
            )
            self.plt.plot(
                list(range(0, self._epoch + 1)),  # TODO: start epoch
                self._val_losses,
                # color=project_conf.Theme.VALIDATION.value, # TODO:
                color="green",
                label="Validation loss",
                marker=self.marker,
            )
        self.refresh()

    def update(
        self, epoch: int, train_losses: List[float], val_losses: List[float]
    ) -> None:
        """Update the data for the training curves plot.

        Args:
            epoch: (int) The current epoch number.
            train_losses: (List[float]) The list of training losses.
            val_losses: (List[float]) The list of validation losses.
        """
        # TODO: We only need to append to the losses. Do we need to keep track of them in the
        # trianing loop? If not we should only keep track of the last one and let the GUI keep
        # track of them all.
        self._epoch = epoch
        self._train_losses = train_losses
        self._val_losses = val_losses
        self.replot()

    def _watch_marker(self) -> None:
        """React to the marker being changed."""
        self.replot()


# TODO: Also make a Rich renderable for Tensors


class TensorWidget(Static):  # TODO: PRETTYER
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def render(self) -> RenderResult:
        return Group(
            Text(
                datetime.now().strftime("[%H:%M] "),
                style="dim cyan",
                end="",
            ),
            Pretty(self.tensor),
        )


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

    # def __init__(self):
    #     self.description = None
    #     self.total = None
    #     self.progress = 0

    def compose(self) -> ComposeResult:
        with Horizontal():
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

        def update_hook(loss: Optional[float] = None):
            self.query_one(ProgressBar).advance()
            if loss is not None:
                self.query_one("#progress_label").update(
                    self.DESCRIPTIONS[task] + f"[loss={loss:.4f}]"
                )

        def reset_hook(total: int):
            sleep(0.5)
            self.query_one(ProgressBar).update(total=100, progress=0)
            self.query_one("#progress_label").update(self.DESCRIPTIONS[Task.IDLE])

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
        self.query_one("#progress_label").update(self.DESCRIPTIONS[task])
        return wrapper, wrapper.update_loss_hook


class GUI(App):
    """A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework."""

    CSS_PATH = "style.css"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("m", "marker", "Cycle example markers"),
        ("ctrl+z", "suspend_progress"),
    ]

    MARKERS = {
        "dot": "Dot",
        "hd": "High Definition",
        "fhd": "Higher Definition",
        "braille": "Braille",
        "sd": "Standard Definition",
    }

    marker: var[str] = var("sd")

    def __init__(self) -> None:
        """Initialise the application."""
        super().__init__()
        self._markers = cycle(self.MARKERS.keys())

    def compose(self) -> ComposeResult:
        yield Header()
        yield PlotterWidget(title="Trainign curves for run-name", classes="box")
        yield RichLog(
            highlight=True, markup=True, wrap=True, id="logger", classes="box"
        )
        # yield Placeholder(classes="box")
        yield DatasetProgressBar()
        yield Placeholder(classes="box")
        yield Footer()

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def watch_marker(self) -> None:
        """React to the marker type being changed."""
        self.sub_title = self.MARKERS[self.marker]
        self.query_one(PlotterWidget).marker = self.marker

    def action_marker(self) -> None:
        """Cycle to the next marker type."""
        self.marker = next(self._markers)

    def on_key(self, event) -> None:
        logger: RichLog = self.query_one(RichLog)
        logger.write(
            Group(
                Text(datetime.now().strftime("[%H:%M] "), style="dim cyan", end=""),
                f"Key pressed: {event.key!r}",
            ),
        )
        if event.key == "t":
            logger.write(
                Group(
                    Text(
                        datetime.now().strftime("[%H:%M] "),
                        style="dim cyan",
                        end="",
                    ),
                    Pretty(torch.rand(randint(1, 12), randint(1, 12))),
                )
            )
        elif event.key == "p":
            self.query_one(PlotterWidget).update(
                epoch=9,
                train_losses=[random() for _ in range(10)],
                val_losses=[random() for _ in range(10)],
            )

    def print(self, message: RenderableType | str | torch.Tensor | np.ndarray):
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
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TRAINING, total
        )

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.VALIDATION, total
        )

    def track_testing(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TESTING, total
        )

    def plot(self, epoch: int, train_losses: List[float], val_losses: List[float]):
        self.query_one(PlotterWidget).update(epoch, train_losses, val_losses)


async def run_my_app():
    gui = GUI()
    task = asyncio.create_task(gui.run_async())
    await asyncio.sleep(0.1)  # Wait for the app to start up
    gui.print("Hello, World!")
    # await asyncio.sleep(2)
    # gui.print(Text("Let's log some tensors :)", style="bold magenta"))
    # await asyncio.sleep(0.5)
    # gui.print(torch.rand(2, 4))
    # await asyncio.sleep(2)
    # gui.print(Text("How about some numpy arrays?!", style="italic green"))
    # await asyncio.sleep(1)
    # gui.print(np.random.rand(3, 3))
    # await asyncio.sleep(3)
    # gui.print("...")
    # await asyncio.sleep(3)
    # gui.print("Go on... Press 'p'! I know you want to!")
    # await asyncio.sleep(4)
    # gui.print("COME ON PRESS P!!!!")
    # await asyncio.sleep(1)
    # pbar, update_progress_loss = gui.track_training(range(10), 10)
    # for i, e in enumerate(pbar):
    #     gui.print(f"[{i+1}/10]: We can iterate over iterables")
    #     gui.print(e)
    #     # sleep(0.1)
    #     await asyncio.sleep(0.1)
    # await asyncio.sleep(5)
    mnist = MNIST(root="data", train=False, download=True, transform=to_tensor)
    dataloader = DataLoader(mnist, 32, shuffle=True)
    train_losses, val_losses = [], []
    pbar, update_progress_loss = gui.track_validation(dataloader, len(dataloader))
    for i, batch in enumerate(pbar):
        if i % 10 == 0:
            await asyncio.sleep(0.01)
            gui.print(batch)
            train_losses.append(random())
            val_losses.append(random())
            update_progress_loss(random())
            gui.plot(epoch=i, train_losses=train_losses, val_losses=val_losses)
            gui.print(
                f"[{i+1}/{len(dataloader)}]: We can also iterate over PyTorch dataloaders!"
            )
        if i == 0:
            gui.print(batch)
    gui.print("Goodbye, world!")
    _ = await task


if __name__ == "__main__":
    asyncio.run(run_my_app())
