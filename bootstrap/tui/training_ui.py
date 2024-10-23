import asyncio
from itertools import cycle
from random import random
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
)

import numpy as np
import torch
import torch.multiprocessing as mp
from rich.text import Text
from textual.app import App, ComposeResult
from textual.reactive import var
from textual.widgets import (
    Footer,
    Header,
    Placeholder,
)
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
from bootstrap.tui import Plot_BestModel, Task
from bootstrap.tui.logger import Logger
from bootstrap.tui.widgets.plotting import PlotterWidget
from bootstrap.tui.widgets.progress import DatasetProgressBar


class TrainingUI(App):
    """
    A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework.
    """

    TITLE = "Matchbox Training TUI"
    CSS_PATH = "styles/training_ui.css"

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
        yield Logger(id="logger", classes="box")
        yield DatasetProgressBar()
        yield Placeholder(classes="box")
        yield Footer()

    def on_mount(self):
        self.query_one(PlotterWidget).loading = True

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark  # skipcq: PYL-W0201

    def watch_marker(self) -> None:
        """React to the marker type being changed."""
        self.sub_title = self.MARKERS[self.marker]  # skipcq: PYL-W0201
        self.query_one(PlotterWidget).marker = self.marker

    def action_marker(self) -> None:
        """Cycle to the next marker type."""
        self.marker = next(self._markers)  # skipcq: PTC-W0063

    def print_rich(self, message: Any):
        self.query_one(Logger).wite(message, is_stderr=False)

    def track_training(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the training process, and a
        progress bar hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TRAINING, total
        )

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the validation process, and a
        progress bar hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.VALIDATION, total
        )

    def track_testing(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        """Return an iterable that tracks the progress of the testing process, and a
        progress bar hook to update the loss value at each iteration."""
        return self.query_one(DatasetProgressBar).track_iterable(
            iterable, Task.TESTING, total
        )

    def plot(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        best_model: Optional[Plot_BestModel] = None,
    ) -> None:
        """Plot the training and validation losses for the current epoch."""
        self.query_one(PlotterWidget).loading = False
        self.query_one(PlotterWidget).update(epoch, train_loss, val_loss, best_model)

    def set_start_epoch(self, start_epoch: int) -> None:
        """Set the starting epoch for the plotter widget."""
        self.query_one(PlotterWidget).set_start_epoch(start_epoch)


async def run_my_app():
    gui = TrainingUI("test-run", log_scale=False)
    task = asyncio.create_task(gui.run_async())
    while not gui.is_running:
        await asyncio.sleep(0.01)  # Wait for the app to start up
    gui.print_rich("Hello, World!")
    await asyncio.sleep(2)
    gui.print_rich(Text("Let's log some tensors :)", style="bold magenta"))
    await asyncio.sleep(0.5)
    gui.print_rich(torch.rand(2, 4))
    await asyncio.sleep(2)
    gui.print_rich(Text("How about some numpy arrays?!", style="italic green"))
    await asyncio.sleep(1)
    gui.print_rich(np.random.rand(3, 3))
    pbar, update_progress_loss = gui.track_training(range(10), 10)
    for i, e in enumerate(pbar):
        gui.print_rich(f"[{i+1}/10]: We can iterate over iterables")
        gui.print_rich(e)
        await asyncio.sleep(0.1)
    await asyncio.sleep(2)
    mnist = MNIST(root="data", train=False, download=True, transform=to_tensor)
    # Somehow, the dataloader will crash if it's not forked when using multiprocessing
    # along with Textual.
    mp.set_start_method("fork")
    dataloader = DataLoader(mnist, 32, shuffle=True, num_workers=2)
    pbar, update_progress_loss = gui.track_validation(dataloader, len(dataloader))
    for i, batch in enumerate(pbar):
        await asyncio.sleep(0.01)
        if i % 10 == 0:
            gui.print_rich(batch)
            update_progress_loss(random())
            gui.plot(epoch=i, train_loss=random(), val_loss=random())
            gui.print_rich(
                f"[{i+1}/{len(dataloader)}]: "
                + "We can also iterate over PyTorch dataloaders!"
            )
        if i == 0:
            gui.print_rich(batch)
    gui.print_rich("Goodbye, world!")
    _ = await task


if __name__ == "__main__":
    asyncio.run(run_my_app())
