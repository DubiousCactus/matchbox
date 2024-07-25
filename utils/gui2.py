# Let's use Textual to rewrite the GUI with better features.

import asyncio
from datetime import datetime
from itertools import cycle
from random import random
from typing import (
    List,
)

import numpy as np
import torch
from rich.console import Group, RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult, RenderResult
from textual.reactive import var
from textual.widgets import (
    Footer,
    Header,
    Placeholder,
    RichLog,
    Static,
)
from textual_plotext import PlotextPlot


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
        # self.plt.plot(self._time, self._data, marker=self.marker)
        if len(self._train_losses) > 0:
            assert (self._epoch + 1) == len(self._train_losses)
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
        yield Placeholder(classes="box")
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
        if isinstance(message, (str, RenderableType)):
            logger.write(
                Group(
                    Text(datetime.now().strftime("[%H:%M] "), style="dim cyan", end=""),
                    message,
                ),
            )
        elif isinstance(message, (torch.Tensor, np.ndarray)):
            logger.write(
                Group(
                    Text(
                        datetime.now().strftime("[%H:%M] "),
                        style="dim cyan",
                        end="",
                    ),
                    Pretty(message),
                )
            )


async def run_my_app():
    gui = GUI()
    task = asyncio.create_task(gui.run_async())
    await asyncio.sleep(1)  # Wait for the app to start up
    gui.print("Hello, World!")
    await asyncio.sleep(2)
    gui.print(Text("Let's log some tensors :)", style="bold magenta"))
    await asyncio.sleep(0.5)
    gui.print(torch.rand(2, 4))
    await asyncio.sleep(2)
    gui.print(Text("How about some numpy arrays?!", style="italic green"))
    await asyncio.sleep(1)
    gui.print(np.random.rand(3, 3))
    await asyncio.sleep(3)
    gui.print("...")
    await asyncio.sleep(3)
    gui.print("Go on... Press 'p'! I know you want to!")
    await asyncio.sleep(4)
    gui.print("COME ON PRESS P!!!!")
    _ = await task


if __name__ == "__main__":
    asyncio.run(run_my_app())
