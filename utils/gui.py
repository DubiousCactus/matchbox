#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
The fancy new GUI.
"""

from collections import abc
from functools import partial
from time import sleep
from typing import Callable, Iterable, Iterator, Sequence

from rich.layout import Layout
from rich.live import Live
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
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor


class GUI:
    def __init__(self) -> None:
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
        self._live = Live(self._layout, screen=True)
        self._console = self._live.console
        self._pbar = Progress(
            SpinnerColumn(spinner_name="monkey"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self._live.console,
            # expand=True,
        )
        self._main_progress = Panel(
            self._pbar,
            title="Training epoch ?/?",
            expand=True,
        )
        self._layout["footer"].update(self._pbar)
        self._layout["header"].update(Panel("Stuff", title="Training run ..."))
        self.tasks = {
            "training": self._pbar.add_task("Training", visible=False),
            "validation": self._pbar.add_task("Validation", visible=False),
            "testing": self._pbar.add_task("Testing", visible=False),
        }

    @property
    def console(self):
        return self._console

    def open(self) -> None:
        self._live.__enter__()

    def close(self) -> None:
        self._live.__exit__(None, None, None)

    def _track_iterable(self, iterable, task, total) -> Iterable:
        class SeqWrapper(abc.Iterator):
            def __init__(
                self,
                seq: Sequence,
                len: int,
                main_progress,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                self._sequence = seq
                self._idx = 0
                self._len = len
                self.__main_progress = main_progress
                self._update_hook = update_hook
                self._reset_hook = reset_hook

            def __next__(self):
                if self._idx >= self._len:
                    self._reset_hook()
                    raise StopIteration
                item = self._sequence[self._idx]
                self._update_hook()
                # self.__main_progress.update(self.__pbar)
                self._idx += 1
                return item

        class IteratorWrapper(abc.Iterator):
            def __init__(
                self,
                iterator: Iterator | DataLoader,
                len: int,
                main_progress,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                self._iterator = iter(iterator)
                self._len = len
                self.__main_progress = main_progress
                self._update_hook = update_hook
                self._reset_hook = reset_hook

            def __next__(self):
                try:
                    item = next(self._iterator)
                    self._update_hook()
                    return item
                except StopIteration:
                    self._reset_hook()
                    raise StopIteration

        def update_hook(task_id: TaskID):
            self._pbar.advance(task_id)

        def reset_hook(task_id: TaskID, total: int):
            self._pbar.reset(task_id, total=total, visible=False)

        wrapper = None
        update_p, reset_p = (
            partial(update_hook, task),
            partial(reset_hook, task, total),
        )
        if isinstance(iterable, abc.Sequence):
            wrapper = SeqWrapper(
                iterable,
                total,
                self._main_progress,
                update_p,
                reset_p,
            )
        elif isinstance(iterable, (abc.Iterator, DataLoader)):
            wrapper = IteratorWrapper(
                iterable,
                total,
                self._main_progress,
                update_p,
                reset_p,
            )
        else:
            raise ValueError(
                f"iterable must be a Sequence or an Iterator, got {type(iterable)}"
            )
        self._pbar.reset(task, total=total, visible=True)
        return wrapper

    def track_training(self, iterable, description: str, total: int) -> Iterable:
        task = self.tasks["training"]
        return self._track_iterable(iterable, task, total)

    def track_validation(self, iterable, description: str, total: int) -> Iterable:
        task = self.tasks["validation"]
        return self._track_iterable(iterable, task, total)

    def print_footer(self, text: str):
        self._layout["footer"].update(text)

    def print_header(self, text: str):
        self._layout["header"].update(text)

    def print(self, text: str):
        """
        Print text to the side panel.
        """
        # NOTE: We could use a table to append messages in the renderable. I don't really know of
        # another way to print stuff in a specific panel.
        self._layout["side"].update(Panel(text, title="Logs"))


if __name__ == "__main__":
    mnist = MNIST(root="data", train=False, download=True, transform=to_tensor)
    dataloader = DataLoader(mnist, 32, shuffle=True)
    gui = GUI()
    gui.open()
    try:
        for i, e in enumerate(gui.track_training(range(10), "Training", 10)):
            gui.print(f"{i}/10")
            sleep(0.1)
        for i, e in enumerate(
            gui.track_validation(dataloader, "Validation", len(dataloader))
        ):
            gui.print(e)  # TODO: Make this work!
            gui.print(f"{i}/{len(dataloader)}")
            sleep(0.01)
    except Exception:
        gui.close()
    gui.close()
