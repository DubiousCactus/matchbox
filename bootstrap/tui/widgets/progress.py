from collections import abc
from functools import partial
from time import sleep
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Center
from textual.widgets import (
    Label,
    ProgressBar,
    Static,
)
from torch.utils.data.dataloader import DataLoader

from bootstrap.tui import Task


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

            def update_loss_hook(self, loss: float) -> None:
                """Update the loss value in the progress bar."""
                # TODO: min_val_loss during validation, val_loss during training.
                # Ideally the second parameter would be super flexible (use a dict
                # then).
                self._loss = loss

        class SeqWrapper(abc.Iterator, LossHook):
            def __init__(
                self,
                seq: Sequence,
                length: int,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                super().__init__()
                self._sequence = seq
                self._idx = 0
                self._len = length
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
                length: int,
                update_hook: Callable,
                reset_hook: Callable,
            ):
                super().__init__()
                self._iterator = iter(iterator)
                self._len = length
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
                self.query_one("#progress_label", Label).update(
                    self.DESCRIPTIONS[task] + f"[loss={loss:.4f}]"
                )

        def reset_hook():
            sleep(0.5)
            self.query_one(ProgressBar).update(total=100, progress=0)
            self.query_one("#progress_label", Label).update(
                self.DESCRIPTIONS[Task.IDLE]
            )

        wrapper = None
        update_p, reset_p = partial(update_hook), partial(reset_hook)
        if isinstance(iterable, abc.Sequence):
            wrapper = SeqWrapper(iterable, total, update_p, reset_p)
        elif isinstance(iterable, (abc.Iterator, DataLoader)):
            wrapper = IteratorWrapper(iterable, total, update_p, reset_p)
        else:
            raise ValueError(
                f"iterable must be a Sequence or an Iterator, got {type(iterable)}"
            )
        self.query_one(ProgressBar).update(total=total, progress=0)
        self.query_one("#progress_label", Label).update(self.DESCRIPTIONS[task])
        return wrapper, wrapper.update_loss_hook
