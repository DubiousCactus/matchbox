from typing import (
    Callable,
    Iterable,
    Tuple,
)

from textual.app import App, ComposeResult
from textual.reactive import var
from textual.widgets import (
    Checkbox,
    Footer,
    Header,
    Placeholder,
    RichLog,
)

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
from bootstrap.tui.widgets.checkbox_panel import CheckboxPanel
from bootstrap.tui.widgets.editor import CodeEditor
from bootstrap.tui.widgets.files_tree import FilesTree
from bootstrap.tui.widgets.tracer import Tracer


class BuilderUI(App):
    """
    A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework.
    """

    TITLE = "Matchbox Builder TUI"
    CSS_PATH = "styles/builder_ui.css"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("r", "reload", "Reload hot code"),
    ]

    # TODO: Make the follwing dynamically created based on the experiment config
    dataset_is_frozen: var[bool] = var(False)
    model_is_frozen: var[bool] = var(False)
    loss_is_frozen: var[bool] = var(False)
    trainer_is_frozen: var[bool] = var(False)

    def compose(self) -> ComposeResult:
        yield Header()
        # yield Placeholder("Checkbox panel", classes="box")
        yield CheckboxPanel(classes="box")
        yield CodeEditor(classes="box", id="code")
        logs = RichLog(classes="box", id="logger")
        logs.border_title = "User logs"
        logs.styles.border = ("solid", "gray")
        yield logs
        ftree = FilesTree(classes="box")
        ftree.border_title = "Project tree"
        ftree.styles.border = ("solid", "gray")
        yield ftree
        lcls = Placeholder("Locals area", classes="box")
        lcls.loading = True
        lcls.border_title = "Frame locals"
        yield lcls
        yield Tracer(classes="box")
        yield Placeholder(classes="box")
        yield Footer()

    def watch_dataset_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(  # type: ignore
            "Dataset is frozen" if checked else "Dataset is executable"
        )

    def watch_model_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(  # type: ignore
            "Model is frozen" if checked else "Model is executable"
        )

    def watch_loss_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(  # type: ignore
            "Loss is frozen" if checked else "Loss is executable"
        )

    def watch_trainer_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(  # type: ignore
            "Trainer is frozen" if checked else "Trainer is executable"
        )

    def action_reload(self) -> None:
        self.query_one("#logger").write("Reloading hot code...")  # type: ignore
        self.query_one(CheckboxPanel).ready()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        self.query_one("#logger").write(  # type: ignore
            f"Checkbox {message.checkbox.id} changed to: {message.value}"
        )
        setattr(self, f"{message.checkbox.id}_is_frozen", message.value)

    def print_log(self, message: str) -> None:
        self.query_one("#logger").write(message)  # type: ignore

    def print(self, message: str) -> None:
        # TODO: Remove this by merging main into this branch
        self.print_log(message)

    def set_start_epoch(self, *args, **kwargs):
        pass

    def track_training(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        def noop(*args, **kwargs):
            pass

        return iterable, noop

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        def noop(*args, **kwargs):
            pass

        return iterable, noop


if __name__ == "__main__":
    BuilderUI().run()
