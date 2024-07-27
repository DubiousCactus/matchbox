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
    dataset_is_frozen: var[bool] = var(True)
    model_is_frozen: var[bool] = var(True)
    loss_is_frozen: var[bool] = var(True)
    trainer_is_frozen: var[bool] = var(True)

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
        self.query_one("#logger").write(
            "Dataset is frozen" if checked else "Dataset is thawed"
        )

    def watch_model_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(
            "Model is frozen" if checked else "Model is thawed"
        )

    def watch_loss_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(
            "Loss is frozen" if checked else "Loss is thawed"
        )

    def watch_trainer_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger").write(
            "Trainer is frozen" if checked else "Trainer is thawed"
        )

    def action_reload(self) -> None:
        self.query_one("#logger").write("Reloading hot code...")
        self.query_one(CheckboxPanel).ready()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        # TODO: Update the corresponding reactive var
        self.query_one("#logger").write(
            f"Checkbox {message.checkbox.id} changed to: {message.value}"
        )


if __name__ == "__main__":
    BuilderUI().run()
