from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Checkbox, Static


class CheckboxPanel(Static):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="tickers")

    def add_checkbox(self, label: str) -> None:
        checkbox = Checkbox(label, value=False, id=label)
        # yield Switch() # TODO: Use switches!!
        self.query_one("#tickers").mount(checkbox)

    def on_mount(self):
        self.ready()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        _ = message
        self.due()

    def due(self) -> None:
        # TODO: Blink the border
        self.styles.border = ("dashed", "yellow")
        self.styles.opacity = 0.8
        self.border_title = "Frozen modules: due for reloading"

    def hang(self, threw: bool) -> None:
        if threw:
            self.styles.border = ("dashed", "red")
            self.border_title = "Frozen modules: exception was thrown"
        else:
            self.due()

    def ready(self) -> None:
        self.styles.border = ("solid", "green")
        self.styles.opacity = 1.0
        self.border_title = "Frozen modules: active"
