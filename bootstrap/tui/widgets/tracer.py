from textual.app import ComposeResult
from textual.widgets import RichLog, Static


class Tracer(Static):
    def compose(self) -> ComposeResult:
        yield RichLog()

    def on_mount(self):
        self.run()

    def hang(self) -> None:
        # TODO: Blink the border
        self.styles.border = ("dashed", "red")
        self.border_title = "Exception trace: hanged"

    def run(self) -> None:
        self.loading = True
        self.styles.border = ("solid", "green")
        self.border_title = "Exception trace: running"
