from rich.console import RenderableType
from textual.app import ComposeResult
from textual.widgets import RichLog, Static


class Tracer(Static):
    def compose(self) -> ComposeResult:
        yield RichLog()

    def on_mount(self):
        self.ready()

    def hang(self, threw: bool) -> None:
        # TODO: Blink the border
        self.styles.border = ("dashed", "red" if threw else "yellow")
        self.border_title = "Exception trace: hanged" + (
            "(exception thrown)" if threw else "(no exception thrown)"
        )

    def ready(self) -> None:
        self.loading = True
        self.styles.border = ("solid", "green")
        self.border_title = "Exception trace: running"

    def write(self, message: str | RenderableType) -> None:
        self.loading = False
        self.query_one(RichLog).write(message)

    def clear(self) -> None:
        self.query_one(RichLog).clear()
