from datetime import datetime
from typing import (
    Any,
)

from rich.console import Group, RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import ComposeResult
from textual.events import Print
from textual.widgets import (
    RichLog,
    Static,
)


class Logger(Static):
    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True)

    def on_mount(self):
        self.begin_capture_print()

    def on_print(self, event: Print) -> None:
        if event.text.strip() != "":
            # FIXME: Why do we need this hack?!
            self.wite(event.text, event.stderr)

    def wite(self, message: Any, is_stderr: bool):
        logger: RichLog = self.query_one(RichLog)
        if isinstance(message, (RenderableType, str)):
            logger.write(
                Group(
                    Text(
                        datetime.now().strftime("[%H:%M] "),
                        style="dim cyan" if not is_stderr else "bold red",
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
