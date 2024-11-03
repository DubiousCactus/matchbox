from typing import Any, Dict

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Pretty, Static


class LocalsPanel(Static):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="locals")

    async def add_locals(self, locals: Dict[str, Any]) -> None:
        for name, val in locals.items():
            await self.query_one("#locals").mount(
                Collapsible(
                    Pretty(val),
                    title=name,
                )
            )

    async def clear(self) -> None:
        await self.query_one("#locals", VerticalScroll).recompose()
