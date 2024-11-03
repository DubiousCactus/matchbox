from typing import Any, Dict, List

import numpy as np
import torch
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Pretty, Static


class LocalsPanel(Static):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="locals")

    async def add_locals(self, locals: Dict[str, Any]) -> None:
        for name, val in locals.items():
            title = f"{name}: {type(val).__name__}"
            if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
                title += f", {val.shape}"
            elif isinstance(val, List):
                title += f", {len(val)} elements"
            elif isinstance(val, Dict):
                title += f", {len(val)} keys"
            await self.query_one("#locals").mount(
                Collapsible(
                    Pretty(val),
                    title=title,
                )
            )

    async def clear(self) -> None:
        await self.query_one("#locals", VerticalScroll).recompose()
