from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

from hydra_zen.typing import Partial


class MatchboxModule:
    PREV = "MatchboxModule.PREV"  # TODO: This is used as an enum value. Should figure it out

    def __init__(self, name: str, fn: Callable | Partial, *args, **kwargs):
        # TODO: Figure out this entire class. It's a hack, I'm still figuring things
        # out as I go.
        self._str_rep = name
        self.underlying_fn = fn.func if isinstance(fn, partial) else fn
        self.partial = partial(fn, *args, **kwargs)

    def __call__(self, prev_result: Any) -> Any:
        # TODO: Replace .PREV in any of the function's args/kwargs with prev_result
        for i, arg in enumerate(self.partial.args):
            if arg == self.PREV:
                assert prev_result is not None
                self.partial.args[i] = prev_result
        for key, value in self.partial.keywords.items():
            if value == self.PREV:
                assert prev_result is not None
                self.partial.keywords[key] = prev_result
        return self.partial()

    def __str__(self) -> str:
        return self._str_rep


@dataclass
class MatchboxModuleState:
    first_run: bool
    result: Any
