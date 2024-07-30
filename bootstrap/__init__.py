from functools import partial
from typing import Any, Callable


class MatchboxModule:
    PREV = None  # TODO: This is used as an enum value. Should figure it out

    def __init__(self, name: str, fn: Callable, *args, **kwargs):
        # TODO: Figure out this entire class. It's a hack, I'm still figuring things
        # out as I go.
        self._name = name
        self._fn = partial(fn, *args, **kwargs)
        # self._fn = fn
        # self._args: List[Any] = *args
        # self._kwargs: Dict[str, Any] = **kwargs

    def __call__(self, prev_result: Any) -> Any:
        # TODO: Replace .PREV in any of the function's args/kwargs with prev_result
        return self._fn()

    def __str__(self) -> str:
        return self._name
