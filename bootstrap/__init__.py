import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional

from hydra_zen.typing import Partial


class MatchboxModule:
    def __init__(self, name: str, fn: Callable | Partial, *args, **kwargs):
        self._str_rep = name
        self._uid = uuid.uuid4().hex
        self.underlying_fn = fn.func if isinstance(fn, partial) else fn
        self.partial = partial(fn, *args, **kwargs)
        self.first_run = True
        self.result = None
        self.is_frozen = False

    def __call__(self, module_chain: List) -> Any:
        """
        Args:
            module_chain: List[MatchboxModule]
        """

        def _find_module_result(module_chain: List, uid: str) -> Any:
            for module in module_chain:
                if module.uid == uid:
                    return module.result
            return None

        for i, arg in enumerate(self.partial.args):
            if isinstance(arg, MatchboxModule):
                self.partial.args[i] = _find_module_result(module_chain, arg.uid)
        for key, value in self.partial.keywords.items():
            if isinstance(value, MatchboxModule):
                self.partial.keywords[key] = _find_module_result(
                    module_chain, value.uid
                )
        return self.partial()

    def __str__(self) -> str:
        return self._str_rep

    @property
    def uid(self) -> str:
        return f"uid-{self._uid}"
