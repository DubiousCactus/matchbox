import uuid
from dataclasses import dataclass
from functools import partial
from types import FrameType
from typing import Any, Callable, List, Optional

from hydra_zen.typing import Partial


class MatchboxModule:
    def __init__(self, name: str, fn: Callable | Partial, *args, **kwargs):
        self._str_rep = name
        self._uid = uuid.uuid4().hex
        self.underlying_fn: Callable = fn.func if isinstance(fn, partial) else fn
        self.partial = partial(fn, *args, **kwargs)
        self.to_reload = False
        self.result = None
        self.is_frozen = False
        self.throw_frame: Optional[FrameType] = None
        self.throw_lambda_argname: Optional[str] = None

    def reload(self, new_func: Callable) -> None:
        self.underlying_fn = new_func
        self.partial = partial(
            self.underlying_fn, *self.partial.args, **self.partial.keywords
        )
        self.to_reload = False

    def reload_surgically(self, method_name: str, method: Callable) -> None:
        setattr(self.underlying_fn, method_name, method)
        self.partial = partial(
            self.underlying_fn, *self.partial.args, **self.partial.keywords
        )
        self.to_reload = False

    def reload_surgically_in_lambda(
        self, arg_name: str, method_name: str, method: Callable
    ) -> None:
        if arg_name not in self.partial.keywords.keys():
            raise KeyError(
                "Could not find the argument to replace in the partial kwargs!"
            )
        for k, v in self.partial.keywords.items():
            setattr(v, method_name, partial(method, v))
            self.partial.keywords[k] = v  # Need to update when using dict iterator
        self.partial = partial(
            self.underlying_fn, *self.partial.args, **self.partial.keywords
        )
        self.to_reload = False

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
