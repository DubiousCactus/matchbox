import asyncio
import importlib
import inspect
import sys
import traceback
from types import FrameType
from typing import (
    Callable,
    Optional,
    Tuple,
)

from rich.text import Text
from textual.app import App
from textual.widgets import (
    RichLog,
)

from bootstrap.hot_reloading.module import MatchboxModule


class HotReloadingEngine:
    def __init__(self, ui: App):
        self.ui = ui

    @classmethod
    def get_class_frame(
        cls, func: Callable, exc_traceback
    ) -> Tuple[Optional[FrameType], Optional[FrameType]]:
        """
        Find the frame of the last callable within the scope of the MatchboxModule in
        the traceback. In this instance, the MatchboxModule is a class so we want to find
        the frame of the method that either (1) threw the exception, or (2) called a
        function that threw (or originated) the exception.
        """
        print("============= get_class_frame() =========")
        last_frame_in_scope, last_frame = None, None
        for frame, _ in traceback.walk_tb(exc_traceback):
            last_frame = frame
            print(frame.f_code.co_qualname)
            if frame.f_code.co_qualname == func.__name__:
                print(
                    f"Found module.underlying_fn ({func.__name__}) in traceback, continuing..."
                )
            for name, val in inspect.getmembers(func):
                if (
                    name == frame.f_code.co_name
                    and "self" in inspect.getargs(frame.f_code).args
                ):
                    print(f"Found method {val} in traceback, continuing...")
                    last_frame_in_scope = frame
        print("============================================")
        return last_frame_in_scope, last_frame

    @classmethod
    def get_lambda_child_frame(
        cls, func: Callable, exc_traceback
    ) -> Tuple[Optional[FrameType], Optional[str]]:
        """
        Find the frame of the last callable within the scope of the MatchboxModule in
        the traceback. In this instance, the MatchboxModule is a lambda function so we want
        to find the frame of the first function called by the lambda.
        """
        lambda_args = inspect.getargs(func.__code__)
        potential_matches = {}
        print("============= get_lambda_child_frame() =========")
        print(f"Lambda args: {lambda_args}")
        for frame, _ in traceback.walk_tb(exc_traceback):
            print(frame.f_code.co_qualname)
            assert lambda_args is not None
            frame_args = inspect.getargvalues(frame)
            for name, val in potential_matches.items():
                print(
                    f"Checking candidate {name}={val} to match against {frame.f_code.co_qualname}"
                )
                if val == frame.f_code.co_qualname:
                    print(f"Matched {name}={val} to {frame.f_code.co_qualname}")
                    return frame, name
                elif hasattr(val, frame.f_code.co_name):
                    print(f"Matched {frame.f_code.co_qualname} to member of {val}")
                    return frame, name
            for name in lambda_args.args:
                print(f"Lambda arg '{name}'")
                if name in frame_args.args:
                    print(f"Frame has arg {name} with value {frame_args.locals[name]}")
                    # TODO: Find the argument which initiated the call that threw!
                    # Which is somewhere deeper in the stack, which
                    # frame.f_code.co_qualname must match one of the
                    # frame_args.args!
                    # NOTE: We know the next frame in the loop WILL match one of
                    # this frame's arguments, either in the qual_name directly or in
                    # the qual_name base (the class)
                    potential_matches[name] = frame_args.locals[name]
        print("============================================")
        return None, None

    @classmethod
    def get_function_frame(cls, func: Callable, exc_traceback) -> Optional[FrameType]:
        print("============= get_function_frame() =========")
        last_frame = None
        for frame, _ in traceback.walk_tb(exc_traceback):
            print(frame.f_code.co_qualname)
            if frame.f_code.co_qualname == func.__name__:
                print(
                    f"Found module.underlying_fn ({func.__name__}) in traceback, continuing..."
                )
            for name, val in inspect.getmembers(func.__module__):
                if name == frame.f_code.co_name:
                    print(f"Found function {val} in traceback, continuing...")
                    last_frame = frame
        print("============================================")
        return last_frame

    async def catch_and_hang(self, module: MatchboxModule, *args, **kwargs):
        try:
            self.ui.print_info(f"Calling MatchboxModule({module.underlying_fn}) with")
            self.ui.print_pretty(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "partial.args": module.partial.args,
                    "partial.kwargs": module.partial.keywords,
                }
            )
            output = await asyncio.to_thread(module, *args, **kwargs)
            self.ui.print_info("Output:")
            self.ui.print_pretty(output)
            return output
        except Exception as exception:
            # If the exception came from the wrapper itself, we should not catch it!
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback.tb_next is None:
                self.ui.exit(1)
                raise RuntimeError("Could not find the next frame in the call stack!")
            elif exc_traceback.tb_next.tb_frame.f_code.co_name == "catch_and_hang":
                self.ui.exit(1)
                raise RuntimeError(
                    f"Caught exception in the Builder: {exception}",
                )
            else:
                self.ui.print_err(
                    f"Caught exception: {exception}",
                )
                self.ui.query_one("#traceback", RichLog).write(traceback.format_exc())
                func = module.underlying_fn
                # NOTE: This frame is for the given function, which is the root of the
                # call tree (our MatchboxModule's underlying function). What we want is
                # to go down to the function that threw, and reload that only if it
                # wasn't called anywhere in the frozen module's call tree.
                root_frame: Optional[FrameType] = None
                throwing_frame: Optional[FrameType] = None
                if inspect.isclass(func):
                    root_frame, throwing_frame = self.get_class_frame(
                        func, exc_traceback
                    )
                elif inspect.isfunction(func) and func.__name__ == "<lambda>":
                    root_frame, lambda_argname = self.get_lambda_child_frame(
                        func, exc_traceback
                    )
                    module.throw_lambda_argname = lambda_argname
                elif inspect.isfunction(func):
                    root_frame = self.get_function_frame(func, exc_traceback)
                else:
                    self.ui.exit(1)
                    raise NotImplementedError()
                locals_f = root_frame if throwing_frame is None else throwing_frame
                if not root_frame:
                    self.ui.exit(1)
                    raise RuntimeError(
                        f"Could not find the frame of the original function {func} in the traceback."
                    )
                else:
                    # NOTE: Here we reloaded the root frame of the throwing call, i.e.
                    # the frame that's in the scope of our MatchboxModule so a class
                    # method if the module is a class.
                    await self.ui.set_locals(
                        locals_f.f_locals, locals_f.f_code.co_qualname
                    )
                    module.throw_frame = root_frame
                    info = (
                        (
                            f"Exception thrown in <{locals_f.f_code.co_qualname}>"
                            + f" with MatchboxModule root <{root_frame.f_code.co_qualname}>:"
                        )
                        if locals_f is not root_frame
                        else (
                            "Exception thrown in MatchboxModule root"
                            + f" <{locals_f.f_code.co_qualname}>:"
                        )
                    )

                    self.ui.print_info(info)
                    self.ui.print_pretty(root_frame)
            module.to_reload = True
            self.ui.print_info("Hanged.")
            await self.ui.hang(threw=True)

    async def reload_module(self, module: MatchboxModule):
        if module.to_reload and module.throw_frame is None:
            self.ui.exit(1)
            raise RuntimeError(
                f"Module {module} is set to reload but we don't have the frame that threw!"
            )
        elif not module.to_reload:
            # TODO: This works as long as we init the builder UI with skip_frozen=True
            # so that we can get the root frame at least once, but it will fail if the
            # module never throws in the first place. We should fix it by decoupling
            # root frame finding from the catch_and_hang() method above. Or ideally by
            # finding the code object more efficiently!
            self.ui.print_info(f"Reloading MatchboxModule({module.underlying_fn})...")
            self.ui.print_err("Hot reloading without throwing is not implemented yet.")
            code_obj = None
            await self.ui.hang(threw=False)
        self.ui.log_tracer(
            Text(
                f"Reloading code from {module.throw_frame.f_code.co_filename}",
                style="purple",
            )
        )
        code_obj = module.throw_frame.f_code
        print(code_obj.co_qualname, inspect.getmodule(code_obj))
        code_module = inspect.getmodule(code_obj)
        if code_module is None:
            self.ui.exit(1)
            raise RuntimeError(
                f"Could not find the module for the code object {code_obj}."
            )
        rld_module = importlib.reload(code_module)
        if code_obj.co_qualname.endswith("__init__"):
            class_name = code_obj.co_qualname.split(".")[0]
            self.ui.log_tracer(
                Text(
                    f"-> Reloading class {class_name} from module {code_module}",
                    style="purple",
                )
            )
            rld_callable = getattr(rld_module, class_name)
            if rld_callable is not None:
                self.ui.log_tracer(
                    Text(
                        f"-> Reloaded class {code_obj.co_qualname} from module {code_module.__name__}",
                        style="cyan",
                    )
                )
                print(inspect.getsource(rld_callable))
                module.reload(rld_callable)
                return

        else:
            if code_obj.co_qualname.find(".") != -1:
                class_name, _ = code_obj.co_qualname.split(".")
                self.ui.log_tracer(
                    Text(
                        f"-> Reloading class {class_name} from module {code_module}",
                        style="purple",
                    )
                )
                rld_class = getattr(rld_module, class_name)
                rld_callable = None
                # Now find the method in the reloaded class, and replace the
                # with the reloaded one.
                for name, val in inspect.getmembers(rld_class):
                    if inspect.isfunction(val) and val.__name__ == code_obj.co_name:
                        self.ui.print_info(
                            f"  -> Reloading method '{name}'",
                        )
                        rld_callable = val
                if rld_callable is not None:
                    self.ui.log_tracer(
                        Text(
                            f"-> Reloaded class-level method {code_obj.co_qualname} from module {code_module.__name__}",
                            style="cyan",
                        )
                    )
                    if module.underlying_fn.__name__ == "<lambda>":
                        assert module.throw_lambda_argname is not None
                        module.reload_surgically_in_lambda(
                            module.throw_lambda_argname, code_obj.co_name, rld_callable
                        )
                    else:
                        module.reload_surgically(code_obj.co_name, rld_callable)
                    return
            else:
                print(code_module, code_obj, code_obj.co_name)
                self.ui.log_tracer(
                    Text(
                        f"-> Reloading module-level function {code_obj.co_name} from module {code_module.__name__}",
                        style="purple",
                    )
                )
                func = getattr(rld_module, code_obj.co_name)
                if func is not None:
                    self.ui.print_info(
                        f"  -> Reloaded module level function {code_obj.co_name}",
                    )
                    print(inspect.getsource(func))
                module.reload(func)
                return
        while True:
            await asyncio.sleep(1)
