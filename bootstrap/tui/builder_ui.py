import asyncio
import importlib
import inspect
import sys
import traceback
from types import FrameType
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
)

from rich.console import RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import (
    Checkbox,
    Footer,
    Header,
    Placeholder,
    RichLog,
)

from bootstrap.tui.widgets.logger import Logger
from bootstrap.tui.widgets.tracer import Tracer

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
from bootstrap import MatchboxModule
from bootstrap.tui.widgets.checkbox_panel import CheckboxPanel
from bootstrap.tui.widgets.editor import CodeEditor
from bootstrap.tui.widgets.files_tree import FilesTree


class BuilderUI(App):
    """
    A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework.
    """

    TITLE = "Matchbox Builder TUI"
    CSS_PATH = "styles/builder_ui.css"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("r", "reload", "Reload hot code"),
    ]

    def __init__(self):
        super().__init__()
        self._module_chain: List[MatchboxModule] = []
        self._runner_task = None

    async def chain_up(self, modules_seq: List[MatchboxModule]) -> None:
        """Add a module (callable to interactively implement and debug) to the
        run-reload chain."""
        keys = []
        for module in modules_seq:
            await self.query_one(CheckboxPanel).add_checkbox(str(module), module.uid)
            if module.uid in keys:
                raise ValueError(f"Duplicate module '{module}' with uid {module.uid}")
            keys.append(module.uid)
        self._module_chain = modules_seq

    async def _run_chain(self) -> None:
        self.log_tracer("Running the chain...")
        if len(self._module_chain) == 0:
            self.log_tracer(Text("The chain is empty!", style="bold red"))
        for module in self._module_chain:
            if module.is_frozen:
                self.log_tracer(Text(f"Skipping frozen module {module}", style="green"))
                continue
            if module.to_reload:
                self.log_tracer(Text(f"Reloading module: {module}", style="yellow"))
                await self._reload_module(module)
            self.log_tracer(Text(f"Running module: {module}", style="yellow"))
            module.result = await self._catch_and_hang(module, self._module_chain)
            self.log_tracer(Text(f"{module} ran sucessfully!", style="bold green"))
            self.print_info("Hanged.")
            self.query_one("#traceback", RichLog).clear()
            await self.hang(threw=False)

    def run_chain(self) -> None:
        if self._runner_task is not None:
            self._runner_task.cancel()
            self._runner_task = None
        self._runner_task = asyncio.create_task(self._run_chain(), name="run_chain")

    def compose(self) -> ComposeResult:
        yield Header()
        yield CheckboxPanel(classes="box")
        yield CodeEditor(classes="box", id="code")
        logs = Logger(classes="box", id="logger")
        logs.border_title = "User logs"
        logs.styles.border = ("solid", "gray")
        yield logs
        ftree = FilesTree(classes="box")
        ftree.border_title = "Project tree"
        ftree.styles.border = ("solid", "gray")
        yield ftree
        lcls = Placeholder("Locals area", classes="box")
        lcls.loading = True
        lcls.border_title = "Frame locals"
        yield lcls
        yield Tracer(classes="box")
        traceback = RichLog(
            classes="box", id="traceback", highlight=True, markup=True, wrap=False
        )
        traceback.border_title = "Exception traceback"
        traceback.styles.border = ("solid", "gray")
        yield traceback
        yield Footer()

    def action_reload(self) -> None:
        self.query_one(Tracer).clear()
        self.log_tracer("Reloading hot code...")
        self.query_one(CheckboxPanel).ready()
        self.query_one(Tracer).ready()
        # self.query_one(CheckboxPanel).hang(threw)
        self.query_one(CodeEditor).ready()
        self.run_chain()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        assert message.checkbox.id is not None
        for module in self._module_chain:
            if module.uid == message.checkbox.id:
                module.is_frozen = bool(message.value)

    def set_start_epoch(self, *args, **kwargs):
        _ = args
        _ = kwargs
        pass

    def track_training(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        _ = total

        def noop(*args, **kwargs):
            _ = args
            _ = kwargs
            pass

        return iterable, noop

    def track_validation(self, iterable, total: int) -> Tuple[Iterable, Callable]:
        _ = total

        def noop(*args, **kwargs):
            _ = args
            _ = kwargs
            pass

        return iterable, noop

    def log_tracer(self, message: str | RenderableType) -> None:
        self.query_one(Tracer).write(message)

    async def hang(self, threw: bool) -> None:
        """
        Give visual signal that the builder is hung, either due to an exception or
        because the function ran successfully.
        """
        self.query_one(Tracer).hang(threw)
        self.query_one(CodeEditor).hang(threw)
        while self.is_running:
            await asyncio.sleep(1)

    def print_err(self, msg: str | Exception) -> None:
        self.log_tracer(
            Text("[!] " + msg, style="bold red")
            if isinstance(msg, str)
            else Pretty(msg)
        )

    def print_warn(self, msg: str) -> None:
        self.log_tracer(Text("[!] " + msg, style="bold yellow"))

    def prompt(self, msg: str) -> str:
        # TODO: We need to use a popup callback
        self.log_tracer(Text("[?] " + msg, style="italic pink"))
        return "y"

    def print_info(self, msg: str) -> None:
        self.log_tracer(Text(msg, style="bold blue"))

    def print_pretty(self, msg: Any) -> None:
        self.log_tracer(Pretty(msg))

    @classmethod
    def get_class_frame(cls, func: Callable, exc_traceback) -> Optional[FrameType]:
        """
        Find the frame of the last callable within the scope of the MatchboxModule in
        the traceback. In this instance, the MatchboxModule is a class so we want to find
        the frame of the method that either (1) threw the exception, or (2) called a
        function that threw (or originated) the exception.
        """
        last_frame = None
        for frame, _ in traceback.walk_tb(exc_traceback):
            for name, val in inspect.getmembers(func):
                if (
                    name == frame.f_code.co_name
                    and "self" in inspect.getargs(frame.f_code).args
                ):
                    print(f"Found method {val} in traceback, continuing...")
                    last_frame = frame
        return last_frame

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
        for frame, _ in traceback.walk_tb(exc_traceback):
            assert lambda_args is not None
            frame_args = inspect.getargvalues(frame)
            for name, val in potential_matches.items():
                if val == frame.f_code.co_qualname:
                    return frame, name
                elif hasattr(val, frame.f_code.co_name):
                    return frame, name
            for name in lambda_args.args:
                if name in frame_args.args:
                    # NOTE: Now we need to find the argument which initiated the call
                    # that threw! Which is somewhere deeper in the stack, which
                    # frame.f_code.co_qualname must match one of the frame_args.args!
                    # NOTE: We know the next frame in the loop WILL match one of
                    # this frame's arguments, either in the qual_name directly or in
                    # the qual_name base (the class)
                    potential_matches[name] = frame_args.locals[name]
        return None, None

    @classmethod
    def get_function_frame(cls, func: Callable, exc_traceback) -> Optional[FrameType]:
        raise NotImplementedError()

    async def _catch_and_hang(self, module: MatchboxModule, *args, **kwargs):
        try:
            self.print_info(f"Calling MatchboxModule({module.underlying_fn}) with")
            self.print_pretty(
                {
                    "args": args,
                    "kwargs": kwargs,
                    "partial.args": module.partial.args,
                    "partial.kwargs": module.partial.keywords,
                }
            )
            output = await asyncio.to_thread(module, *args, **kwargs)
            self.print_info("Output:")
            self.print_pretty(output)
            return output
        except Exception as exception:
            # If the exception came from the wrapper itself, we should not catch it!
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback.tb_next is None:
                self.print_err(
                    "[ERROR] Could not find the next frame in the call stack!"
                )
            elif exc_traceback.tb_next.tb_frame.f_code.co_name == "catch_and_hang":
                self.print_err(
                    f"[ERROR] Caught exception in the Builder: {exception}",
                )
            else:
                self.print_err(
                    f"Caught exception: {exception}",
                )
                self.query_one("#traceback", RichLog).write(traceback.format_exc())
                func = module.underlying_fn
                # NOTE: This frame is for the given function, which is the root of the
                # call tree (our MatchboxModule's underlying function). What we want is
                # to go down to the function that threw, and reload that only if it
                # wasn't called anywhere in the frozen module's call tree.
                frame = None
                if inspect.isclass(func):
                    frame = self.get_class_frame(func, exc_traceback)
                elif inspect.isfunction(func) and func.__name__ == "<lambda>":
                    frame, lambda_argname = self.get_lambda_child_frame(
                        func, exc_traceback
                    )
                    module.throw_lambda_argname = lambda_argname
                elif inspect.isfunction(func):
                    frame = self.get_function_frame(func, exc_traceback)
                else:
                    raise NotImplementedError()
                if not frame:
                    self.print_err(
                        f"Could not find the frame of the original function {func} in the traceback."
                    )
                module.throw_frame = frame
                self.print_info("Exception thrown in:")
                self.print_pretty(frame)
            module.to_reload = True
            self.print_info("Hanged.")
            await self.hang(threw=True)

    async def _reload_module(self, module: MatchboxModule):
        if module.throw_frame is None:
            self.exit(1)
            raise RuntimeError(
                f"Module {module} is set to reload but we don't have the frame that threw!"
            )
        self.log_tracer(
            Text(
                f"Reloading code from {module.throw_frame.f_code.co_filename}",
                style="purple",
            )
        )
        code_obj = module.throw_frame.f_code
        code_module = inspect.getmodule(code_obj)
        if code_module is None:
            self.exit(1)
            raise RuntimeError(
                f"Could not find the module for the code object {code_obj}."
            )
        rld_module = importlib.reload(code_module)
        if code_obj.co_qualname.endswith("__init__"):
            class_name = code_obj.co_qualname.split(".")[0]
            self.log_tracer(
                Text(
                    f"-> Reloading class {class_name} from module {code_module}",
                    style="purple",
                )
            )
            rld_callable = getattr(rld_module, class_name)
            if rld_callable is not None:
                self.log_tracer(
                    Text(
                        f"-> Reloaded class {code_obj.co_qualname} from module {code_module.__name__}",
                        style="cyan",
                    )
                )
                module.reload(rld_callable)
                return

        else:
            if code_obj.co_qualname.find(".") != -1:
                class_name, _ = code_obj.co_qualname.split(".")
                self.log_tracer(
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
                        self.print_info(
                            f"  -> Reloading method '{name}'",
                        )
                        rld_callable = val
                if rld_callable is not None:
                    self.log_tracer(
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
                self.log_tracer(
                    Text(
                        f"-> Reloading module-level function {code_obj.co_name} from module {code_module.__name__}",
                        style="purple",
                    )
                )
                func = getattr(rld_module, code_obj.co_name)
                if func is not None:
                    self.print_info(
                        f"  -> Reloaded module level function {code_obj.co_name}",
                    )
                    print(inspect.getsource(func))
                module.reload(func)
                return
        while True:
            await asyncio.sleep(1)
