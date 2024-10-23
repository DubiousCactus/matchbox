import asyncio
import importlib
import inspect
import sys
import traceback
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from rich.console import RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual import log
from textual.app import App, ComposeResult
from textual.widgets import (
    Checkbox,
    Footer,
    Header,
    Placeholder,
    RichLog,
)

from bootstrap.tui.widgets.tracer import Tracer

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
from bootstrap import MatchboxModule, MatchboxModuleState
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
        # TODO: Unify the module chain and the module states!
        self._module_chain: List[MatchboxModule] = []
        self._runner_task = None
        self._module_states: Dict[str, MatchboxModuleState] = {}

    def chain_up(self, modules_seq: List[MatchboxModule]) -> None:
        """Add a module (callable to interactively implement and debug) to the
        run-reload chain."""
        self._module_chain = modules_seq
        for module in modules_seq:
            self._module_states[str(module)] = MatchboxModuleState(
                first_run=True, result=None, is_frozen=False
            )
            self.query_one(CheckboxPanel).add_checkbox(str(module))

    async def _run_chain(self) -> None:
        log("_run_chain()")
        self.log_tracer("Running the chain...")
        if len(self._module_chain) == 0:
            self.log_tracer(Text("The chain is empty!", style="bold red"))
        # TODO: Should we reset all modules to "first_run"? Because if we restart the
        # chain from a previously frozen step, we should run it as a first run, right?
        # Not sure about this.
        for module_idx, module in enumerate(self._module_chain):
            log(f"Running module: {module}...")
            initial_run = self._module_states[str(module)].first_run
            self._module_states[str(module)].first_run = False
            if self._module_states[str(module)].is_frozen:
                self.log_tracer(Text(f"Skipping frozen module {module}", style="green"))
                continue
            self.log_tracer(Text(f"Running module: {module}", style="yellow"))
            prev_result = self._module_states[
                list(self._module_states.keys())[module_idx - 1]
            ].result
            self._module_states[str(module)].result = await self.catch_and_hang(
                module, initial_run, prev_result
            )
            self.log_tracer(Text(f"{module} ran sucessfully!", style="bold green"))
            self.print_info("Hanged.")
            await self.hang(threw=False)
            self.query_one(Tracer).clear()

    def run_chain(self) -> None:
        if self._runner_task is not None:
            log("Cancelling previous chain run...")
            self._runner_task.cancel()
            self._runner_task = None
        log("Starting new chain run...")
        self._runner_task = asyncio.create_task(self._run_chain(), name="run_chain")

    def compose(self) -> ComposeResult:
        yield Header()
        yield CheckboxPanel(classes="box")
        yield CodeEditor(classes="box", id="code")
        logs = RichLog(classes="box", id="logger")
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
        yield Placeholder(classes="box")
        yield Footer()

    def action_reload(self) -> None:
        log("Reloading...")
        self.query_one(Tracer).clear()
        self.log_tracer("Reloading hot code...")
        self.query_one(CheckboxPanel).ready()
        self.query_one(Tracer).ready()
        # self.query_one(CheckboxPanel).hang(threw)
        self.query_one(CodeEditor).ready()
        self.run_chain()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        self.query_one("#logger", RichLog).write(
            f"Checkbox {message.checkbox.id} changed to: {message.value}"
        )
        assert message.checkbox.id is not None
        self._module_states[message.checkbox.id].is_frozen = bool(message.value)
        # setattr(self, f"{message.checkbox.id}_is_frozen", message.value)

    def print_log(self, message: str) -> None:
        self.query_one("#logger", RichLog).write(message)

    def print(self, message: str) -> None:
        # TODO: Remove this by merging main into this branch
        self.print_log(message)

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
    def get_function_frame(cls, func, exc_traceback):
        """
        Find the frame of the original function in the traceback.
        """
        for frame, _ in traceback.walk_tb(exc_traceback):
            if frame.f_code.co_name == func.__name__:
                return frame
        return None

    # TODO: Refactor this
    async def catch_and_hang(
        self, callable: MatchboxModule | Callable, reload_code: bool, *args, **kwargs
    ):
        if not reload_code:  # Take out this block. This is just code reloading.
            _callable = (
                callable.underlying_fn
                if isinstance(callable, MatchboxModule)
                else callable
            )
            # I think it's not a good idea to let the user reload other modules, because it could
            # lead to unexpected behavior across the codebase (e.g. if the function called by the
            # callable is used elsewhere where the reference to the function is not updated,
            # which probably do not want to do).
            self.print_info(
                f"[*] Reloading callable '{_callable.__name__}'. (Anything outside this scope will not be reloaded)"
            )

            # This is super interesting stuff about Python's inner workings! Look at the
            # documentation for more information:
            # https://docs.python.org/3/reference/datamodel.html?highlight=__func__#instance-methods
            importlib.reload(sys.modules[_callable.__module__])
            reloaded_module = importlib.import_module(_callable.__module__)
            rld_callable = None
            # First case, it's a class that we're trying to instantiate. We just need to
            # reload the class:
            if inspect.isclass(_callable) or _callable.__name__.endswith("__init__"):
                self.print_info(
                    f"  -> Reloading class '{_callable.__name__}' from module '{_callable.__module__}'",
                )
                reloaded_class = getattr(reloaded_module, _callable.__name__)
                self.print_log(reloaded_class)
                self.print_log(inspect.getsource(reloaded_class))
                rld_callable = reloaded_class
            elif hasattr(_callable, "__self__"):
                # _callable is a *bound* class method, so we can retrieve the class and reload it
                self.print_info(
                    f"  -> Reloading class '{_callable.__self__.__class__.__name__}'"
                )
                reloaded_class = getattr(
                    reloaded_module, _callable.__self__.__class__.__name__
                )
                # Now find the method in the reloaded class, and replace the
                # with the reloaded one.
                for name, val in inspect.getmembers(reloaded_class):
                    if inspect.isfunction(val) and val.__name__ == _callable.__name__:
                        self.print_info(
                            f"  -> Reloading method '{name}'",
                        )
                        rld_callable = val
            else:
                # Most likely we end up here because _callable is the function object of the
                # called method, not the method itself. Is there even a case where we end up
                # with the method object? First we can try to reload it directly if it was a
                # module level function:
                try:
                    self.print_info(
                        f"  -> Reloading module level function '{_callable.__name__}'",
                    )
                    rld_callable = getattr(reloaded_module, _callable.__name__)
                except AttributeError:
                    self.print_info(
                        f"  -> Could not find '{_callable.__name__}' in module '{_callable.__module__}'. "
                        + "Looking for a class method...",
                    )
                    # Ok that failed, so we need to find the class of the method and reload it,
                    # then find the method in the reloaded class and replace the function with
                    # the method's function object; this is the same as above.
                    # TODO: This feels very hacky! Can we find the class in a better way, maybe
                    # without going through all classes in the module? Because I'm not sure if the
                    # qualname always contains the class name in this way; like what about
                    # inheritance?
                    self.print_info(
                        f"  -> Reloading class {_callable.__qualname__.split('.')[0]}",
                    )
                    reloaded_class = getattr(
                        reloaded_module, _callable.__qualname__.split(".")[0]
                    )
                    for name, val in inspect.getmembers(reloaded_class):
                        if inspect.isfunction(val) and name == _callable.__name__:
                            self.print_info(
                                f"  -> Reloading method {name}",
                            )
                            rld_callable = val
                            break
            if rld_callable is None:
                self.print_err(
                    f"Could not reload callable {_callable}!",
                )
                await self.hang(threw=True)
            self.print_info(
                f":) Reloaded callable {_callable.__name__}! Retrying the call...",
            )
            _callable = rld_callable
            if isinstance(callable, MatchboxModule):
                # callable.underlying_fn = _callable
                callable = MatchboxModule(
                    callable._str_rep,
                    _callable,
                    *callable.partial.args,
                    **callable.partial.keywords,
                )
            else:
                raise NotImplementedError()
            # TODO: What if the user modified other methods/functions called by the _callable?
            # Should we find them and recursively reload them? Maybe we can keep track of
            # every called *user* function, and if the user modifies any after an exception is
            # caught, we can first ask if we should reload it, warning them about the previous
            # calls that will be affected by the reload.
            # TODO: Check if we changed the function signature and if so, backtrace the call
            # and update the arguments by re-running the routine that generated them and made
            # the call.
            # TODO: Free memory allocation (open file descriptors, etc.) before retrying the
            # call.
        try:
            if isinstance(callable, partial):
                self.print_info(f"Calling {callable.func} with")
                self.print_pretty({"args": callable.args, "kwargs": callable.keywords})
            elif isinstance(callable, MatchboxModule):
                self.print_info(
                    f"Calling MatchboxModule({callable.underlying_fn}) with"
                )
                self.print_pretty(
                    {
                        "args": args,
                        "kwargs": kwargs,
                        "partial.args": callable.partial.args,
                        "partial.kwargs": callable.partial.keywords,
                    }
                )
            else:
                self.print_info(f"Calling {callable} with")
                self.print_pretty({"args": args, "kwargs": kwargs})
            output = await asyncio.to_thread(callable, *args, **kwargs)
            self.print_info("Output:")
            self.print_pretty(output)
            return output
        except Exception as exception:
            # If the exception came from the wrapper itself, we should not catch it!
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback.tb_next is None:
                self.print_warn("Could not find the next frame!")
                self.print_pretty(traceback.format_exc())
            elif exc_traceback.tb_next.tb_frame.f_code.co_name == "catch_and_hang":
                self.print_warn(
                    f"Caught exception in 'debug_trace': {exception}",
                )
                raise exception
            else:
                self.print_err(
                    f"Caught exception: {exception}",
                )
                self.print_err(traceback.format_exc())
                func = (
                    callable.underlying_fn
                    if isinstance(callable, MatchboxModule)
                    else callable
                )
                # NOTE: This frame is for the given function, which is the root of the
                # call tree (our MatchboxModule's underlying function). What we want is
                # to go down to the function that threw, and reload that only if it
                # wasn't called anywhere in the frozen module's call tree.
                frame = self.get_function_frame(func, exc_traceback)
                if not frame:
                    self.print_err(
                        f"Could not find the frame of the original function {func} in the traceback."
                    )
                # self.print_info("Exception thrown in:")
                # self.print_pretty(frame)
            self.print_info("Hanged.")
            await self.hang(threw=True)
            # reload = self.prompt(
            #     "Take action? ([L]aunch IPython shell and reload the code/[r]eload the code/[a]bort) ",
            # )
            # if reload.lower() in ("l", ""):
            #     # Drop into an IPython shell to inspect the callable and its context.
            #     # Get the frame of the original callable
            #     frame = get_function_frame(callable, exc_traceback)
            #     if not frame:
            #         raise Exception(
            #             f"Could not find the frame of the original function {callable} in the traceback."
            #         )
            #     interactive_shell = IPython.terminal.embed.InteractiveShellEmbed(
            #         cfg=IPython.terminal.embed.load_default_config(),
            #         banner1=Text(
            #             f"[*] Dropping into an IPython shell to inspect {callable} "
            #             + "with the locals as they were at the time of the exception "
            #             + f"thrown at line {frame.f_lineno} of {frame.f_code.co_filename}."
            #             + "\n============================== TIPS =============================="
            #             + "\n -> Use '%whos' to list variables in the current scope."
            #             + "\n -> Use '%debug' to launch the debugger."
            #             + "\n -> Use '<variable_name>' to display the value of a variable. "
            #             + "Add a '?' to display the type."
            #             + "\n -> Use '<function_name>?' to display the function's docstring. "
            #             + "Add a '?' to display the source code."
            #             + "\n -> Use 'frame??' to display the source code of the current frame which threw the exception."
            #             + "\n==================================================================",
            #             style="green",
            #         ),
            #         exit_msg=Text("Leaving IPython shell.", style="yellow"),
            #     )
            #     interactive_shell(local_ns={**frame.f_locals, "frame": frame})
