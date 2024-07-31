import asyncio
import importlib
import inspect
import sys
import traceback
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Tuple,
)

import IPython
from rich.console import RenderableType
from rich.pretty import Pretty
from rich.text import Text
from textual.app import App, ComposeResult
from textual.reactive import var
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

    # TODO: Make the follwing dynamically created based on the experiment config
    dataset_is_frozen: var[bool] = var(False)
    model_is_frozen: var[bool] = var(False)
    loss_is_frozen: var[bool] = var(False)
    trainer_is_frozen: var[bool] = var(False)

    def __init__(self):
        super().__init__()
        self._module_chain: List[MatchboxModule] = []
        self._restart = False
        self._runner_task = None

    def chain_up(self, modules_seq: List[MatchboxModule]) -> None:
        """Add a module (callable to interactively implement and debug) to the
        run-reload chain."""
        self._module_chain = modules_seq

    async def _run_chain(self) -> None:
        self.log_tracer("Running the chain...")
        result = None
        if len(self._module_chain) == 0:
            self.log_tracer(Text("The chain is empty!", style="bold red"))
        for module in self._module_chain:
            self.log_tracer(f"Running module {module}")
            result = await self.catch_and_hang(module, result)

    def run_chain(self) -> None:
        if self._runner_task is not None:
            self._runner_task.cancel()
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

    def watch_dataset_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger", expect_type=RichLog).write(
            "Dataset is frozen" if checked else "Dataset is executable"
        )

    def watch_model_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger", RichLog).write(
            "Model is frozen" if checked else "Model is executable"
        )

    def watch_loss_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger", RichLog).write(
            "Loss is frozen" if checked else "Loss is executable"
        )

    def watch_trainer_is_frozen(self, checked: bool) -> None:
        self.query_one("#logger", RichLog).write(
            "Trainer is frozen" if checked else "Trainer is executable"
        )

    def action_reload(self) -> None:
        self.query_one(Tracer).clear()
        self.log_tracer("Reloading hot code...")
        self.query_one(CheckboxPanel).ready()
        self.run_chain()

    def on_checkbox_changed(self, message: Checkbox.Changed):
        self.query_one("#logger", RichLog).write(
            f"Checkbox {message.checkbox.id} changed to: {message.value}"
        )
        setattr(self, f"{message.checkbox.id}_is_frozen", message.value)

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

    def hang(self, threw: bool) -> None:
        """
        Give visual signal that the builder is hung, either due to an exception or
        because the function ran successfully.
        """
        self.query_one(Tracer).hang(threw)
        self.query_one(CheckboxPanel).hang(threw)
        self.query_one(CodeEditor).hang(threw)

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
        self.log_tracer(Text("[*] " + msg, style="bold blue"))

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
    async def catch_and_hang(self, callable: Callable, *args, **kwargs):
        """
        Decorator to call a callable and launch an IPython shell after an exception is thrown. This
        lets the user debug the callable in the context of the exception and fix the function/method. It will
        then retry the call until no exception is thrown, after reloading the function/method code.
        """
        while self.is_running:  # TODO: What's this in the state machine?
            try:
                self.print_info(f"Calling {callable} with")
                self.print_pretty({"args": args, "kwargs": kwargs})
                output = await asyncio.to_thread(callable, *args, **kwargs)
                self.print_info("Output:")
                self.print_pretty(output)
                self.print_info("Hanged.")
                self.hang(threw=False)
                while self.is_running:
                    # _ = input()
                    await asyncio.sleep(1)
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
                else:
                    self.print_err(
                        f"Caught exception: {exception}",
                    )
                    self.print_err(traceback.format_exc())
                # reload = self.prompt(
                #     "Take action? ([L]aunch IPython shell and reload the code/[r]eload the code/[a]bort) ",
                # )
                self.print_info("Hanged.")
                self.hang(threw=True)
                while self.is_running:
                    # _ = input()
                    await asyncio.sleep(1)
                return
                if reload.lower() not in ("l", "", "r"):
                    print("[!] Aborting")
                    # TODO: Why can't I just raise the exception? It's weird but it gets caught by
                    # the wrapper a few times until it finally gets raised.
                    sys.exit(1)
                if reload.lower() in ("l", ""):
                    # Drop into an IPython shell to inspect the callable and its context.
                    # Get the frame of the original callable
                    frame = get_function_frame(callable, exc_traceback)
                    if not frame:
                        raise Exception(
                            f"Could not find the frame of the original function {callable} in the traceback."
                        )
                    interactive_shell = IPython.terminal.embed.InteractiveShellEmbed(
                        cfg=IPython.terminal.embed.load_default_config(),
                        banner1=Text(
                            f"[*] Dropping into an IPython shell to inspect {callable} "
                            + "with the locals as they were at the time of the exception "
                            + f"thrown at line {frame.f_lineno} of {frame.f_code.co_filename}."
                            + "\n============================== TIPS =============================="
                            + "\n -> Use '%whos' to list variables in the current scope."
                            + "\n -> Use '%debug' to launch the debugger."
                            + "\n -> Use '<variable_name>' to display the value of a variable. "
                            + "Add a '?' to display the type."
                            + "\n -> Use '<function_name>?' to display the function's docstring. "
                            + "Add a '?' to display the source code."
                            + "\n -> Use 'frame??' to display the source code of the current frame which threw the exception."
                            + "\n==================================================================",
                            style="green",
                        ),
                        exit_msg=Text("Leaving IPython shell.", style="yellow"),
                    )
                    interactive_shell(local_ns={**frame.f_locals, "frame": frame})
                # I think it's not a good idea to let the user reload other modules, because it could
                # lead to unexpected behavior across the codebase (e.g. if the function called by the
                # callable is used elsewhere where the reference to the function is not updated,
                # which probably do not want to do).
                self.print_info(
                    f"Reloading callable {callable.__name__}. (Anything outside this scope will not be reloaded)"
                )

                # This is super interesting stuff about Python's inner workings! Look at the
                # documentation for more information:
                # https://docs.python.org/3/reference/datamodel.html?highlight=__func__#instance-methods
                importlib.reload(sys.modules[callable.__module__])
                reloaded_module = importlib.import_module(callable.__module__)
                rld_callable = None
                if hasattr(callable, "__self__"):
                    # callable is a *bound* class method, so we can retrieve the class and reload it
                    self.print_info(
                        f"-> Reloading class {callable.__self__.__class__.__name__}"
                    )
                    reloaded_class = getattr(
                        reloaded_module, callable.__self__.__class__.__name__
                    )
                    # Now find the method in the reloaded class, and replace the
                    # with the reloaded one.
                    for name, val in inspect.getmembers(reloaded_class):
                        if (
                            inspect.isfunction(val)
                            and val.__name__ == callable.__name__
                        ):
                            self.print_info(
                                f"-> Reloading method {name}",
                            )
                            rld_callable = val
                else:
                    # Most likely we end up here because callable is the function object of the
                    # called method, not the method itself. Is there even a case where we end up
                    # with the method object? First we can try to reload it directly if it was a
                    # module level function:
                    try:
                        self.print_info(
                            f"-> Reloading module level function {callable.__name__}",
                        )
                        callable = getattr(reloaded_module, callable.__name__)
                    except AttributeError:
                        self.print_info(
                            f"-> Could not find {callable.__name__} in module {callable.__module__}. "
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
                            f"-> Reloading class {callable.__qualname__.split('.')[0]}",
                        )
                        reloaded_class = getattr(
                            reloaded_module, callable.__qualname__.split(".")[0]
                        )
                        for name, val in inspect.getmembers(reloaded_class):
                            if inspect.isfunction(val) and name == callable.__name__:
                                self.print_info(
                                    f"-> Reloading method {name}",
                                )
                                rld_callable = val
                                break
                if rld_callable is None:
                    self.print_err(
                        f"[!] Could not reload callable {callable}!",
                    )
                    sys.exit(1)
                self.print_info(
                    f"[*] Reloaded callable {callable.__name__}! Retrying the call...",
                )
                callable = rld_callable
                # TODO: What if the user modified other methods/functions called by the callable?
                # Should we find them and recursively reload them? Maybe we can keep track of
                # every called *user* function, and if the user modifies any after an exception is
                # caught, we can first ask if we should reload it, warning them about the previous
                # calls that will be affected by the reload.
                # TODO: Check if we changed the function signature and if so, backtrace the call
                # and update the arguments by re-running the routine that generated them and made
                # the call.
                # TODO: Free memory allocation (open file descriptors, etc.) before retrying the
                # call.
