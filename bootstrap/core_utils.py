import asyncio
import importlib
import inspect
import sys
import traceback
from time import sleep
from typing import Any, Callable

import IPython
from rich.pretty import Pretty
from rich.text import Text


class TraceCatcher:
    def __init__(self, log_callback: Callable, hang_callback: Callable):
        self._log_callback = log_callback
        self._hang_callback = hang_callback

    def print_err(self, msg: str) -> None:
        self._log_callback(Text("[!] " + msg, style="bold red"))

    def print_warn(self, msg: str) -> None:
        self._log_callback(Text("[!] " + msg, style="bold yellow"))

    def prompt(self, msg: str) -> str:
        # TODO: We need to use a popup callback
        self._log_callback(Text("[?] " + msg, style="italic pink"))
        return "y"

    def print_info(self, msg: str) -> None:
        self._log_callback(Text("[*] " + msg, style="bold blue"))

    def print_pretty(self, msg: Any) -> None:
        self._log_callback(Pretty(msg))

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
        while True:  # TODO: What's this in the state machine?
            try:
                self.print_info(f"Calling {callable} with")
                self.print_pretty({"args": args, "kwargs": kwargs})
                output = await asyncio.to_thread(callable, *args, **kwargs)
                self.print_info("Output:")
                self.print_pretty(output)
                self.print_info("Hanged.")
                self._hang_callback(threw=False)
                while True:
                    # _ = input()
                    await asyncio.sleep(1)
                return output
            except Exception as exception:
                # If the exception came from the wrapper itself, we should not catch it!
                exc_type, exc_value, exc_traceback = sys.exc_info()
                if exc_traceback.tb_next is None:
                    traceback.print_exc()
                    sys.exit(1)
                elif exc_traceback.tb_next.tb_frame.f_code.co_name == "catch_and_hang":
                    self.print_err(
                        f"Caught exception in 'debug_trace': {exception}",
                    )
                    sys.exit(1)
                self.print_err(
                    f"Caught exception: {exception}",
                )
                full_traceback = self.prompt(
                    "Display full traceback? (y/[N]) ",
                )
                if full_traceback.lower() == "y":
                    traceback.print_exc()
                reload = self.prompt(
                    "Take action? ([L]aunch IPython shell and reload the code/[r]eload the code/[a]bort) ",
                )
                sleep(100)

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


# def debug_methods(cls):
#     """
#     Decorator to debug all methods of a class using the debug_trace decorator if the DEBUG environment variable is set.
#     """
#     if not project_conf.DEBUG:
#         return cls
#     for key, val in vars(cls).items():
#         if callable(val):
#             setattr(cls, key, debug_trace(val))
#     return cls
#


# class DebugMetaclass(type):
#     """
#     We can use this metaclass to automatically decorate all methods of a class with the debug_trace
#     decorator, making it simpler with inheritance.
#     """
#
#     def __new__(cls, name, bases, dct):
#         obj = super().__new__(cls, name, bases, dct)
#         obj = debug_methods(obj)
#         return obj
