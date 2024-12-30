import asyncio
from typing import (
    Any,
    Callable,
    Iterable,
    List,
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
    RichLog,
)

from bootstrap.hot_reloading.engine import HotReloadingEngine
from bootstrap.hot_reloading.module import MatchboxModule
from bootstrap.tui.widgets.checkbox_panel import CheckboxPanel
from bootstrap.tui.widgets.editor import CodeEditor
from bootstrap.tui.widgets.files_tree import FilesTree
from bootstrap.tui.widgets.locals_panel import LocalsPanel
from bootstrap.tui.widgets.logger import Logger
from bootstrap.tui.widgets.tracer import Tracer


class BuilderUI(App):
    """
    A Textual app to serve as *useful* GUI/TUI for my pytorch-based micro framework.
    """

    TITLE = "Matchbox Builder TUI"
    CSS_PATH = "styles/builder_ui.css"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("r", "reload", "Hot reload"),
    ]

    def __init__(self, chain: List[MatchboxModule]):
        super().__init__()
        self._module_chain: List[MatchboxModule] = chain
        self._runner_task = None
        self._engine = HotReloadingEngine(self)

    async def on_mount(self):
        await self._chain_up()
        self.run_chain()

    async def _chain_up(self) -> None:
        keys = []
        for module in self._module_chain:
            if not isinstance(module, MatchboxModule):
                self.exit(1)
                raise TypeError(f"Expected MatchboxModule, got {type(module)}")
            await self.query_one(CheckboxPanel).add_checkbox(str(module), module.uid)
            if module.uid in keys:
                raise ValueError(f"Duplicate module '{module}' with uid {module.uid}")
            keys.append(module.uid)
        self.query_one(CheckboxPanel).ready()

    async def _run_chain(self) -> None:
        self.log_tracer("Running the chain...")
        if len(self._module_chain) == 0:
            self.log_tracer(Text("The chain is empty!", style="bold red"))
        for module in self._module_chain:
            await self.query_one(LocalsPanel).clear()
            self.query_one(Tracer).clear()
            if module.is_frozen:
                self.log_tracer(Text(f"Skipping frozen module {module}", style="green"))
                continue
            if module.to_reload:
                self.log_tracer(Text(f"Reloading module: {module}", style="yellow"))
                await self._engine.reload_module(module)
            self.log_tracer(Text(f"Running module: {module}", style="yellow"))
            module.result = await self._engine.catch_and_hang(
                module, self._module_chain
            )
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
        checkboxes = CheckboxPanel(classes="box")
        checkboxes.loading = True
        yield checkboxes
        yield CodeEditor(classes="box", id="code")
        logs = Logger(classes="box", id="logger")
        logs.border_title = "User logs"
        logs.styles.border = ("solid", "gray")
        yield logs
        ftree = FilesTree(classes="box")
        ftree.border_title = "Project tree"
        ftree.styles.border = ("solid", "gray")
        yield ftree
        lcls = LocalsPanel(classes="box")
        lcls.styles.border = ("solid", "gray")
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

    async def set_locals(self, locals: Dict[str, Any], frame_name: str) -> None:
        self.query_one(LocalsPanel).set_frame_name(frame_name)
        await self.query_one(LocalsPanel).add_locals(locals)

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

    def print_info(self, msg: str) -> None:
        self.log_tracer(Text(msg, style="bold blue"))

    def print_pretty(self, msg: Any) -> None:
        self.log_tracer(Pretty(msg))
