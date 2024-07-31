from textual.app import ComposeResult
from textual.widgets import Static, TextArea

TEXT = """\
from developer import Help

class App:
    def __init__(self):
        # This code editor is not yet operational, it is a placeholder for a future
        # feature. You may use the editor of your choosing to edit your code in the
        # meantime. Press 'r' or click on the footer button to reload the training
        # program.

    def help_im_stuck(self) -> ?:
        self.escape_key = '<Esc>'
        # Press <Esc> if you are stuck in this text area!


    def how_does_it_work(self) -> str:
        # Using the checkboxes on the top left, you can freeze/unfreeze modules of your
        # PyTorch program.
        self.instructions()

    def instructions(self) -> Help:
        # 1. The frozen modules will remain in memory and the code will
        # only be executed once, which will save you precious time in your research.

        # 2. The unfrozen modules will be entirely reloaded from disk and re-run when you
        # press 'r' or click on the footer button. This is called 'hot code reloading'.

        # 3. When your program crashes or throws an uncaught exception (i.e. bad tensor
        # operation), Matchbox will catch it and display the trace below this text area.
        # You will be able to debug it easily with the frame locals at time of death on
        # the lower left corner. We will soon introduce a REPL to aid in debugging.

        # 4. You may as well call builder.print() in your code to log anything on the
        # right Rich log panel. You can use any Rich renderables in addition to strings,
        # tensors and what not.
        # Hope this helps you become a more pragmatic and faster deep learning researcher :)
"""


class CodeEditor(Static):
    def compose(self) -> ComposeResult:
        yield TextArea.code_editor(TEXT, language="python")

    def on_mount(self):
        self.border_title = "Code editor"
        self.ready()

    def hang(self, threw: bool) -> None:
        self.styles.border = ("dashed", "red" if threw else "yellow")

    def ready(self) -> None:
        self.styles.border = ("solid", "green")
