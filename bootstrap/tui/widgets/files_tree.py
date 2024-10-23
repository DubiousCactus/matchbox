from textual.app import ComposeResult
from textual.widgets import (
    Static,
    Tree,
)


class FilesTree(Static):
    def compose(self) -> ComposeResult:
        tree: Tree[dict] = Tree("root")
        tree.root.expand()
        src = tree.root.add("src", expand=True)
        src.add_leaf("base_trainer.py")
        src.add_leaf("base_tester.py")
        src.add_leaf("this_is_dummy.py")
        other = tree.root.add("it_is_all_dummy", expand=True)
        other.add_leaf("even_this.py")
        other.add_leaf("the_whole_tree.py")
        yield tree
