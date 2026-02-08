import os
import difflib
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax

console = Console()

def is_safe_path(base_path: str, target_path: str) -> bool:
    """
    Check if the target path is within the base path.
    """
    base = Path(base_path).resolve()
    target = Path(target_path).resolve()
    return base in target.parents or base == target

def show_diff(file_path: str, new_content: str):
    """
    Show a diff between the current file content and the new content.
    """
    try:
        with open(file_path, "r") as f:
            old_content = f.read()
    except FileNotFoundError:
        old_content = ""

    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )

    diff_text = "".join(diff)
    if not diff_text:
        return

    syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
    console.print(syntax)
