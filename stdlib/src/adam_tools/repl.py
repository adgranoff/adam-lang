"""Interactive REPL for the Adam language.

Uses prompt_toolkit for line editing, syntax highlighting, history,
and multiline input. Each expression is compiled to bytecode, executed
in the VM, and the result is printed.

Usage:
    adam-repl
    uv run adam-repl
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexer import RegexLexer
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    String as StringToken,
    Token,
)

from adam_tools.runner import find_compiler, find_repo_root, find_vm


class AdamLexer(RegexLexer):
    """Pygments lexer for Adam language syntax highlighting in the REPL."""

    name = "Adam"
    filenames = ["*.adam"]

    tokens = {
        "root": [
            (r"//.*$", Comment.Single),
            (r"\b(fn|let|if|else|while|for|in|match|return|break|continue|true|false|type|struct|impl)\b", Keyword),
            (r"\b(Int|Float|String|Bool|Nil)\b", Name.Builtin),
            (r"\b(println|len|push|clock)\b", Name.Builtin),
            (r'"[^"]*"', StringToken),
            (r"\b\d+\.\d+\b", Number.Float),
            (r"\b\d+\b", Number.Integer),
            (r"(\|>|=>|->|&&|\|\||[+\-*/%=<>!]=?)", Operator),
            (r"[a-zA-Z_]\w*", Name),
            (r"\s+", Token.Text),
            (r".", Token.Text),
        ]
    }


STYLE = Style.from_dict(
    {
        "prompt": "bold #6c9ef8",
        "continuation": "#666666",
    }
)

BANNER = """\
Adam Language REPL v0.1.0
Type expressions to evaluate. Use Ctrl+D to exit.
Multiline: end a line with { or \\ to continue.
"""


def is_incomplete(text: str) -> bool:
    """Heuristic: does the input look like it needs more lines?"""
    stripped = text.rstrip()
    if not stripped:
        return False
    # Open braces without matching close
    opens = stripped.count("{") - stripped.count("}")
    if opens > 0:
        return True
    # Explicit line continuation
    if stripped.endswith("\\"):
        return True
    return False


def run_snippet(
    source: str, compiler: Path, vm: Path, counter: int
) -> tuple[str, str]:
    """Compile and run a REPL snippet.

    Wraps the input so bare expressions get printed. Returns (output, errors).
    """
    # Wrap bare expressions in println() so the result is visible.
    # If the source already contains println or is a statement (let/fn/etc.),
    # don't wrap.
    lines = source.strip().splitlines()
    last_line = lines[-1].strip() if lines else ""
    is_statement = any(
        last_line.startswith(kw)
        for kw in ("let ", "fn ", "while ", "for ", "type ", "struct ", "impl ")
    )
    has_println = "println" in source

    if not is_statement and not has_println and not last_line.endswith("}"):
        # Wrap last expression in println
        if len(lines) == 1:
            source = f"println({source})"
        else:
            lines[-1] = f"println({last_line})"
            source = "\n".join(lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".adam", delete=False, prefix=f"repl_{counter}_"
    ) as src_file:
        src_file.write(source)
        src_path = Path(src_file.name)

    bytecode_path = src_path.with_suffix(".adamb")

    try:
        # Compile
        result = subprocess.run(
            [str(compiler), "compile", str(src_path), "-o", str(bytecode_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return "", result.stderr

        # Execute
        result = subprocess.run(
            [str(vm), str(bytecode_path)],
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr
    finally:
        src_path.unlink(missing_ok=True)
        bytecode_path.unlink(missing_ok=True)


def main() -> None:
    try:
        root = find_repo_root()
        compiler = find_compiler(root)
        vm = find_vm(root)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Build the project first: just build", file=sys.stderr)
        sys.exit(1)

    history_path = Path.home() / ".adam_repl_history"
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        lexer=PygmentsLexer(AdamLexer),
        style=STYLE,
    )

    print(BANNER)
    counter = 0

    while True:
        try:
            # Collect input (with multiline support)
            text = session.prompt("adam> ")
            while is_incomplete(text):
                continuation = session.prompt("  ... ")
                text += "\n" + continuation

            if not text.strip():
                continue

            counter += 1
            stdout, stderr = run_snippet(text, compiler, vm, counter)

            if stderr:
                # Filter out compiler warnings, show errors
                for line in stderr.splitlines():
                    if line.startswith("warning:"):
                        continue
                    if "Compiled to" in line:
                        continue
                    print(f"\033[31m{line}\033[0m", file=sys.stderr)

            if stdout:
                print(stdout, end="")

        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
