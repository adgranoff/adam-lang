"""End-to-end integration tests for Adam.

Compiles .adam source files → bytecode → runs in VM → checks output.
Uses pytest for discovery and assertions.

Run with:
    cd tests && uv run pytest test_e2e.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add stdlib to path so we can import adam_tools
sys.path.insert(0, str(Path(__file__).parent.parent / "stdlib" / "src"))
from adam_tools.runner import find_repo_root, run_file


ROOT = find_repo_root()
EXAMPLES_DIR = ROOT / "examples"
SNAPSHOTS_DIR = ROOT / "tests" / "snapshots"


def load_snapshot(name: str) -> str | None:
    """Load expected output from a snapshot file."""
    snapshot = SNAPSHOTS_DIR / f"{name}.expected"
    if snapshot.exists():
        return snapshot.read_text().rstrip("\n")
    return None


def load_inline_expectations(source: str) -> list[str] | None:
    """Extract expected output from // expect: comments."""
    expectations = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("// expect:"):
            expectations.append(stripped[len("// expect:") :].strip())
    return expectations if expectations else None


# ── Parametrized tests from examples/ ────────────────────────────────

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.adam"))


@pytest.mark.parametrize(
    "adam_file",
    EXAMPLE_FILES,
    ids=[f.stem for f in EXAMPLE_FILES],
)
def test_example(adam_file: Path) -> None:
    """Compile and run each example, verify output matches expectations."""
    source = adam_file.read_text()
    name = adam_file.stem

    stdout, stderr, code = run_file(adam_file, ROOT)
    actual = stdout.rstrip("\n")

    # Try inline expectations first
    inline = load_inline_expectations(source)
    if inline is not None:
        expected = "\n".join(inline)
        assert code == 0, f"Runtime error: {stderr}"
        assert actual == expected, f"Output mismatch for {name}"
        return

    # Fall back to snapshot
    expected = load_snapshot(name)
    if expected is not None:
        assert code == 0, f"Runtime error: {stderr}"
        assert actual == expected, f"Output mismatch for {name}"
        return

    # No expectations — just verify it doesn't crash
    assert code == 0, f"Unexpected error running {name}: {stderr}"


# ── Specific feature tests ───────────────────────────────────────────


class TestArithmetic:
    def test_integer_math(self) -> None:
        stdout, _, code = run_file(EXAMPLES_DIR / "test_basic.adam", ROOT)
        assert code == 0
        assert stdout.strip() == "10"

    def test_pipe_chain(self) -> None:
        stdout, _, code = run_file(EXAMPLES_DIR / "test_chain.adam", ROOT)
        assert code == 0
        assert stdout.strip() == "11"


class TestFibonacci:
    def test_fib_20(self) -> None:
        stdout, _, code = run_file(EXAMPLES_DIR / "fibonacci.adam", ROOT)
        assert code == 0
        assert stdout.strip() == "6765"


class TestCalculator:
    def test_full_program(self) -> None:
        stdout, _, code = run_file(EXAMPLES_DIR / "calculator.adam", ROOT)
        assert code == 0
        lines = stdout.strip().splitlines()
        assert lines == ["11", "5", "30", "15", "Hello World!", "false", "true"]
