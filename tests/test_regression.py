"""Regression tests — verify all existing examples produce unchanged output.

Compiles each .adam file directly (no build step) and compares output
against baseline snapshots. Uses the adam_tools.runner for compilation.

Run with:
    cd stdlib && uv run --with pytest pytest "../tests/test_regression.py" -v
Or:
    just test-regression
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "stdlib" / "src"))
from adam_tools.runner import find_repo_root, run_file


ROOT = find_repo_root()
EXAMPLES_DIR = ROOT / "examples"
SNAPSHOTS_DIR = ROOT / "tests" / "snapshots"

# Files that require external data or special setup
SKIP_FILES = {"mnist", "mnist_smoke"}

BASELINE_FILES = sorted(
    f
    for f in SNAPSHOTS_DIR.glob("*_baseline.txt")
    if f.stem.replace("_baseline", "") not in SKIP_FILES
)


@pytest.mark.parametrize(
    "baseline_file",
    BASELINE_FILES,
    ids=[f.stem.replace("_baseline", "") for f in BASELINE_FILES],
)
def test_regression(baseline_file: Path) -> None:
    """Verify example output matches its baseline snapshot."""
    name = baseline_file.stem.replace("_baseline", "")
    adam_file = EXAMPLES_DIR / f"{name}.adam"

    if not adam_file.exists():
        pytest.skip(f"No .adam file for {name}")

    expected = baseline_file.read_text().rstrip("\n")
    stdout, stderr, code = run_file(adam_file, ROOT)
    actual = stdout.rstrip("\n")

    assert code == 0, f"Runtime error in {name}: {stderr}"
    assert actual == expected, (
        f"Regression in {name}:\n"
        f"  Expected: {expected[:200]}\n"
        f"  Actual:   {actual[:200]}"
    )
