"""Test runner for Adam programs.

Compiles and executes .adam test files, comparing actual output against
expected output stored in snapshot files. Supports three test modes:

1. **Snapshot tests**: Compare output against .expected files
2. **Inline tests**: Parse expected output from `// expect: <value>` comments
3. **Error tests**: Verify that specific type errors are reported

Usage:
    adam-test                          # Run all tests in tests/
    adam-test tests/test_arithmetic.adam  # Run a single test
    adam-test --update                 # Update snapshot files
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from adam_tools.runner import find_compiler, find_repo_root, find_vm, run_file


@dataclass
class TestResult:
    name: str
    passed: bool
    expected: str
    actual: str
    error: str
    duration_ms: float


def extract_inline_expectations(source: str) -> list[str]:
    """Extract expected output lines from `// expect: <value>` comments."""
    expectations = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("// expect:"):
            expected_value = stripped[len("// expect:") :].strip()
            expectations.append(expected_value)
    return expectations


def extract_error_expectations(source: str) -> list[str]:
    """Extract expected error patterns from `// error: <pattern>` comments."""
    errors = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("// error:"):
            pattern = stripped[len("// error:") :].strip()
            errors.append(pattern)
    return errors


def run_test(
    test_file: Path,
    snapshot_dir: Path,
    root: Path,
) -> TestResult:
    """Run a single test file and check against expectations.

    Checks (in order):
    1. Inline `// expect:` comments
    2. Inline `// error:` comments (for type error tests)
    3. Snapshot file in snapshot_dir
    """
    name = test_file.stem
    source = test_file.read_text()

    start = time.perf_counter()
    stdout, stderr, code = run_file(test_file, root)
    duration_ms = (time.perf_counter() - start) * 1000

    actual_output = stdout.rstrip("\n")

    # Check for error expectations
    error_expectations = extract_error_expectations(source)
    if error_expectations:
        all_output = stderr.rstrip("\n")
        missing = []
        for pattern in error_expectations:
            if pattern not in all_output:
                missing.append(pattern)
        if missing:
            return TestResult(
                name=name,
                passed=False,
                expected="\n".join(f"error containing: {p}" for p in error_expectations),
                actual=all_output,
                error=f"Missing error patterns: {missing}",
                duration_ms=duration_ms,
            )
        return TestResult(
            name=name,
            passed=True,
            expected="",
            actual="",
            error="",
            duration_ms=duration_ms,
        )

    # Check for inline expectations
    inline_expected = extract_inline_expectations(source)
    if inline_expected:
        expected_output = "\n".join(inline_expected)
        passed = actual_output == expected_output
        return TestResult(
            name=name,
            passed=passed,
            expected=expected_output,
            actual=actual_output,
            error=stderr if code != 0 else "",
            duration_ms=duration_ms,
        )

    # Check for snapshot file
    snapshot_file = snapshot_dir / f"{name}.expected"
    if snapshot_file.exists():
        expected_output = snapshot_file.read_text().rstrip("\n")
        passed = actual_output == expected_output
        return TestResult(
            name=name,
            passed=passed,
            expected=expected_output,
            actual=actual_output,
            error=stderr if code != 0 else "",
            duration_ms=duration_ms,
        )

    # No expectations found — mark as missing
    return TestResult(
        name=name,
        passed=False,
        expected="<no snapshot or inline expectations>",
        actual=actual_output,
        error="No expected output defined. Run with --update to create snapshots.",
        duration_ms=duration_ms,
    )


def update_snapshot(test_file: Path, snapshot_dir: Path, root: Path) -> str:
    """Run a test and write its output as the new snapshot."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stdout, stderr, code = run_file(test_file, root)
    if code != 0:
        return f"SKIP (compile/run error): {stderr.strip()}"

    snapshot_file = snapshot_dir / f"{test_file.stem}.expected"
    snapshot_file.write_text(stdout.rstrip("\n") + "\n")
    return f"Updated {snapshot_file}"


def discover_tests(test_dir: Path) -> list[Path]:
    """Find all .adam test files in a directory."""
    if not test_dir.exists():
        return []
    return sorted(test_dir.glob("**/*.adam"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adam-test",
        description="Test runner for Adam programs.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific test files to run (default: all in tests/ and examples/)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update snapshot files with current output",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="Snapshot directory (default: tests/snapshots/)",
    )
    args = parser.parse_args()

    root = find_repo_root()
    snapshot_dir = args.snapshots or (root / "tests" / "snapshots")

    # Discover test files
    if args.files:
        test_files = args.files
    else:
        test_files = discover_tests(root / "examples")

    if not test_files:
        print("No test files found.")
        sys.exit(0)

    if args.update:
        print(f"Updating snapshots for {len(test_files)} files...\n")
        for tf in test_files:
            msg = update_snapshot(tf, snapshot_dir, root)
            print(f"  {tf.name}: {msg}")
        print("\nDone.")
        return

    # Run tests
    results: list[TestResult] = []
    for tf in test_files:
        result = run_test(tf, snapshot_dir, root)
        results.append(result)

        status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
        print(f"  {status}  {result.name} ({result.duration_ms:.0f}ms)")

        if not result.passed:
            if result.error:
                print(f"         Error: {result.error}")
            if result.expected != result.actual:
                print(f"         Expected: {result.expected!r}")
                print(f"         Actual:   {result.actual!r}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_ms = sum(r.duration_ms for r in results)

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed ({total_ms:.0f}ms total)")
    if failed > 0:
        print(f"  Failed: {', '.join(r.name for r in results if not r.passed)}")
    print(f"{'=' * 50}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
