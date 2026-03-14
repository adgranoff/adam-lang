"""Runner — compile and execute .adam files.

Orchestrates the Rust compiler (adamc) and the C virtual machine (adam-vm)
as subprocesses. Locates binaries by walking up from the stdlib directory
to the repo root.

Usage:
    adam run examples/fibonacci.adam
    adam compile examples/fibonacci.adam -o out.adamb
    adam check examples/fibonacci.adam
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains compiler/ and vm/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "compiler").is_dir() and (current / "vm").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Cannot find repo root (expected directories: compiler/, vm/)"
    )


def find_compiler(root: Path) -> Path:
    """Locate the adamc compiler binary."""
    candidates = [
        root / "compiler" / "target" / "release" / "adamc.exe",
        root / "compiler" / "target" / "release" / "adamc",
        root / "compiler" / "target" / "debug" / "adamc.exe",
        root / "compiler" / "target" / "debug" / "adamc",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find adamc binary. Build with: cargo build --manifest-path {root / 'compiler' / 'Cargo.toml'}"
    )


def find_vm(root: Path) -> Path:
    """Locate the adam-vm binary."""
    candidates = [
        root / "vm" / "build" / "adam-vm.exe",
        root / "vm" / "build" / "adam-vm",
        root / "vm" / "build" / "Release" / "adam-vm.exe",
        root / "vm" / "build" / "Debug" / "adam-vm.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot find adam-vm binary. Build with: cmake --build {root / 'vm' / 'build'}"
    )


def run_compile(
    compiler: Path,
    source: Path,
    output: Path | None = None,
    *,
    check_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the compiler on a source file.

    Args:
        compiler: Path to the adamc binary.
        source: Path to the .adam source file.
        output: Path for the .adamb output (None = derive from source).
        check_only: If True, only type-check (don't generate bytecode).

    Returns:
        CompletedProcess with stdout/stderr.
    """
    if check_only:
        cmd = [str(compiler), "check", str(source)]
    else:
        cmd = [str(compiler), "compile", str(source)]
        if output:
            cmd.extend(["-o", str(output)])

    return subprocess.run(cmd, capture_output=True, text=True)


def run_vm(
    vm: Path, bytecode: Path, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute a .adamb bytecode file in the VM.

    Args:
        vm: Path to the adam-vm binary.
        bytecode: Path to the .adamb file.
        cwd: Working directory for the VM process (None = inherit).

    Returns:
        CompletedProcess with stdout/stderr.
    """
    return subprocess.run(
        [str(vm), str(bytecode)], capture_output=True, text=True, cwd=cwd
    )


def run_file(
    source: Path,
    root: Path | None = None,
) -> tuple[str, str, int]:
    """Compile and run a .adam file end-to-end.

    Args:
        source: Path to the .adam source file.
        root: Repo root (auto-detected if None).

    Returns:
        Tuple of (stdout, stderr, return_code).
    """
    if root is None:
        root = find_repo_root()
    compiler = find_compiler(root)
    vm = find_vm(root)

    with tempfile.NamedTemporaryFile(suffix=".adamb", delete=False) as tmp:
        bytecode_path = Path(tmp.name)

    try:
        result = run_compile(compiler, source, bytecode_path)
        if result.returncode != 0:
            return "", result.stderr, result.returncode

        result = run_vm(vm, bytecode_path, cwd=root)
        return result.stdout, result.stderr, result.returncode
    finally:
        bytecode_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adam",
        description="Adam language runner — compile and execute .adam files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # adam run <file>
    run_parser = subparsers.add_parser("run", help="Compile and execute a .adam file")
    run_parser.add_argument("file", type=Path, help="Path to .adam source file")

    # adam compile <file> [-o output]
    compile_parser = subparsers.add_parser(
        "compile", help="Compile a .adam file to bytecode"
    )
    compile_parser.add_argument("file", type=Path, help="Path to .adam source file")
    compile_parser.add_argument(
        "-o", "--output", type=Path, help="Output .adamb file path"
    )

    # adam check <file>
    check_parser = subparsers.add_parser("check", help="Type-check a .adam file")
    check_parser.add_argument("file", type=Path, help="Path to .adam source file")

    args = parser.parse_args()
    root = find_repo_root()

    if args.command == "run":
        stdout, stderr, code = run_file(args.file, root)
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if stdout:
            print(stdout, end="")
        sys.exit(code)

    elif args.command == "compile":
        compiler = find_compiler(root)
        result = run_compile(compiler, args.file, args.output)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        sys.exit(result.returncode)

    elif args.command == "check":
        compiler = find_compiler(root)
        result = run_compile(compiler, args.file, check_only=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
