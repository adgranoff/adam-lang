"""Benchmark runner for Adam programs.

Compiles and times execution of .adam benchmark files, running each
multiple times to get stable measurements. Outputs results as a
formatted table and optionally writes JSON for charts.

Usage:
    adam-bench                           # Run all benchmarks
    adam-bench benchmarks/fib35.adam      # Run a single benchmark
    adam-bench --iterations 20           # Custom iteration count
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from adam_tools.runner import (
    find_compiler,
    find_repo_root,
    find_vm,
    run_compile,
    run_vm,
)


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    compile_ms: float


def run_benchmark(
    source_file: Path,
    compiler: Path,
    vm: Path,
    iterations: int,
) -> BenchmarkResult:
    """Run a benchmark: compile once, execute N times, collect timings."""
    name = source_file.stem
    bytecode_path = source_file.with_suffix(".adamb")

    # Compile (timed separately)
    compile_start = time.perf_counter()
    result = run_compile(compiler, source_file, bytecode_path)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    if result.returncode != 0:
        print(f"  SKIP {name}: compilation failed", file=sys.stderr)
        print(f"    {result.stderr.strip()}", file=sys.stderr)
        return BenchmarkResult(
            name=name,
            iterations=0,
            mean_ms=0,
            median_ms=0,
            min_ms=0,
            max_ms=0,
            stddev_ms=0,
            compile_ms=compile_ms,
        )

    # Execute N times
    timings: list[float] = []
    for i in range(iterations):
        start = time.perf_counter()
        run_vm(vm, bytecode_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    # Clean up bytecode
    bytecode_path.unlink(missing_ok=True)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=statistics.mean(timings),
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        stddev_ms=statistics.stdev(timings) if len(timings) > 1 else 0,
        compile_ms=compile_ms,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    if not results:
        print("No benchmark results.")
        return

    header = f"{'Benchmark':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10} {'Compile':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r.iterations == 0:
            print(f"{r.name:<25} {'SKIPPED':>10}")
            continue
        print(
            f"{r.name:<25} {r.mean_ms:>9.1f}ms {r.median_ms:>9.1f}ms "
            f"{r.min_ms:>9.1f}ms {r.max_ms:>9.1f}ms {r.stddev_ms:>9.1f}ms "
            f"{r.compile_ms:>9.1f}ms"
        )


def save_results(results: list[BenchmarkResult], output: Path) -> None:
    """Save benchmark results as JSON."""
    data = [
        {
            "name": r.name,
            "iterations": r.iterations,
            "mean_ms": round(r.mean_ms, 2),
            "median_ms": round(r.median_ms, 2),
            "min_ms": round(r.min_ms, 2),
            "max_ms": round(r.max_ms, 2),
            "stddev_ms": round(r.stddev_ms, 2),
            "compile_ms": round(r.compile_ms, 2),
        }
        for r in results
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nResults saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adam-bench",
        description="Benchmark runner for Adam programs.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Benchmark files (default: all in benchmarks/)",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10, help="Iterations per benchmark"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Save results as JSON"
    )
    args = parser.parse_args()

    root = find_repo_root()
    compiler = find_compiler(root)
    vm = find_vm(root)

    # Discover benchmark files
    if args.files:
        bench_files = args.files
    else:
        bench_dir = root / "benchmarks"
        bench_files = sorted(bench_dir.glob("*.adam")) if bench_dir.exists() else []

    if not bench_files:
        print("No benchmark files found.")
        sys.exit(0)

    print(f"Running {len(bench_files)} benchmarks ({args.iterations} iterations each)\n")

    results: list[BenchmarkResult] = []
    for bf in bench_files:
        result = run_benchmark(bf, compiler, vm, args.iterations)
        results.append(result)

    print_results(results)

    if args.output:
        save_results(results, args.output)
    else:
        default_output = root / "benchmarks" / "results" / "latest.json"
        save_results(results, default_output)


if __name__ == "__main__":
    main()
