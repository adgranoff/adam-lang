# Adam Language — Cross-language build orchestration
# Usage: just <recipe>

# Default PATH setup for Windows (MSYS2 MinGW + Cargo)
export PATH := env("HOME") + "/.cargo/bin:/c/msys64/mingw64/bin:" + env("PATH")

# Default recipe: build everything
default: build

# ─── Build ───────────────────────────────────────────────────────────────────

# Build all components
build: build-vm build-compiler

# Build everything including TypeScript tooling
build-all: build build-lsp build-playground

# Build the C virtual machine
build-vm:
    cd vm && cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Build the Rust compiler
build-compiler:
    cd compiler && cargo build --release

# Install Python tooling dependencies
setup-python:
    cd stdlib && uv sync

# ─── Test ────────────────────────────────────────────────────────────────────

# Run all test suites
test: test-vm test-compiler test-e2e

# Run C VM tests
test-vm: build-vm
    cd vm/build && ctest --output-on-failure

# Run Rust compiler tests (lexer, parser, type checker, codegen)
test-compiler:
    cd compiler && cargo test

# Run end-to-end integration tests (Python pytest)
test-e2e: build
    cd stdlib && uv run --with pytest pytest "../tests/test_e2e.py" -v

# Run Adam test runner (inline expectations)
test-adam: build
    cd stdlib && uv run adam-test

# ─── Dev ─────────────────────────────────────────────────────────────────────

# Run a .adam file end-to-end (compile + execute)
run file: build
    cd stdlib && uv run adam run "../{{file}}"

# Type-check a .adam file
check file: build-compiler
    cd stdlib && uv run adam check "../{{file}}"

# Launch the interactive REPL
repl: build
    cd stdlib && uv run adam-repl

# Run benchmarks
bench: build
    cd stdlib && uv run adam-bench

# Build TypeScript LSP server
build-lsp:
    cd tools/lsp && npm install && npx tsc

# Build web playground
build-playground:
    cd tools/playground && npm install && npx vite build

# Start playground dev server
playground-dev:
    cd tools/playground && npm run dev

# Clean all build artifacts
clean:
    rm -rf vm/build
    cd compiler && cargo clean

# Format all code
fmt:
    cd compiler && cargo fmt

# Lint all code
lint:
    cd compiler && cargo clippy -- -D warnings
