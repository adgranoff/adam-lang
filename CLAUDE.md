# Adam Language — Claude Code Project Guide

## What This Is

Adam is a statically-typed, expression-oriented programming language with:
- **Rust compiler**: lexer, Pratt parser, Hindley-Milner type inference, autograd pass, bytecode codegen
- **C virtual machine**: NaN-boxed values, computed goto dispatch, tri-color mark-and-sweep GC
- **Python tooling**: CLI runner, REPL, test framework, benchmarks
- **TypeScript editor**: LSP server, browser playground with tree-walking interpreter

## Build Commands

```bash
# Build everything (requires cargo, cmake/ninja, MinGW on Windows)
just build

# Build individual components
just build-vm          # C VM (cmake + ninja)
just build-compiler    # Rust compiler (cargo)

# Run a program end-to-end
just run examples/fibonacci.adam

# Type check only
just check examples/calculator.adam

# Run all tests
just test

# Run specific test suites
cd compiler && cargo test          # Rust unit tests (75 tests)
cd tests && uv run pytest -v      # Python E2E tests
cd tools/playground && npx tsc --noEmit  # TypeScript type check
```

## Architecture

```
Source.adam → Lexer → Parser → Type Checker → Autograd → Codegen → .adamb → VM → Output
              Rust                                                          C
```

The compiler and VM are separate executables. Python orchestrates them.

## Key Files

### Compiler (Rust) — `compiler/src/`
| File | Purpose |
|------|---------|
| `token.rs` | Token types and spans |
| `lexer.rs` | Character-by-character tokenizer |
| `parser.rs` | Pratt parser (expressions) + recursive descent (statements) |
| `ast.rs` | AST node types: Stmt, Expr, BinOp, Pattern, TypeExpr, DimExpr |
| `types.rs` | HM type inference with Algorithm W, tensor shape checking |
| `autograd.rs` | Reverse-mode AD via AST source transformation (grad() intrinsic) |
| `compiler.rs` | AST → bytecode emission, variable resolution, jump patching |
| `bytecode.rs` | Op enum, .adamb serialization format |

### VM (C) — `vm/`
| File | Purpose |
|------|---------|
| `include/adam/value.h` | NaN-boxed Value (uint64_t) — all values in 8 bytes |
| `include/adam/object.h` | ObjType enum, ObjString/Function/Closure/Array/Struct/Tensor |
| `include/adam/chunk.h` | OpCode enum (must match Rust Op enum exactly) |
| `src/vm.c` | Dispatch loop with computed goto / switch fallback |
| `src/gc.c` | Tri-color mark-and-sweep garbage collector |
| `src/native.c` | Built-in functions: clock, println, len, tensor_* |
| `src/object.c` | Object allocation, deallocation, printing |

### Tooling
| File | Purpose |
|------|---------|
| `tools/playground/src/interpreter.ts` | Browser tree-walking interpreter |
| `tools/playground/src/examples.ts` | Playground example programs |
| `stdlib/src/adam_tools/runner.py` | CLI compile + execute orchestrator |
| `tests/test_e2e.py` | Pytest E2E tests with inline `// expect:` assertions |

## Critical Invariants

1. **Opcode sync**: `Op` enum in `bytecode.rs` and `OpCode` enum in `chunk.h` MUST have identical discriminant order. Adding an opcode to one requires adding it to the other at the same position.

2. **NaN boxing**: Values are 64-bit `uint64_t`. Floats are raw doubles. Non-floats use quiet NaN bit patterns. Object pointers use the sign bit. See `value.h` for the bit layout.

3. **Computed goto dispatch table** in `vm.c` must list labels in the same order as the OpCode enum. Adding a new opcode requires adding its label to `dispatch_table[]`.

4. **Type variable IDs** are shared between type variables and dimension variables in `types.rs`. The `next_var` counter is used for both. Dimension bindings use a separate `dim_bindings` map.

5. **String interning**: All ObjStrings are interned in `vm->strings`. Pointer equality == string equality.

## Tensor System

- **Type checking**: `Tensor<DType, [dims]>` syntax with dimension variables (N, M, K). Compile-time shape verification via dimension unification in `types.rs`.
- **Runtime**: `ObjTensor` in C with `ndim`, `shape[]`, `data[]`. Naive triple-loop matmul.
- **Autograd**: `grad(f)` is a compiler intrinsic. `autograd.rs` transforms the AST of `f` into forward + backward passes. No runtime tape.

## Testing Conventions

- **Compiler unit tests**: In each `.rs` file's `#[cfg(test)] mod tests`. Run with `cargo test`.
- **E2E tests**: `tests/test_e2e.py` compiles `.adam` files and checks output against `// expect:` comments.
- **Example programs**: `examples/*.adam` — used by E2E tests. Each should have inline expectations.

## Common Tasks

### Adding a new opcode
1. Add to `Op` enum in `compiler/src/bytecode.rs`
2. Add to `OpCode` enum in `vm/include/adam/chunk.h` (same position)
3. Add handler in `vm/src/vm.c` (both switch case and computed goto label in dispatch table)
4. Emit in `compiler/src/compiler.rs`

### Adding a native function
1. Implement in `vm/src/native.c` with signature `Value fn(VM*, int, Value*)`
2. Register in `adam_register_natives()`
3. Add type to type checker in `types.rs` `register_builtins()`
4. Add to playground interpreter in `tools/playground/src/interpreter.ts`

### Adding a new AST node
1. Add variant to `ExprKind` or `StmtKind` in `ast.rs`
2. Parse in `parser.rs`
3. Type check in `types.rs`
4. Compile in `compiler.rs`
5. Handle in playground `interpreter.ts`
