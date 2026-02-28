# Adam

A statically-typed, expression-oriented programming language with Hindley-Milner type inference, compiled to bytecode and executed on a custom virtual machine.

```
fn fibonacci(n) {
    if n <= 1 { n }
    else { fibonacci(n - 1) + fibonacci(n - 2) }
}

println(fibonacci(20))
```

## Architecture

```mermaid
graph LR
    A[Source .adam] --> B[Lexer]
    B --> C[Parser]
    C --> D[Type Checker]
    D --> D2[Autograd]
    D2 --> E[Codegen]
    E --> F[Bytecode .adamb]
    F --> G[Virtual Machine]
    G --> H[Output]

    style B fill:#dea584,color:#000
    style C fill:#dea584,color:#000
    style D fill:#dea584,color:#000
    style E fill:#dea584,color:#000
    style G fill:#7ec8e3,color:#000
```

| Component | Language | Description |
|-----------|----------|-------------|
| **Compiler** | Rust | Lexer, Pratt parser, HM type inference, autograd, bytecode codegen |
| **VM** | C | NaN-boxed values, computed goto dispatch, tri-color GC |
| **Tooling** | Python | CLI runner, interactive REPL, test framework, benchmarks |
| **Editor** | TypeScript | LSP server, browser playground with Monaco editor |

## Language Features

### Type Inference

Types are inferred via Hindley-Milner Algorithm W. No type annotations required.

```
let x = 42          // inferred as Int
let name = "hello"  // inferred as String

fn id(x) { x }      // polymorphic: forall a. a -> a
id(42)               // Int
id(true)             // Bool
```

### First-Class Functions and Closures

```
fn make_counter(start) {
    let count = start
    |step| {
        count = count + step
        count
    }
}

let counter = make_counter(0)
println(counter(1))   // 1
println(counter(1))   // 2
println(counter(5))   // 7
```

### Pipe Operator

```
fn double(x) { x * 2 }
fn add_one(x) { x + 1 }
fn square(x) { x * x }

let result = 5 |> double |> add_one |> square
println(result)  // 121
```

### Arrays

```
let numbers = [10, 20, 30, 40, 50]
println(len(numbers))   // 5
println(numbers[2])     // 30

let total = 0
for x in numbers {
    total = total + x
}
println(total)  // 150
```

### Shape-Typed Tensors

Built-in tensor operations with compile-time dimension checking:

```
// Dimension variables (N) are checked at compile time
fn predict(images: Tensor<Float, [N, 784]>) -> Tensor<Float, [N, 10]> {
    images @@ weights + bias   // [N,784] @@ [784,10] -> [N,10]
}

// Matrix multiply, element-wise ops, and tensor utilities
let a = tensor_from_array([1, 2, 3, 4, 5, 6], [2, 3])
let b = tensor_ones([3, 2])
let c = a @@ b         // matrix multiply
let d = c + c          // element-wise add
println(tensor_sum(d)) // reduce to scalar
```

### Automatic Differentiation

Compile-time reverse-mode AD via AST source transformation:

```
fn loss(x) {
    let h = x @@ w1
    tensor_sum(h @@ w2)
}

// grad() is a compiler intrinsic -- no runtime tape
let grad_loss = grad(loss)
let grads = grad_loss(input)
```

No shipped language combines shape-dependent types, compiler-pass AD, and simplicity. Adam achieves this by restricting dimension variables to integers (not full dependent types) and limiting AD to a tractable set of tensor operations.

### MNIST Handwritten Digit Recognition

Adam's tensor system is powerful enough to train a real neural network. A 2-layer network (784->128->10) achieves **97.2% accuracy** on MNIST, trained entirely in Adam with manual backpropagation:

```
let w1 = tensor_randn([784, 128]) * 0.047
let b1 = tensor_zeros([1, 128])

// Forward pass
let z1 = x @@ w1 + b1
let h = tensor_relu(z1)
let logits = h @@ w2 + b2
let probs = tensor_exp(logits) / tensor_sum_axis(tensor_exp(logits), 1)

// Backward pass (manual)
let d_logits = (probs - targets) / 32.0
let d_w2 = tensor_transpose(h) @@ d_logits
w2 = w2 - lr * d_w2
```

See [MNIST Demo](docs/mnist.md) for full details. Run with:

```bash
just prepare-mnist   # download MNIST data
just mnist           # train and evaluate (~2 minutes)
```

### Expression-Oriented

Everything returns a value. `if/else`, blocks, and functions all evaluate to their last expression.

```
let max = if a > b { a } else { b }

let result = {
    let x = 10
    let y = 20
    x + y
}
```

## Quick Start

### Prerequisites

- Rust (cargo)
- C compiler (GCC or MinGW) + CMake
- Python 3.11+ with [UV](https://docs.astral.sh/uv/)
- [just](https://github.com/casey/just) command runner

### Build and Run

```bash
# Build the compiler and VM
just build

# Run a program
just run examples/fibonacci.adam

# Launch the REPL
just repl

# Run all tests
just test
```

## Build Commands

| Command | Description |
|---------|-------------|
| `just build` | Build VM and compiler |
| `just test` | Run all test suites (VM, compiler, E2E) |
| `just run <file>` | Compile and execute an .adam file |
| `just check <file>` | Type-check without executing |
| `just repl` | Launch interactive REPL |
| `just bench` | Run benchmark suite |
| `just test-adam` | Run inline expectation tests |
| `just test-e2e` | Run pytest E2E integration tests |
| `just fmt` | Format Rust code |
| `just lint` | Lint Rust code |
| `just prepare-mnist` | Download and prepare MNIST dataset |
| `just mnist` | Train MNIST neural network (~2 min) |
| `just clean` | Remove build artifacts |

## Project Structure

```
adam-lang/
├── compiler/                # Rust compiler
│   └── src/
│       ├── lexer.rs         # Tokenizer with span tracking
│       ├── parser.rs        # Pratt parser + recursive descent
│       ├── types.rs         # Hindley-Milner type inference
│       ├── autograd.rs      # Reverse-mode AD via AST transformation
│       ├── compiler.rs      # AST to bytecode compilation
│       ├── bytecode.rs      # Bytecode format and serialization
│       ├── ast.rs           # AST node definitions
│       ├── token.rs         # Token types
│       └── errors.rs        # Error types with source spans
├── vm/                      # C virtual machine
│   ├── include/adam/        # Public headers
│   │   ├── value.h          # NaN-boxed value representation
│   │   ├── vm.h             # VM API
│   │   ├── gc.h             # Garbage collector
│   │   ├── table.h          # Robin Hood hash table
│   │   └── object.h         # Heap-allocated objects
│   └── src/                 # Implementation
│       ├── vm.c             # Dispatch loop (computed goto)
│       ├── gc.c             # Tri-color mark-and-sweep
│       ├── table.c          # Hash table with Robin Hood hashing
│       └── value.c          # NaN boxing encode/decode
├── tools/                   # TypeScript editor tooling
│   ├── lsp/                 # Language Server Protocol server
│   ├── playground/          # Browser playground (Monaco + interpreter)
│   └── syntax/              # TextMate grammar
├── stdlib/                  # Python tooling
│   └── src/adam_tools/
│       ├── runner.py        # CLI: compile + execute
│       ├── repl.py          # Interactive REPL
│       ├── test_runner.py   # Test framework
│       ├── benchmark.py     # Benchmark suite
│       └── prepare_mnist.py # MNIST data pipeline
├── examples/                # Example Adam programs
├── benchmarks/              # Performance benchmarks
├── tests/                   # E2E integration tests
├── docs/                    # Technical documentation
└── Justfile                 # Build orchestration
```

## Documentation

- [Architecture Overview](docs/architecture.md) -- system design and data flow
- [Language Reference](docs/language-spec.md) -- grammar, types, and semantics
- [VM Internals](docs/vm-internals.md) -- NaN boxing, GC, dispatch loop
- [Compiler Pipeline](docs/compiler-pipeline.md) -- lexer, parser, type inference, codegen
- [Tensor Types](docs/tensor-types.md) -- shape-dependent type system for tensors
- [Autograd](docs/autograd.md) -- compile-time reverse-mode automatic differentiation
- [MNIST Demo](docs/mnist.md) -- training a neural network from scratch in Adam
