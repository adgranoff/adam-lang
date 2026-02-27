# Automatic Differentiation in Adam

Adam implements reverse-mode automatic differentiation (AD) via compile-time AST source transformation. When the compiler encounters `grad(f)`, it generates a new function that computes the gradient of `f`.

## How It Works

### 1. Detection

The autograd pass (`autograd.rs`) runs after type checking and before code generation. It scans for `let g = grad(f)` patterns where `f` is a named function.

### 2. Forward Tape Construction

The body of `f` is walked, building a "tape" -- a sequence of SSA-like intermediate values. Each operation is recorded with its inputs and operation type:

```
// Original function body:
let h = x @@ w1
let out = h @@ w2
tensor_sum(out)

// Tape:
_ad_v0 = x @@ w1        (MatMul, inputs: [x, w1])
_ad_v1 = _ad_v0 @@ w2   (MatMul, inputs: [_ad_v0, w2])
_ad_v2 = tensor_sum(_ad_v1) (Sum, inputs: [_ad_v1])
```

### 3. Backward Pass Generation

The tape is walked in reverse. For each operation, the adjoint (chain rule) is applied:

| Forward Op | Backward Rule |
|-----------|---------------|
| `c = a + b` | `d_a += d_c`, `d_b += d_c` |
| `c = a - b` | `d_a += d_c`, `d_b += -d_c` |
| `c = a * b` | `d_a += d_c * b`, `d_b += d_c * a` |
| `c = a @@ b` | `d_a += d_c @@ transpose(b)`, `d_b += transpose(a) @@ d_c` |
| `c = -a` | `d_a += -d_c` |
| `c = sum(a)` | `d_a += ones_like(a)` |
| `c = transpose(a)` | `d_a += transpose(d_c)` |

The output adjoint is seeded with `tensor_ones(tensor_shape(result))`.

### 4. Generated Function

The result is a new function with the same signature as the original, but returning the gradient with respect to the first parameter:

```adam
// grad(loss) generates approximately:
fn grad_loss(x) {
    // Forward pass
    let _ad_v0 = x @@ w1
    let _ad_v1 = _ad_v0 @@ w2
    let _ad_v2 = tensor_sum(_ad_v1)

    // Backward pass (reverse order)
    let _ad_d__ad_v2 = tensor_ones(tensor_shape(_ad_v2))
    let _ad_ones0 = tensor_ones(tensor_shape(_ad_v1))   // d(sum) = ones
    let _ad_bt0 = tensor_transpose(w2)                   // for d(@@)
    let _ad_mm_a0 = _ad_ones0 @@ _ad_bt0                // d_h
    let _ad_bt1 = tensor_transpose(w1)
    let _ad_mm_a1 = _ad_mm_a0 @@ _ad_bt1                // d_x

    _ad_mm_a1   // return gradient w.r.t. x
}
```

## Design Decisions

### Source Transformation vs Runtime Tape

Most AD systems (PyTorch, JAX) use runtime tape recording. Adam uses compile-time AST transformation instead:

- **No runtime overhead**: The gradient function is just regular code, no tape allocation or traversal
- **Optimizable**: The generated code goes through the normal compilation pipeline (type checking, bytecode generation)
- **Debuggable**: The gradient function is visible as normal Adam code

The tradeoff is that we can only differentiate through statically-known computation graphs. Loops and data-dependent control flow are not differentiated (they're treated as opaque).

### Adjoint Accumulation

When a variable is used multiple times, its adjoint contributions are accumulated:

```adam
// c = x + x  (x used twice)
// d_x = d_c + d_c  (two contributions added)
```

The accumulation uses fresh variables to maintain SSA form in the generated code.

### Scope

Only tensor operations are differentiated. The supported operations are:
- Binary: `+`, `-`, `*`, `@@`
- Unary: `-`
- Functions: `tensor_sum`, `tensor_transpose`

Scalar operations, control flow, and function calls to non-differentiable functions are treated as opaque (constant during differentiation).

## Comparison

| System | AD Method | Timing | Overhead |
|--------|-----------|--------|----------|
| Adam | Source transformation | Compile-time | Zero runtime |
| PyTorch | Dynamic tape | Runtime | Tape allocation per forward pass |
| JAX | Tracing + transformation | JIT compile | One-time compilation cost |
| Swift for TensorFlow | Compiler transformation | Compile-time | Zero runtime |
| Dex | Compiler transformation | Compile-time | Zero runtime |

Adam's approach is closest to Swift for TensorFlow and Dex, but with a simpler implementation (AST-to-AST rather than IR-level transformation).
