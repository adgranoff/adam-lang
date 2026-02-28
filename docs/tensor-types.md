# Shape-Dependent Tensor Types

Adam's type system extends Hindley-Milner inference with dimension variables for compile-time tensor shape checking.

## Dimension Variables

Dimension variables are uppercase identifiers (`N`, `M`, `K`) that appear in tensor type annotations:

```
fn predict(images: Tensor<Float, [N, 784]>) -> Tensor<Float, [N, 10]> {
    images @@ weights + bias
}
```

During type checking, dimension variables behave like type variables: they unify with concrete integers or other dimension variables. The rules are:

| Left | Right | Result |
|------|-------|--------|
| `DimLit(784)` | `DimLit(784)` | OK (equal) |
| `DimLit(784)` | `DimLit(10)` | Shape mismatch error |
| `DimVar(N)` | `DimLit(784)` | `N` binds to `784` |
| `DimVar(N)` | `DimVar(M)` | `N` and `M` unify |

This is a restricted form of dependent types -- dimension variables are integers with unification, not arbitrary terms. This keeps the type system decidable while catching shape errors.

## Type System Implementation

In `types.rs`, the `Type` enum includes:

```rust
Type::Tensor {
    dtype: Box<Type>,           // Float or Int
    shape: Vec<DimExprType>,    // e.g., [DimVar(0), DimLit(784)]
}
```

`DimExprType` mirrors the AST's `DimExpr` but uses numeric IDs for variables (shared ID space with type variables). The substitution structure has a separate `dim_bindings: HashMap<u32, DimExprType>` for dimension unification.

### Tensor Unification

When unifying `Tensor<D1, S1>` with `Tensor<D2, S2>`:
1. Unify `D1` with `D2` (element type must match)
2. Shapes must have the same number of dimensions
3. Each dimension is unified pairwise

### MatMul Shape Checking

`Tensor<F, [.., M, K]> @@ Tensor<F, [K, N]>` produces `Tensor<F, [.., M, N]>`:
1. Left operand must have >= 2 dimensions
2. Right operand must be exactly 2D
3. Inner dimensions (`K`) must unify
4. Result shape: left's batch dimensions + `[M, N]`

## Comparison With Other Approaches

| System | Approach | Status |
|--------|----------|--------|
| Adam | Dimension variables in HM type system | Shipping |
| Dex (Google) | Full dependent types for arrays | Discontinued |
| Swift for TensorFlow | Shape checking via library protocols | Discontinued |
| jaxtyping (Python) | Runtime shape assertions | Active, runtime-only |
| Idris/Agda | Full dependent types | Not practical for ML |

Adam's approach is right-sized: dimension variables are powerful enough to catch real shape errors but simple enough to integrate into an existing HM type checker without making inference undecidable.

## Runtime Broadcasting

At runtime, the VM supports NumPy-style broadcasting for 2D tensor arithmetic:

| Operand A | Operand B | Broadcast Rule |
|-----------|-----------|----------------|
| `[M, N]` | `[M, N]` | Element-wise (no broadcast) |
| `[M, N]` | `[1, N]` | Row broadcast — repeat b's row M times |
| `[M, N]` | `[M, 1]` | Column broadcast — repeat b's column N times |
| `[M, N]` | scalar | Scalar broadcast — apply to every element |

The broadcast index is computed by `broadcast_idx()` in `vm.c`, which checks shapes to distinguish row vs column broadcasting. This is essential for operations like softmax normalization (`exps / sum_exps` where shapes are `[32,10] / [32,1]`).

## Practical Validation

The tensor type system and runtime are validated by the MNIST demo (`examples/mnist.adam`), which trains a 784->128->10 neural network to 97.2% accuracy using tensor operations for forward pass, backpropagation, and SGD weight updates. See [MNIST Demo](mnist.md) for details.
