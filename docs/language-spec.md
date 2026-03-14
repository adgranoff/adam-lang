# Adam Language Reference

## Types

| Type | Examples | Description |
|------|----------|-------------|
| Int | `42`, `-7`, `0` | 32-bit signed integer |
| Float | `3.14`, `0.5` | 64-bit IEEE 754 double |
| String | `"hello"`, `""` | Immutable string (double-quoted) |
| Bool | `true`, `false` | Boolean |
| Nil | (implicit) | Absence of value |
| Array | `[1, 2, 3]` | Ordered, mutable, homogeneous |
| Function | `fn f(x) { x }` | First-class, supports closures |
| Tensor | `tensor_zeros([2, 3])` | Multi-dimensional array of doubles |

## Grammar

```
program     = statement* EOF

statement   = fn_decl | let_decl | expr_stmt
fn_decl     = "fn" IDENT "(" params? ")" block
let_decl    = "let" IDENT "=" expr
expr_stmt   = expr NEWLINE?

expr        = assignment
assignment  = (IDENT | index_expr) "=" assignment | pipe_expr
pipe_expr   = or_expr ("|>" or_expr)*
or_expr     = and_expr ("||" and_expr)*
and_expr    = equality ("&&" equality)*
equality    = comparison (("==" | "!=") comparison)*
comparison  = addition (("<" | ">" | "<=" | ">=") addition)*
addition    = multiply (("+" | "-") multiply)*
multiply    = power (("*" | "/" | "%" | "@@") power)*
power       = unary ("**" power)?                    // right-associative
unary       = ("-" | "!") unary | call
call        = primary ("(" args? ")" | "[" expr "]" | "." IDENT)*
primary     = INT | FLOAT | STRING | "true" | "false"
            | IDENT | "(" expr ")" | block | if_expr
            | while_expr | for_expr | lambda | array_lit

block       = "{" statement* expr? "}"
if_expr     = "if" expr block ("else" (if_expr | block))?
while_expr  = "while" expr block
for_expr    = "for" IDENT "in" expr block
lambda      = "|" params? "|" (block | expr)
array_lit   = "[" (expr ("," expr)*)? "]"

params      = IDENT ("," IDENT)*
args        = expr ("," expr)*
```

## Variables

Variables are declared with `let` and are block-scoped:

```
let x = 42
let name = "Adam"
x = x + 1          // reassignment
```

## Functions

Functions are declared with `fn`. The return value is the last expression in the body (no explicit `return` needed):

```
fn add(a, b) {
    a + b
}

fn factorial(n) {
    if n <= 1 { 1 }
    else { n * factorial(n - 1) }
}
```

## Closures and Lambdas

Lambda syntax uses `|params| body`:

```
let double = |x| x * 2
let add = |a, b| a + b

// Multi-statement lambda
let greet = |name| {
    let msg = "Hello, " + name
    println(msg)
    msg
}
```

Closures capture their enclosing environment by reference:

```
fn make_adder(n) {
    |x| n + x
}

let add5 = make_adder(5)
println(add5(10))   // 15
```

## Operators

### Arithmetic

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition / string concatenation | `1 + 2`, `"a" + "b"` |
| `-` | Subtraction | `5 - 3` |
| `*` | Multiplication | `4 * 7` |
| `/` | Division | `10 / 3` |
| `%` | Modulo | `10 % 3` |
| `**` | Exponentiation (right-associative) | `2 ** 10` |
| `@@` | Matrix multiply (tensors) | `a @@ b` |
| `-` (unary) | Negation | `-x` |

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

### Logical

| Operator | Description |
|----------|-------------|
| `&&` | Logical AND (short-circuit) |
| `\|\|` | Logical OR (short-circuit) |
| `!` | Logical NOT |

### Pipe

The pipe operator passes the left-hand value as the argument to the right-hand function:

```
// x |> f  is equivalent to  f(x)
// x |> f |> g  is equivalent to  g(f(x))

fn double(x) { x * 2 }
fn add_one(x) { x + 1 }

5 |> double |> add_one   // 11
```

Pipe is left-associative: `a |> b |> c` is `(a |> b) |> c`.

## Control Flow

### if/else

`if/else` is an expression that returns a value:

```
let max = if a > b { a } else { b }

if condition {
    println("yes")
}
```

### while

```
let i = 0
while i < 10 {
    println(i)
    i = i + 1
}
```

### for..in

Iterates over arrays:

```
for x in [1, 2, 3, 4, 5] {
    println(x)
}
```

## Arrays

```
let arr = [10, 20, 30]
println(arr[0])        // 10
println(len(arr))      // 3
arr[1] = 99            // mutation
push(arr, 40)          // append
```

## Strings

Double-quoted, immutable. Concatenated with `+`:

```
let name = "World"
let greeting = "Hello, " + name + "!"
println(greeting)   // Hello, World!
```

## Built-in Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `println` | `(value) -> Nil` | Print value to stdout with newline |
| `len` | `(array) -> Int` | Return array length |
| `clock` | `() -> Float` | Return current time in seconds |
| `push` | `(array, value) -> Nil` | Append value to array |

## Expression-Oriented Design

Blocks, `if/else`, and function bodies all evaluate to their last expression:

```
let x = {
    let a = 10
    let b = 20
    a + b       // block evaluates to 30
}

let sign = if n > 0 { "positive" }
           else if n < 0 { "negative" }
           else { "zero" }
```

## Tensor Types

Adam has built-in tensor support with shape-typed dimensions and compile-time shape checking.

### Tensor Creation

```
let zeros = tensor_zeros([2, 3])          // 2x3 tensor of zeros
let ones = tensor_ones([3, 2])            // 3x2 tensor of ones
let rand = tensor_randn([784, 10])        // random normal
let data = tensor_from_array([1, 2, 3, 4], [2, 2])   // from flat data + shape
```

### Shape-Typed Annotations

Tensor type annotations use `Tensor<DType, [dims]>` syntax with dimension variables:

```
fn predict(images: Tensor<Float, [N, 784]>) -> Tensor<Float, [N, 10]> {
    images @@ weights + bias
}
```

Dimension variables (`N`, `M`, `K`) are uppercase identifiers that unify like type variables but are restricted to integer kinds. The compiler checks shapes at compile time:
- `Tensor<F, [N, 784]> @@ Tensor<F, [784, 10]>` unifies to `Tensor<F, [N, 10]>` (valid)
- `Tensor<F, [N, 3]> @@ Tensor<F, [5, M]>` produces a shape mismatch error (3 != 5)

### Matrix Multiply (@@)

The `@@` operator performs matrix multiplication:

```
let a = tensor_from_array([1, 2, 3, 4, 5, 6], [2, 3])  // [2, 3]
let b = tensor_from_array([1, 0, 0, 1, 0, 0], [3, 2])  // [3, 2]
let c = a @@ b                                           // [2, 2]
```

Binding power is the same as `*` (level 8). The inner dimensions must match.

### Element-wise Operations

Standard arithmetic operators work element-wise on tensors of the same shape:

```
let x = tensor_ones([2, 2])
let y = tensor_from_array([10, 20, 30, 40], [2, 2])
x + y    // element-wise add
y - x    // element-wise subtract
x * y    // element-wise multiply (Hadamard product)
-x       // element-wise negate
```

### Tensor-Scalar Broadcasting

Arithmetic operators support mixed tensor-scalar operands:

```
let w = tensor_randn([784, 128]) * 0.047  // scale every element
let shifted = logits - 5.0                 // subtract scalar from tensor
```

### 2D Broadcasting

Tensors with compatible shapes broadcast automatically:

```
let a = tensor_ones([4, 10])        // [4, 10]
let b = tensor_ones([1, 10])        // [1, 10] — row broadcast
let c = tensor_ones([4, 1])         // [4, 1]  — column broadcast
a + b    // [4,10] + [1,10] -> repeats b across 4 rows
a / c    // [4,10] / [4,1]  -> repeats c across 10 columns
```

### Tensor Built-in Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor_zeros` | `([Int]) -> Tensor` | Create zero tensor from shape array |
| `tensor_ones` | `([Int]) -> Tensor` | Create ones tensor from shape array |
| `tensor_randn` | `([Int]) -> Tensor` | Create random normal tensor |
| `tensor_from_array` | `([Float], [Int]) -> Tensor` | Create tensor from flat data + shape |
| `tensor_shape` | `(Tensor) -> [Int]` | Get shape as array |
| `tensor_reshape` | `(Tensor, [Int]) -> Tensor` | Reshape (element count must match) |
| `tensor_sum` | `(Tensor) -> Float` | Sum all elements to scalar |
| `tensor_sum_axis` | `(Tensor, Int) -> Tensor` | Sum along axis (0=columns, 1=rows) |
| `tensor_transpose` | `(Tensor) -> Tensor` | Transpose 2D tensor |
| `tensor_exp` | `(Tensor) -> Tensor` | Element-wise exp |
| `tensor_log` | `(Tensor) -> Tensor` | Element-wise natural log |
| `tensor_relu` | `(Tensor) -> Tensor` | Element-wise max(0, x) |
| `tensor_relu_backward` | `(Tensor, Tensor) -> Tensor` | ReLU gradient (grad where z > 0) |
| `tensor_max` | `(Tensor) -> Float` | Maximum element |
| `tensor_slice` | `(Tensor, Int, Int) -> Tensor` | Extract rows [start, start+count) |
| `tensor_one_hot` | `(Tensor, Int) -> Tensor` | Convert label indices to one-hot |
| `tensor_load` | `(String) -> Tensor` | Load tensor from binary file |

## Automatic Differentiation

The `grad()` compiler intrinsic transforms a function into its reverse-mode derivative at compile time:

```
fn loss(x) {
    let h = x @@ w1
    let out = h @@ w2
    tensor_sum(out)
}

// grad(loss) generates a gradient function via AST transformation
let grad_loss = grad(loss)
let grads = grad_loss(input)   // gradient of loss w.r.t. input
```

This is a **source transformation**, not a runtime tape. The compiler walks the function's AST, builds a computation tape, and emits a new function with both forward and backward passes.

Supported differentiation rules:
- `a + b`: adjoints pass through
- `a - b`: adjoint of b is negated
- `a * b` (element-wise): `d_a = adj * b`, `d_b = adj * a`
- `a @@ b` (matmul): `d_a = adj @@ transpose(b)`, `d_b = transpose(a) @@ adj`
- `-a`: adjoint is negated
- `tensor_sum(a)`: adjoint broadcasts to `ones_like(a)`
- `tensor_transpose(a)`: adjoint is transposed

## Type Inference

Adam uses Hindley-Milner type inference. Types are never declared -- they are inferred from usage:

```
fn id(x) { x }         // inferred: forall a. a -> a
fn add(a, b) { a + b } // inferred: (Int, Int) -> Int

id(42)                  // Int
id("hello")             // String -- polymorphic
```

Let-bindings are generalized (polymorphic). Function arguments within a body are monomorphic.
