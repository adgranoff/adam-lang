# Adam Language Reference

## Types

| Type | Examples | Description |
|------|----------|-------------|
| Int | `42`, `-7`, `0` | 32-bit signed integer |
| Float | `3.14`, `0.5` | 64-bit IEEE 754 double |
| String | `"hello"`, `""` | Immutable string (double-quoted) |
| Bool | `true`, `false` | Boolean |
| Nil | (implicit) | Absence of value |
| Array | `[1, 2, 3]` | Ordered, mutable, heterogeneous |
| Function | `fn f(x) { x }` | First-class, supports closures |

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
multiply    = power (("*" | "/" | "%") power)*
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

## Type Inference

Adam uses Hindley-Milner type inference. Types are never declared -- they are inferred from usage:

```
fn id(x) { x }         // inferred: forall a. a -> a
fn add(a, b) { a + b } // inferred: (Int, Int) -> Int

id(42)                  // Int
id("hello")             // String -- polymorphic
```

Let-bindings are generalized (polymorphic). Function arguments within a body are monomorphic.
