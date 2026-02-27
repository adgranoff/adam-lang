# Compiler Pipeline

The Adam compiler transforms source code into bytecode through four stages: lexing, parsing, type checking, and code generation.

## Stage 1: Lexer

**File**: `compiler/src/lexer.rs`

The lexer converts source text into a stream of tokens. Each token carries:
- A type tag (keyword, operator, literal, etc.)
- The lexeme (source text)
- A span (byte offset + length) for error reporting

### Token Types

- **Literals**: `Int(i64)`, `Float(f64)`, `Str(String)`, `True`, `False`
- **Identifiers**: `Ident(String)`
- **Keywords**: `Fn`, `Let`, `If`, `Else`, `While`, `For`, `In`, `Return`, `Break`, `Continue`, `Match`, `Type`, `Struct`, `Impl`
- **Operators**: `Plus`, `Minus`, `Star`, `Slash`, `Percent`, `StarStar`, `Eq`, `EqEq`, `BangEq`, `Lt`, `Gt`, `LtEq`, `GtEq`, `And`, `Or`, `Bang`, `PipeGt`, `Pipe`, `Arrow`, `FatArrow`
- **Delimiters**: `LParen`, `RParen`, `LBrace`, `RBrace`, `LBracket`, `RBracket`, `Comma`, `Dot`, `Colon`, `Semicolon`, `Newline`

The lexer skips comments (`//` to end of line) and whitespace (except newlines, which are significant as statement terminators).

## Stage 2: Parser

**File**: `compiler/src/parser.rs`

The parser produces an AST from the token stream using two complementary techniques:

### Pratt Parsing (Expressions)

Expressions use a Pratt parser (top-down operator precedence). Each operator has a binding power that determines precedence:

| Binding Power | Operators |
|--------------|-----------|
| 1 | `\|\|` |
| 2 | `&&` |
| 3 | `==`, `!=` |
| 4 | `<`, `>`, `<=`, `>=` |
| 5 | `+`, `-` |
| 6 | `*`, `/`, `%` |
| 7 | `**` (right-associative) |
| 5 | `\|>` (pipe, left-associative) |

The core algorithm:

```
fn expr(min_bp):
    left = prefix()              // parse atom or prefix op
    loop:
        op = peek()
        op_bp = binding_power(op)
        if op_bp < min_bp: break
        right_bp = if left_assoc: op_bp + 1 else: op_bp
        right = expr(right_bp)   // recurse
        left = BinaryExpr(op, left, right)
    return left
```

Left-associative operators pass `op_bp + 1` as the right-side minimum, so same-precedence operators don't nest. Right-associative operators (like `**`) pass `op_bp` unchanged.

### Recursive Descent (Statements)

Statements and declarations use recursive descent:

- `fn name(params) block` -> FnDecl
- `let name = expr` -> LetDecl
- `expr` -> ExprStmt

### AST Nodes

**Expressions** (`ast.rs`):
- `IntLit`, `FloatLit`, `StrLit`, `BoolLit`
- `Var(name)` -- variable reference
- `Binary(op, left, right)` -- arithmetic, comparison, logical
- `Unary(op, operand)` -- negation, not
- `Call(callee, args)` -- function call
- `Pipe(left, right)` -- pipe operator
- `Index(object, index)` -- array indexing
- `Field(object, name)` -- struct field access
- `Array(elements)` -- array literal
- `Lambda(params, body)` -- closure
- `If(cond, then, else)` -- conditional (expression)
- `While(cond, body)` -- while loop
- `For(var, iter, body)` -- for-in loop
- `Block(stmts, expr)` -- block with optional trailing expression
- `Assign(target, value)` -- assignment

**Statements**:
- `FnDecl(name, params, body)`
- `LetDecl(name, value)`
- `ExprStmt(expr)`
- `Return(value)`

## Stage 3: Type Checker

**File**: `compiler/src/types.rs`

The type checker implements Hindley-Milner type inference using Algorithm W.

### Type Representation

```
Type = Int | Float | Str | Bool | Nil
     | Array(element_type)
     | Fn(param_types, return_type)
     | Var(id)                        // unification variable
```

### Algorithm W

For each expression, `infer(expr, env)` returns a `Type` and updates the substitution:

1. **Literals**: Return their concrete type (Int, Str, etc.)
2. **Variables**: Look up in environment. If the binding is polymorphic (a `Scheme`), instantiate it with fresh type variables.
3. **Lambda**: Create fresh type variables for parameters, infer the body in an extended environment, return `Fn(param_types, body_type)`.
4. **Application**: Infer callee type, infer argument types, create a fresh variable for the result, unify callee with `Fn(arg_types, result_var)`.
5. **Let**: Infer the bound expression, generalize its type (quantify over free variables not in the environment), add the polymorphic scheme to the environment.

### Unification

Unification finds a substitution that makes two types equal:

```
unify(T1, T2):
    T1, T2 = apply current substitution
    if T1 == T2: done
    if T1 is Var(a): bind a -> T2 (with occurs check)
    if T2 is Var(a): bind a -> T1 (with occurs check)
    if T1 = Fn(p1, r1) and T2 = Fn(p2, r2):
        unify each p1[i] with p2[i]
        unify r1 with r2
    else: type error
```

The **occurs check** prevents creating infinite types: before binding `a -> T`, verify that `a` does not appear in `T`.

### Substitution

Type variable bindings use a union-find structure for efficient lookup. `find(var_id)` follows the chain of bindings to the representative type.

### Generalization and Instantiation

- **Generalize**: Quantify over type variables in a type that are NOT free in the current environment. This creates a `Scheme` (polymorphic type).
- **Instantiate**: Replace quantified variables with fresh type variables. Each use of a polymorphic binding gets its own variables.

This enables let-polymorphism:

```
let id = |x| x    // generalized to forall a. a -> a
id(42)             // instantiated: Int -> Int
id("hello")        // instantiated: String -> String
```

### Recursive Functions

Recursive functions are typed by first binding the function name to a fresh monomorphic type variable, inferring the body, then generalizing. The monomorphic binding is removed before generalization to prevent its type variables from appearing in the environment's free variables.

## Stage 4: Code Generation

**File**: `compiler/src/compiler.rs`

The code generator walks the AST and emits stack-based bytecode.

### Local Variables

Local variables are assigned stack slots at compile time. A scope stack tracks which names map to which slots. When a scope ends, its slots are freed.

### Upvalue Capture

When a closure references a variable from an enclosing function, the compiler resolves it as an upvalue:

1. Walk up the scope chain to find which function owns the variable.
2. If the variable is a local in the immediate parent, capture it directly (marks it as "captured" so the VM knows to close it).
3. If the variable is itself an upvalue of the parent, capture the parent's upvalue index.

Each closure's upvalue descriptors are emitted after the `CLOSURE` instruction.

### Control Flow

- **if/else**: Emit condition, `JUMP_IF_FALSE` to else branch (or end), then branch body, `JUMP` over else branch, else body.
- **while**: Mark loop start, emit condition, `JUMP_IF_FALSE` to end, body, `LOOP` back to start.
- **for-in**: Desugar to array iteration with an index counter.

### Constant Pool

String literals, large integers, and function prototypes are stored in a constant pool. The `CONST` instruction references entries by index.

### Bytecode Serialization

The compiler produces an in-memory `Chunk` (constant pool + code bytes), which is serialized to the `.adamb` binary format:

```
"ADAM" (4 bytes magic)
0x01   (1 byte version)
[constant pool length: u32][entries...]
[code length: u32][bytecode...]
```
