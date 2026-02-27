//! Abstract Syntax Tree — Typed nodes for the Adam language.
//!
//! Every AST node carries a [`Span`] for error reporting. The tree is
//! designed to be expression-oriented: `if`, `match`, and blocks are
//! all expressions that produce values.
//!
//! This module defines the grammar of Adam as Rust types. The parser
//! constructs this tree; the type checker annotates it; the compiler
//! walks it to emit bytecode.

use crate::token::Span;

/// A complete program is a sequence of top-level declarations/statements.
pub type Program = Vec<Stmt>;

// ── Statements ───────────────────────────────────────────────────────

/// Top-level and block-level statements.
#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// `let x = expr` or `let x: Type = expr`
    Let {
        name: String,
        type_ann: Option<TypeExpr>,
        value: Expr,
    },

    /// `fn name(params) -> RetType { body }`
    Function {
        name: String,
        params: Vec<Param>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },

    /// `type Name<T> { Variant1(T), Variant2 }`
    TypeDecl {
        name: String,
        type_params: Vec<String>,
        variants: Vec<Variant>,
    },

    /// `struct Name { field1: Type, field2: Type }`
    StructDecl {
        name: String,
        type_params: Vec<String>,
        fields: Vec<Field>,
    },

    /// `impl TypeName { fn method(...) { ... } ... }`
    ImplBlock {
        type_name: String,
        methods: Vec<Stmt>,
    },

    /// Expression used as a statement (the value is discarded).
    Expression(Expr),

    /// `return expr`
    Return(Option<Expr>),
}

/// A function/method parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_ann: Option<TypeExpr>,
    pub span: Span,
}

/// An algebraic data type variant: `Some(T)` or `None`.
#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<TypeExpr>,
    pub span: Span,
}

/// A struct field: `name: Type`.
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub type_ann: TypeExpr,
    pub span: Span,
}

// ── Expressions ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Integer literal: `42`
    IntLit(i64),

    /// Float literal: `3.14`
    FloatLit(f64),

    /// String literal: `"hello"`
    StringLit(String),

    /// Boolean literal: `true` / `false`
    BoolLit(bool),

    /// Variable reference: `x`
    Var(String),

    /// Binary operation: `a + b`, `a == b`, etc.
    Binary {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },

    /// Unary operation: `-x`, `!b`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// Function call: `f(a, b)`
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Field access: `point.x`
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },

    /// Index: `arr[i]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },

    /// Block (sequence of statements, last expression is the value):
    /// `{ stmt1; stmt2; expr }`
    Block {
        stmts: Vec<Stmt>,
        expr: Option<Box<Expr>>,
    },

    /// If expression: `if cond { then } else { else }`
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    /// Match expression: `match expr { pattern => body, ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// Closure / lambda: `|x, y| x + y`
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
    },

    /// Pipe: `expr |> func` (desugars to `func(expr)`)
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Array literal: `[1, 2, 3]`
    Array(Vec<Expr>),

    /// Struct literal: `Point { x: 1.0, y: 2.0 }`
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
    },

    /// Variant constructor: `Some(42)`
    VariantConstruct {
        name: String,
        args: Vec<Expr>,
    },

    /// While loop: `while cond { body }`
    While {
        condition: Box<Expr>,
        body: Box<Expr>,
    },

    /// For loop: `for x in iter { body }`
    For {
        var: String,
        iterator: Box<Expr>,
        body: Box<Expr>,
    },

    /// Assignment: `x = expr`
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },

    /// Break from a loop
    Break,

    /// Continue to next iteration
    Continue,
}

/// A match arm: `Pattern => Expr`
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

// ── Patterns ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Pattern {
    /// Wildcard: `_`
    Wildcard(Span),

    /// Variable binding: `x`
    Var(String, Span),

    /// Literal: `42`, `"hello"`, `true`
    Literal(Expr),

    /// Variant destructure: `Some(x)`, `None`
    Variant {
        name: String,
        fields: Vec<Pattern>,
        span: Span,
    },
}

// ── Operators ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

// ── Type expressions ─────────────────────────────────────────────────

/// Type annotations in the source: `Int`, `Option<T>`, `fn(Int) -> Bool`
#[derive(Debug, Clone)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeExprKind {
    /// Named type: `Int`, `String`, `Bool`, `Float`
    Named(String),

    /// Generic type: `Option<T>`, `Map<K, V>`
    Generic {
        name: String,
        args: Vec<TypeExpr>,
    },

    /// Function type: `fn(Int, Int) -> Bool`
    Function {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
    },

    /// Array type: `[Int]`
    Array(Box<TypeExpr>),
}
