//! Hindley-Milner Type Inference — Phase 3
//!
//! Implements Algorithm W for the Adam language. This is the core
//! type-theoretic component: given an untyped (or partially annotated)
//! program, it infers the most general type for every expression and
//! reports type errors with source spans.
//!
//! # Key Concepts
//!
//! **Type Variables**: Fresh unknowns (α, β, ...) generated during
//! inference. They are unified as constraints are discovered.
//!
//! **Unification**: Given two types τ₁ and τ₂, find a substitution σ
//! such that σ(τ₁) = σ(τ₂), or report a type error. Uses a persistent
//! union-find structure for efficiency.
//!
//! **Generalization**: After inferring the type of a `let` binding,
//! quantify free type variables to produce a polymorphic type scheme
//! (∀α. α → α). This enables let-polymorphism: `let id = |x| x`
//! can be used at different types.
//!
//! **Instantiation**: When referencing a polymorphic binding, replace
//! its quantified variables with fresh type variables so each use-site
//! can unify independently.
//!
//! # References
//!
//! - Damas & Milner, "Principal type-schemes for functional programs" (1982)
//! - Heeren, Hage & Swierstra, "Generalizing Hindley-Milner Type Inference Algorithms" (2002)

use std::collections::HashMap;
use std::fmt;

use crate::ast::*;
use crate::token::Span;

// ── Type representation ─────────────────────────────────────────────

/// An Adam type. Type variables (`Var`) are placeholders resolved by
/// unification during inference.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Machine integer (64-bit).
    Int,
    /// IEEE 754 double.
    Float,
    /// UTF-8 string.
    Str,
    /// Boolean.
    Bool,
    /// The unit/nil type (no meaningful value).
    Nil,
    /// Homogeneous array: `[T]`.
    Array(Box<Type>),
    /// Function type: `(T₁, T₂, ...) → R`.
    Fn {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    /// Named struct type.
    Struct {
        name: String,
        fields: Vec<(String, Type)>,
    },
    /// Named algebraic data type (sum type).
    Adt {
        name: String,
        variants: Vec<(String, Vec<Type>)>,
    },
    /// Shape-typed tensor: `Tensor<Float, [784, N, 10]>`.
    /// dtype is Float or Int, shape is a list of dimension expressions.
    Tensor {
        dtype: Box<Type>,
        shape: Vec<DimExprType>,
    },
    /// Type variable — a placeholder for an unknown type. The `u32`
    /// is a unique ID into the substitution table.
    Var(u32),
}

/// Dimension expression in the type system.
/// Restricted dependent types — integer constants or unification variables.
#[derive(Debug, Clone, PartialEq)]
pub enum DimExprType {
    /// Known dimension: `784`, `10`
    Lit(i64),
    /// Dimension variable (unification variable for dimensions).
    /// Uses the same ID space as type variables for simplicity.
    Var(u32),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Str => write!(f, "String"),
            Type::Bool => write!(f, "Bool"),
            Type::Nil => write!(f, "Nil"),
            Type::Array(t) => write!(f, "[{}]", t),
            Type::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Struct { name, .. } => write!(f, "{}", name),
            Type::Adt { name, .. } => write!(f, "{}", name),
            Type::Tensor { dtype, shape } => {
                write!(f, "Tensor<{}, [", dtype)?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", dim)?;
                }
                write!(f, "]>")
            }
            Type::Var(id) => write!(f, "?{}", id),
        }
    }
}

impl fmt::Display for DimExprType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExprType::Lit(n) => write!(f, "{}", n),
            DimExprType::Var(id) => write!(f, "?d{}", id),
        }
    }
}

// ── Type Scheme (polymorphic type) ──────────────────────────────────

/// A type scheme ∀ α₁ α₂ ... αₙ. τ — a type with universally
/// quantified variables. Monomorphic types have an empty `vars` set.
#[derive(Debug, Clone)]
struct Scheme {
    /// Quantified type variable IDs.
    vars: Vec<u32>,
    /// The underlying type (may reference vars in `vars`).
    ty: Type,
}

impl Scheme {
    /// Create a monomorphic scheme (no quantified variables).
    fn mono(ty: Type) -> Self {
        Scheme {
            vars: Vec::new(),
            ty,
        }
    }
}

// ── Type Error ──────────────────────────────────────────────────────

/// A type error with source location and human-readable message.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Type error at {}..{}: {}",
            self.span.start, self.span.end, self.message
        )
    }
}

// ── Substitution Table (Union-Find) ─────────────────────────────────

/// Manages type variable bindings using a persistent substitution table.
///
/// When a type variable is unified with a concrete type or another
/// variable, the binding is recorded here. `resolve` follows the chain
/// to find the most concrete type known for any variable.
struct Substitution {
    /// Map from type variable ID → its binding (if any).
    bindings: HashMap<u32, Type>,
    /// Map from dimension variable ID → its dimension binding.
    dim_bindings: HashMap<u32, DimExprType>,
    /// Counter for generating fresh type variable IDs.
    next_id: u32,
}

impl Substitution {
    fn new() -> Self {
        Substitution {
            bindings: HashMap::new(),
            dim_bindings: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a fresh, unbound type variable.
    fn fresh_var(&mut self) -> Type {
        let id = self.next_id;
        self.next_id += 1;
        Type::Var(id)
    }

    /// Create a fresh dimension variable.
    fn fresh_dim_var(&mut self) -> DimExprType {
        let id = self.next_id;
        self.next_id += 1;
        DimExprType::Var(id)
    }

    /// Walk the substitution chain to find the most resolved form of a type.
    /// If `ty` is `Var(id)` and `id` is bound to some other type, follow
    /// that binding recursively. Non-variable types are returned as-is.
    fn resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(bound) = self.bindings.get(id) {
                    self.resolve(bound)
                } else {
                    ty.clone()
                }
            }
            _ => ty.clone(),
        }
    }

    /// Fully resolve a type, including inside compound types (arrays,
    /// function parameters, etc.). This produces the final inferred type
    /// with no remaining resolvable variables.
    fn deep_resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(bound) = self.bindings.get(id) {
                    self.deep_resolve(bound)
                } else {
                    ty.clone()
                }
            }
            Type::Array(elem) => Type::Array(Box::new(self.deep_resolve(elem))),
            Type::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.deep_resolve(p)).collect(),
                ret: Box::new(self.deep_resolve(ret)),
            },
            Type::Struct { name, fields } => Type::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(n, t)| (n.clone(), self.deep_resolve(t)))
                    .collect(),
            },
            Type::Adt { name, variants } => Type::Adt {
                name: name.clone(),
                variants: variants
                    .iter()
                    .map(|(n, ts)| {
                        (
                            n.clone(),
                            ts.iter().map(|t| self.deep_resolve(t)).collect(),
                        )
                    })
                    .collect(),
            },
            Type::Tensor { dtype, shape } => Type::Tensor {
                dtype: Box::new(self.deep_resolve(dtype)),
                shape: shape.iter().map(|d| self.deep_resolve_dim(d)).collect(),
            },
            _ => ty.clone(),
        }
    }

    /// Resolve a dimension expression through the dim_bindings.
    fn deep_resolve_dim(&self, dim: &DimExprType) -> DimExprType {
        self.resolve_dim(dim)
    }

    /// Unify two types: make them equal by adding substitution bindings.
    ///
    /// Returns `Ok(())` if unification succeeds, or a descriptive error
    /// message if the types are incompatible.
    fn unify(&mut self, a: &Type, b: &Type) -> Result<(), String> {
        let a = self.resolve(a);
        let b = self.resolve(b);

        match (&a, &b) {
            // Identical concrete types — trivially unified.
            (Type::Int, Type::Int)
            | (Type::Float, Type::Float)
            | (Type::Str, Type::Str)
            | (Type::Bool, Type::Bool)
            | (Type::Nil, Type::Nil) => Ok(()),

            // A type variable unifies with anything (occurs check first).
            (Type::Var(id), other) | (other, Type::Var(id)) => {
                if let Type::Var(other_id) = other {
                    if id == other_id {
                        return Ok(()); // Same variable — nothing to do.
                    }
                }
                // Occurs check: prevent infinite types like α = [α].
                if self.occurs(*id, other) {
                    return Err(format!("Infinite type: ?{} occurs in {}", id, other));
                }
                self.bindings.insert(*id, other.clone());
                Ok(())
            }

            // Array types: unify element types.
            (Type::Array(a_elem), Type::Array(b_elem)) => self.unify(a_elem, b_elem),

            // Function types: unify parameter lists and return types.
            (
                Type::Fn {
                    params: a_params,
                    ret: a_ret,
                },
                Type::Fn {
                    params: b_params,
                    ret: b_ret,
                },
            ) => {
                if a_params.len() != b_params.len() {
                    return Err(format!(
                        "Function arity mismatch: expected {} parameters, got {}",
                        a_params.len(),
                        b_params.len()
                    ));
                }
                for (ap, bp) in a_params.iter().zip(b_params.iter()) {
                    self.unify(ap, bp)?;
                }
                self.unify(a_ret, b_ret)
            }

            // Named struct types: must be the same struct, unify field types.
            (
                Type::Struct {
                    name: a_name,
                    fields: a_fields,
                },
                Type::Struct {
                    name: b_name,
                    fields: b_fields,
                },
            ) => {
                if a_name != b_name {
                    return Err(format!(
                        "Type mismatch: struct {} vs struct {}",
                        a_name, b_name
                    ));
                }
                for (af, bf) in a_fields.iter().zip(b_fields.iter()) {
                    self.unify(&af.1, &bf.1)?;
                }
                Ok(())
            }

            // Named ADT types: must be the same ADT.
            (Type::Adt { name: a_name, .. }, Type::Adt { name: b_name, .. }) => {
                if a_name != b_name {
                    return Err(format!("Type mismatch: {} vs {}", a_name, b_name));
                }
                Ok(())
            }

            // Tensor types: unify dtypes and each dimension.
            (
                Type::Tensor {
                    dtype: a_dtype,
                    shape: a_shape,
                },
                Type::Tensor {
                    dtype: b_dtype,
                    shape: b_shape,
                },
            ) => {
                self.unify(a_dtype, b_dtype)?;
                if a_shape.len() != b_shape.len() {
                    return Err(format!(
                        "Tensor rank mismatch: expected {} dimensions, got {}",
                        a_shape.len(),
                        b_shape.len()
                    ));
                }
                for (ad, bd) in a_shape.iter().zip(b_shape.iter()) {
                    self.unify_dim(ad, bd)?;
                }
                Ok(())
            }

            // Everything else is a mismatch.
            _ => Err(format!("Type mismatch: expected {}, got {}", a, b)),
        }
    }

    /// Unify two dimension expressions.
    fn unify_dim(&mut self, a: &DimExprType, b: &DimExprType) -> Result<(), String> {
        match (a, b) {
            (DimExprType::Lit(x), DimExprType::Lit(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err(format!("Tensor shape mismatch: dimension {} vs {}", x, y))
                }
            }
            // Dim variables unify with anything — bind via the same substitution
            // table, using a special sentinel type to store the dim binding.
            (DimExprType::Var(id), other) | (other, DimExprType::Var(id)) => {
                if let DimExprType::Var(other_id) = other {
                    if id == other_id {
                        return Ok(());
                    }
                }
                // Store dim binding: we encode DimExprType::Lit(n) as Type::Int
                // in the substitution table won't work cleanly. Instead, we use
                // a separate dim_bindings map.
                self.bind_dim(*id, other.clone());
                Ok(())
            }
        }
    }

    /// Bind a dimension variable to a dimension expression.
    fn bind_dim(&mut self, id: u32, dim: DimExprType) {
        self.dim_bindings.insert(id, dim);
    }

    /// Resolve a dimension variable through dim_bindings.
    fn resolve_dim(&self, dim: &DimExprType) -> DimExprType {
        match dim {
            DimExprType::Var(id) => {
                if let Some(bound) = self.dim_bindings.get(id) {
                    self.resolve_dim(bound)
                } else {
                    dim.clone()
                }
            }
            DimExprType::Lit(_) => dim.clone(),
        }
    }

    /// Occurs check: does type variable `id` appear anywhere inside `ty`?
    /// If so, unifying them would create an infinite type.
    fn occurs(&self, id: u32, ty: &Type) -> bool {
        match ty {
            Type::Var(other_id) => {
                if *other_id == id {
                    return true;
                }
                if let Some(bound) = self.bindings.get(other_id) {
                    self.occurs(id, bound)
                } else {
                    false
                }
            }
            Type::Array(elem) => self.occurs(id, elem),
            Type::Fn { params, ret } => {
                params.iter().any(|p| self.occurs(id, p)) || self.occurs(id, ret)
            }
            Type::Struct { fields, .. } => fields.iter().any(|(_, t)| self.occurs(id, t)),
            Type::Adt { variants, .. } => variants
                .iter()
                .any(|(_, ts)| ts.iter().any(|t| self.occurs(id, t))),
            Type::Tensor { dtype, .. } => self.occurs(id, dtype),
            _ => false,
        }
    }

    /// Collect free (unbound) type variable IDs in a type.
    fn free_vars(&self, ty: &Type) -> Vec<u32> {
        let mut vars = Vec::new();
        self.collect_free_vars(ty, &mut vars);
        vars
    }

    fn collect_free_vars(&self, ty: &Type, out: &mut Vec<u32>) {
        match ty {
            Type::Var(id) => {
                if let Some(bound) = self.bindings.get(id) {
                    self.collect_free_vars(bound, out);
                } else if !out.contains(id) {
                    out.push(*id);
                }
            }
            Type::Array(elem) => self.collect_free_vars(elem, out),
            Type::Fn { params, ret } => {
                for p in params {
                    self.collect_free_vars(p, out);
                }
                self.collect_free_vars(ret, out);
            }
            Type::Struct { fields, .. } => {
                for (_, t) in fields {
                    self.collect_free_vars(t, out);
                }
            }
            Type::Adt { variants, .. } => {
                for (_, ts) in variants {
                    for t in ts {
                        self.collect_free_vars(t, out);
                    }
                }
            }
            Type::Tensor { dtype, shape } => {
                self.collect_free_vars(dtype, out);
                for dim in shape {
                    if let DimExprType::Var(id) = dim {
                        if !self.dim_bindings.contains_key(id) && !out.contains(id) {
                            out.push(*id);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ── Type Environment ────────────────────────────────────────────────

/// Scoped type environment mapping variable names to type schemes.
///
/// Supports nested scopes (blocks, function bodies) via a stack of
/// hash maps. Inner scopes shadow outer bindings.
struct TypeEnv {
    scopes: Vec<HashMap<String, Scheme>>,
}

impl TypeEnv {
    fn new() -> Self {
        TypeEnv {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Look up a name, searching from innermost to outermost scope.
    fn lookup(&self, name: &str) -> Option<&Scheme> {
        for scope in self.scopes.iter().rev() {
            if let Some(scheme) = scope.get(name) {
                return Some(scheme);
            }
        }
        None
    }

    /// Bind a name in the current (innermost) scope.
    fn bind(&mut self, name: String, scheme: Scheme) {
        self.scopes.last_mut().unwrap().insert(name, scheme);
    }

    /// Collect all free type variables across all scopes.
    fn free_vars(&self, subst: &Substitution) -> Vec<u32> {
        let mut vars = Vec::new();
        for scope in &self.scopes {
            for scheme in scope.values() {
                let fv = subst.free_vars(&scheme.ty);
                for v in fv {
                    if !scheme.vars.contains(&v) && !vars.contains(&v) {
                        vars.push(v);
                    }
                }
            }
        }
        vars
    }
}

// ── Struct & ADT Registries ─────────────────────────────────────────

/// Stores declared struct shapes for field type lookups.
struct StructDef {
    fields: Vec<(String, Type)>,
}

/// Stores declared ADT shapes for variant constructors and match.
struct AdtDef {
    name: String,
    variants: Vec<(String, Vec<Type>)>,
}

// ── The Inference Engine ────────────────────────────────────────────

/// The Hindley-Milner type inference engine.
///
/// Walks the AST, generating type constraints via unification, and
/// builds up a substitution that maps type variables to concrete types.
/// After inference, `deep_resolve` produces fully inferred types.
pub struct TypeChecker {
    subst: Substitution,
    env: TypeEnv,
    errors: Vec<TypeError>,
    /// Declared struct types for field lookups.
    structs: HashMap<String, StructDef>,
    /// Declared ADT types for variant constructors and match.
    adts: HashMap<String, AdtDef>,
    /// Mapping from dimension variable names (N, M, K) to their IDs.
    dim_var_names: HashMap<String, u32>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut tc = TypeChecker {
            subst: Substitution::new(),
            env: TypeEnv::new(),
            errors: Vec::new(),
            structs: HashMap::new(),
            adts: HashMap::new(),
            dim_var_names: HashMap::new(),
        };
        // Register built-in functions.
        tc.register_builtins();
        tc
    }

    fn register_builtins(&mut self) {
        // println: accepts any type, returns Nil.
        let a = self.subst.fresh_var();
        let a_id = match &a {
            Type::Var(id) => *id,
            _ => unreachable!(),
        };
        self.env.bind(
            "println".into(),
            Scheme {
                vars: vec![a_id],
                ty: Type::Fn {
                    params: vec![a],
                    ret: Box::new(Type::Nil),
                },
            },
        );

        // len: [α] → Int
        let b = self.subst.fresh_var();
        let b_id = match &b {
            Type::Var(id) => *id,
            _ => unreachable!(),
        };
        self.env.bind(
            "len".into(),
            Scheme {
                vars: vec![b_id],
                ty: Type::Fn {
                    params: vec![Type::Array(Box::new(b))],
                    ret: Box::new(Type::Int),
                },
            },
        );

        // push: ([α], α) → Nil
        let c = self.subst.fresh_var();
        let c_id = match &c {
            Type::Var(id) => *id,
            _ => unreachable!(),
        };
        self.env.bind(
            "push".into(),
            Scheme {
                vars: vec![c_id],
                ty: Type::Fn {
                    params: vec![Type::Array(Box::new(c.clone())), c],
                    ret: Box::new(Type::Nil),
                },
            },
        );

        // clock: () → Float
        self.env.bind(
            "clock".into(),
            Scheme::mono(Type::Fn {
                params: vec![],
                ret: Box::new(Type::Float),
            }),
        );

        // Tensor builtins — use polymorphic types with fresh vars.
        // tensor_zeros: [Int] → Tensor (shape determined at runtime)
        let tz = self.subst.fresh_var();
        let tz_id = match &tz { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_zeros".into(),
            Scheme {
                vars: vec![tz_id],
                ty: Type::Fn {
                    params: vec![Type::Array(Box::new(Type::Int))],
                    ret: Box::new(tz),
                },
            },
        );
        let to = self.subst.fresh_var();
        let to_id = match &to { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_ones".into(),
            Scheme {
                vars: vec![to_id],
                ty: Type::Fn {
                    params: vec![Type::Array(Box::new(Type::Int))],
                    ret: Box::new(to),
                },
            },
        );
        let tr = self.subst.fresh_var();
        let tr_id = match &tr { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_randn".into(),
            Scheme {
                vars: vec![tr_id],
                ty: Type::Fn {
                    params: vec![Type::Array(Box::new(Type::Int))],
                    ret: Box::new(tr),
                },
            },
        );
        // tensor_sum: Tensor → Float
        let ts = self.subst.fresh_var();
        let ts_id = match &ts { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_sum".into(),
            Scheme {
                vars: vec![ts_id],
                ty: Type::Fn {
                    params: vec![ts],
                    ret: Box::new(Type::Float),
                },
            },
        );
        // tensor_transpose: Tensor → Tensor
        let tt1 = self.subst.fresh_var();
        let tt2 = self.subst.fresh_var();
        let tt1_id = match &tt1 { Type::Var(id) => *id, _ => unreachable!() };
        let tt2_id = match &tt2 { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_transpose".into(),
            Scheme {
                vars: vec![tt1_id, tt2_id],
                ty: Type::Fn {
                    params: vec![tt1],
                    ret: Box::new(tt2),
                },
            },
        );
        // tensor_shape: Tensor → [Int]
        let tsh = self.subst.fresh_var();
        let tsh_id = match &tsh { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_shape".into(),
            Scheme {
                vars: vec![tsh_id],
                ty: Type::Fn {
                    params: vec![tsh],
                    ret: Box::new(Type::Array(Box::new(Type::Int))),
                },
            },
        );
        // tensor_reshape: (Tensor, [Int]) → Tensor
        let trsh1 = self.subst.fresh_var();
        let trsh2 = self.subst.fresh_var();
        let trsh1_id = match &trsh1 { Type::Var(id) => *id, _ => unreachable!() };
        let trsh2_id = match &trsh2 { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_reshape".into(),
            Scheme {
                vars: vec![trsh1_id, trsh2_id],
                ty: Type::Fn {
                    params: vec![trsh1, Type::Array(Box::new(Type::Int))],
                    ret: Box::new(trsh2),
                },
            },
        );
        // tensor_from_array: ([Float], [Int]) → Tensor
        let tfa = self.subst.fresh_var();
        let tfa_id = match &tfa { Type::Var(id) => *id, _ => unreachable!() };
        self.env.bind(
            "tensor_from_array".into(),
            Scheme {
                vars: vec![tfa_id],
                ty: Type::Fn {
                    params: vec![
                        Type::Array(Box::new(Type::Float)),
                        Type::Array(Box::new(Type::Int)),
                    ],
                    ret: Box::new(tfa),
                },
            },
        );
    }

    // ── Generalization & Instantiation ──────────────────────────────

    /// Generalize a type into a scheme by quantifying type variables
    /// that are free in `ty` but not free in the environment.
    ///
    /// This is the key operation for let-polymorphism: after inferring
    /// `let id = |x| x : ?3 → ?3`, we generalize to `∀?3. ?3 → ?3`.
    fn generalize(&self, ty: &Type) -> Scheme {
        let env_fv = self.env.free_vars(&self.subst);
        let ty_fv = self.subst.free_vars(ty);
        let vars: Vec<u32> = ty_fv.into_iter().filter(|v| !env_fv.contains(v)).collect();
        Scheme {
            vars,
            ty: ty.clone(),
        }
    }

    /// Instantiate a scheme by replacing each quantified variable with
    /// a fresh type variable. This ensures each use-site of a polymorphic
    /// binding gets independent inference.
    fn instantiate(&mut self, scheme: &Scheme) -> Type {
        let mapping: HashMap<u32, Type> = scheme
            .vars
            .iter()
            .map(|&v| (v, self.subst.fresh_var()))
            .collect();
        self.apply_mapping(&scheme.ty, &mapping)
    }

    /// Apply a variable-to-type mapping to a type (used during instantiation).
    fn apply_mapping(&self, ty: &Type, mapping: &HashMap<u32, Type>) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(replacement) = mapping.get(id) {
                    replacement.clone()
                } else if let Some(bound) = self.subst.bindings.get(id) {
                    self.apply_mapping(bound, mapping)
                } else {
                    ty.clone()
                }
            }
            Type::Array(elem) => Type::Array(Box::new(self.apply_mapping(elem, mapping))),
            Type::Fn { params, ret } => Type::Fn {
                params: params
                    .iter()
                    .map(|p| self.apply_mapping(p, mapping))
                    .collect(),
                ret: Box::new(self.apply_mapping(ret, mapping)),
            },
            Type::Struct { name, fields } => Type::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(n, t)| (n.clone(), self.apply_mapping(t, mapping)))
                    .collect(),
            },
            Type::Adt { name, variants } => Type::Adt {
                name: name.clone(),
                variants: variants
                    .iter()
                    .map(|(n, ts)| {
                        (
                            n.clone(),
                            ts.iter().map(|t| self.apply_mapping(t, mapping)).collect(),
                        )
                    })
                    .collect(),
            },
            Type::Tensor { dtype, shape } => Type::Tensor {
                dtype: Box::new(self.apply_mapping(dtype, mapping)),
                shape: shape.clone(), // Dim vars are separate from type vars
            },
            _ => ty.clone(),
        }
    }

    // ── Unification with error reporting ────────────────────────────

    fn unify(&mut self, a: &Type, b: &Type, span: Span) {
        if let Err(msg) = self.subst.unify(a, b) {
            self.errors.push(TypeError {
                message: msg,
                span,
            });
        }
    }

    // ── Convert source type annotations to Types ────────────────────

    fn resolve_type_expr(&mut self, texpr: &TypeExpr) -> Type {
        match &texpr.kind {
            TypeExprKind::Named(name) => match name.as_str() {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "String" => Type::Str,
                "Bool" => Type::Bool,
                "Nil" => Type::Nil,
                other => {
                    // Could be a struct or ADT name.
                    if self.structs.contains_key(other) {
                        Type::Struct {
                            name: other.to_string(),
                            fields: self.structs[other]
                                .fields
                                .iter()
                                .map(|(n, t)| (n.clone(), t.clone()))
                                .collect(),
                        }
                    } else if self.adts.contains_key(other) {
                        let adt = &self.adts[other];
                        Type::Adt {
                            name: adt.name.clone(),
                            variants: adt.variants.clone(),
                        }
                    } else {
                        // Unknown type — treat as a type variable name.
                        // This handles generic type parameters like T.
                        Type::Var(0) // Placeholder; proper generics would
                                     // use a mapping from names to var IDs.
                    }
                }
            },
            TypeExprKind::Array(elem) => {
                Type::Array(Box::new(self.resolve_type_expr(elem)))
            }
            TypeExprKind::Function { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.resolve_type_expr(p)).collect(),
                ret: Box::new(self.resolve_type_expr(ret)),
            },
            TypeExprKind::Generic { name, .. } => {
                // Simplified: just resolve the base name for now.
                // Full generic instantiation would create fresh vars.
                if self.adts.contains_key(name.as_str()) {
                    let adt = &self.adts[name];
                    Type::Adt {
                        name: adt.name.clone(),
                        variants: adt.variants.clone(),
                    }
                } else {
                    Type::Nil // Placeholder for unresolved generics.
                }
            }
            TypeExprKind::TensorType { dtype, dims } => {
                let dtype_type = self.resolve_type_expr(dtype);
                let shape: Vec<DimExprType> = dims
                    .iter()
                    .map(|d| self.resolve_dim_expr(d))
                    .collect();
                Type::Tensor {
                    dtype: Box::new(dtype_type),
                    shape,
                }
            }
        }
    }

    /// Convert a source-level DimExpr to a type-level DimExprType.
    fn resolve_dim_expr(&mut self, dim: &crate::ast::DimExpr) -> DimExprType {
        match dim {
            crate::ast::DimExpr::Lit(n) => DimExprType::Lit(*n),
            crate::ast::DimExpr::Var(name) => {
                // Look up or create a dimension variable for this name.
                // We use the dim_var_names map to reuse the same ID for
                // the same name within a scope.
                if let Some(&id) = self.dim_var_names.get(name) {
                    DimExprType::Var(id)
                } else {
                    let dim_var = self.subst.fresh_dim_var();
                    if let DimExprType::Var(id) = dim_var {
                        self.dim_var_names.insert(name.clone(), id);
                        dim_var
                    } else {
                        unreachable!()
                    }
                }
            }
        }
    }

    // ── Program-level inference ─────────────────────────────────────

    /// Type-check an entire program. Returns accumulated errors.
    pub fn check_program(&mut self, program: &Program) -> Vec<TypeError> {
        // First pass: register all struct and ADT declarations so they
        // can be referenced in any order.
        for stmt in program {
            match &stmt.kind {
                StmtKind::StructDecl { name, fields, .. } => {
                    let field_types: Vec<(String, Type)> = fields
                        .iter()
                        .map(|f| (f.name.clone(), self.resolve_type_expr(&f.type_ann)))
                        .collect();
                    self.structs.insert(
                        name.clone(),
                        StructDef {
                            fields: field_types,
                        },
                    );
                }
                StmtKind::TypeDecl {
                    name, variants, ..
                } => {
                    let variant_types: Vec<(String, Vec<Type>)> = variants
                        .iter()
                        .map(|v| {
                            let field_types: Vec<Type> =
                                v.fields.iter().map(|f| self.resolve_type_expr(f)).collect();
                            (v.name.clone(), field_types)
                        })
                        .collect();
                    self.adts.insert(
                        name.clone(),
                        AdtDef {
                            name: name.clone(),
                            variants: variant_types,
                        },
                    );
                }
                _ => {}
            }
        }

        // Second pass: infer types for all statements.
        for stmt in program {
            self.infer_stmt(stmt);
        }

        self.errors.clone()
    }

    // ── Statement inference ─────────────────────────────────────────

    fn infer_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let {
                name,
                type_ann,
                value,
            } => {
                let inferred = self.infer_expr(value);
                if let Some(ann) = type_ann {
                    let expected = self.resolve_type_expr(ann);
                    self.unify(&inferred, &expected, value.span);
                }
                // Generalize: this is where let-polymorphism happens.
                // The inferred type of the value may contain free type
                // variables. By generalizing, we allow the binding to
                // be used polymorphically.
                let scheme = self.generalize(&inferred);
                self.env.bind(name.clone(), scheme);
            }

            StmtKind::Function {
                name,
                params,
                return_type,
                body,
            } => {
                // Create fresh type variables for each parameter.
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        if let Some(ann) = &p.type_ann {
                            self.resolve_type_expr(ann)
                        } else {
                            self.subst.fresh_var()
                        }
                    })
                    .collect();

                let ret_var = if let Some(ret_ann) = return_type {
                    self.resolve_type_expr(ret_ann)
                } else {
                    self.subst.fresh_var()
                };

                // Bind the function name to its type (for recursion).
                let fn_type = Type::Fn {
                    params: param_types.clone(),
                    ret: Box::new(ret_var.clone()),
                };
                // Use a monomorphic scheme for the recursive binding
                // (recursive calls see the same type variables, not
                // fresh ones via instantiation).
                self.env.bind(name.clone(), Scheme::mono(fn_type.clone()));

                // Open a new scope for the function body.
                self.env.push_scope();
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    self.env.bind(param.name.clone(), Scheme::mono(ty.clone()));
                }

                let body_type = self.infer_expr(body);
                self.unify(&body_type, &ret_var, body.span);

                self.env.pop_scope();

                // Remove the monomorphic recursive binding before
                // generalizing. If it stays, its free type variables
                // appear in the environment, blocking quantification
                // and preventing let-polymorphism.
                self.env.scopes.last_mut().unwrap().remove(name);
                let scheme = self.generalize(&fn_type);
                self.env.bind(name.clone(), scheme);
            }

            StmtKind::Expression(expr) => {
                self.infer_expr(expr);
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.infer_expr(e);
                }
                // Return type checking is handled by the enclosing function's
                // body unification. A more complete implementation would track
                // a "current return type" and unify here.
            }

            StmtKind::TypeDecl { .. } | StmtKind::StructDecl { .. } => {
                // Already processed in the first pass.
            }

            StmtKind::ImplBlock { methods, .. } => {
                for method in methods {
                    self.infer_stmt(method);
                }
            }
        }
    }

    // ── Expression inference ────────────────────────────────────────

    /// Infer the type of an expression. This is the heart of Algorithm W.
    ///
    /// For each expression form, we:
    /// 1. Recursively infer types of sub-expressions
    /// 2. Generate fresh type variables where types are unknown
    /// 3. Unify to establish constraints
    /// 4. Return the expression's inferred type
    fn infer_expr(&mut self, expr: &Expr) -> Type {
        match &expr.kind {
            // ── Literals ────────────────────────────────────────────
            ExprKind::IntLit(_) => Type::Int,
            ExprKind::FloatLit(_) => Type::Float,
            ExprKind::StringLit(_) => Type::Str,
            ExprKind::BoolLit(_) => Type::Bool,

            // ── Variables ───────────────────────────────────────────
            ExprKind::Var(name) => {
                if let Some(scheme) = self.env.lookup(name) {
                    let scheme = scheme.clone();
                    self.instantiate(&scheme)
                } else {
                    self.errors.push(TypeError {
                        message: format!("Undefined variable: {}", name),
                        span: expr.span,
                    });
                    self.subst.fresh_var()
                }
            }

            // ── Binary operations ───────────────────────────────────
            ExprKind::Binary { left, op, right } => {
                let lt = self.infer_expr(left);
                let rt = self.infer_expr(right);

                match op {
                    // Arithmetic: both operands must be numeric, result is same type.
                    BinOp::Add => {
                        // Add is special: works on Int, Float, or String.
                        // Use a fresh var and unify both sides.
                        let result = self.subst.fresh_var();
                        self.unify(&lt, &rt, expr.span);
                        self.unify(&lt, &result, expr.span);
                        // Verify it's a valid type for addition.
                        let resolved = self.subst.resolve(&result);
                        match resolved {
                            Type::Int | Type::Float | Type::Str | Type::Var(_) => {}
                            _ => {
                                self.errors.push(TypeError {
                                    message: format!(
                                        "Operator + not supported for type {}",
                                        resolved
                                    ),
                                    span: expr.span,
                                });
                            }
                        }
                        result
                    }
                    // Matrix multiply: Tensor<F, [.., M, K]> @@ Tensor<F, [K, N]> → Tensor<F, [.., M, N]>
                    BinOp::MatMul => {
                        let lt_resolved = self.subst.deep_resolve(&lt);
                        let rt_resolved = self.subst.deep_resolve(&rt);
                        match (&lt_resolved, &rt_resolved) {
                            (
                                Type::Tensor {
                                    dtype: a_dtype,
                                    shape: a_shape,
                                },
                                Type::Tensor {
                                    dtype: b_dtype,
                                    shape: b_shape,
                                },
                            ) => {
                                // dtypes must match
                                self.unify(a_dtype, b_dtype, expr.span);
                                // a must be at least 2D, b must be exactly 2D
                                if a_shape.len() < 2 {
                                    self.errors.push(TypeError {
                                        message: format!(
                                            "Left operand of @@ must have at least 2 dimensions, got {}",
                                            a_shape.len()
                                        ),
                                        span: left.span,
                                    });
                                    return lt;
                                }
                                if b_shape.len() != 2 {
                                    self.errors.push(TypeError {
                                        message: format!(
                                            "Right operand of @@ must have exactly 2 dimensions, got {}",
                                            b_shape.len()
                                        ),
                                        span: right.span,
                                    });
                                    return lt;
                                }
                                // Inner dimensions must match: a[.., M, K] @@ b[K, N]
                                let a_inner = &a_shape[a_shape.len() - 1]; // K
                                let b_outer = &b_shape[0]; // K
                                if let Err(msg) = self.subst.unify_dim(a_inner, b_outer) {
                                    self.errors.push(TypeError {
                                        message: format!("Matrix multiply shape error: {}", msg),
                                        span: expr.span,
                                    });
                                }
                                // Result: [.., M, N]
                                let mut result_shape: Vec<DimExprType> =
                                    a_shape[..a_shape.len() - 1].to_vec();
                                result_shape.push(b_shape[1].clone());
                                Type::Tensor {
                                    dtype: a_dtype.clone(),
                                    shape: result_shape,
                                }
                            }
                            // If either isn't a resolved tensor yet, use fresh vars
                            _ => {
                                if !matches!(lt_resolved, Type::Var(_))
                                    && !matches!(lt_resolved, Type::Tensor { .. })
                                {
                                    self.errors.push(TypeError {
                                        message: format!(
                                            "Operator @@ requires tensor operands, got {}",
                                            lt_resolved
                                        ),
                                        span: expr.span,
                                    });
                                }
                                self.subst.fresh_var()
                            }
                        }
                    }

                    BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                        // Numeric only. Both sides must unify.
                        self.unify(&lt, &rt, expr.span);
                        let resolved = self.subst.resolve(&lt);
                        match resolved {
                            Type::Int | Type::Float | Type::Var(_) => {}
                            _ => {
                                self.errors.push(TypeError {
                                    message: format!(
                                        "Arithmetic operator not supported for type {}",
                                        resolved
                                    ),
                                    span: expr.span,
                                });
                            }
                        }
                        lt
                    }

                    // Comparison: both operands must be same type, result is Bool.
                    BinOp::Eq | BinOp::Neq => {
                        self.unify(&lt, &rt, expr.span);
                        Type::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        self.unify(&lt, &rt, expr.span);
                        Type::Bool
                    }

                    // Logical: both operands must be Bool, result is Bool.
                    BinOp::And | BinOp::Or => {
                        self.unify(&lt, &Type::Bool, left.span);
                        self.unify(&rt, &Type::Bool, right.span);
                        Type::Bool
                    }
                }
            }

            // ── Unary operations ────────────────────────────────────
            ExprKind::Unary { op, operand } => {
                let t = self.infer_expr(operand);
                match op {
                    UnaryOp::Neg => {
                        let resolved = self.subst.resolve(&t);
                        match resolved {
                            Type::Int | Type::Float | Type::Var(_) => {}
                            _ => {
                                self.errors.push(TypeError {
                                    message: format!(
                                        "Unary - not supported for type {}",
                                        resolved
                                    ),
                                    span: expr.span,
                                });
                            }
                        }
                        t
                    }
                    UnaryOp::Not => {
                        self.unify(&t, &Type::Bool, operand.span);
                        Type::Bool
                    }
                }
            }

            // ── Function calls ──────────────────────────────────────
            ExprKind::Call { callee, args } => {
                let callee_type = self.infer_expr(callee);
                let arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();
                let ret_var = self.subst.fresh_var();

                let expected_fn = Type::Fn {
                    params: arg_types,
                    ret: Box::new(ret_var.clone()),
                };
                self.unify(&callee_type, &expected_fn, expr.span);
                ret_var
            }

            // ── Pipe operator ───────────────────────────────────────
            ExprKind::Pipe { left, right } => {
                // `left |> right` desugars to `right(left)`
                let arg_type = self.infer_expr(left);
                let fn_type = self.infer_expr(right);
                let ret_var = self.subst.fresh_var();

                let expected_fn = Type::Fn {
                    params: vec![arg_type],
                    ret: Box::new(ret_var.clone()),
                };
                self.unify(&fn_type, &expected_fn, expr.span);
                ret_var
            }

            // ── Blocks ─────────────────────────────────────────────
            ExprKind::Block { stmts, expr: tail } => {
                self.env.push_scope();
                for s in stmts {
                    self.infer_stmt(s);
                }
                let ty = if let Some(e) = tail {
                    self.infer_expr(e)
                } else {
                    Type::Nil
                };
                self.env.pop_scope();
                ty
            }

            // ── If expressions ──────────────────────────────────────
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_type = self.infer_expr(condition);
                self.unify(&cond_type, &Type::Bool, condition.span);

                let then_type = self.infer_expr(then_branch);
                if let Some(else_b) = else_branch {
                    let else_type = self.infer_expr(else_b);
                    self.unify(&then_type, &else_type, expr.span);
                    then_type
                } else {
                    // No else branch — result is Nil.
                    self.unify(&then_type, &Type::Nil, then_branch.span);
                    Type::Nil
                }
            }

            // ── While loops ─────────────────────────────────────────
            ExprKind::While { condition, body } => {
                let cond_type = self.infer_expr(condition);
                self.unify(&cond_type, &Type::Bool, condition.span);
                self.infer_expr(body);
                Type::Nil
            }

            // ── For loops ───────────────────────────────────────────
            ExprKind::For {
                var,
                iterator,
                body,
            } => {
                let iter_type = self.infer_expr(iterator);
                let elem_var = self.subst.fresh_var();
                self.unify(
                    &iter_type,
                    &Type::Array(Box::new(elem_var.clone())),
                    iterator.span,
                );

                self.env.push_scope();
                self.env.bind(var.clone(), Scheme::mono(elem_var));
                self.infer_expr(body);
                self.env.pop_scope();
                Type::Nil
            }

            // ── Lambdas / closures ──────────────────────────────────
            ExprKind::Lambda { params, body } => {
                self.env.push_scope();
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        let ty = if let Some(ann) = &p.type_ann {
                            self.resolve_type_expr(ann)
                        } else {
                            self.subst.fresh_var()
                        };
                        self.env.bind(p.name.clone(), Scheme::mono(ty.clone()));
                        ty
                    })
                    .collect();

                let body_type = self.infer_expr(body);
                self.env.pop_scope();

                Type::Fn {
                    params: param_types,
                    ret: Box::new(body_type),
                }
            }

            // ── Arrays ─────────────────────────────────────────────
            ExprKind::Array(elements) => {
                let elem_var = self.subst.fresh_var();
                for elem in elements {
                    let et = self.infer_expr(elem);
                    self.unify(&et, &elem_var, elem.span);
                }
                Type::Array(Box::new(elem_var))
            }

            // ── Array indexing ──────────────────────────────────────
            ExprKind::Index { object, index } => {
                let obj_type = self.infer_expr(object);
                let idx_type = self.infer_expr(index);
                self.unify(&idx_type, &Type::Int, index.span);

                let elem_var = self.subst.fresh_var();
                self.unify(
                    &obj_type,
                    &Type::Array(Box::new(elem_var.clone())),
                    object.span,
                );
                elem_var
            }

            // ── Struct literals ─────────────────────────────────────
            ExprKind::StructLit { name, fields } => {
                if let Some(sdef) = self.structs.get(name) {
                    let expected_fields = sdef.fields.clone();
                    for (fname, fexpr) in fields {
                        let inferred = self.infer_expr(fexpr);
                        if let Some((_, expected_ty)) =
                            expected_fields.iter().find(|(n, _)| n == fname)
                        {
                            self.unify(&inferred, expected_ty, fexpr.span);
                        } else {
                            self.errors.push(TypeError {
                                message: format!(
                                    "Unknown field '{}' for struct {}",
                                    fname, name
                                ),
                                span: fexpr.span,
                            });
                        }
                    }
                    Type::Struct {
                        name: name.clone(),
                        fields: expected_fields,
                    }
                } else {
                    self.errors.push(TypeError {
                        message: format!("Unknown struct: {}", name),
                        span: expr.span,
                    });
                    self.subst.fresh_var()
                }
            }

            // ── Field access ────────────────────────────────────────
            ExprKind::FieldAccess { object, field } => {
                let obj_type = self.infer_expr(object);
                let resolved = self.subst.deep_resolve(&obj_type);
                match &resolved {
                    Type::Struct { fields, .. } => {
                        if let Some((_, ft)) = fields.iter().find(|(n, _)| n == field) {
                            ft.clone()
                        } else {
                            self.errors.push(TypeError {
                                message: format!(
                                    "No field '{}' on type {}",
                                    field, resolved
                                ),
                                span: expr.span,
                            });
                            self.subst.fresh_var()
                        }
                    }
                    Type::Var(_) => {
                        // Type not yet resolved — can't check field access.
                        // Return a fresh variable; may cause downstream errors.
                        self.subst.fresh_var()
                    }
                    _ => {
                        self.errors.push(TypeError {
                            message: format!(
                                "Cannot access field '{}' on non-struct type {}",
                                field, resolved
                            ),
                            span: expr.span,
                        });
                        self.subst.fresh_var()
                    }
                }
            }

            // ── Assignment ──────────────────────────────────────────
            ExprKind::Assign { target, value } => {
                let target_type = self.infer_expr(target);
                let value_type = self.infer_expr(value);
                self.unify(&target_type, &value_type, expr.span);
                value_type
            }

            // ── Match expressions ───────────────────────────────────
            ExprKind::Match { scrutinee, arms } => {
                let scrut_type = self.infer_expr(scrutinee);
                let result_var = self.subst.fresh_var();

                for arm in arms {
                    self.check_pattern(&arm.pattern, &scrut_type, arm.span);
                    self.env.push_scope();
                    self.bind_pattern(&arm.pattern, &scrut_type);
                    let body_type = self.infer_expr(&arm.body);
                    self.unify(&body_type, &result_var, arm.body.span);
                    self.env.pop_scope();
                }

                result_var
            }

            // ── Variant constructors ────────────────────────────────
            ExprKind::VariantConstruct { name, args } => {
                // Find which ADT this variant belongs to.
                let mut found: Option<(String, Vec<Type>)> = None;
                for (adt_name, adt_def) in &self.adts {
                    for (vname, vtypes) in &adt_def.variants {
                        if vname == name {
                            found = Some((adt_name.clone(), vtypes.clone()));
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }

                if let Some((adt_name, expected_types)) = found {
                    if args.len() != expected_types.len() {
                        self.errors.push(TypeError {
                            message: format!(
                                "Variant {} expects {} arguments, got {}",
                                name,
                                expected_types.len(),
                                args.len()
                            ),
                            span: expr.span,
                        });
                    } else {
                        for (arg, expected) in args.iter().zip(expected_types.iter()) {
                            let at = self.infer_expr(arg);
                            self.unify(&at, expected, arg.span);
                        }
                    }
                    let adt_def = &self.adts[&adt_name];
                    Type::Adt {
                        name: adt_name,
                        variants: adt_def.variants.clone(),
                    }
                } else {
                    self.errors.push(TypeError {
                        message: format!("Unknown variant: {}", name),
                        span: expr.span,
                    });
                    self.subst.fresh_var()
                }
            }

            // ── Break / Continue ────────────────────────────────────
            ExprKind::Break | ExprKind::Continue => Type::Nil,
        }
    }

    // ── Pattern type checking ───────────────────────────────────────

    fn check_pattern(&mut self, pattern: &Pattern, expected: &Type, span: Span) {
        match pattern {
            Pattern::Wildcard(_) => {} // Matches anything.
            Pattern::Var(_, _) => {}   // Binds to any type.
            Pattern::Literal(lit) => {
                let lit_type = self.infer_expr(lit);
                self.unify(&lit_type, expected, span);
            }
            Pattern::Variant { name, fields, .. } => {
                // Verify the variant exists and field count matches.
                let resolved = self.subst.deep_resolve(expected);
                if let Type::Adt { variants, .. } = &resolved {
                    if let Some((_, vtypes)) = variants.iter().find(|(n, _)| n == name) {
                        if fields.len() != vtypes.len() {
                            self.errors.push(TypeError {
                                message: format!(
                                    "Pattern {} expects {} fields, got {}",
                                    name,
                                    vtypes.len(),
                                    fields.len()
                                ),
                                span,
                            });
                        }
                    } else {
                        self.errors.push(TypeError {
                            message: format!("Unknown variant: {}", name),
                            span,
                        });
                    }
                }
                // If the scrutinee type isn't resolved to an ADT yet,
                // we can't check field counts — that's OK for now.
            }
        }
    }

    /// Bind pattern variables into the current scope.
    fn bind_pattern(&mut self, pattern: &Pattern, scrutinee_type: &Type) {
        match pattern {
            Pattern::Wildcard(_) | Pattern::Literal(_) => {}
            Pattern::Var(name, _) => {
                self.env
                    .bind(name.clone(), Scheme::mono(scrutinee_type.clone()));
            }
            Pattern::Variant { name, fields, .. } => {
                // Look up the variant's field types and bind each pattern variable.
                let resolved = self.subst.deep_resolve(scrutinee_type);
                if let Type::Adt { variants, .. } = &resolved {
                    if let Some((_, vtypes)) = variants.iter().find(|(n, _)| n == name) {
                        for (field_pat, vtype) in fields.iter().zip(vtypes.iter()) {
                            self.bind_pattern(field_pat, vtype);
                        }
                    }
                }
            }
        }
    }

    // ── Public query interface ──────────────────────────────────────

    /// Fully resolve a type to its most concrete form. Useful for
    /// displaying inferred types after analysis is complete.
    pub fn resolve_type(&self, ty: &Type) -> Type {
        self.subst.deep_resolve(ty)
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Type-check a parsed program, returning any type errors found.
///
/// The type checker infers types for all expressions using Algorithm W
/// (Hindley-Milner type inference). It does not modify the AST or
/// produce output — it only validates types and reports errors.
///
/// # Example
///
/// ```ignore
/// let errors = adam_compiler::types::check(&program);
/// for err in &errors {
///     eprintln!("{}", err);
/// }
/// ```
pub fn check(program: &Program) -> Vec<TypeError> {
    let mut checker = TypeChecker::new();
    checker.check_program(program)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(src: &str) -> Program {
        let mut lexer = Lexer::new(src);
        let tokens = lexer.scan_tokens();
        assert!(lexer.errors().is_empty(), "Lex errors: {:?}", lexer.errors());
        let mut parser = Parser::new(tokens);
        parser.parse().expect("Parse failed")
    }

    fn check_ok(src: &str) {
        let program = parse(src);
        let errors = check(&program);
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    fn check_err(src: &str) -> Vec<TypeError> {
        let program = parse(src);
        let errors = check(&program);
        assert!(!errors.is_empty(), "Expected type errors for: {}", src);
        errors
    }

    // ── Literals ────────────────────────────────────────────────────

    #[test]
    fn test_literal_int() {
        check_ok("let x = 42");
    }

    #[test]
    fn test_literal_string() {
        check_ok("let s = \"hello\"");
    }

    #[test]
    fn test_literal_bool() {
        check_ok("let b = true");
    }

    // ── Arithmetic ─────────────────────────────────────────────────

    #[test]
    fn test_arithmetic_same_type() {
        check_ok("let x = 1 + 2");
        check_ok("let x = 1.0 + 2.0");
    }

    #[test]
    fn test_arithmetic_type_mismatch() {
        let errors = check_err("let x = 1 + true");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Functions ──────────────────────────────────────────────────

    #[test]
    fn test_function_basic() {
        check_ok("fn double(x) { x * 2 }\nlet y = double(5)");
    }

    #[test]
    fn test_function_wrong_arg_count() {
        let errors = check_err("fn f(x) { x }\nf(1, 2)");
        assert!(errors[0].message.contains("arity") || errors[0].message.contains("mismatch"));
    }

    #[test]
    fn test_function_with_annotations() {
        check_ok("fn add(a: Int, b: Int) -> Int { a + b }");
    }

    #[test]
    fn test_function_annotation_mismatch() {
        let errors = check_err("fn f(x: Int) -> Bool { x + 1 }");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Let-polymorphism ───────────────────────────────────────────

    #[test]
    fn test_let_polymorphism() {
        // `id` should be usable at both Int and Bool.
        check_ok("fn id(x) { x }\nlet a = id(42)\nlet b = id(true)");
    }

    #[test]
    fn test_lambda_polymorphism() {
        check_ok("let id = |x| x\nlet a = id(42)\nlet b = id(true)");
    }

    // ── Pipe operator ──────────────────────────────────────────────

    #[test]
    fn test_pipe_basic() {
        check_ok("fn double(x) { x * 2 }\nlet r = 5 |> double");
    }

    #[test]
    fn test_pipe_chain() {
        check_ok(
            "fn double(x) { x * 2 }\nfn add_one(x) { x + 1 }\nlet r = 5 |> double |> add_one",
        );
    }

    // ── Arrays ─────────────────────────────────────────────────────

    #[test]
    fn test_array_homogeneous() {
        check_ok("let arr = [1, 2, 3]");
    }

    #[test]
    fn test_array_heterogeneous() {
        let errors = check_err("let arr = [1, true, 3]");
        assert!(errors[0].message.contains("mismatch"));
    }

    #[test]
    fn test_array_index() {
        check_ok("let arr = [1, 2, 3]\nlet x = arr[0]");
    }

    #[test]
    fn test_array_index_non_int() {
        let errors = check_err("let arr = [1, 2, 3]\nlet x = arr[true]");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── If expressions ─────────────────────────────────────────────

    #[test]
    fn test_if_condition_bool() {
        check_ok("let x = if true { 1 } else { 2 }");
    }

    #[test]
    fn test_if_condition_non_bool() {
        let errors = check_err("let x = if 42 { 1 } else { 2 }");
        assert!(errors[0].message.contains("mismatch"));
    }

    #[test]
    fn test_if_branches_same_type() {
        check_ok("let x = if true { 1 } else { 2 }");
    }

    #[test]
    fn test_if_branches_different_type() {
        let errors = check_err("let x = if true { 1 } else { \"no\" }");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Closures ───────────────────────────────────────────────────

    #[test]
    fn test_closure_captures() {
        check_ok("fn make_adder(n) { |x| n + x }\nlet add5 = make_adder(5)\nadd5(10)");
    }

    // ── While / For ────────────────────────────────────────────────

    #[test]
    fn test_while_condition_bool() {
        check_ok("while false { 1 }");
    }

    #[test]
    fn test_while_condition_non_bool() {
        let errors = check_err("while 42 { 1 }");
        assert!(errors[0].message.contains("mismatch"));
    }

    #[test]
    fn test_for_loop() {
        check_ok("for x in [1, 2, 3] { x + 1 }");
    }

    #[test]
    fn test_for_non_array() {
        let errors = check_err("for x in 42 { x }");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Structs ────────────────────────────────────────────────────

    #[test]
    fn test_struct_literal() {
        check_ok("struct Point { x: Int, y: Int }\nlet p = Point { x: 1, y: 2 }");
    }

    #[test]
    fn test_struct_field_access() {
        check_ok("struct Point { x: Int, y: Int }\nlet p = Point { x: 1, y: 2 }\nlet a = p.x");
    }

    #[test]
    fn test_struct_wrong_field_type() {
        let errors =
            check_err("struct Point { x: Int, y: Int }\nlet p = Point { x: true, y: 2 }");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Match expressions ──────────────────────────────────────────

    #[test]
    fn test_match_literal() {
        check_ok("let x = match 1 { 1 => \"one\", _ => \"other\" }");
    }

    #[test]
    fn test_match_arms_same_type() {
        let errors = check_err("let x = match 1 { 1 => 42, _ => \"other\" }");
        assert!(errors[0].message.contains("mismatch"));
    }

    // ── Recursion ──────────────────────────────────────────────────

    #[test]
    fn test_recursion() {
        check_ok(
            "fn fib(n) { if n <= 1 { n } else { fib(n - 1) + fib(n - 2) } }\nfib(10)",
        );
    }

    // ── String operations ──────────────────────────────────────────

    #[test]
    fn test_string_concat() {
        check_ok("let s = \"hello\" + \" \" + \"world\"");
    }

    // ── Builtin functions ──────────────────────────────────────────

    #[test]
    fn test_println() {
        check_ok("println(42)");
        check_ok("println(\"hello\")");
    }

    #[test]
    fn test_len() {
        check_ok("let n = len([1, 2, 3])");
    }

    // ── Full programs ──────────────────────────────────────────────

    #[test]
    fn test_calculator_program() {
        check_ok(
            r#"
fn double(x) { x * 2 }
fn add_one(x) { x + 1 }

let result = 5 |> double |> add_one
println(result)

let arr = [10, 20, 30, 40, 50]
println(len(arr))
println(arr[2])

fn make_adder(n) {
    |x| n + x
}
let add5 = make_adder(5)
println(add5(10))

let greeting = "Hello" + " " + "World!"
println(greeting)

let x = true && false
let y = true || false
println(x)
println(y)
"#,
        );
    }
}
