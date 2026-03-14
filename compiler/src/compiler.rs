//! Compiler — AST to bytecode compilation (Phase 4).
//!
//! Walks the AST emitted by the parser and produces bytecode instructions
//! for the C VM. Uses a stack of [`FnCompiler`] states to handle nested
//! function declarations and closures.
//!
//! Variable resolution follows three tiers:
//!   1. **Locals** — stack-allocated in the current function frame
//!   2. **Upvalues** — captured from enclosing functions (closure mechanism)
//!   3. **Globals** — stored by name in the VM's global hash table
//!
//! Control flow uses forward jumps ([`Op::Jump`], [`Op::JumpIfFalse`])
//! patched after the target is known, and backward jumps ([`Op::Loop`])
//! for loops. Jump offsets are 2-byte big-endian (matching the C VM).

use crate::ast::*;
use crate::bytecode::{CompiledProgram, Constant, Op};

// ── Local variable tracking ─────────────────────────────────────────

/// A local variable in the current function scope.
struct Local {
    name: String,
    /// Scope depth. -1 means declared but not yet defined (prevents
    /// `let x = x` from referencing itself during initialization).
    depth: i32,
    /// Set to true when a nested closure captures this local.
    is_captured: bool,
}

/// An upvalue reference captured by a closure.
#[derive(Clone)]
struct UpvalueRef {
    /// Index into the enclosing function's locals (if `is_local`) or
    /// upvalues (if `!is_local`).
    index: u8,
    /// True if this upvalue captures a local from the immediately
    /// enclosing function. False if it captures from further up.
    is_local: bool,
}

// ── Per-function compilation state ──────────────────────────────────

/// Compilation state for a single function (or the top-level script).
struct FnCompiler {
    name: Option<String>,
    arity: u8,
    locals: Vec<Local>,
    upvalues: Vec<UpvalueRef>,
    scope_depth: i32,
    code: Vec<u8>,
    constants: Vec<Constant>,
    /// Stack of loop-start code offsets for `OP_LOOP` (innermost last).
    loop_starts: Vec<usize>,
    /// For each active loop, offsets of break-jumps to patch at loop end.
    break_jumps: Vec<Vec<usize>>,
}

impl FnCompiler {
    fn new(name: Option<String>, arity: u8) -> Self {
        // Slot 0 is reserved for the function/closure object itself.
        let locals = vec![Local {
            name: String::new(),
            depth: 0,
            is_captured: false,
        }];
        Self {
            name,
            arity,
            locals,
            upvalues: Vec::new(),
            scope_depth: 0,
            code: Vec::new(),
            constants: Vec::new(),
            loop_starts: Vec::new(),
            break_jumps: Vec::new(),
        }
    }
}

// ── Main compiler ───────────────────────────────────────────────────

/// The bytecode compiler. Walks an AST and emits bytecode for the C VM.
struct Compiler {
    /// Stack of function compilation states. Entry 0 is always the
    /// top-level "script" function.
    functions: Vec<FnCompiler>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            functions: vec![FnCompiler::new(None, 0)],
        }
    }

    fn current(&self) -> &FnCompiler {
        self.functions.last().expect("compiler function stack empty")
    }

    fn current_mut(&mut self) -> &mut FnCompiler {
        self.functions.last_mut().expect("compiler function stack empty")
    }

    // ── Bytecode emission ───────────────────────────────────────────

    fn emit_byte(&mut self, byte: u8) {
        self.current_mut().code.push(byte);
    }

    fn emit_op(&mut self, op: Op) {
        self.emit_byte(op as u8);
    }

    fn add_constant(&mut self, constant: Constant) -> u8 {
        let f = self.current_mut();
        // Deduplicate Int, Float, and String constants
        for (i, existing) in f.constants.iter().enumerate() {
            let is_match = match (&constant, existing) {
                (Constant::Int(a), Constant::Int(b)) => a == b,
                (Constant::Float(a), Constant::Float(b)) => a.to_bits() == b.to_bits(),
                (Constant::String(a), Constant::String(b)) => a == b,
                _ => false,
            };
            if is_match {
                return i as u8;
            }
        }
        let idx = f.constants.len();
        assert!(idx <= u8::MAX as usize, "Too many constants in one function (max 256)");
        f.constants.push(constant);
        idx as u8
    }

    fn emit_constant(&mut self, constant: Constant) {
        let idx = self.add_constant(constant);
        self.emit_op(Op::Const);
        self.emit_byte(idx);
    }

    /// Emit a jump instruction with a placeholder 2-byte offset.
    /// Returns the code index of the first offset byte for later patching.
    fn emit_jump(&mut self, op: Op) -> usize {
        self.emit_op(op);
        self.emit_byte(0xff);
        self.emit_byte(0xff);
        self.current().code.len() - 2
    }

    /// Patch a previously-emitted forward jump to land at the current offset.
    fn patch_jump(&mut self, offset: usize) {
        let f = self.current_mut();
        let jump = f.code.len() - offset - 2;
        assert!(jump <= u16::MAX as usize, "Jump offset too large");
        f.code[offset] = ((jump >> 8) & 0xff) as u8;
        f.code[offset + 1] = (jump & 0xff) as u8;
    }

    /// Emit a backward loop jump to `loop_start`.
    fn emit_loop(&mut self, loop_start: usize) {
        self.emit_op(Op::Loop);
        let offset = self.current().code.len() - loop_start + 2; // +2 for the 2 offset bytes
        assert!(offset <= u16::MAX as usize, "Loop body too large");
        self.emit_byte(((offset >> 8) & 0xff) as u8);
        self.emit_byte((offset & 0xff) as u8);
    }

    fn code_len(&self) -> usize {
        self.current().code.len()
    }

    // ── Scope management ────────────────────────────────────────────

    fn begin_scope(&mut self) {
        self.current_mut().scope_depth += 1;
    }

    fn end_scope(&mut self) {
        let f = self.current_mut();
        f.scope_depth -= 1;

        // Pop locals that went out of scope, closing upvalues as needed.
        while let Some(local) = f.locals.last() {
            if local.depth <= f.scope_depth {
                break;
            }
            if local.is_captured {
                f.code.push(Op::CloseUpvalue as u8);
            } else {
                f.code.push(Op::Pop as u8);
            }
            f.locals.pop();
        }
    }

    fn declare_local(&mut self, name: &str) {
        let f = self.current_mut();
        if f.scope_depth == 0 {
            return; // Globals aren't tracked as locals.
        }
        f.locals.push(Local {
            name: name.to_string(),
            depth: -1, // Not yet initialized.
            is_captured: false,
        });
    }

    fn define_local(&mut self) {
        let f = self.current_mut();
        if f.scope_depth == 0 {
            return;
        }
        let len = f.locals.len();
        f.locals[len - 1].depth = f.scope_depth;
    }

    // ── Variable resolution ─────────────────────────────────────────

    fn resolve_local(&self, fn_idx: usize, name: &str) -> Option<u8> {
        let f = &self.functions[fn_idx];
        for (i, local) in f.locals.iter().enumerate().rev() {
            if local.name == name && local.depth != -1 {
                return Some(i as u8);
            }
        }
        None
    }

    fn resolve_upvalue(&mut self, fn_idx: usize, name: &str) -> Option<u8> {
        if fn_idx == 0 {
            return None; // Top-level script has no enclosing scope.
        }
        // Check if it's a local in the immediately enclosing function.
        if let Some(local_idx) = self.resolve_local(fn_idx - 1, name) {
            self.functions[fn_idx - 1].locals[local_idx as usize].is_captured = true;
            return Some(self.add_upvalue(fn_idx, local_idx, true));
        }
        // Check if it's already an upvalue in the enclosing function.
        if let Some(upvalue_idx) = self.resolve_upvalue(fn_idx - 1, name) {
            return Some(self.add_upvalue(fn_idx, upvalue_idx, false));
        }
        None
    }

    fn add_upvalue(&mut self, fn_idx: usize, index: u8, is_local: bool) -> u8 {
        let f = &mut self.functions[fn_idx];
        // Check for duplicate.
        for (i, uv) in f.upvalues.iter().enumerate() {
            if uv.index == index && uv.is_local == is_local {
                return i as u8;
            }
        }
        let count = f.upvalues.len();
        assert!(count <= u8::MAX as usize, "Too many upvalues (max 256)");
        f.upvalues.push(UpvalueRef { index, is_local });
        count as u8
    }

    /// Emit instructions to load a variable by name.
    fn emit_load(&mut self, name: &str) -> Result<(), String> {
        let fn_idx = self.functions.len() - 1;
        if let Some(slot) = self.resolve_local(fn_idx, name) {
            self.emit_op(Op::LoadLocal);
            self.emit_byte(slot);
        } else if let Some(slot) = self.resolve_upvalue(fn_idx, name) {
            self.emit_op(Op::LoadUpvalue);
            self.emit_byte(slot);
        } else {
            let idx = self.add_constant(Constant::String(name.to_string()));
            self.emit_op(Op::LoadGlobal);
            self.emit_byte(idx);
        }
        Ok(())
    }

    /// Emit instructions to store TOS into a variable by name.
    /// The value remains on the stack (peek semantics).
    fn emit_store(&mut self, name: &str) -> Result<(), String> {
        let fn_idx = self.functions.len() - 1;
        if let Some(slot) = self.resolve_local(fn_idx, name) {
            self.emit_op(Op::StoreLocal);
            self.emit_byte(slot);
        } else if let Some(slot) = self.resolve_upvalue(fn_idx, name) {
            self.emit_op(Op::StoreUpvalue);
            self.emit_byte(slot);
        } else {
            let idx = self.add_constant(Constant::String(name.to_string()));
            self.emit_op(Op::StoreGlobal);
            self.emit_byte(idx);
        }
        Ok(())
    }

    // ── Statement compilation ───────────────────────────────────────

    fn compile_program(&mut self, program: &Program) -> Result<(), String> {
        for stmt in program {
            self.compile_stmt(stmt)?;
        }
        self.emit_op(Op::Nil);
        self.emit_op(Op::Return);
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                self.declare_local(name);
                self.compile_expr(value)?;
                if self.current().scope_depth > 0 {
                    // Local: the value on the stack IS the local's slot.
                    self.define_local();
                } else {
                    // Global: store by name, then pop (statements don't leave values).
                    let idx = self.add_constant(Constant::String(name.clone()));
                    self.emit_op(Op::StoreGlobal);
                    self.emit_byte(idx);
                    self.emit_op(Op::Pop);
                }
            }

            StmtKind::Function {
                name, params, body, ..
            } => {
                self.declare_local(name);
                self.compile_function_body(Some(name.clone()), params, body)?;
                if self.current().scope_depth > 0 {
                    self.define_local();
                } else {
                    let idx = self.add_constant(Constant::String(name.clone()));
                    self.emit_op(Op::StoreGlobal);
                    self.emit_byte(idx);
                    self.emit_op(Op::Pop);
                }
            }

            StmtKind::Expression(expr) => {
                self.compile_expr(expr)?;
                self.emit_op(Op::Pop);
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr_in_tail(e)?;
                } else {
                    self.emit_op(Op::Nil);
                }
                self.emit_op(Op::Return);
            }

            StmtKind::TypeDecl { .. } | StmtKind::StructDecl { .. } => {
                // Type/struct declarations define shapes used by the type
                // checker. No runtime code is emitted.
            }

            StmtKind::ImplBlock { methods, .. } => {
                for method in methods {
                    self.compile_stmt(method)?;
                }
            }
        }
        Ok(())
    }

    // ── Expression compilation ──────────────────────────────────────

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), String> {
        match &expr.kind {
            ExprKind::IntLit(n) => {
                self.emit_constant(Constant::Int(*n as i32));
            }

            ExprKind::FloatLit(n) => {
                self.emit_constant(Constant::Float(*n));
            }

            ExprKind::StringLit(s) => {
                self.emit_constant(Constant::String(s.clone()));
            }

            ExprKind::BoolLit(true) => self.emit_op(Op::True),
            ExprKind::BoolLit(false) => self.emit_op(Op::False),

            ExprKind::Var(name) => {
                self.emit_load(name)?;
            }

            ExprKind::Binary { left, op, right } => {
                self.compile_binary(left, *op, right)?;
            }

            ExprKind::Unary { op, operand } => {
                self.compile_expr(operand)?;
                match op {
                    UnaryOp::Neg => self.emit_op(Op::Neg),
                    UnaryOp::Not => self.emit_op(Op::Not),
                }
            }

            ExprKind::Call { callee, args } => {
                self.compile_expr(callee)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                assert!(args.len() <= u8::MAX as usize, "Too many arguments (max 255)");
                self.emit_op(Op::Call);
                self.emit_byte(args.len() as u8);
            }

            ExprKind::Pipe { left, right } => {
                // `left |> right` desugars to `right(left)`
                self.compile_expr(right)?;
                self.compile_expr(left)?;
                self.emit_op(Op::Call);
                self.emit_byte(1);
            }

            ExprKind::Block { stmts, expr } => {
                self.begin_scope();
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                if let Some(e) = expr {
                    self.compile_expr(e)?;
                } else {
                    self.emit_op(Op::Nil);
                }
                self.end_scope();
            }

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.compile_if(condition, then_branch, else_branch.as_deref())?;
            }

            ExprKind::While { condition, body } => {
                self.compile_while(condition, body)?;
            }

            ExprKind::For {
                var,
                iterator,
                body,
            } => {
                self.compile_for(var, iterator, body)?;
            }

            ExprKind::Lambda { params, body } => {
                self.compile_function_body(None, params, body)?;
            }

            ExprKind::Array(elements) => {
                for elem in elements {
                    self.compile_expr(elem)?;
                }
                assert!(
                    elements.len() <= u8::MAX as usize,
                    "Array literal too large (max 255)"
                );
                self.emit_op(Op::ArrayNew);
                self.emit_byte(elements.len() as u8);
            }

            ExprKind::Index { object, index } => {
                self.compile_expr(object)?;
                self.compile_expr(index)?;
                self.emit_op(Op::ArrayGet);
            }

            ExprKind::StructLit { name, fields } => {
                // Push field values in source order.
                for (_, value) in fields {
                    self.compile_expr(value)?;
                }
                let name_idx = self.add_constant(Constant::String(name.clone()));
                let count = fields.len() as u8;
                self.emit_op(Op::StructNew);
                self.emit_byte(name_idx);
                self.emit_byte(count);
                // Field name constant indices follow in bytecode.
                for (field_name, _) in fields {
                    let fidx = self.add_constant(Constant::String(field_name.clone()));
                    self.emit_byte(fidx);
                }
            }

            ExprKind::FieldAccess { object, field } => {
                self.compile_expr(object)?;
                let fidx = self.add_constant(Constant::String(field.clone()));
                self.emit_op(Op::StructGet);
                self.emit_byte(fidx);
            }

            ExprKind::Assign { target, value } => {
                self.compile_assign(target, value)?;
            }

            ExprKind::Match { scrutinee, arms } => {
                self.compile_match(scrutinee, arms)?;
            }

            ExprKind::VariantConstruct { name, args } => {
                // Build the payload. For 0 args → nil payload.
                // For 1 arg → the arg is the payload.
                // For multiple args → wrap in an array.
                match args.len() {
                    0 => self.emit_op(Op::Nil),
                    1 => self.compile_expr(&args[0])?,
                    _ => {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit_op(Op::ArrayNew);
                        self.emit_byte(args.len() as u8);
                    }
                }
                // OP_VARIANT_NEW reads a tag string constant, pops the payload,
                // and creates an ObjVariant(tag, payload).
                let tag_idx = self.add_constant(Constant::String(name.clone()));
                self.emit_op(Op::VariantNew);
                self.emit_byte(tag_idx);
            }

            ExprKind::Break => {
                assert!(
                    !self.current().break_jumps.is_empty(),
                    "'break' outside of loop"
                );
                let jump = self.emit_jump(Op::Jump);
                self.current_mut()
                    .break_jumps
                    .last_mut()
                    .unwrap()
                    .push(jump);
            }

            ExprKind::Continue => {
                assert!(
                    !self.current().loop_starts.is_empty(),
                    "'continue' outside of loop"
                );
                let loop_start = *self.current().loop_starts.last().unwrap();
                self.emit_loop(loop_start);
            }
        }
        Ok(())
    }

    // ── Binary operation compilation ────────────────────────────────

    fn compile_binary(
        &mut self,
        left: &Expr,
        op: BinOp,
        right: &Expr,
    ) -> Result<(), String> {
        match op {
            // Short-circuit: if left is falsey, skip right
            BinOp::And => {
                self.compile_expr(left)?;
                let end_jump = self.emit_jump(Op::JumpIfFalse);
                self.emit_op(Op::Pop);
                self.compile_expr(right)?;
                self.patch_jump(end_jump);
            }
            // Short-circuit: if left is truthy, skip right
            BinOp::Or => {
                self.compile_expr(left)?;
                let else_jump = self.emit_jump(Op::JumpIfFalse);
                let end_jump = self.emit_jump(Op::Jump);
                self.patch_jump(else_jump);
                self.emit_op(Op::Pop);
                self.compile_expr(right)?;
                self.patch_jump(end_jump);
            }
            _ => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                match op {
                    BinOp::Add => self.emit_op(Op::Add),
                    BinOp::Sub => self.emit_op(Op::Sub),
                    BinOp::Mul => self.emit_op(Op::Mul),
                    BinOp::Div => self.emit_op(Op::Div),
                    BinOp::Mod => self.emit_op(Op::Mod),
                    BinOp::Pow => self.emit_op(Op::Pow),
                    BinOp::MatMul => self.emit_op(Op::TensorMatMul),
                    BinOp::Eq => self.emit_op(Op::Eq),
                    BinOp::Neq => self.emit_op(Op::Neq),
                    BinOp::Lt => self.emit_op(Op::Lt),
                    BinOp::Gt => self.emit_op(Op::Gt),
                    BinOp::Lte => self.emit_op(Op::Lte),
                    BinOp::Gte => self.emit_op(Op::Gte),
                    BinOp::And | BinOp::Or => unreachable!(),
                }
            }
        }
        Ok(())
    }

    // ── Assignment compilation ──────────────────────────────────────

    fn compile_assign(&mut self, target: &Expr, value: &Expr) -> Result<(), String> {
        match &target.kind {
            ExprKind::Var(name) => {
                self.compile_expr(value)?;
                let name = name.clone();
                self.emit_store(&name)?;
            }
            ExprKind::Index { object, index } => {
                self.compile_expr(object)?;
                self.compile_expr(index)?;
                self.compile_expr(value)?;
                self.emit_op(Op::ArraySet);
            }
            ExprKind::FieldAccess { object, field } => {
                self.compile_expr(object)?;
                self.compile_expr(value)?;
                let fidx = self.add_constant(Constant::String(field.clone()));
                self.emit_op(Op::StructSet);
                self.emit_byte(fidx);
            }
            _ => return Err("Invalid assignment target".into()),
        }
        Ok(())
    }

    // ── Control flow ────────────────────────────────────────────────

    fn compile_if(
        &mut self,
        condition: &Expr,
        then_branch: &Expr,
        else_branch: Option<&Expr>,
    ) -> Result<(), String> {
        self.compile_expr(condition)?;
        let then_jump = self.emit_jump(Op::JumpIfFalse);
        self.emit_op(Op::Pop); // Pop condition (truthy path)
        self.compile_expr(then_branch)?;
        let else_jump = self.emit_jump(Op::Jump);
        self.patch_jump(then_jump);
        self.emit_op(Op::Pop); // Pop condition (falsey path)
        if let Some(else_b) = else_branch {
            self.compile_expr(else_b)?;
        } else {
            self.emit_op(Op::Nil);
        }
        self.patch_jump(else_jump);
        Ok(())
    }

    fn compile_while(&mut self, condition: &Expr, body: &Expr) -> Result<(), String> {
        let loop_start = self.code_len();
        self.current_mut().loop_starts.push(loop_start);
        self.current_mut().break_jumps.push(Vec::new());

        self.compile_expr(condition)?;
        let exit_jump = self.emit_jump(Op::JumpIfFalse);
        self.emit_op(Op::Pop); // Pop condition
        self.compile_expr(body)?;
        self.emit_op(Op::Pop); // Discard body value each iteration
        self.emit_loop(loop_start);
        self.patch_jump(exit_jump);
        self.emit_op(Op::Pop); // Pop condition (false exit)
        self.emit_op(Op::Nil); // While evaluates to nil

        self.current_mut().loop_starts.pop();
        let breaks = self.current_mut().break_jumps.pop().unwrap();
        for b in breaks {
            self.patch_jump(b);
        }
        Ok(())
    }

    fn compile_for(
        &mut self,
        var: &str,
        iterator: &Expr,
        body: &Expr,
    ) -> Result<(), String> {
        // Desugar:
        //   for x in arr { body }
        // → { let __arr = arr; let __i = 0;
        //     while __i < len(__arr) { let x = __arr[__i]; body; __i++ } }
        self.begin_scope();

        // __arr = <iterator>
        self.compile_expr(iterator)?;
        self.declare_local("__arr");
        self.define_local();
        let arr_slot = (self.current().locals.len() - 1) as u8;

        // __i = 0
        self.emit_constant(Constant::Int(0));
        self.declare_local("__i");
        self.define_local();
        let idx_slot = (self.current().locals.len() - 1) as u8;

        let loop_start = self.code_len();
        self.current_mut().loop_starts.push(loop_start);
        self.current_mut().break_jumps.push(Vec::new());

        // Condition: __i < len(__arr)
        self.emit_op(Op::LoadLocal);
        self.emit_byte(idx_slot);
        self.emit_op(Op::LoadLocal);
        self.emit_byte(arr_slot);
        self.emit_op(Op::ArrayLen);
        self.emit_op(Op::Lt);

        let exit_jump = self.emit_jump(Op::JumpIfFalse);
        self.emit_op(Op::Pop); // Pop condition

        // Inner scope: let x = __arr[__i]
        self.begin_scope();
        self.emit_op(Op::LoadLocal);
        self.emit_byte(arr_slot);
        self.emit_op(Op::LoadLocal);
        self.emit_byte(idx_slot);
        self.emit_op(Op::ArrayGet);
        self.declare_local(var);
        self.define_local();

        // Body
        self.compile_expr(body)?;
        self.emit_op(Op::Pop); // Discard body value

        self.end_scope(); // Pops loop variable

        // __i = __i + 1
        self.emit_op(Op::LoadLocal);
        self.emit_byte(idx_slot);
        self.emit_constant(Constant::Int(1));
        self.emit_op(Op::Add);
        self.emit_op(Op::StoreLocal);
        self.emit_byte(idx_slot);
        self.emit_op(Op::Pop); // StoreLocal peeks; pop the duplicate

        self.emit_loop(loop_start);
        self.patch_jump(exit_jump);
        self.emit_op(Op::Pop); // Pop condition (false exit)
        self.emit_op(Op::Nil); // For evaluates to nil

        self.current_mut().loop_starts.pop();
        let breaks = self.current_mut().break_jumps.pop().unwrap();
        for b in breaks {
            self.patch_jump(b);
        }

        self.end_scope(); // Pops __arr and __i
        Ok(())
    }

    // ── Function compilation ────────────────────────────────────────

    fn compile_function_body(
        &mut self,
        name: Option<String>,
        params: &[Param],
        body: &Expr,
    ) -> Result<(), String> {
        let arity = params.len() as u8;
        self.functions.push(FnCompiler::new(name.clone(), arity));
        self.begin_scope();

        // Declare parameters as locals (slot 0 = function itself).
        for param in params {
            self.declare_local(&param.name);
            self.define_local();
        }

        // If body is a Block, inline its contents directly into the function
        // scope. This avoids the Block's end_scope popping locals from the
        // top of the stack, which would clobber the tail expression result.
        // OP_RETURN already cleans up the entire frame.
        match &body.kind {
            ExprKind::Block { stmts, expr } => {
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                if let Some(e) = expr {
                    self.compile_expr_in_tail(e)?;
                } else {
                    self.emit_op(Op::Nil);
                }
            }
            _ => {
                self.compile_expr_in_tail(body)?;
            }
        }
        self.emit_op(Op::Return);

        // No end_scope — the function's frame is cleaned up by OP_RETURN.
        let fn_compiler = self.functions.pop().unwrap();
        let upvalue_count = fn_compiler.upvalues.len() as u8;

        let fn_const = Constant::Function {
            name,
            arity,
            upvalue_count,
            code: fn_compiler.code,
            constants: fn_compiler.constants,
        };

        let fn_idx = self.add_constant(fn_const);
        self.emit_op(Op::Closure);
        self.emit_byte(fn_idx);

        // Emit upvalue descriptors for the VM to process.
        for uv in &fn_compiler.upvalues {
            self.emit_byte(if uv.is_local { 1 } else { 0 });
            self.emit_byte(uv.index);
        }

        Ok(())
    }

    // ── Tail call optimization ────────────────────────────────────

    /// Compile an expression in tail position. Calls become TailCall,
    /// and If/Block propagate tail position into their sub-expressions.
    fn compile_expr_in_tail(&mut self, expr: &Expr) -> Result<(), String> {
        match &expr.kind {
            ExprKind::Call { callee, args } => {
                self.compile_expr(callee)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                assert!(args.len() <= u8::MAX as usize, "Too many arguments (max 255)");
                self.emit_op(Op::TailCall);
                self.emit_byte(args.len() as u8);
            }

            ExprKind::Pipe { left, right } => {
                self.compile_expr(right)?;
                self.compile_expr(left)?;
                self.emit_op(Op::TailCall);
                self.emit_byte(1);
            }

            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.compile_expr(condition)?;
                let then_jump = self.emit_jump(Op::JumpIfFalse);
                self.emit_op(Op::Pop);
                self.compile_expr_in_tail(then_branch)?;
                let else_jump = self.emit_jump(Op::Jump);
                self.patch_jump(then_jump);
                self.emit_op(Op::Pop);
                if let Some(else_b) = else_branch {
                    self.compile_expr_in_tail(else_b)?;
                } else {
                    self.emit_op(Op::Nil);
                }
                self.patch_jump(else_jump);
            }

            ExprKind::Block { stmts, expr } => {
                self.begin_scope();
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                if let Some(e) = expr {
                    self.compile_expr_in_tail(e)?;
                } else {
                    self.emit_op(Op::Nil);
                }
                self.end_scope();
            }

            // All other expressions: not tail-call-eligible, compile normally.
            _ => {
                self.compile_expr(expr)?;
            }
        }
        Ok(())
    }

    // ── Pattern matching compilation ────────────────────────────────

    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
    ) -> Result<(), String> {
        // Store the scrutinee in a hidden local so we can re-load it
        // for each arm (the VM has no OP_DUP).
        self.begin_scope();
        self.compile_expr(scrutinee)?;
        self.declare_local("__match");
        self.define_local();
        let match_slot = (self.current().locals.len() - 1) as u8;

        let mut end_jumps = Vec::new();

        for arm in arms {
            match &arm.pattern {
                // Wildcard always matches — compile body directly.
                Pattern::Wildcard(_) => {
                    self.compile_expr(&arm.body)?;
                    let j = self.emit_jump(Op::Jump);
                    end_jumps.push(j);
                }

                // Variable binding: bind scrutinee to name, compile body.
                Pattern::Var(name, _) => {
                    self.begin_scope();
                    self.emit_op(Op::LoadLocal);
                    self.emit_byte(match_slot);
                    self.declare_local(name);
                    self.define_local();
                    self.compile_expr(&arm.body)?;
                    self.end_scope();
                    let j = self.emit_jump(Op::Jump);
                    end_jumps.push(j);
                }

                // Literal pattern: compare scrutinee to literal.
                Pattern::Literal(lit) => {
                    self.emit_op(Op::LoadLocal);
                    self.emit_byte(match_slot);
                    self.compile_expr(lit)?;
                    self.emit_op(Op::Eq);
                    let skip = self.emit_jump(Op::JumpIfFalse);
                    self.emit_op(Op::Pop); // Pop comparison result
                    self.compile_expr(&arm.body)?;
                    let j = self.emit_jump(Op::Jump);
                    end_jumps.push(j);
                    self.patch_jump(skip);
                    self.emit_op(Op::Pop); // Pop comparison result (false)
                }

                // Variant pattern: check tag, bind fields.
                Pattern::Variant {
                    name, fields, ..
                } => {
                    let tag_idx = self.add_constant(Constant::String(name.clone()));
                    // Load scrutinee, use OP_MATCH to check tag.
                    self.emit_op(Op::LoadLocal);
                    self.emit_byte(match_slot);
                    self.emit_op(Op::Match);
                    self.emit_byte(tag_idx);
                    // OP_MATCH emits a 2-byte jump offset if no match.
                    let skip_hi = self.code_len();
                    self.emit_byte(0xff);
                    self.emit_byte(0xff);

                    // Match succeeded — bind fields.
                    self.begin_scope();
                    if fields.len() == 1 {
                        if let Pattern::Var(field_name, _) = &fields[0] {
                            // Load variant payload: scrutinee is still on stack
                            // after OP_MATCH success. Pop it, the payload is
                            // accessible via the variant. For now, we need a way
                            // to extract the payload. Since ObjVariant has a single
                            // payload field, we can use OP_STRUCT_GET with "_0".
                            // Actually OP_MATCH leaves the value on the stack,
                            // so the scrutinee (variant) is TOS. We need to
                            // extract .payload from it. Without a dedicated opcode,
                            // use struct field access with a known field name.
                            let payload_name =
                                self.add_constant(Constant::String("_0".to_string()));
                            self.emit_op(Op::StructGet);
                            self.emit_byte(payload_name);
                            self.declare_local(field_name);
                            self.define_local();
                        }
                    }
                    // Pop the variant from stack (OP_MATCH leaves it).
                    // If we bound a field, we consumed it with StructGet.
                    // If no fields, just pop.
                    if fields.is_empty() {
                        self.emit_op(Op::Pop);
                    }
                    self.compile_expr(&arm.body)?;
                    self.end_scope();
                    let j = self.emit_jump(Op::Jump);
                    end_jumps.push(j);

                    // Patch the skip for OP_MATCH failure.
                    let f = self.current_mut();
                    let skip_target = f.code.len() - skip_hi - 2;
                    f.code[skip_hi] = ((skip_target >> 8) & 0xff) as u8;
                    f.code[skip_hi + 1] = (skip_target & 0xff) as u8;
                    // OP_MATCH doesn't pop on failure, so pop scrutinee copy.
                    self.emit_op(Op::Pop);
                }
            }
        }

        // Patch all end jumps to land here.
        for j in end_jumps {
            self.patch_jump(j);
        }

        self.end_scope(); // Pops __match
        Ok(())
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Compile an AST into bytecode ready for the C VM.
pub fn compile(program: &Program) -> Result<CompiledProgram, String> {
    let mut compiler = Compiler::new();
    compiler.compile_program(program)?;

    let top = compiler.functions.pop().unwrap();
    Ok(CompiledProgram {
        constants: top.constants,
        code: top.code,
    })
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn compile_source(src: &str) -> CompiledProgram {
        let mut lexer = Lexer::new(src);
        let tokens = lexer.scan_tokens();
        assert!(lexer.errors().is_empty(), "Lex errors: {:?}", lexer.errors());
        let mut parser = Parser::new(tokens);
        let program = parser.parse().expect("Parse failed");
        compile(&program).expect("Compile failed")
    }

    fn find_op(code: &[u8], op: Op) -> bool {
        code.iter().any(|&b| b == op as u8)
    }

    #[test]
    fn test_integer_literal() {
        let prog = compile_source("42");
        assert!(find_op(&prog.code, Op::Const));
        assert!(find_op(&prog.code, Op::Pop)); // expression statement
        assert_eq!(prog.constants.len(), 1);
    }

    #[test]
    fn test_binary_add() {
        let prog = compile_source("1 + 2");
        assert!(find_op(&prog.code, Op::Add));
    }

    #[test]
    fn test_let_global() {
        let prog = compile_source("let x = 10");
        assert!(find_op(&prog.code, Op::StoreGlobal));
    }

    #[test]
    fn test_if_expression() {
        let prog = compile_source("if true { 1 } else { 2 }");
        assert!(find_op(&prog.code, Op::JumpIfFalse));
        assert!(find_op(&prog.code, Op::Jump));
    }

    #[test]
    fn test_while_loop() {
        let prog = compile_source("while false { 1 }");
        assert!(find_op(&prog.code, Op::Loop));
        assert!(find_op(&prog.code, Op::JumpIfFalse));
    }

    #[test]
    fn test_function_declaration() {
        let prog = compile_source("fn add(a, b) { a + b }");
        assert!(find_op(&prog.code, Op::Closure));
        assert!(find_op(&prog.code, Op::StoreGlobal));
        // The function constant should exist.
        assert!(prog.constants.iter().any(|c| matches!(c, Constant::Function { .. })));
    }

    #[test]
    fn test_function_call() {
        let prog = compile_source("fn f() { 42 }\nf()");
        assert!(find_op(&prog.code, Op::Call));
    }

    #[test]
    fn test_array_literal() {
        let prog = compile_source("[1, 2, 3]");
        assert!(find_op(&prog.code, Op::ArrayNew));
    }

    #[test]
    fn test_struct_literal() {
        let prog = compile_source("struct Point { x: Int, y: Int }\nPoint { x: 1, y: 2 }");
        assert!(find_op(&prog.code, Op::StructNew));
    }

    #[test]
    fn test_field_access() {
        let prog = compile_source(
            "struct Point { x: Int, y: Int }\nlet p = Point { x: 1, y: 2 }\np.x",
        );
        assert!(find_op(&prog.code, Op::StructGet));
    }

    #[test]
    fn test_closure_upvalue() {
        let prog = compile_source(
            "fn outer() {\n  let x = 10\n  fn inner() { x }\n  inner()\n}",
        );
        // The inner function should have an upvalue.
        let has_closure_with_upvalues = prog.constants.iter().any(|c| {
            if let Constant::Function { constants, .. } = c {
                constants.iter().any(|inner| {
                    matches!(inner, Constant::Function { upvalue_count, .. } if *upvalue_count > 0)
                })
            } else {
                false
            }
        });
        assert!(has_closure_with_upvalues);
    }

    #[test]
    fn test_pipe_operator() {
        let prog = compile_source("fn double(x) { x * 2 }\n5 |> double");
        assert!(find_op(&prog.code, Op::Call));
    }

    #[test]
    fn test_for_loop() {
        let prog = compile_source("for x in [1, 2, 3] { x }");
        assert!(find_op(&prog.code, Op::ArrayLen));
        assert!(find_op(&prog.code, Op::Loop));
    }

    #[test]
    fn test_short_circuit_and() {
        let prog = compile_source("true && false");
        assert!(find_op(&prog.code, Op::JumpIfFalse));
        // No OP_AND exists — it's compiled to jumps.
    }

    #[test]
    fn test_tail_call_optimization() {
        // Tail call: last expression in function body is a call
        let prog = compile_source("fn f(n) { f(n) }");
        let fn_const = prog.constants.iter().find(|c| matches!(c, Constant::Function { .. })).unwrap();
        if let Constant::Function { code, .. } = fn_const {
            assert!(find_op(code, Op::TailCall), "Expected TailCall in tail position");
            assert!(!find_op(code, Op::Call), "Call should be replaced by TailCall");
        }
    }

    #[test]
    fn test_tail_call_in_if_branches() {
        // Both branches of an if in tail position should get TailCall
        let prog = compile_source("fn f(n) { if n > 0 { f(n - 1) } else { f(0) } }");
        let fn_const = prog.constants.iter().find(|c| matches!(c, Constant::Function { .. })).unwrap();
        if let Constant::Function { code, .. } = fn_const {
            assert!(find_op(code, Op::TailCall), "Expected TailCall in if branches");
            assert!(!find_op(code, Op::Call), "All calls should be TailCall");
        }
    }

    #[test]
    fn test_non_tail_call_not_optimized() {
        // f(n) + 1 is NOT a tail call — result is used in addition
        let prog = compile_source("fn f(n) { f(n) + 1 }");
        let fn_const = prog.constants.iter().find(|c| matches!(c, Constant::Function { .. })).unwrap();
        if let Constant::Function { code, .. } = fn_const {
            assert!(find_op(code, Op::Call), "Non-tail call should remain Call");
            assert!(!find_op(code, Op::TailCall), "Should not have TailCall");
        }
    }

    #[test]
    fn test_explicit_return_tail_call() {
        // return f(n) is a tail call
        let prog = compile_source("fn f(n) {\n  return f(n)\n}");
        let fn_const = prog.constants.iter().find(|c| matches!(c, Constant::Function { .. })).unwrap();
        if let Constant::Function { code, .. } = fn_const {
            assert!(find_op(code, Op::TailCall), "return f(n) should be a tail call");
        }
    }

    #[test]
    fn test_return_statement() {
        // return is a statement, so it must be followed by a semicolon/newline
        // or be the only thing in the block. The block's tail expression is
        // implicit nil after the return statement.
        let prog = compile_source("fn f(x) {\n  return x + 1\n}");
        // The function body should contain OP_RETURN.
        let has_return = prog.constants.iter().any(|c| {
            if let Constant::Function { code, .. } = c {
                find_op(code, Op::Return)
            } else {
                false
            }
        });
        assert!(has_return);
    }
}
