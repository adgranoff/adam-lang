//! Parser — Pratt parser (expressions) + recursive descent (statements).
//!
//! The parser converts a flat token stream into a structured AST. It uses
//! two complementary techniques:
//!
//! **Pratt parsing** (aka precedence climbing) for expressions:
//!   Each token has a "binding power" (precedence). When parsing a binary
//!   expression, we continue consuming operators as long as their binding
//!   power exceeds the current minimum. This elegantly handles precedence
//!   and associativity without a separate grammar rule per precedence level.
//!   See Vaughan Pratt's 1973 paper "Top Down Operator Precedence."
//!
//! **Recursive descent** for statements and declarations:
//!   Each grammar production (let, fn, struct, etc.) maps to a function
//!   that consumes tokens and returns an AST node. This is straightforward
//!   and produces excellent error messages because we always know what
//!   we were trying to parse when something goes wrong.
//!
//! Error recovery:
//!   On a parse error, we record the error and attempt to synchronize to
//!   the next statement boundary (semicolon, `}`, or keyword). This lets
//!   us report multiple errors per parse attempt.

use crate::ast::*;
use crate::token::{Span, Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    errors: Vec<String>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            errors: Vec::new(),
        }
    }

    /// Parse a complete program. Returns Ok(stmts) or Err(errors).
    pub fn parse(&mut self) -> Result<Program, Vec<String>> {
        let mut stmts = Vec::new();
        while !self.is_at_end() {
            match self.declaration() {
                Ok(stmt) => stmts.push(stmt),
                Err(msg) => {
                    self.errors.push(msg);
                    self.synchronize();
                }
            }
        }
        if self.errors.is_empty() {
            Ok(stmts)
        } else {
            Err(self.errors.clone())
        }
    }

    // ── Declarations ─────────────────────────────────────────────────

    fn declaration(&mut self) -> Result<Stmt, String> {
        match self.peek_kind() {
            TokenKind::Let => self.let_declaration(),
            TokenKind::Fn => self.function_declaration(),
            TokenKind::Type => self.type_declaration(),
            TokenKind::Struct => self.struct_declaration(),
            TokenKind::Impl => self.impl_block(),
            _ => self.statement(),
        }
    }

    fn let_declaration(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Let)?;
        let name = self.expect_identifier()?;

        let type_ann = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.expression()?;
        let span = start.merge(value.span);
        Ok(Stmt {
            kind: StmtKind::Let {
                name,
                type_ann,
                value,
            },
            span,
        })
    }

    fn function_declaration(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Fn)?;
        let name = self.expect_identifier()?;

        self.expect(TokenKind::LParen)?;
        let params = self.param_list()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.block_expr()?;
        let span = start.merge(body.span);
        Ok(Stmt {
            kind: StmtKind::Function {
                name,
                params,
                return_type,
                body: Box::new(body),
            },
            span,
        })
    }

    fn type_declaration(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Type)?;
        let name = self.expect_identifier()?;

        let type_params = if self.match_token(TokenKind::Lt) {
            let params = self.comma_separated(|p| p.expect_identifier())?;
            self.expect(TokenKind::Gt)?;
            params
        } else {
            vec![]
        };

        self.expect(TokenKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let vstart = self.current_span();
            let vname = self.expect_identifier()?;
            let fields = if self.match_token(TokenKind::LParen) {
                let types = self.comma_separated(|p| p.parse_type())?;
                self.expect(TokenKind::RParen)?;
                types
            } else {
                vec![]
            };
            let vspan = vstart.merge(self.previous_span());
            variants.push(Variant {
                name: vname,
                fields,
                span: vspan,
            });
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Stmt {
            kind: StmtKind::TypeDecl {
                name,
                type_params,
                variants,
            },
            span: start.merge(end),
        })
    }

    fn struct_declaration(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Struct)?;
        let name = self.expect_identifier()?;

        let type_params = if self.match_token(TokenKind::Lt) {
            let params = self.comma_separated(|p| p.expect_identifier())?;
            self.expect(TokenKind::Gt)?;
            params
        } else {
            vec![]
        };

        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let fstart = self.current_span();
            let fname = self.expect_identifier()?;
            self.expect(TokenKind::Colon)?;
            let ftype = self.parse_type()?;
            let fspan = fstart.merge(ftype.span);
            fields.push(Field {
                name: fname,
                type_ann: ftype,
                span: fspan,
            });
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Stmt {
            kind: StmtKind::StructDecl {
                name,
                type_params,
                fields,
            },
            span: start.merge(end),
        })
    }

    fn impl_block(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Impl)?;
        let type_name = self.expect_identifier()?;
        self.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            methods.push(self.function_declaration()?);
        }
        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Stmt {
            kind: StmtKind::ImplBlock {
                type_name,
                methods,
            },
            span: start.merge(end),
        })
    }

    // ── Statements ───────────────────────────────────────────────────

    fn statement(&mut self) -> Result<Stmt, String> {
        if self.check(TokenKind::Return) {
            return self.return_statement();
        }

        let expr = self.expression()?;
        let span = expr.span;
        Ok(Stmt {
            kind: StmtKind::Expression(expr),
            span,
        })
    }

    fn return_statement(&mut self) -> Result<Stmt, String> {
        let start = self.current_span();
        self.expect(TokenKind::Return)?;

        let value = if self.check(TokenKind::RBrace) || self.is_at_end() {
            None
        } else {
            Some(self.expression()?)
        };

        let span = start.merge(value.as_ref().map_or(start, |e| e.span));
        Ok(Stmt {
            kind: StmtKind::Return(value),
            span,
        })
    }

    // ── Pratt parser for expressions ─────────────────────────────────

    fn expression(&mut self) -> Result<Expr, String> {
        self.parse_expr(0)
    }

    /// Core Pratt parsing loop.
    ///
    /// 1. Parse the "prefix" (literal, unary, parenthesized, etc.)
    /// 2. While the next token's binding power > min_bp, consume it
    ///    and parse the right-hand side at the appropriate binding power.
    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, String> {
        let mut left = self.prefix()?;

        loop {
            let (op_bp, assoc) = match self.peek_kind() {
                // Assignment (right-associative, lowest precedence)
                TokenKind::Eq => (1, Assoc::Right),

                // Pipe operator
                TokenKind::PipeGt => (2, Assoc::Left),

                // Logical or
                TokenKind::Or => (3, Assoc::Left),

                // Logical and
                TokenKind::And => (4, Assoc::Left),

                // Equality
                TokenKind::EqEq | TokenKind::BangEq => (5, Assoc::Left),

                // Comparison
                TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq => {
                    (6, Assoc::Left)
                }

                // Addition/subtraction
                TokenKind::Plus | TokenKind::Minus => (7, Assoc::Left),

                // Multiplication/division
                TokenKind::Star | TokenKind::Slash | TokenKind::Percent => (8, Assoc::Left),

                // Power (right-associative)
                TokenKind::StarStar => (9, Assoc::Right),

                // Postfix: call, index, field access
                TokenKind::LParen => (10, Assoc::Left),
                TokenKind::LBracket => (10, Assoc::Left),
                TokenKind::Dot => (10, Assoc::Left),

                _ => break,
            };

            if op_bp < min_bp {
                break;
            }

            // Left-associative: right side parses at op_bp + 1 so same-precedence
            // operators stop (become the next iteration of the outer loop).
            // Right-associative: right side parses at op_bp so same-precedence
            // operators are consumed recursively.
            let right_bp = match assoc {
                Assoc::Left => op_bp + 1,
                Assoc::Right => op_bp,
            };

            left = self.infix(left, right_bp)?;
        }

        Ok(left)
    }

    /// Parse a prefix expression (the "nud" in Pratt terminology).
    fn prefix(&mut self) -> Result<Expr, String> {
        let token = self.advance_token();
        let start = token.span;

        match token.kind {
            TokenKind::Int(n) => Ok(Expr {
                kind: ExprKind::IntLit(n),
                span: start,
            }),
            TokenKind::Float(n) => Ok(Expr {
                kind: ExprKind::FloatLit(n),
                span: start,
            }),
            TokenKind::Str(s) => Ok(Expr {
                kind: ExprKind::StringLit(s),
                span: start,
            }),
            TokenKind::True => Ok(Expr {
                kind: ExprKind::BoolLit(true),
                span: start,
            }),
            TokenKind::False => Ok(Expr {
                kind: ExprKind::BoolLit(false),
                span: start,
            }),
            TokenKind::Identifier(name) => {
                // Check for struct literal: Name { ... }
                if self.check(TokenKind::LBrace) && name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return self.struct_literal(name, start);
                }
                // Check for variant constructor: Name(args)
                if self.check(TokenKind::LParen) && name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    self.advance_token(); // consume (
                    let args = if self.check(TokenKind::RParen) {
                        vec![]
                    } else {
                        self.comma_separated(|p| p.expression())?
                    };
                    let end = self.current_span();
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expr {
                        kind: ExprKind::VariantConstruct { name, args },
                        span: start.merge(end),
                    });
                }
                Ok(Expr {
                    kind: ExprKind::Var(name),
                    span: start,
                })
            }
            TokenKind::Minus => {
                let operand = self.parse_expr(9)?; // high precedence for unary
                let span = start.merge(operand.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::Bang => {
                let operand = self.parse_expr(9)?;
                let span = start.merge(operand.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::LParen => {
                let expr = self.expression()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => {
                let elements = if self.check(TokenKind::RBracket) {
                    vec![]
                } else {
                    self.comma_separated(|p| p.expression())?
                };
                let end = self.current_span();
                self.expect(TokenKind::RBracket)?;
                Ok(Expr {
                    kind: ExprKind::Array(elements),
                    span: start.merge(end),
                })
            }
            TokenKind::LBrace => {
                self.current -= 1; // put back the brace
                self.block_expr()
            }
            TokenKind::If => {
                self.current -= 1;
                self.if_expr()
            }
            TokenKind::Match => {
                self.current -= 1;
                self.match_expr()
            }
            TokenKind::While => {
                self.current -= 1;
                self.while_expr()
            }
            TokenKind::For => {
                self.current -= 1;
                self.for_expr()
            }
            TokenKind::Pipe => {
                // Lambda: |params| body
                let params = if self.check(TokenKind::Pipe) {
                    vec![]
                } else {
                    self.lambda_params()?
                };
                self.expect(TokenKind::Pipe)?;
                let body = self.expression()?;
                let span = start.merge(body.span);
                Ok(Expr {
                    kind: ExprKind::Lambda {
                        params,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            TokenKind::Or => {
                // Empty lambda: || body
                let body = self.expression()?;
                let span = start.merge(body.span);
                Ok(Expr {
                    kind: ExprKind::Lambda {
                        params: vec![],
                        body: Box::new(body),
                    },
                    span,
                })
            }
            TokenKind::Break => Ok(Expr {
                kind: ExprKind::Break,
                span: start,
            }),
            TokenKind::Continue => Ok(Expr {
                kind: ExprKind::Continue,
                span: start,
            }),
            _ => Err(format!(
                "Expected expression, found '{}' at byte {}",
                token.kind, token.span.start
            )),
        }
    }

    /// Parse an infix expression (the "led" in Pratt terminology).
    fn infix(&mut self, left: Expr, bp: u8) -> Result<Expr, String> {
        let token = self.advance_token();
        let op_span = token.span;

        match token.kind {
            // Binary operators
            TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Percent
            | TokenKind::StarStar
            | TokenKind::EqEq
            | TokenKind::BangEq
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::LtEq
            | TokenKind::GtEq
            | TokenKind::And
            | TokenKind::Or => {
                let op = match token.kind {
                    TokenKind::Plus => BinOp::Add,
                    TokenKind::Minus => BinOp::Sub,
                    TokenKind::Star => BinOp::Mul,
                    TokenKind::Slash => BinOp::Div,
                    TokenKind::Percent => BinOp::Mod,
                    TokenKind::StarStar => BinOp::Pow,
                    TokenKind::EqEq => BinOp::Eq,
                    TokenKind::BangEq => BinOp::Neq,
                    TokenKind::Lt => BinOp::Lt,
                    TokenKind::Gt => BinOp::Gt,
                    TokenKind::LtEq => BinOp::Lte,
                    TokenKind::GtEq => BinOp::Gte,
                    TokenKind::And => BinOp::And,
                    TokenKind::Or => BinOp::Or,
                    _ => unreachable!(),
                };
                let right = self.parse_expr(bp)?;
                let span = left.span.merge(right.span);
                Ok(Expr {
                    kind: ExprKind::Binary {
                        left: Box::new(left),
                        op,
                        right: Box::new(right),
                    },
                    span,
                })
            }

            // Pipe operator: a |> f  desugars to f(a)
            TokenKind::PipeGt => {
                let right = self.parse_expr(bp)?;
                let span = left.span.merge(right.span);
                Ok(Expr {
                    kind: ExprKind::Pipe {
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                })
            }

            // Assignment
            TokenKind::Eq => {
                let value = self.parse_expr(0)?;
                let span = left.span.merge(value.span);
                Ok(Expr {
                    kind: ExprKind::Assign {
                        target: Box::new(left),
                        value: Box::new(value),
                    },
                    span,
                })
            }

            // Function call: f(args)
            TokenKind::LParen => {
                let left_span = left.span;
                let args = if self.check(TokenKind::RParen) {
                    vec![]
                } else {
                    self.comma_separated(|p| p.expression())?
                };
                let end = self.current_span();
                self.expect(TokenKind::RParen)?;
                Ok(Expr {
                    kind: ExprKind::Call {
                        callee: Box::new(left),
                        args,
                    },
                    span: left_span.merge(end),
                })
            }

            // Index: a[i]
            TokenKind::LBracket => {
                let left_span = left.span;
                let index = self.expression()?;
                let end = self.current_span();
                self.expect(TokenKind::RBracket)?;
                Ok(Expr {
                    kind: ExprKind::Index {
                        object: Box::new(left),
                        index: Box::new(index),
                    },
                    span: left_span.merge(end),
                })
            }

            // Field access: a.field
            TokenKind::Dot => {
                let left_span = left.span;
                let field = self.expect_identifier()?;
                let end = self.previous_span();
                Ok(Expr {
                    kind: ExprKind::FieldAccess {
                        object: Box::new(left),
                        field,
                    },
                    span: left_span.merge(end),
                })
            }

            _ => Err(format!(
                "Unexpected infix operator '{}' at byte {}",
                token.kind, op_span.start
            )),
        }
    }

    // ── Compound expressions ─────────────────────────────────────────

    fn block_expr(&mut self) -> Result<Expr, String> {
        let start = self.current_span();
        self.expect(TokenKind::LBrace)?;

        let mut stmts = Vec::new();
        let mut final_expr: Option<Box<Expr>> = None;

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            // Try to parse as a declaration first
            if matches!(
                self.peek_kind(),
                TokenKind::Let | TokenKind::Fn | TokenKind::Type | TokenKind::Struct
            ) {
                stmts.push(self.declaration()?);
                continue;
            }

            // Return statement
            if self.check(TokenKind::Return) {
                stmts.push(self.return_statement()?);
                continue;
            }

            // Parse expression
            let expr = self.expression()?;

            // If next is `}`, this expression is the block's value
            if self.check(TokenKind::RBrace) {
                final_expr = Some(Box::new(expr));
                break;
            }

            // Otherwise it's an expression statement
            let span = expr.span;
            stmts.push(Stmt {
                kind: StmtKind::Expression(expr),
                span,
            });
        }

        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Expr {
            kind: ExprKind::Block {
                stmts,
                expr: final_expr,
            },
            span: start.merge(end),
        })
    }

    fn if_expr(&mut self) -> Result<Expr, String> {
        let start = self.current_span();
        self.expect(TokenKind::If)?;

        let condition = self.expression()?;
        let then_branch = self.block_expr()?;

        let else_branch = if self.match_token(TokenKind::Else) {
            if self.check(TokenKind::If) {
                Some(Box::new(self.if_expr()?))
            } else {
                Some(Box::new(self.block_expr()?))
            }
        } else {
            None
        };

        let span = start.merge(
            else_branch
                .as_ref()
                .map_or(then_branch.span, |e| e.span),
        );
        Ok(Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch,
            },
            span,
        })
    }

    fn match_expr(&mut self) -> Result<Expr, String> {
        let start = self.current_span();
        self.expect(TokenKind::Match)?;
        let scrutinee = self.expression()?;
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let arm_start = self.current_span();
            let pattern = self.parse_pattern()?;
            self.expect(TokenKind::FatArrow)?;
            let body = self.expression()?;
            let arm_span = arm_start.merge(body.span);
            arms.push(MatchArm {
                pattern,
                body,
                span: arm_span,
            });
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;

        Ok(Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
            span: start.merge(end),
        })
    }

    fn while_expr(&mut self) -> Result<Expr, String> {
        let start = self.current_span();
        self.expect(TokenKind::While)?;
        let condition = self.expression()?;
        let body = self.block_expr()?;
        let span = start.merge(body.span);
        Ok(Expr {
            kind: ExprKind::While {
                condition: Box::new(condition),
                body: Box::new(body),
            },
            span,
        })
    }

    fn for_expr(&mut self) -> Result<Expr, String> {
        let start = self.current_span();
        self.expect(TokenKind::For)?;
        let var = self.expect_identifier()?;
        self.expect(TokenKind::In)?;
        let iterator = self.expression()?;
        let body = self.block_expr()?;
        let span = start.merge(body.span);
        Ok(Expr {
            kind: ExprKind::For {
                var,
                iterator: Box::new(iterator),
                body: Box::new(body),
            },
            span,
        })
    }

    fn struct_literal(&mut self, name: String, start: Span) -> Result<Expr, String> {
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let fname = self.expect_identifier()?;
            self.expect(TokenKind::Colon)?;
            let fvalue = self.expression()?;
            fields.push((fname, fvalue));
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        let end = self.current_span();
        self.expect(TokenKind::RBrace)?;
        Ok(Expr {
            kind: ExprKind::StructLit { name, fields },
            span: start.merge(end),
        })
    }

    // ── Patterns ─────────────────────────────────────────────────────

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        let token = self.advance_token();
        match token.kind {
            TokenKind::Identifier(ref name) if name == "_" => {
                Ok(Pattern::Wildcard(token.span))
            }
            TokenKind::Identifier(name) => {
                if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    // Variant pattern: Some(x)
                    if self.match_token(TokenKind::LParen) {
                        let fields = self.comma_separated(|p| p.parse_pattern())?;
                        let end = self.current_span();
                        self.expect(TokenKind::RParen)?;
                        Ok(Pattern::Variant {
                            name,
                            fields,
                            span: token.span.merge(end),
                        })
                    } else {
                        // Variant without fields: None
                        Ok(Pattern::Variant {
                            name,
                            fields: vec![],
                            span: token.span,
                        })
                    }
                } else {
                    Ok(Pattern::Var(name, token.span))
                }
            }
            TokenKind::Int(n) => Ok(Pattern::Literal(Expr {
                kind: ExprKind::IntLit(n),
                span: token.span,
            })),
            TokenKind::Float(n) => Ok(Pattern::Literal(Expr {
                kind: ExprKind::FloatLit(n),
                span: token.span,
            })),
            TokenKind::Str(s) => Ok(Pattern::Literal(Expr {
                kind: ExprKind::StringLit(s),
                span: token.span,
            })),
            TokenKind::True => Ok(Pattern::Literal(Expr {
                kind: ExprKind::BoolLit(true),
                span: token.span,
            })),
            TokenKind::False => Ok(Pattern::Literal(Expr {
                kind: ExprKind::BoolLit(false),
                span: token.span,
            })),
            _ => Err(format!(
                "Expected pattern, found '{}' at byte {}",
                token.kind, token.span.start
            )),
        }
    }

    // ── Type annotations ─────────────────────────────────────────────

    fn parse_type(&mut self) -> Result<TypeExpr, String> {
        let start = self.current_span();

        // Array type: [T]
        if self.match_token(TokenKind::LBracket) {
            let inner = self.parse_type()?;
            let end = self.current_span();
            self.expect(TokenKind::RBracket)?;
            return Ok(TypeExpr {
                kind: TypeExprKind::Array(Box::new(inner)),
                span: start.merge(end),
            });
        }

        // Function type: fn(T, U) -> V
        if self.match_token(TokenKind::Fn) {
            self.expect(TokenKind::LParen)?;
            let params = self.comma_separated(|p| p.parse_type())?;
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Arrow)?;
            let ret = self.parse_type()?;
            let span = start.merge(ret.span);
            return Ok(TypeExpr {
                kind: TypeExprKind::Function {
                    params,
                    ret: Box::new(ret),
                },
                span,
            });
        }

        // Named type, possibly generic: Int, Option<T>
        let name = self.expect_identifier()?;
        if self.match_token(TokenKind::Lt) {
            let args = self.comma_separated(|p| p.parse_type())?;
            let end = self.current_span();
            self.expect(TokenKind::Gt)?;
            Ok(TypeExpr {
                kind: TypeExprKind::Generic { name, args },
                span: start.merge(end),
            })
        } else {
            let end = self.previous_span();
            Ok(TypeExpr {
                kind: TypeExprKind::Named(name),
                span: start.merge(end),
            })
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn param_list(&mut self) -> Result<Vec<Param>, String> {
        if self.check(TokenKind::RParen) {
            return Ok(vec![]);
        }
        self.comma_separated(|p| {
            let start = p.current_span();
            // Handle `self` parameter
            let name = p.expect_identifier()?;
            let type_ann = if p.match_token(TokenKind::Colon) {
                Some(p.parse_type()?)
            } else {
                None
            };
            let span = start.merge(p.previous_span());
            Ok(Param {
                name,
                type_ann,
                span,
            })
        })
    }

    fn lambda_params(&mut self) -> Result<Vec<Param>, String> {
        self.comma_separated(|p| {
            let start = p.current_span();
            let name = p.expect_identifier()?;
            let span = start.merge(p.previous_span());
            Ok(Param {
                name,
                type_ann: None,
                span,
            })
        })
    }

    fn comma_separated<T>(
        &mut self,
        mut parse_fn: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let mut items = vec![parse_fn(self)?];
        while self.match_token(TokenKind::Comma) {
            // Allow trailing comma
            if self.check(TokenKind::RParen)
                || self.check(TokenKind::RBracket)
                || self.check(TokenKind::RBrace)
                || self.check(TokenKind::Gt)
                || self.check(TokenKind::Pipe)
            {
                break;
            }
            items.push(parse_fn(self)?);
        }
        Ok(items)
    }

    // ── Token manipulation ───────────────────────────────────────────

    fn advance_token(&mut self) -> Token {
        let token = self.tokens[self.current].clone();
        if !self.is_at_end() {
            self.current += 1;
        }
        token
    }

    fn peek_kind(&self) -> TokenKind {
        self.tokens[self.current].kind.clone()
    }

    fn check(&self, kind: TokenKind) -> bool {
        std::mem::discriminant(&self.tokens[self.current].kind)
            == std::mem::discriminant(&kind)
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.current += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<(), String> {
        if self.check(kind.clone()) {
            self.current += 1;
            Ok(())
        } else {
            Err(format!(
                "Expected '{}', found '{}' at byte {}",
                kind,
                self.tokens[self.current].kind,
                self.tokens[self.current].span.start,
            ))
        }
    }

    fn expect_identifier(&mut self) -> Result<String, String> {
        let token = self.advance_token();
        match token.kind {
            TokenKind::Identifier(name) => Ok(name),
            // Allow 'self' as identifier in parameter position
            _ => Err(format!(
                "Expected identifier, found '{}' at byte {}",
                token.kind, token.span.start
            )),
        }
    }

    fn current_span(&self) -> Span {
        self.tokens[self.current].span
    }

    fn previous_span(&self) -> Span {
        self.tokens[self.current.saturating_sub(1)].span
    }

    fn is_at_end(&self) -> bool {
        matches!(self.tokens[self.current].kind, TokenKind::Eof)
    }

    /// Panic-mode error recovery: skip tokens until we find a statement
    /// boundary (newline after expression, semicolon, closing brace, or
    /// the start of a new declaration keyword).
    fn synchronize(&mut self) {
        while !self.is_at_end() {
            match self.peek_kind() {
                TokenKind::Let
                | TokenKind::Fn
                | TokenKind::Type
                | TokenKind::Struct
                | TokenKind::Impl
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return => return,
                TokenKind::RBrace => {
                    self.current += 1;
                    return;
                }
                _ => {
                    self.current += 1;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Assoc {
    Left,
    Right,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.scan_tokens();
        assert!(lexer.errors().is_empty(), "Lexer errors: {:?}", lexer.errors());
        let mut parser = Parser::new(tokens);
        parser.parse().expect("Parse errors")
    }

    #[test]
    fn test_let_binding() {
        let program = parse("let x = 42");
        assert_eq!(program.len(), 1);
        assert!(matches!(program[0].kind, StmtKind::Let { .. }));
    }

    #[test]
    fn test_let_with_type() {
        let program = parse("let x: Int = 42");
        assert_eq!(program.len(), 1);
        if let StmtKind::Let { type_ann, .. } = &program[0].kind {
            assert!(type_ann.is_some());
        } else {
            panic!("Expected Let");
        }
    }

    #[test]
    fn test_function() {
        let program = parse("fn add(a: Int, b: Int) -> Int { a + b }");
        assert_eq!(program.len(), 1);
        if let StmtKind::Function { params, .. } = &program[0].kind {
            assert_eq!(params.len(), 2);
        } else {
            panic!("Expected Function");
        }
    }

    #[test]
    fn test_binary_precedence() {
        let program = parse("1 + 2 * 3");
        if let StmtKind::Expression(expr) = &program[0].kind {
            // Should be Add(1, Mul(2, 3))
            if let ExprKind::Binary { op, right, .. } = &expr.kind {
                assert_eq!(*op, BinOp::Add);
                assert!(matches!(right.kind, ExprKind::Binary { op: BinOp::Mul, .. }));
            } else {
                panic!("Expected Binary");
            }
        }
    }

    #[test]
    fn test_if_expression() {
        let program = parse("if x > 0 { x } else { -x }");
        assert!(matches!(
            program[0].kind,
            StmtKind::Expression(Expr { kind: ExprKind::If { .. }, .. })
        ));
    }

    #[test]
    fn test_match_expression() {
        let program = parse(r#"
            match opt {
                Some(x) => x,
                None => 0,
            }
        "#);
        if let StmtKind::Expression(Expr { kind: ExprKind::Match { arms, .. }, .. }) = &program[0].kind {
            assert_eq!(arms.len(), 2);
        } else {
            panic!("Expected Match");
        }
    }

    #[test]
    fn test_type_declaration() {
        let program = parse(r#"
            type Option<T> {
                Some(T),
                None,
            }
        "#);
        if let StmtKind::TypeDecl { variants, type_params, .. } = &program[0].kind {
            assert_eq!(type_params.len(), 1);
            assert_eq!(variants.len(), 2);
        } else {
            panic!("Expected TypeDecl");
        }
    }

    #[test]
    fn test_struct_declaration() {
        let program = parse("struct Point { x: Float, y: Float }");
        if let StmtKind::StructDecl { fields, .. } = &program[0].kind {
            assert_eq!(fields.len(), 2);
        } else {
            panic!("Expected StructDecl");
        }
    }

    #[test]
    fn test_pipe_operator() {
        let program = parse("x |> foo |> bar");
        if let StmtKind::Expression(expr) = &program[0].kind {
            // Should be Pipe(Pipe(x, foo), bar) — left-associative
            assert!(matches!(expr.kind, ExprKind::Pipe { .. }));
        }
    }

    #[test]
    fn test_lambda() {
        let program = parse("|x| x + 1");
        if let StmtKind::Expression(expr) = &program[0].kind {
            if let ExprKind::Lambda { params, .. } = &expr.kind {
                assert_eq!(params.len(), 1);
            } else {
                panic!("Expected Lambda");
            }
        }
    }

    #[test]
    fn test_array_literal() {
        let program = parse("[1, 2, 3]");
        if let StmtKind::Expression(Expr { kind: ExprKind::Array(elements), .. }) = &program[0].kind {
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Array");
        }
    }

    #[test]
    fn test_full_program() {
        let source = r#"
            fn fibonacci(n: Int) -> Int {
                if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
            }

            let result = fibonacci(10)
        "#;
        let program = parse(source);
        assert_eq!(program.len(), 2);
    }
}
