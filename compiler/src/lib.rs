//! Adam Compiler — Frontend and backend for the Adam programming language.
//!
//! # Compiler Pipeline
//!
//! ```text
//! Source Code (.adam)
//!     │
//!     ▼
//! ┌──────────┐
//! │  Lexer    │  Tokenizes source into a stream of tokens with spans
//! └────┬─────┘
//!      │
//!      ▼
//! ┌──────────┐
//! │  Parser   │  Pratt parser (expressions) + recursive descent (statements)
//! └────┬─────┘
//!      │
//!      ▼
//! ┌──────────┐
//! │  Types    │  Hindley-Milner type inference + checking (Phase 3)
//! └────┬─────┘
//!      │
//!      ▼
//! ┌──────────┐
//! │ Compiler  │  AST → bytecode compilation (Phase 4)
//! └────┬─────┘
//!      │
//!      ▼
//! Bytecode (.adamb)
//! ```

pub mod ast;
pub mod autograd;
pub mod bytecode;
pub mod compiler;
pub mod errors;
pub mod lexer;
pub mod parser;
pub mod token;
pub mod types;
