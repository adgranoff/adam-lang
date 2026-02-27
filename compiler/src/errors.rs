//! Rich error reporting with source spans.
//!
//! Uses miette for beautiful terminal error output with source code
//! context, underlines, and helpful suggestions.

use crate::token::Span;
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// A compiler error with source location information.
#[derive(Error, Debug, Diagnostic)]
#[error("{message}")]
pub struct CompileError {
    pub message: String,

    #[source_code]
    pub src: String,

    #[label("{label}")]
    pub span: SourceSpan,

    pub label: String,
}

impl CompileError {
    pub fn new(message: impl Into<String>, src: &str, span: Span, label: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            src: src.to_string(),
            span: (span.start, span.end - span.start).into(),
            label: label.into(),
        }
    }
}
