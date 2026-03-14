//! Token types for the Adam language.
//!
//! Each token carries its type, lexeme (the raw source text), and a span
//! indicating its position in the source. Spans enable precise error
//! reporting: we can underline the exact characters that caused an error.

use std::fmt;

/// Byte offset range in the source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Merge two spans into one that covers both.
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// All token types in the Adam language.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    True,
    False,

    // Identifiers
    Identifier(String),

    // Keywords
    Let,
    Fn,
    If,
    Else,
    Match,
    Return,
    Type,
    Struct,
    Impl,
    For,
    In,
    While,
    Break,
    Continue,

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    StarStar,   // **
    Eq,         // =
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    Gt,         // >
    LtEq,       // <=
    GtEq,       // >=
    Bang,       // !
    And,        // &&
    Or,         // ||
    Pipe,       // |
    PipeGt,     // |>
    AtAt,       // @@
    Arrow,      // ->
    FatArrow,   // =>
    Dot,        // .
    DotDot,     // ..
    Colon,      // :
    ColonColon, // ::
    Semicolon,  // ;
    Comma,      // ,

    // Delimiters
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]

    // Special
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Int(n) => write!(f, "{}", n),
            TokenKind::Float(n) => write!(f, "{}", n),
            TokenKind::Str(s) => write!(f, "\"{}\"", s),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Identifier(s) => write!(f, "{}", s),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Struct => write!(f, "struct"),
            TokenKind::Impl => write!(f, "impl"),
            TokenKind::For => write!(f, "for"),
            TokenKind::In => write!(f, "in"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::StarStar => write!(f, "**"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::BangEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::PipeGt => write!(f, "|>"),
            TokenKind::AtAt => write!(f, "@@"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::ColonColon => write!(f, "::"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::Eof => write!(f, "EOF"),
        }
    }
}
