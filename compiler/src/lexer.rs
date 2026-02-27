//! Lexer — Tokenizes Adam source code with error recovery.
//!
//! The lexer scans the source string character by character, producing a
//! vector of tokens. Key design decisions:
//!
//! - **Error recovery**: On an unexpected character, we emit an error and
//!   skip it, continuing to tokenize the rest. This lets us report multiple
//!   errors in a single pass rather than stopping at the first one.
//!
//! - **Span tracking**: Every token records its byte offset range in the
//!   source. This enables precise error messages that underline the exact
//!   problematic characters (like Rust/Elm-quality diagnostics).
//!
//! - **Keyword recognition**: After scanning an identifier, we check it
//!   against a keyword table. This is simpler than reserving keywords in
//!   the character-scanning phase and handles contextual keywords cleanly.

use crate::token::{Span, Token, TokenKind};

pub struct Lexer<'src> {
    source: &'src str,
    chars: Vec<char>,
    start: usize,    // Start of current token (byte offset)
    current: usize,  // Current position (char index)
    byte_pos: usize, // Current byte position
    tokens: Vec<Token>,
    errors: Vec<String>,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Self {
            source,
            chars: source.chars().collect(),
            start: 0,
            current: 0,
            byte_pos: 0,
            tokens: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn scan_tokens(&mut self) -> Vec<Token> {
        while !self.is_at_end() {
            self.start = self.byte_pos;
            self.scan_token();
        }
        self.tokens.push(Token::new(
            TokenKind::Eof,
            Span::new(self.byte_pos, self.byte_pos),
        ));
        self.tokens.clone()
    }

    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    fn scan_token(&mut self) {
        let c = self.advance();
        match c {
            // Whitespace — skip
            ' ' | '\t' | '\r' | '\n' => {}

            // Single-line comments
            '/' if self.peek() == '/' => {
                while !self.is_at_end() && self.peek() != '\n' {
                    self.advance();
                }
            }

            // Block comments
            '/' if self.peek() == '*' => {
                self.advance(); // consume *
                let mut depth = 1;
                while !self.is_at_end() && depth > 0 {
                    if self.peek() == '/' && self.peek_next() == '*' {
                        self.advance();
                        self.advance();
                        depth += 1;
                    } else if self.peek() == '*' && self.peek_next() == '/' {
                        self.advance();
                        self.advance();
                        depth -= 1;
                    } else {
                        self.advance();
                    }
                }
                if depth > 0 {
                    self.errors.push(format!(
                        "Unterminated block comment at byte {}",
                        self.start
                    ));
                }
            }

            // Single-character tokens
            '(' => self.add_token(TokenKind::LParen),
            ')' => self.add_token(TokenKind::RParen),
            '{' => self.add_token(TokenKind::LBrace),
            '}' => self.add_token(TokenKind::RBrace),
            '[' => self.add_token(TokenKind::LBracket),
            ']' => self.add_token(TokenKind::RBracket),
            ',' => self.add_token(TokenKind::Comma),
            ';' => self.add_token(TokenKind::Semicolon),

            // One-or-two character tokens
            '+' => self.add_token(TokenKind::Plus),
            '%' => self.add_token(TokenKind::Percent),

            '-' => {
                if self.match_char('>') {
                    self.add_token(TokenKind::Arrow);
                } else {
                    self.add_token(TokenKind::Minus);
                }
            }

            '*' => {
                if self.match_char('*') {
                    self.add_token(TokenKind::StarStar);
                } else {
                    self.add_token(TokenKind::Star);
                }
            }

            '/' => self.add_token(TokenKind::Slash),

            '=' => {
                if self.match_char('=') {
                    self.add_token(TokenKind::EqEq);
                } else if self.match_char('>') {
                    self.add_token(TokenKind::FatArrow);
                } else {
                    self.add_token(TokenKind::Eq);
                }
            }

            '!' => {
                if self.match_char('=') {
                    self.add_token(TokenKind::BangEq);
                } else {
                    self.add_token(TokenKind::Bang);
                }
            }

            '<' => {
                if self.match_char('=') {
                    self.add_token(TokenKind::LtEq);
                } else {
                    self.add_token(TokenKind::Lt);
                }
            }

            '>' => {
                if self.match_char('=') {
                    self.add_token(TokenKind::GtEq);
                } else {
                    self.add_token(TokenKind::Gt);
                }
            }

            '&' => {
                if self.match_char('&') {
                    self.add_token(TokenKind::And);
                } else {
                    self.errors.push(format!(
                        "Unexpected '&' at byte {}. Did you mean '&&'?",
                        self.start
                    ));
                }
            }

            '|' => {
                if self.match_char('|') {
                    self.add_token(TokenKind::Or);
                } else if self.match_char('>') {
                    self.add_token(TokenKind::PipeGt);
                } else {
                    self.add_token(TokenKind::Pipe);
                }
            }

            '.' => {
                if self.match_char('.') {
                    self.add_token(TokenKind::DotDot);
                } else {
                    self.add_token(TokenKind::Dot);
                }
            }

            ':' => {
                if self.match_char(':') {
                    self.add_token(TokenKind::ColonColon);
                } else {
                    self.add_token(TokenKind::Colon);
                }
            }

            // String literals
            '"' => self.string(),

            // Number literals
            c if c.is_ascii_digit() => self.number(c),

            // Identifiers and keywords
            c if c.is_alphabetic() || c == '_' => self.identifier(c),

            _ => {
                self.errors.push(format!(
                    "Unexpected character '{}' at byte {}",
                    c, self.start
                ));
            }
        }
    }

    // ── Literal scanners ─────────────────────────────────────────────

    fn string(&mut self) {
        let mut value = String::new();
        while !self.is_at_end() && self.peek() != '"' {
            let c = self.advance();
            if c == '\\' && !self.is_at_end() {
                match self.advance() {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    other => {
                        self.errors.push(format!(
                            "Unknown escape sequence '\\{}' at byte {}",
                            other, self.byte_pos
                        ));
                        value.push(other);
                    }
                }
            } else {
                value.push(c);
            }
        }

        if self.is_at_end() {
            self.errors
                .push(format!("Unterminated string at byte {}", self.start));
            return;
        }

        self.advance(); // closing "
        self.add_token(TokenKind::Str(value));
    }

    fn number(&mut self, first: char) {
        let mut s = String::new();
        s.push(first);
        let mut is_float = false;

        while !self.is_at_end() && self.peek().is_ascii_digit() {
            s.push(self.advance());
        }

        // Check for fractional part
        if !self.is_at_end() && self.peek() == '.' && self.peek_next().is_ascii_digit() {
            is_float = true;
            s.push(self.advance()); // consume .
            while !self.is_at_end() && self.peek().is_ascii_digit() {
                s.push(self.advance());
            }
        }

        // Check for exponent
        if !self.is_at_end() && (self.peek() == 'e' || self.peek() == 'E') {
            is_float = true;
            s.push(self.advance());
            if !self.is_at_end() && (self.peek() == '+' || self.peek() == '-') {
                s.push(self.advance());
            }
            while !self.is_at_end() && self.peek().is_ascii_digit() {
                s.push(self.advance());
            }
        }

        if is_float {
            match s.parse::<f64>() {
                Ok(n) => self.add_token(TokenKind::Float(n)),
                Err(_) => self.errors.push(format!(
                    "Invalid float literal '{}' at byte {}",
                    s, self.start
                )),
            }
        } else {
            match s.parse::<i64>() {
                Ok(n) => self.add_token(TokenKind::Int(n)),
                Err(_) => self.errors.push(format!(
                    "Invalid integer literal '{}' at byte {}",
                    s, self.start
                )),
            }
        }
    }

    fn identifier(&mut self, first: char) {
        let mut name = String::new();
        name.push(first);
        while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            name.push(self.advance());
        }

        let kind = match name.as_str() {
            "let" => TokenKind::Let,
            "fn" => TokenKind::Fn,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "return" => TokenKind::Return,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "type" => TokenKind::Type,
            "struct" => TokenKind::Struct,
            "impl" => TokenKind::Impl,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "while" => TokenKind::While,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            _ => TokenKind::Identifier(name),
        };
        self.add_token(kind);
    }

    // ── Character-level helpers ──────────────────────────────────────

    fn advance(&mut self) -> char {
        let c = self.chars[self.current];
        self.current += 1;
        self.byte_pos += c.len_utf8();
        c
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.chars[self.current]
        }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.chars.len() {
            '\0'
        } else {
            self.chars[self.current + 1]
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.chars[self.current] != expected {
            return false;
        }
        self.current += 1;
        self.byte_pos += expected.len_utf8();
        true
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.chars.len()
    }

    fn add_token(&mut self, kind: TokenKind) {
        self.tokens.push(Token::new(kind, Span::new(self.start, self.byte_pos)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.scan_tokens();
        assert!(lexer.errors().is_empty(), "Lexer errors: {:?}", lexer.errors());
        tokens.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_numbers() {
        assert_eq!(lex("42"), vec![TokenKind::Int(42), TokenKind::Eof]);
        assert_eq!(lex("3.14"), vec![TokenKind::Float(3.14), TokenKind::Eof]);
        assert_eq!(lex("1e10"), vec![TokenKind::Float(1e10), TokenKind::Eof]);
    }

    #[test]
    fn test_strings() {
        assert_eq!(
            lex(r#""hello""#),
            vec![TokenKind::Str("hello".into()), TokenKind::Eof]
        );
        assert_eq!(
            lex(r#""line\nbreak""#),
            vec![TokenKind::Str("line\nbreak".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex("+ - * / ** |> => -> == != <= >=");
        assert!(tokens.contains(&TokenKind::Plus));
        assert!(tokens.contains(&TokenKind::StarStar));
        assert!(tokens.contains(&TokenKind::PipeGt));
        assert!(tokens.contains(&TokenKind::FatArrow));
        assert!(tokens.contains(&TokenKind::Arrow));
    }

    #[test]
    fn test_keywords() {
        let tokens = lex("let fn if else match return type struct impl");
        assert!(tokens.contains(&TokenKind::Let));
        assert!(tokens.contains(&TokenKind::Fn));
        assert!(tokens.contains(&TokenKind::Match));
        assert!(tokens.contains(&TokenKind::Struct));
        assert!(tokens.contains(&TokenKind::Impl));
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(
            lex("foo bar_baz _x"),
            vec![
                TokenKind::Identifier("foo".into()),
                TokenKind::Identifier("bar_baz".into()),
                TokenKind::Identifier("_x".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comments() {
        assert_eq!(
            lex("42 // this is a comment\n 7"),
            vec![TokenKind::Int(42), TokenKind::Int(7), TokenKind::Eof]
        );
        assert_eq!(
            lex("1 /* block */ 2"),
            vec![TokenKind::Int(1), TokenKind::Int(2), TokenKind::Eof]
        );
    }

    #[test]
    fn test_error_recovery() {
        let mut lexer = Lexer::new("42 @ 7");
        let tokens = lexer.scan_tokens();
        assert_eq!(tokens.len(), 3); // 42, 7, EOF
        assert_eq!(lexer.errors().len(), 1); // one error for @
    }

    #[test]
    fn test_full_program() {
        let source = r#"
            fn fibonacci(n: Int) -> Int {
                if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
            }
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.scan_tokens();
        assert!(lexer.errors().is_empty());
        assert!(tokens.len() > 10);
    }
}
