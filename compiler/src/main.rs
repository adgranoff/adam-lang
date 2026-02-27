//! Adam compiler CLI entry point.
//!
//! Usage:
//!   adamc compile <input.adam> -o <output.adamb>
//!   adamc check <input.adam>    (type-check only)
//!   adamc parse <input.adam>    (dump AST)
//!   adamc lex <input.adam>      (dump tokens)

use adam_compiler::{lexer::Lexer, parser::Parser};
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: adamc <command> <file.adam>");
        eprintln!("Commands: lex, parse, check, compile");
        process::exit(64);
    }

    let command = &args[1];
    let filename = &args[2];

    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading '{}': {}", filename, e);
            process::exit(74);
        }
    };

    match command.as_str() {
        "lex" => {
            let mut lexer = Lexer::new(&source);
            let tokens = lexer.scan_tokens();
            for token in &tokens {
                println!("{:?}", token);
            }
            if !lexer.errors().is_empty() {
                for err in lexer.errors() {
                    eprintln!("{}", err);
                }
                process::exit(65);
            }
        }
        "parse" => {
            let program = lex_and_parse(&source);
            for stmt in &program {
                println!("{:#?}", stmt);
            }
        }
        "check" => {
            let program = lex_and_parse(&source);
            let errors = adam_compiler::types::check(&program);
            if errors.is_empty() {
                println!("No type errors.");
            } else {
                for err in &errors {
                    eprintln!("{}", err);
                }
                process::exit(65);
            }
        }
        "compile" => {
            let program = lex_and_parse(&source);

            // Run type checker (errors are warnings — compilation proceeds).
            let type_errors = adam_compiler::types::check(&program);
            for err in &type_errors {
                eprintln!("warning: {}", err);
            }

            // Run autograd pass — transforms grad(f) calls into gradient functions.
            let program = adam_compiler::autograd::transform(&program);

            let bytecode = match adam_compiler::compiler::compile(&program) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Compilation error: {}", e);
                    process::exit(65);
                }
            };
            let output = if args.len() > 4 && args[3] == "-o" {
                args[4].clone()
            } else {
                filename.replace(".adam", ".adamb")
            };
            match adam_compiler::bytecode::write_bytecode(&output, &bytecode) {
                Ok(()) => println!("Compiled to {}", output),
                Err(e) => {
                    eprintln!("Error writing output: {}", e);
                    process::exit(74);
                }
            }
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            process::exit(64);
        }
    }
}

/// Lex and parse source code, exiting on errors.
fn lex_and_parse(source: &str) -> adam_compiler::ast::Program {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.scan_tokens();
    if !lexer.errors().is_empty() {
        for err in lexer.errors() {
            eprintln!("{}", err);
        }
        process::exit(65);
    }
    let mut parser = Parser::new(tokens);
    match parser.parse() {
        Ok(program) => program,
        Err(errors) => {
            for err in &errors {
                eprintln!("{}", err);
            }
            process::exit(65);
        }
    }
}
