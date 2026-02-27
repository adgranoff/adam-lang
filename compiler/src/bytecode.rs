//! Bytecode instruction definitions and serialization.
//!
//! This module defines the bytecode instruction set shared between the
//! Rust compiler and the C VM. The instruction set matches the OpCode
//! enum in `vm/include/adam/chunk.h` exactly — the compiler produces
//! bytes that the VM directly interprets.
//!
//! Binary format (.adamb):
//!   - 4 bytes: magic "ADAM"
//!   - 1 byte: version (currently 1)
//!   - Constant pool (length-prefixed)
//!   - Code section (length-prefixed)

use std::fs;
use std::io;

/// Opcodes matching the C VM's OpCode enum. The discriminant values
/// MUST match the C enum order exactly.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Const = 0,
    Nil,
    True,
    False,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Neg,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Not,
    LoadLocal,
    StoreLocal,
    LoadGlobal,
    StoreGlobal,
    LoadUpvalue,
    StoreUpvalue,
    CloseUpvalue,
    Jump,
    JumpIfFalse,
    Loop,
    Call,
    Closure,
    Return,
    ArrayNew,
    ArrayGet,
    ArraySet,
    ArrayLen,
    StructNew,
    StructGet,
    StructSet,
    Match,
    Print,
    Pop,
}

/// A constant value in the bytecode constant pool.
#[derive(Debug, Clone)]
pub enum Constant {
    Int(i32),
    Float(f64),
    String(String),
    Function {
        name: Option<String>,
        arity: u8,
        upvalue_count: u8,
        code: Vec<u8>,
        constants: Vec<Constant>,
    },
}

/// A compiled program ready for serialization.
#[derive(Debug)]
pub struct CompiledProgram {
    pub constants: Vec<Constant>,
    pub code: Vec<u8>,
}

/// Write compiled bytecode to a .adamb file.
pub fn write_bytecode(path: &str, program: &CompiledProgram) -> io::Result<()> {
    let mut bytes = Vec::new();

    // Magic number
    bytes.extend_from_slice(b"ADAM");
    // Version
    bytes.push(1);

    // Constant pool (simplified for now)
    let const_count = program.constants.len() as u32;
    bytes.extend_from_slice(&const_count.to_le_bytes());
    for constant in &program.constants {
        write_constant(&mut bytes, constant);
    }

    // Code section
    let code_len = program.code.len() as u32;
    bytes.extend_from_slice(&code_len.to_le_bytes());
    bytes.extend_from_slice(&program.code);

    fs::write(path, bytes)
}

fn write_constant(bytes: &mut Vec<u8>, constant: &Constant) {
    match constant {
        Constant::Int(n) => {
            bytes.push(0); // type tag
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        Constant::Float(n) => {
            bytes.push(1);
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        Constant::String(s) => {
            bytes.push(2);
            let len = s.len() as u32;
            bytes.extend_from_slice(&len.to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        }
        Constant::Function {
            name,
            arity,
            upvalue_count,
            code,
            constants,
        } => {
            bytes.push(3);
            // Name
            match name {
                Some(n) => {
                    bytes.push(1);
                    let len = n.len() as u32;
                    bytes.extend_from_slice(&len.to_le_bytes());
                    bytes.extend_from_slice(n.as_bytes());
                }
                None => bytes.push(0),
            }
            bytes.push(*arity);
            bytes.push(*upvalue_count);
            // Nested constants
            let const_count = constants.len() as u32;
            bytes.extend_from_slice(&const_count.to_le_bytes());
            for c in constants {
                write_constant(bytes, c);
            }
            // Code
            let code_len = code.len() as u32;
            bytes.extend_from_slice(&code_len.to_le_bytes());
            bytes.extend_from_slice(code);
        }
    }
}
