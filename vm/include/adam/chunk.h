/*
 * chunk.h — Bytecode chunk layout
 *
 * A "chunk" is a contiguous block of bytecode instructions plus a constant
 * pool. Each function in an Adam program compiles to one chunk. The VM
 * executes chunks by walking the code array byte by byte.
 *
 * Instruction encoding:
 *   - Stack-based: operands are implicitly on the value stack.
 *   - Variable-length: each instruction is an opcode byte (1 byte) followed
 *     by zero or more operand bytes. Most operands are 1 byte (constant
 *     index, local slot). Jump offsets are 2 bytes (big-endian uint16).
 *   - The constant pool stores literals (numbers, strings) referenced by
 *     index from OP_CONST and similar instructions.
 *
 * Line number tracking:
 *   We store a parallel `lines` array (1:1 with code bytes) so every
 *   bytecode position maps back to a source line for error reporting.
 *   This is memory-inefficient (could use run-length encoding) but simple.
 */

#pragma once

#include "adam/common.h"
#include "adam/value.h"

/* ── Opcodes ───────────────────────────────────────────────────────── */

typedef enum {
    /* Constants and literals */
    OP_CONST,           /* [index]         Push constants[index]               */
    OP_NIL,             /*                 Push nil                            */
    OP_TRUE,            /*                 Push true                           */
    OP_FALSE,           /*                 Push false                          */

    /* Arithmetic (operate on top-of-stack) */
    OP_ADD,             /*                 a + b  (also string concatenation)  */
    OP_SUB,             /*                 a - b                               */
    OP_MUL,             /*                 a * b                               */
    OP_DIV,             /*                 a / b                               */
    OP_MOD,             /*                 a % b                               */
    OP_POW,             /*                 a ** b                              */
    OP_NEG,             /*                 -a                                  */

    /* Comparison */
    OP_EQ,              /*                 a == b                              */
    OP_NEQ,             /*                 a != b                              */
    OP_LT,              /*                 a < b                               */
    OP_GT,              /*                 a > b                               */
    OP_LTE,             /*                 a <= b                              */
    OP_GTE,             /*                 a >= b                              */

    /* Logical */
    OP_NOT,             /*                 !a                                  */

    /* Variables */
    OP_LOAD_LOCAL,      /* [slot]          Push locals[slot]                   */
    OP_STORE_LOCAL,     /* [slot]          Store top into locals[slot]         */
    OP_LOAD_GLOBAL,     /* [name_idx]      Push globals[constants[name_idx]]   */
    OP_STORE_GLOBAL,    /* [name_idx]      Store top into globals[name]        */
    OP_LOAD_UPVALUE,    /* [index]         Push upvalues[index]                */
    OP_STORE_UPVALUE,   /* [index]         Store top into upvalues[index]      */
    OP_CLOSE_UPVALUE,   /*                 Close upvalue at stack top, pop     */

    /* Control flow */
    OP_JUMP,            /* [hi][lo]        ip += offset                        */
    OP_JUMP_IF_FALSE,   /* [hi][lo]        if falsey(top) ip += offset         */
    OP_LOOP,            /* [hi][lo]        ip -= offset                        */

    /* Functions */
    OP_CALL,            /* [arg_count]     Call function with N arguments       */
    OP_CLOSURE,         /* [fn_idx] then pairs of [is_local][index]            */
    OP_RETURN,          /*                 Return top of stack to caller        */

    /* Collections */
    OP_ARRAY_NEW,       /* [count]         Pop N values, create array           */
    OP_ARRAY_GET,       /*                 Pop index, pop array, push element   */
    OP_ARRAY_SET,       /*                 Pop value, pop index, pop array, set */
    OP_ARRAY_LEN,       /*                 Pop array, push length               */

    /* Structs */
    OP_STRUCT_NEW,      /* [name_idx][field_count] Pop fields, create struct    */
    OP_STRUCT_GET,      /* [field_idx]     Pop struct, push field               */
    OP_STRUCT_SET,      /* [field_idx]     Pop value, pop struct, set field     */

    /* Pattern matching */
    OP_MATCH,           /* [tag_idx][hi][lo] Check variant tag, jump if no match */

    /* I/O and stack */
    OP_PRINT,           /*                 Pop and print value                  */
    OP_POP,             /*                 Discard top of stack                 */
} OpCode;

/* ── Chunk ─────────────────────────────────────────────────────────── */

typedef struct {
    int count;               /* Number of bytecode bytes written */
    int capacity;            /* Allocated size of code array */
    uint8_t* code;           /* Bytecode instructions */
    int* lines;              /* Source line numbers (parallel to code) */

    int constant_count;      /* Number of constants in pool */
    int constant_capacity;   /* Allocated size of constants array */
    Value* constants;        /* Constant pool (literals, function objects) */
} Chunk;

/* Forward declaration — VM is needed for GC-aware allocation */
typedef struct VM VM;

void adam_chunk_init(Chunk* chunk);
void adam_chunk_write(VM* vm, Chunk* chunk, uint8_t byte, int line);
int  adam_chunk_add_constant(VM* vm, Chunk* chunk, Value value);
void adam_chunk_free(VM* vm, Chunk* chunk);
