/*
 * debug.h — Bytecode disassembler
 *
 * Pretty-prints bytecode instructions for debugging and inspection.
 * Shows opcode names, operand values, constant pool contents, and
 * source line numbers. Essential during development and useful as
 * a learning tool for understanding the compilation output.
 */

#pragma once

#include "adam/chunk.h"

/* Disassemble an entire chunk (all instructions). */
void adam_disassemble_chunk(Chunk* chunk, const char* name);

/* Disassemble a single instruction at the given offset.
 * Returns the offset of the NEXT instruction. */
int adam_disassemble_instruction(Chunk* chunk, int offset);
