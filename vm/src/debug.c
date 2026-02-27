/*
 * debug.c — Bytecode disassembler
 *
 * Pretty-prints bytecode for debugging and inspection. Each instruction
 * is shown with its offset, source line number, opcode name, and operands.
 * Line numbers use `|` for instructions on the same line as the previous
 * one, making it easy to see statement boundaries.
 */

#include "adam/common.h"
#include "adam/debug.h"
#include "adam/value.h"
#include "adam/object.h"

/* ── Instruction printers ──────────────────────────────────────────── */

static int simple_instruction(const char* name, int offset) {
    printf("%s\n", name);
    return offset + 1;
}

static int constant_instruction(const char* name, Chunk* chunk, int offset) {
    uint8_t constant = chunk->code[offset + 1];
    printf("%-20s %4d '", name, constant);
    adam_print_value(chunk->constants[constant]);
    printf("'\n");
    return offset + 2;
}

static int byte_instruction(const char* name, Chunk* chunk, int offset) {
    uint8_t slot = chunk->code[offset + 1];
    printf("%-20s %4d\n", name, slot);
    return offset + 2;
}

static int jump_instruction(const char* name, int sign, Chunk* chunk,
                            int offset) {
    uint16_t jump = (uint16_t)(chunk->code[offset + 1] << 8);
    jump |= chunk->code[offset + 2];
    printf("%-20s %4d -> %d\n", name, offset, offset + 3 + sign * jump);
    return offset + 3;
}

static int struct_instruction(const char* name, Chunk* chunk, int offset) {
    uint8_t name_idx = chunk->code[offset + 1];
    uint8_t field_count = chunk->code[offset + 2];
    printf("%-20s name=", name);
    adam_print_value(chunk->constants[name_idx]);
    printf(" fields=%d\n", field_count);
    return offset + 3;
}

static int match_instruction(const char* name, Chunk* chunk, int offset) {
    uint8_t tag_idx = chunk->code[offset + 1];
    uint16_t jump = (uint16_t)(chunk->code[offset + 2] << 8);
    jump |= chunk->code[offset + 3];
    printf("%-20s tag=", name);
    adam_print_value(chunk->constants[tag_idx]);
    printf(" else -> %d\n", offset + 4 + jump);
    return offset + 4;
}

/* ── Public API ────────────────────────────────────────────────────── */

void adam_disassemble_chunk(Chunk* chunk, const char* name) {
    printf("== %s ==\n", name);
    for (int offset = 0; offset < chunk->count; ) {
        offset = adam_disassemble_instruction(chunk, offset);
    }
}

int adam_disassemble_instruction(Chunk* chunk, int offset) {
    printf("%04d ", offset);

    /* Show line number, or `|` if same as previous instruction. */
    if (offset > 0 && chunk->lines[offset] == chunk->lines[offset - 1]) {
        printf("   | ");
    } else {
        printf("%4d ", chunk->lines[offset]);
    }

    uint8_t instruction = chunk->code[offset];
    switch (instruction) {
    case OP_CONST:        return constant_instruction("OP_CONST", chunk, offset);
    case OP_NIL:          return simple_instruction("OP_NIL", offset);
    case OP_TRUE:         return simple_instruction("OP_TRUE", offset);
    case OP_FALSE:        return simple_instruction("OP_FALSE", offset);
    case OP_ADD:          return simple_instruction("OP_ADD", offset);
    case OP_SUB:          return simple_instruction("OP_SUB", offset);
    case OP_MUL:          return simple_instruction("OP_MUL", offset);
    case OP_DIV:          return simple_instruction("OP_DIV", offset);
    case OP_MOD:          return simple_instruction("OP_MOD", offset);
    case OP_POW:          return simple_instruction("OP_POW", offset);
    case OP_NEG:          return simple_instruction("OP_NEG", offset);
    case OP_EQ:           return simple_instruction("OP_EQ", offset);
    case OP_NEQ:          return simple_instruction("OP_NEQ", offset);
    case OP_LT:           return simple_instruction("OP_LT", offset);
    case OP_GT:           return simple_instruction("OP_GT", offset);
    case OP_LTE:          return simple_instruction("OP_LTE", offset);
    case OP_GTE:          return simple_instruction("OP_GTE", offset);
    case OP_NOT:          return simple_instruction("OP_NOT", offset);
    case OP_LOAD_LOCAL:   return byte_instruction("OP_LOAD_LOCAL", chunk, offset);
    case OP_STORE_LOCAL:  return byte_instruction("OP_STORE_LOCAL", chunk, offset);
    case OP_LOAD_GLOBAL:  return constant_instruction("OP_LOAD_GLOBAL", chunk, offset);
    case OP_STORE_GLOBAL: return constant_instruction("OP_STORE_GLOBAL", chunk, offset);
    case OP_LOAD_UPVALUE: return byte_instruction("OP_LOAD_UPVALUE", chunk, offset);
    case OP_STORE_UPVALUE:return byte_instruction("OP_STORE_UPVALUE", chunk, offset);
    case OP_CLOSE_UPVALUE:return simple_instruction("OP_CLOSE_UPVALUE", offset);
    case OP_JUMP:         return jump_instruction("OP_JUMP", 1, chunk, offset);
    case OP_JUMP_IF_FALSE:return jump_instruction("OP_JUMP_IF_FALSE", 1, chunk, offset);
    case OP_LOOP:         return jump_instruction("OP_LOOP", -1, chunk, offset);
    case OP_CALL:         return byte_instruction("OP_CALL", chunk, offset);
    case OP_CLOSURE: {
        offset++;
        uint8_t constant = chunk->code[offset++];
        printf("%-20s %4d ", "OP_CLOSURE", constant);
        adam_print_value(chunk->constants[constant]);
        printf("\n");

        ObjFunction* function = AS_FUNCTION(chunk->constants[constant]);
        for (int j = 0; j < function->upvalue_count; j++) {
            int is_local = chunk->code[offset++];
            int index = chunk->code[offset++];
            printf("%04d    |                     %s %d\n",
                   offset - 2, is_local ? "local" : "upvalue", index);
        }
        return offset;
    }
    case OP_RETURN:       return simple_instruction("OP_RETURN", offset);
    case OP_ARRAY_NEW:    return byte_instruction("OP_ARRAY_NEW", chunk, offset);
    case OP_ARRAY_GET:    return simple_instruction("OP_ARRAY_GET", offset);
    case OP_ARRAY_SET:    return simple_instruction("OP_ARRAY_SET", offset);
    case OP_ARRAY_LEN:    return simple_instruction("OP_ARRAY_LEN", offset);
    case OP_STRUCT_NEW:   return struct_instruction("OP_STRUCT_NEW", chunk, offset);
    case OP_STRUCT_GET:   return byte_instruction("OP_STRUCT_GET", chunk, offset);
    case OP_STRUCT_SET:   return byte_instruction("OP_STRUCT_SET", chunk, offset);
    case OP_MATCH:        return match_instruction("OP_MATCH", chunk, offset);
    case OP_PRINT:        return simple_instruction("OP_PRINT", offset);
    case OP_POP:          return simple_instruction("OP_POP", offset);
    default:
        printf("Unknown opcode %d\n", instruction);
        return offset + 1;
    }
}
