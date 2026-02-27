/*
 * chunk.c — Bytecode chunk management
 *
 * A chunk is the unit of compiled code: a dynamic array of bytecode bytes
 * paired with a constant pool. Each function compiles to one chunk.
 *
 * Bytecode format:
 *   Instructions are variable-length: one opcode byte followed by zero
 *   or more operand bytes. Most operands are single bytes (constant pool
 *   indices, local variable slots). Jump offsets are two bytes (big-endian
 *   uint16), giving a maximum jump distance of 65535 bytes.
 *
 * Memory management:
 *   Both the code array and constant pool use the dynamic array pattern:
 *   start at capacity 0, grow by doubling when full. All allocations go
 *   through adam_gc_allocate() so the GC can track total memory usage.
 */

#include "adam/common.h"
#include "adam/chunk.h"
#include "adam/gc.h"
#include "adam/vm.h"

void adam_chunk_init(Chunk* chunk) {
    chunk->count = 0;
    chunk->capacity = 0;
    chunk->code = NULL;
    chunk->lines = NULL;
    chunk->constant_count = 0;
    chunk->constant_capacity = 0;
    chunk->constants = NULL;
}

void adam_chunk_write(VM* vm, Chunk* chunk, uint8_t byte, int line) {
    if (chunk->count >= chunk->capacity) {
        int old_capacity = chunk->capacity;
        chunk->capacity = GROW_CAPACITY(old_capacity);
        chunk->code = GROW_ARRAY(vm, uint8_t, chunk->code,
                                 old_capacity, chunk->capacity);
        chunk->lines = GROW_ARRAY(vm, int, chunk->lines,
                                  old_capacity, chunk->capacity);
    }
    chunk->code[chunk->count] = byte;
    chunk->lines[chunk->count] = line;
    chunk->count++;
}

int adam_chunk_add_constant(VM* vm, Chunk* chunk, Value value) {
    /* Push value onto VM stack temporarily to protect it from GC.
     * The GROW_ARRAY call below may trigger a collection. */
    adam_vm_push(vm, value);

    if (chunk->constant_count >= chunk->constant_capacity) {
        int old_capacity = chunk->constant_capacity;
        chunk->constant_capacity = GROW_CAPACITY(old_capacity);
        chunk->constants = GROW_ARRAY(vm, Value, chunk->constants,
                                      old_capacity, chunk->constant_capacity);
    }
    chunk->constants[chunk->constant_count] = value;
    chunk->constant_count++;

    adam_vm_pop(vm);
    return chunk->constant_count - 1;
}

void adam_chunk_free(VM* vm, Chunk* chunk) {
    FREE_ARRAY(vm, uint8_t, chunk->code, chunk->capacity);
    FREE_ARRAY(vm, int, chunk->lines, chunk->capacity);
    FREE_ARRAY(vm, Value, chunk->constants, chunk->constant_capacity);
    adam_chunk_init(chunk);
}
