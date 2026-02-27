/*
 * main.c — CLI entry point for the Adam VM
 *
 * Supports two modes:
 *   1. Run a .adamb bytecode file: adam-vm <file.adamb>
 *   2. Run built-in test programs: adam-vm --test
 *   3. Disassemble bytecode: adam-vm --disasm <file.adamb>
 *
 * The --test flag runs hand-crafted bytecode programs that verify the
 * VM works correctly before the compiler exists. This is the Phase 1
 * verification step.
 */

#include "adam/common.h"
#include "adam/vm.h"
#include "adam/chunk.h"
#include "adam/value.h"
#include "adam/object.h"
#include "adam/debug.h"
#include "adam/gc.h"
#include "adam/native.h"

#include <stdlib.h>
#include <string.h>

/* ── Bytecode file loader (.adamb) ─────────────────────────────────── */

/*
 * Binary format:
 *   4 bytes: magic "ADAM"
 *   1 byte:  version (1)
 *   u32le:   constant_count
 *   [constants...]
 *   u32le:   code_length
 *   [code bytes...]
 *
 * Constant encoding:
 *   0 = Int(i32le)
 *   1 = Float(f64le)
 *   2 = String(u32le length, bytes)
 *   3 = Function(has_name, [name], arity, upvalue_count, constants, code)
 */

typedef struct {
    const uint8_t* data;
    size_t length;
    size_t pos;
} BytecodeReader;

static uint8_t read_u8(BytecodeReader* r) {
    if (r->pos >= r->length) {
        fprintf(stderr, "Unexpected end of bytecode.\n");
        return 0;
    }
    return r->data[r->pos++];
}

static uint32_t read_u32(BytecodeReader* r) {
    uint32_t v = 0;
    v |= (uint32_t)read_u8(r);
    v |= (uint32_t)read_u8(r) << 8;
    v |= (uint32_t)read_u8(r) << 16;
    v |= (uint32_t)read_u8(r) << 24;
    return v;
}

static int32_t read_i32(BytecodeReader* r) {
    return (int32_t)read_u32(r);
}

static double read_f64(BytecodeReader* r) {
    union { double d; uint8_t bytes[8]; } u;
    for (int i = 0; i < 8; i++) {
        u.bytes[i] = read_u8(r);
    }
    return u.d;
}

static Value load_constant(BytecodeReader* r, VM* vm);

static ObjFunction* load_function_constant(BytecodeReader* r, VM* vm) {
    ObjFunction* fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fn)); /* Protect from GC */

    /* Name */
    uint8_t has_name = read_u8(r);
    if (has_name) {
        uint32_t name_len = read_u32(r);
        char* name_buf = (char*)malloc(name_len + 1);
        for (uint32_t i = 0; i < name_len; i++) {
            name_buf[i] = (char)read_u8(r);
        }
        name_buf[name_len] = '\0';
        fn->name = adam_copy_string(vm, name_buf, name_len);
        free(name_buf);
    }

    fn->arity = read_u8(r);
    fn->upvalue_count = read_u8(r);

    /* Nested constants */
    uint32_t const_count = read_u32(r);
    for (uint32_t i = 0; i < const_count; i++) {
        Value val = load_constant(r, vm);
        adam_chunk_add_constant(vm, &fn->chunk, val);
    }

    /* Code */
    uint32_t code_len = read_u32(r);
    for (uint32_t i = 0; i < code_len; i++) {
        adam_chunk_write(vm, &fn->chunk, read_u8(r), 1);
    }

    adam_vm_pop(vm); /* Pop GC protection */
    return fn;
}

static Value load_constant(BytecodeReader* r, VM* vm) {
    uint8_t tag = read_u8(r);
    switch (tag) {
    case 0: { /* Int */
        int32_t n = read_i32(r);
        return INT_VAL(n);
    }
    case 1: { /* Float */
        double n = read_f64(r);
        return adam_float_to_value(n);
    }
    case 2: { /* String */
        uint32_t len = read_u32(r);
        char* buf = (char*)malloc(len + 1);
        for (uint32_t i = 0; i < len; i++) {
            buf[i] = (char)read_u8(r);
        }
        buf[len] = '\0';
        ObjString* s = adam_copy_string(vm, buf, len);
        free(buf);
        return OBJ_VAL(s);
    }
    case 3: { /* Function */
        ObjFunction* fn = load_function_constant(r, vm);
        return OBJ_VAL(fn);
    }
    default:
        fprintf(stderr, "Unknown constant type tag %d.\n", tag);
        return NIL_VAL;
    }
}

static ObjFunction* load_bytecode(VM* vm, const uint8_t* data, size_t length) {
    BytecodeReader reader = { data, length, 0 };
    BytecodeReader* r = &reader;

    /* Magic number "ADAM" */
    if (length < 5) {
        fprintf(stderr, "File too small to be valid bytecode.\n");
        return NULL;
    }
    if (r->data[0] != 'A' || r->data[1] != 'D' ||
        r->data[2] != 'A' || r->data[3] != 'M') {
        fprintf(stderr, "Invalid magic number (expected 'ADAM').\n");
        return NULL;
    }
    r->pos = 4;

    uint8_t version = read_u8(r);
    if (version != 1) {
        fprintf(stderr, "Unsupported bytecode version %d.\n", version);
        return NULL;
    }

    /* Register native functions so globals like 'print' are available. */
    adam_register_natives(vm);

    /* Build the top-level script function. */
    ObjFunction* script = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(script)); /* GC protection */

    /* Load constants */
    uint32_t const_count = read_u32(r);
    for (uint32_t i = 0; i < const_count; i++) {
        Value val = load_constant(r, vm);
        adam_chunk_add_constant(vm, &script->chunk, val);
    }

    /* Load code */
    uint32_t code_len = read_u32(r);
    for (uint32_t i = 0; i < code_len; i++) {
        adam_chunk_write(vm, &script->chunk, read_u8(r), 1);
    }

    adam_vm_pop(vm);
    return script;
}

/* ── Helper: emit bytecode into a chunk ────────────────────────────── */

static void emit_byte(VM* vm, Chunk* chunk, uint8_t byte, int line) {
    adam_chunk_write(vm, chunk, byte, line);
}

static void emit_bytes(VM* vm, Chunk* chunk, uint8_t b1, uint8_t b2, int line) {
    adam_chunk_write(vm, chunk, b1, line);
    adam_chunk_write(vm, chunk, b2, line);
}

static uint8_t make_constant(VM* vm, Chunk* chunk, Value value) {
    int constant = adam_chunk_add_constant(vm, chunk, value);
    if (constant > UINT8_MAX) {
        fprintf(stderr, "Too many constants in one chunk.\n");
        exit(1);
    }
    return (uint8_t)constant;
}

static void emit_constant(VM* vm, Chunk* chunk, Value value, int line) {
    emit_bytes(vm, chunk, OP_CONST, make_constant(vm, chunk, value), line);
}

static void emit_jump(VM* vm, Chunk* chunk, uint8_t op, int line) {
    /* Emit jump with placeholder offset (to be patched). */
    emit_byte(vm, chunk, op, line);
    emit_byte(vm, chunk, 0xff, line);
    emit_byte(vm, chunk, 0xff, line);
}

static void patch_jump(Chunk* chunk, int offset) {
    /* -2 for the two offset bytes themselves. */
    int jump = chunk->count - offset - 2;
    if (jump > UINT16_MAX) {
        fprintf(stderr, "Jump too large.\n");
        exit(1);
    }
    chunk->code[offset] = (jump >> 8) & 0xff;
    chunk->code[offset + 1] = jump & 0xff;
}

/* ── Test: Arithmetic (1 + 2 * 3 = 7) ─────────────────────────────── */

static void test_arithmetic(VM* vm) {
    printf("--- Test: Arithmetic ---\n");

    ObjFunction* fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fn)); /* Protect from GC */

    Chunk* chunk = &fn->chunk;

    /* 1 + 2 * 3 = 7 */
    emit_constant(vm, chunk, INT_VAL(1), 1);
    emit_constant(vm, chunk, INT_VAL(2), 1);
    emit_constant(vm, chunk, INT_VAL(3), 1);
    emit_byte(vm, chunk, OP_MUL, 1);
    emit_byte(vm, chunk, OP_ADD, 1);
    emit_byte(vm, chunk, OP_PRINT, 1);

    /* 10.5 / 2.0 = 5.25 */
    emit_constant(vm, chunk, FLOAT_VAL(10.5), 2);
    emit_constant(vm, chunk, FLOAT_VAL(2.0), 2);
    emit_byte(vm, chunk, OP_DIV, 2);
    emit_byte(vm, chunk, OP_PRINT, 2);

    /* 2 ** 10 = 1024 */
    emit_constant(vm, chunk, INT_VAL(2), 3);
    emit_constant(vm, chunk, INT_VAL(10), 3);
    emit_byte(vm, chunk, OP_POW, 3);
    emit_byte(vm, chunk, OP_PRINT, 3);

    emit_byte(vm, chunk, OP_NIL, 4);
    emit_byte(vm, chunk, OP_RETURN, 4);

    adam_vm_pop(vm); /* Unprotect fn */

    ObjClosure* closure = adam_new_closure(vm, fn);
    InterpretResult result = adam_vm_interpret(vm, closure);
    printf("Result: %s\n\n", result == INTERPRET_OK ? "OK" : "ERROR");
}

/* ── Test: Fibonacci via recursive function calls ──────────────────── */

static void test_fibonacci(VM* vm) {
    printf("--- Test: Fibonacci ---\n");

    /* Build the fibonacci function:
     *   fn fib(n) {
     *       if (n <= 1) return n;
     *       return fib(n-1) + fib(n-2);
     *   }
     */
    ObjFunction* fib_fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fib_fn));
    fib_fn->arity = 1;
    fib_fn->name = adam_copy_string(vm, "fib", 3);
    Chunk* fc = &fib_fn->chunk;

    /* if (n <= 1) return n */
    emit_bytes(vm, fc, OP_LOAD_LOCAL, 1, 1);        /* load n (slot 1, slot 0 is the fn itself) */
    emit_constant(vm, fc, INT_VAL(1), 1);
    emit_byte(vm, fc, OP_LTE, 1);
    emit_jump(vm, fc, OP_JUMP_IF_FALSE, 1);         /* jump if n > 1 */
    int then_offset = fc->count - 2;
    emit_byte(vm, fc, OP_POP, 1);                   /* pop condition */
    emit_bytes(vm, fc, OP_LOAD_LOCAL, 1, 1);         /* load n */
    emit_byte(vm, fc, OP_RETURN, 1);
    patch_jump(fc, then_offset);
    emit_byte(vm, fc, OP_POP, 1);                   /* pop condition */

    /* fib(n-1) */
    uint8_t fib_name_idx = make_constant(vm, fc, OBJ_VAL(fib_fn->name));
    emit_bytes(vm, fc, OP_LOAD_GLOBAL, fib_name_idx, 2);
    emit_bytes(vm, fc, OP_LOAD_LOCAL, 1, 2);         /* load n */
    emit_constant(vm, fc, INT_VAL(1), 2);
    emit_byte(vm, fc, OP_SUB, 2);
    emit_bytes(vm, fc, OP_CALL, 1, 2);

    /* fib(n-2) */
    emit_bytes(vm, fc, OP_LOAD_GLOBAL, fib_name_idx, 3);
    emit_bytes(vm, fc, OP_LOAD_LOCAL, 1, 3);         /* load n */
    emit_constant(vm, fc, INT_VAL(2), 3);
    emit_byte(vm, fc, OP_SUB, 3);
    emit_bytes(vm, fc, OP_CALL, 1, 3);

    /* return fib(n-1) + fib(n-2) */
    emit_byte(vm, fc, OP_ADD, 4);
    emit_byte(vm, fc, OP_RETURN, 4);

    adam_vm_pop(vm);

    /* Build the main script:
     *   global fib = <closure>
     *   print fib(20)
     */
    ObjFunction* main_fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(main_fn));
    Chunk* mc = &main_fn->chunk;

    /* Define fib as a global */
    uint8_t fib_const = make_constant(vm, mc, OBJ_VAL(fib_fn));
    uint8_t fib_name_const = make_constant(vm, mc,
        OBJ_VAL(adam_copy_string(vm, "fib", 3)));

    emit_bytes(vm, mc, OP_CLOSURE, fib_const, 1);
    emit_bytes(vm, mc, OP_STORE_GLOBAL, fib_name_const, 1);
    emit_byte(vm, mc, OP_POP, 1);

    /* Call fib(20) and print */
    emit_bytes(vm, mc, OP_LOAD_GLOBAL, fib_name_const, 2);
    emit_constant(vm, mc, INT_VAL(20), 2);
    emit_bytes(vm, mc, OP_CALL, 1, 2);
    emit_byte(vm, mc, OP_PRINT, 2);

    emit_byte(vm, mc, OP_NIL, 3);
    emit_byte(vm, mc, OP_RETURN, 3);

    adam_vm_pop(vm);

    /* Disassemble and run */
    adam_disassemble_chunk(&fib_fn->chunk, "fib");
    adam_disassemble_chunk(&main_fn->chunk, "script");

    ObjClosure* closure = adam_new_closure(vm, main_fn);
    InterpretResult result = adam_vm_interpret(vm, closure);
    printf("Result: %s (expected output: 6765)\n\n",
           result == INTERPRET_OK ? "OK" : "ERROR");
}

/* ── Test: String concatenation ────────────────────────────────────── */

static void test_strings(VM* vm) {
    printf("--- Test: String Concatenation ---\n");

    ObjFunction* fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fn));
    Chunk* chunk = &fn->chunk;

    emit_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, "hello", 5)), 1);
    emit_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, " ", 1)), 1);
    emit_byte(vm, chunk, OP_ADD, 1);
    emit_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, "world", 5)), 1);
    emit_byte(vm, chunk, OP_ADD, 1);
    emit_byte(vm, chunk, OP_PRINT, 1);

    emit_byte(vm, chunk, OP_NIL, 2);
    emit_byte(vm, chunk, OP_RETURN, 2);

    adam_vm_pop(vm);

    ObjClosure* closure = adam_new_closure(vm, fn);
    InterpretResult result = adam_vm_interpret(vm, closure);
    printf("Result: %s (expected output: hello world)\n\n",
           result == INTERPRET_OK ? "OK" : "ERROR");
}

/* ── Test: Array operations ────────────────────────────────────────── */

static void test_arrays(VM* vm) {
    printf("--- Test: Arrays ---\n");

    ObjFunction* fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fn));
    Chunk* chunk = &fn->chunk;

    /* Create array [10, 20, 30] */
    emit_constant(vm, chunk, INT_VAL(10), 1);
    emit_constant(vm, chunk, INT_VAL(20), 1);
    emit_constant(vm, chunk, INT_VAL(30), 1);
    emit_bytes(vm, chunk, OP_ARRAY_NEW, 3, 1);
    emit_byte(vm, chunk, OP_PRINT, 1);   /* print [10, 20, 30] */

    /* Recreate array and get element at index 1 */
    emit_constant(vm, chunk, INT_VAL(10), 2);
    emit_constant(vm, chunk, INT_VAL(20), 2);
    emit_constant(vm, chunk, INT_VAL(30), 2);
    emit_bytes(vm, chunk, OP_ARRAY_NEW, 3, 2);
    emit_constant(vm, chunk, INT_VAL(1), 2);
    emit_byte(vm, chunk, OP_ARRAY_GET, 2);
    emit_byte(vm, chunk, OP_PRINT, 2);   /* print 20 */

    emit_byte(vm, chunk, OP_NIL, 3);
    emit_byte(vm, chunk, OP_RETURN, 3);

    adam_vm_pop(vm);

    ObjClosure* closure = adam_new_closure(vm, fn);
    InterpretResult result = adam_vm_interpret(vm, closure);
    printf("Result: %s (expected: [10, 20, 30] then 20)\n\n",
           result == INTERPRET_OK ? "OK" : "ERROR");
}

/* ── Test: Global variables and conditionals ───────────────────────── */

static void test_globals(VM* vm) {
    printf("--- Test: Globals & Conditionals ---\n");

    ObjFunction* fn = adam_new_function(vm);
    adam_vm_push(vm, OBJ_VAL(fn));
    Chunk* chunk = &fn->chunk;

    /* x = 42 */
    uint8_t x_name = make_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, "x", 1)));
    emit_constant(vm, chunk, INT_VAL(42), 1);
    emit_bytes(vm, chunk, OP_STORE_GLOBAL, x_name, 1);
    emit_byte(vm, chunk, OP_POP, 1);

    /* if (x == 42) print "correct" else print "wrong" */
    emit_bytes(vm, chunk, OP_LOAD_GLOBAL, x_name, 2);
    emit_constant(vm, chunk, INT_VAL(42), 2);
    emit_byte(vm, chunk, OP_EQ, 2);
    emit_jump(vm, chunk, OP_JUMP_IF_FALSE, 2);
    int else_offset = chunk->count - 2;
    emit_byte(vm, chunk, OP_POP, 2);
    emit_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, "correct", 7)), 3);
    emit_byte(vm, chunk, OP_PRINT, 3);
    emit_jump(vm, chunk, OP_JUMP, 3);
    int end_offset = chunk->count - 2;
    patch_jump(chunk, else_offset);
    emit_byte(vm, chunk, OP_POP, 4);
    emit_constant(vm, chunk,
        OBJ_VAL(adam_copy_string(vm, "wrong", 5)), 4);
    emit_byte(vm, chunk, OP_PRINT, 4);
    patch_jump(chunk, end_offset);

    emit_byte(vm, chunk, OP_NIL, 5);
    emit_byte(vm, chunk, OP_RETURN, 5);

    adam_vm_pop(vm);

    ObjClosure* closure = adam_new_closure(vm, fn);
    InterpretResult result = adam_vm_interpret(vm, closure);
    printf("Result: %s (expected output: correct)\n\n",
           result == INTERPRET_OK ? "OK" : "ERROR");
}

/* ── Entry point ───────────────────────────────────────────────────── */

static void run_all_tests(void) {
    VM vm;
    adam_vm_init(&vm);
    test_arithmetic(&vm);
    adam_vm_free(&vm);

    adam_vm_init(&vm);
    test_strings(&vm);
    adam_vm_free(&vm);

    adam_vm_init(&vm);
    test_arrays(&vm);
    adam_vm_free(&vm);

    adam_vm_init(&vm);
    test_globals(&vm);
    adam_vm_free(&vm);

    adam_vm_init(&vm);
    test_fibonacci(&vm);
    adam_vm_free(&vm);

    printf("=== All tests complete ===\n");
}

int main(int argc, char* argv[]) {
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        run_all_tests();
        return 0;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: adam-vm [--test | <file.adamb>]\n");
        return 64;
    }

    /* ── Load and execute .adamb bytecode file ──────────────────────── */
    const char* path = argv[1];
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Could not open '%s'.\n", path);
        return 74;
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    uint8_t* buffer = (uint8_t*)malloc(file_size);
    if (!buffer) {
        fprintf(stderr, "Not enough memory to read '%s'.\n", path);
        fclose(file);
        return 74;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    fclose(file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Could not read '%s'.\n", path);
        free(buffer);
        return 74;
    }

    VM vm;
    adam_vm_init(&vm);

    ObjFunction* script = load_bytecode(&vm, buffer, file_size);
    free(buffer);
    if (!script) {
        adam_vm_free(&vm);
        return 65;
    }

    /* Wrap in a closure and execute. */
    adam_vm_push(&vm, OBJ_VAL(script));
    ObjClosure* closure = adam_new_closure(&vm, script);
    adam_vm_pop(&vm);
    adam_vm_push(&vm, OBJ_VAL(closure));

    InterpretResult result = adam_vm_interpret(&vm, closure);
    adam_vm_free(&vm);

    return (result == INTERPRET_OK) ? 0 : 70;
}
