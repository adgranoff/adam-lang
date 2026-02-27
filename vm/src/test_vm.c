/*
 * test_vm.c — Automated tests for the Adam VM
 *
 * These tests verify core VM functionality: NaN boxing, arithmetic,
 * string interning, hash table operations, and bytecode execution.
 * Returns 0 if all pass, 1 on first failure.
 */

#include "adam/common.h"
#include "adam/vm.h"
#include "adam/value.h"
#include "adam/object.h"
#include "adam/chunk.h"
#include "adam/table.h"
#include "adam/gc.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* ── NaN boxing tests ──────────────────────────────────────────────── */

static void test_nan_boxing(void) {
    /* Nil */
    ASSERT(IS_NIL(NIL_VAL), "NIL_VAL is nil");
    ASSERT(!IS_BOOL(NIL_VAL), "NIL_VAL is not bool");
    ASSERT(!IS_INT(NIL_VAL), "NIL_VAL is not int");
    ASSERT(!IS_FLOAT(NIL_VAL), "NIL_VAL is not float");
    ASSERT(!IS_OBJ(NIL_VAL), "NIL_VAL is not obj");

    /* Booleans */
    ASSERT(IS_BOOL(TRUE_VAL), "TRUE_VAL is bool");
    ASSERT(IS_BOOL(FALSE_VAL), "FALSE_VAL is bool");
    ASSERT(AS_BOOL(TRUE_VAL) == true, "TRUE_VAL extracts to true");
    ASSERT(AS_BOOL(FALSE_VAL) == false, "FALSE_VAL extracts to false");
    ASSERT(!IS_NIL(TRUE_VAL), "TRUE_VAL is not nil");

    /* Integers */
    Value i42 = INT_VAL(42);
    ASSERT(IS_INT(i42), "INT_VAL(42) is int");
    ASSERT(AS_INT(i42) == 42, "INT_VAL(42) extracts to 42");
    ASSERT(!IS_FLOAT(i42), "INT_VAL is not float");

    Value i_neg = INT_VAL(-1);
    ASSERT(IS_INT(i_neg), "INT_VAL(-1) is int");
    ASSERT(AS_INT(i_neg) == -1, "INT_VAL(-1) extracts to -1");

    Value i_zero = INT_VAL(0);
    ASSERT(IS_INT(i_zero), "INT_VAL(0) is int");
    ASSERT(AS_INT(i_zero) == 0, "INT_VAL(0) extracts to 0");

    /* Floats */
    Value f = FLOAT_VAL(3.14);
    ASSERT(IS_FLOAT(f), "FLOAT_VAL(3.14) is float");
    ASSERT(AS_FLOAT(f) == 3.14, "FLOAT_VAL(3.14) extracts to 3.14");
    ASSERT(!IS_INT(f), "FLOAT_VAL is not int");

    Value f_zero = FLOAT_VAL(0.0);
    ASSERT(IS_FLOAT(f_zero), "FLOAT_VAL(0.0) is float");

    /* Equality */
    ASSERT(adam_values_equal(NIL_VAL, NIL_VAL), "nil == nil");
    ASSERT(adam_values_equal(TRUE_VAL, TRUE_VAL), "true == true");
    ASSERT(!adam_values_equal(TRUE_VAL, FALSE_VAL), "true != false");
    ASSERT(adam_values_equal(INT_VAL(42), INT_VAL(42)), "42 == 42");
    ASSERT(!adam_values_equal(INT_VAL(1), INT_VAL(2)), "1 != 2");
    ASSERT(adam_values_equal(FLOAT_VAL(3.14), FLOAT_VAL(3.14)), "3.14 == 3.14");
    ASSERT(!adam_values_equal(NIL_VAL, INT_VAL(0)), "nil != 0");
}

/* ── String interning tests ────────────────────────────────────────── */

static void test_string_interning(void) {
    VM vm;
    adam_vm_init(&vm);

    ObjString* a = adam_copy_string(&vm, "hello", 5);
    ObjString* b = adam_copy_string(&vm, "hello", 5);
    ObjString* c = adam_copy_string(&vm, "world", 5);

    ASSERT(a == b, "interned strings with same content share pointer");
    ASSERT(a != c, "different strings have different pointers");
    ASSERT(a->hash == b->hash, "same strings have same hash");
    ASSERT(a->length == 5, "string length is correct");
    ASSERT(strcmp(a->chars, "hello") == 0, "string content is correct");

    adam_vm_free(&vm);
}

/* ── Hash table tests ──────────────────────────────────────────────── */

static void test_table(void) {
    VM vm;
    adam_vm_init(&vm);

    Table table;
    adam_table_init(&table);

    ObjString* key1 = adam_copy_string(&vm, "key1", 4);
    ObjString* key2 = adam_copy_string(&vm, "key2", 4);
    ObjString* key3 = adam_copy_string(&vm, "key3", 4);

    /* Set and get */
    adam_table_set(&vm, &table, key1, INT_VAL(100));
    adam_table_set(&vm, &table, key2, INT_VAL(200));

    Value value;
    ASSERT(adam_table_get(&table, key1, &value), "key1 exists");
    ASSERT(AS_INT(value) == 100, "key1 == 100");
    ASSERT(adam_table_get(&table, key2, &value), "key2 exists");
    ASSERT(AS_INT(value) == 200, "key2 == 200");
    ASSERT(!adam_table_get(&table, key3, &value), "key3 does not exist");

    /* Update */
    adam_table_set(&vm, &table, key1, INT_VAL(999));
    ASSERT(adam_table_get(&table, key1, &value), "key1 still exists");
    ASSERT(AS_INT(value) == 999, "key1 updated to 999");

    /* Delete */
    ASSERT(adam_table_delete(&table, key1), "delete key1 succeeds");
    ASSERT(!adam_table_get(&table, key1, &value), "key1 deleted");
    ASSERT(adam_table_get(&table, key2, &value), "key2 unaffected");

    adam_table_free(&vm, &table);
    adam_vm_free(&vm);
}

/* ── Chunk tests ───────────────────────────────────────────────────── */

static void test_chunk(void) {
    VM vm;
    adam_vm_init(&vm);

    Chunk chunk;
    adam_chunk_init(&chunk);

    adam_chunk_write(&vm, &chunk, OP_CONST, 1);
    int idx = adam_chunk_add_constant(&vm, &chunk, INT_VAL(42));
    adam_chunk_write(&vm, &chunk, (uint8_t)idx, 1);
    adam_chunk_write(&vm, &chunk, OP_RETURN, 1);

    ASSERT(chunk.count == 3, "chunk has 3 bytes");
    ASSERT(chunk.code[0] == OP_CONST, "first byte is OP_CONST");
    ASSERT(chunk.constant_count == 1, "one constant");
    ASSERT(AS_INT(chunk.constants[0]) == 42, "constant is 42");
    ASSERT(chunk.lines[0] == 1, "line number is 1");

    adam_chunk_free(&vm, &chunk);
    adam_vm_free(&vm);
}

/* ── Entry point ───────────────────────────────────────────────────── */

int main(void) {
    printf("Running Adam VM tests...\n\n");

    test_nan_boxing();
    test_string_interning();
    test_table();
    test_chunk();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
