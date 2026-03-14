/*
 * object.h — Heap-allocated objects
 *
 * Values that don't fit in 8 bytes (strings, functions, arrays, etc.)
 * live on the heap as "objects." Each object starts with a common Obj
 * header containing the type tag, GC mark bit, and an intrusive linked
 * list pointer for the garbage collector to walk all live allocations.
 *
 * String interning:
 *   All ObjStrings are interned in a global hash table (vm->strings).
 *   Two strings with the same content share the same ObjString pointer,
 *   enabling O(1) equality checks via pointer comparison. The hash is
 *   computed once at creation time using FNV-1a and cached in the object.
 *
 * Object lifecycle:
 *   1. Allocated via adam_gc_allocate (tracks bytes for GC scheduling)
 *   2. Prepended to vm->objects linked list (GC can find it)
 *   3. Used by the program
 *   4. If unreachable after mark phase, freed during sweep
 */

#pragma once

#include "adam/common.h"
#include "adam/value.h"
#include "adam/chunk.h"

/* Forward declarations */
typedef struct VM VM;

/* ── Object types ──────────────────────────────────────────────────── */

typedef enum {
    OBJ_STRING,
    OBJ_FUNCTION,
    OBJ_CLOSURE,
    OBJ_UPVALUE,
    OBJ_NATIVE,
    OBJ_ARRAY,
    OBJ_STRUCT,
    OBJ_VARIANT,
    OBJ_TENSOR,
} ObjType;

/* ── Base object header ────────────────────────────────────────────── */

struct Obj {
    ObjType type;
    bool is_marked;     /* GC: false=white (unreached), true=black (survived) */
    struct Obj* next;   /* Intrusive list: vm->objects → obj1 → obj2 → ... */
};

/* ── Type checking macros ──────────────────────────────────────────── */

#define OBJ_TYPE(value)     (AS_OBJ(value)->type)
#define IS_STRING(value)    (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_STRING)
#define IS_FUNCTION(value)  (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_FUNCTION)
#define IS_CLOSURE(value)   (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_CLOSURE)
#define IS_NATIVE(value)    (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_NATIVE)
#define IS_ARRAY(value)     (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_ARRAY)
#define IS_STRUCT(value)    (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_STRUCT)
#define IS_VARIANT(value)   (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_VARIANT)
#define IS_TENSOR(value)    (IS_OBJ(value) && OBJ_TYPE(value) == OBJ_TENSOR)

/* ── Concrete object types ─────────────────────────────────────────── */

/*
 * ObjString — Immutable, interned string with cached hash.
 *
 * FNV-1a hash (Fowler–Noll–Vo): simple, fast, good distribution.
 * Offset basis: 2166136261, prime: 16777619.
 * For each byte: hash ^= byte; hash *= prime.
 */
typedef struct ObjString {
    Obj obj;
    int length;
    uint32_t hash;          /* Cached FNV-1a hash */
    char chars[];           /* Flexible array member — string data follows header */
} ObjString;

/*
 * ObjFunction — A compiled function (bytecode chunk + metadata).
 * This is the compile-time representation. At runtime, functions are
 * wrapped in ObjClosure to capture their lexical environment.
 */
typedef struct {
    Obj obj;
    int arity;              /* Number of parameters */
    int upvalue_count;      /* Number of captured variables */
    Chunk chunk;            /* The function's bytecode */
    ObjString* name;        /* Function name (NULL for top-level script) */
} ObjFunction;

/*
 * ObjUpvalue — A captured variable from an enclosing scope.
 *
 * While the variable is still on the stack, `location` points into the
 * stack. When the variable goes out of scope, we "close" the upvalue:
 * copy the value into `closed` and point `location` at `closed`.
 * This is the key mechanism that makes closures work correctly even
 * after the enclosing function returns.
 */
typedef struct ObjUpvalue {
    Obj obj;
    Value* location;        /* Points to stack slot or to `closed` below */
    Value closed;           /* Holds the value after the variable leaves the stack */
    struct ObjUpvalue* next; /* Linked list of open upvalues (sorted by slot) */
} ObjUpvalue;

/*
 * ObjClosure — A function paired with its captured environment.
 * Every function call at runtime goes through a closure, even if
 * the function captures nothing (upvalue_count == 0).
 */
typedef struct ObjClosure {
    Obj obj;
    ObjFunction* function;
    ObjUpvalue** upvalues;  /* Array of pointers to captured upvalues */
    int upvalue_count;
} ObjClosure;

/*
 * ObjNative — A native (C) function callable from Adam code.
 * Used for builtins like clock(), print(), len(), etc.
 */
typedef Value (*NativeFn)(VM* vm, int arg_count, Value* args);

typedef struct {
    Obj obj;
    NativeFn function;
    ObjString* name;        /* Name for error messages */
} ObjNative;

/*
 * ObjArray — Dynamic array of Values.
 */
typedef struct {
    Obj obj;
    int count;
    int capacity;
    Value* elements;
} ObjArray;

/*
 * ObjStruct — Named struct with fixed fields.
 * Stores field names (interned strings) alongside values so the compiler
 * can emit name-based field access without needing type information.
 */
typedef struct {
    Obj obj;
    ObjString* name;
    int field_count;
    ObjString** field_names; /* Interned field name strings */
    Value* fields;           /* Field values */
} ObjStruct;

/*
 * ObjVariant — A tagged variant (algebraic data type constructor).
 * e.g., Some(42) has tag="Some", payload=INT_VAL(42).
 */
typedef struct {
    Obj obj;
    ObjString* tag;
    Value payload;
} ObjVariant;

/*
 * ObjTensor — Multi-dimensional array of doubles.
 * Row-major layout. Shape is stored alongside data so the VM can
 * perform dimension checks at runtime for untyped tensor code.
 */
typedef struct {
    Obj obj;
    int ndim;           /* Number of dimensions */
    int* shape;         /* Shape array [d0, d1, ..., d_{ndim-1}] */
    int count;          /* Total element count (product of shape) */
    double* data;       /* Flat row-major data array */
} ObjTensor;

/* ── Extraction macros ─────────────────────────────────────────────── */

#define AS_STRING(value)    ((ObjString*)AS_OBJ(value))
#define AS_CSTRING(value)   (((ObjString*)AS_OBJ(value))->chars)
#define AS_FUNCTION(value)  ((ObjFunction*)AS_OBJ(value))
#define AS_CLOSURE(value)   ((ObjClosure*)AS_OBJ(value))
#define AS_NATIVE(value)    (((ObjNative*)AS_OBJ(value))->function)
#define AS_NATIVE_OBJ(value) ((ObjNative*)AS_OBJ(value))
#define AS_ARRAY(value)     ((ObjArray*)AS_OBJ(value))
#define AS_STRUCT(value)    ((ObjStruct*)AS_OBJ(value))
#define AS_VARIANT(value)   ((ObjVariant*)AS_OBJ(value))
#define AS_TENSOR(value)    ((ObjTensor*)AS_OBJ(value))

/* ── Allocation functions ──────────────────────────────────────────── */

ObjString*   adam_copy_string(VM* vm, const char* chars, int length);
ObjString*   adam_take_string(VM* vm, char* chars, int length);
ObjFunction* adam_new_function(VM* vm);
ObjClosure*  adam_new_closure(VM* vm, ObjFunction* function);
ObjUpvalue*  adam_new_upvalue(VM* vm, Value* slot);
ObjNative*   adam_new_native(VM* vm, NativeFn function, ObjString* name);
ObjArray*    adam_new_array(VM* vm);
void         adam_array_push(VM* vm, ObjArray* array, Value value);
ObjStruct*   adam_new_struct(VM* vm, ObjString* name, int field_count);
ObjVariant*  adam_new_variant(VM* vm, ObjString* tag, Value payload);
ObjTensor*   adam_new_tensor(VM* vm, int ndim, int* shape);
void         adam_free_object(VM* vm, Obj* object);
void         adam_print_object(Value value);
