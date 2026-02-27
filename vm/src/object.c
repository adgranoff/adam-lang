/*
 * object.c — Heap-allocated object management
 *
 * All objects that don't fit in a NaN-boxed Value live on the heap:
 * strings, functions, closures, arrays, structs, and variants. Each
 * object has a common Obj header for GC tracking.
 *
 * String interning:
 *   Every string is stored in vm->strings (a hash table acting as a set).
 *   Before creating a new ObjString, we check if an identical string
 *   already exists. If so, we return the existing pointer. This gives us:
 *     - O(1) string equality (pointer comparison)
 *     - Memory deduplication (one copy of each unique string)
 *     - Faster hash table lookups (compare hash + pointer, not chars)
 *
 * FNV-1a hash:
 *   Fowler-Noll-Vo variant 1a. For each byte: hash ^= byte, hash *= prime.
 *   The XOR-then-multiply order (variant 1a) has slightly better avalanche
 *   properties than multiply-then-XOR (variant 1). Simple, fast, and
 *   produces good distribution for typical string inputs.
 */

#include "adam/common.h"
#include "adam/object.h"
#include "adam/value.h"
#include "adam/table.h"
#include "adam/gc.h"
#include "adam/vm.h"

/* ── Internal helpers ──────────────────────────────────────────────── */

static Obj* allocate_object(VM* vm, size_t size, ObjType type) {
    Obj* object = (Obj*)adam_gc_reallocate(vm, NULL, 0, size);
    object->type = type;
    object->is_marked = false;

    /* Prepend to the VM's object list so the GC can find it. */
    object->next = vm->objects;
    vm->objects = object;

#ifdef DEBUG_LOG_GC
    printf("%p allocate %zu for type %d\n", (void*)object, size, type);
#endif

    return object;
}

#define ALLOCATE_OBJ(vm, type, obj_type) \
    (type*)allocate_object(vm, sizeof(type), obj_type)

/*
 * FNV-1a hash function.
 * Offset basis: 2166136261 (32-bit FNV offset)
 * Prime: 16777619 (32-bit FNV prime)
 */
static uint32_t hash_string(const char* key, int length) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < length; i++) {
        hash ^= (uint8_t)key[i];
        hash *= 16777619u;
    }
    return hash;
}

/* Allocate an ObjString and intern it. Assumes chars are already owned. */
static ObjString* allocate_string(VM* vm, char* chars, int length,
                                  uint32_t hash) {
    /* Allocate ObjString with space for chars (flexible array member). */
    ObjString* string = (ObjString*)allocate_object(
        vm, sizeof(ObjString) + length + 1, OBJ_STRING);
    string->length = length;
    string->hash = hash;
    memcpy(string->chars, chars, length);
    string->chars[length] = '\0';

    /* Intern: add to the string table. Push/pop to protect from GC. */
    adam_vm_push(vm, OBJ_VAL(string));
    adam_table_set(vm, &vm->strings, string, NIL_VAL);
    adam_vm_pop(vm);

    /* Free the temporary buffer if we own it */
    return string;
}

/* ── Public string API ─────────────────────────────────────────────── */

ObjString* adam_copy_string(VM* vm, const char* chars, int length) {
    uint32_t hash = hash_string(chars, length);

    /* Check if an identical string is already interned. */
    ObjString* interned = adam_table_find_string(&vm->strings, chars,
                                                  length, hash);
    if (interned != NULL) return interned;

    /* Not found — allocate a new string with its own copy of the chars. */
    char* heap_chars = ALLOCATE(vm, char, length + 1);
    memcpy(heap_chars, chars, length);
    heap_chars[length] = '\0';

    ObjString* result = allocate_string(vm, heap_chars, length, hash);
    /* Free the temporary buffer — allocate_string copied into the FAM. */
    FREE_ARRAY(vm, char, heap_chars, length + 1);
    return result;
}

ObjString* adam_take_string(VM* vm, char* chars, int length) {
    uint32_t hash = hash_string(chars, length);

    ObjString* interned = adam_table_find_string(&vm->strings, chars,
                                                  length, hash);
    if (interned != NULL) {
        /* Already interned — free the caller's buffer and return existing. */
        FREE_ARRAY(vm, char, chars, length + 1);
        return interned;
    }

    ObjString* result = allocate_string(vm, chars, length, hash);
    FREE_ARRAY(vm, char, chars, length + 1);
    return result;
}

/* ── Function objects ──────────────────────────────────────────────── */

ObjFunction* adam_new_function(VM* vm) {
    ObjFunction* function = ALLOCATE_OBJ(vm, ObjFunction, OBJ_FUNCTION);
    function->arity = 0;
    function->upvalue_count = 0;
    function->name = NULL;
    adam_chunk_init(&function->chunk);
    return function;
}

ObjClosure* adam_new_closure(VM* vm, ObjFunction* function) {
    /* Allocate upvalue pointer array. */
    ObjUpvalue** upvalues = ALLOCATE(vm, ObjUpvalue*, function->upvalue_count);
    for (int i = 0; i < function->upvalue_count; i++) {
        upvalues[i] = NULL;
    }

    ObjClosure* closure = ALLOCATE_OBJ(vm, ObjClosure, OBJ_CLOSURE);
    closure->function = function;
    closure->upvalues = upvalues;
    closure->upvalue_count = function->upvalue_count;
    return closure;
}

ObjUpvalue* adam_new_upvalue(VM* vm, Value* slot) {
    ObjUpvalue* upvalue = ALLOCATE_OBJ(vm, ObjUpvalue, OBJ_UPVALUE);
    upvalue->location = slot;
    upvalue->closed = NIL_VAL;
    upvalue->next = NULL;
    return upvalue;
}

ObjNative* adam_new_native(VM* vm, NativeFn function, ObjString* name) {
    ObjNative* native = ALLOCATE_OBJ(vm, ObjNative, OBJ_NATIVE);
    native->function = function;
    native->name = name;
    return native;
}

/* ── Array objects ─────────────────────────────────────────────────── */

ObjArray* adam_new_array(VM* vm) {
    ObjArray* array = ALLOCATE_OBJ(vm, ObjArray, OBJ_ARRAY);
    array->count = 0;
    array->capacity = 0;
    array->elements = NULL;
    return array;
}

void adam_array_push(VM* vm, ObjArray* array, Value value) {
    if (array->count >= array->capacity) {
        int old_capacity = array->capacity;
        array->capacity = GROW_CAPACITY(old_capacity);
        array->elements = GROW_ARRAY(vm, Value, array->elements,
                                     old_capacity, array->capacity);
    }
    array->elements[array->count] = value;
    array->count++;
}

/* ── Struct and variant objects ────────────────────────────────────── */

ObjStruct* adam_new_struct(VM* vm, ObjString* name, int field_count) {
    ObjStruct* s = ALLOCATE_OBJ(vm, ObjStruct, OBJ_STRUCT);
    s->name = name;
    s->field_count = field_count;
    s->field_names = ALLOCATE(vm, ObjString*, field_count);
    s->fields = ALLOCATE(vm, Value, field_count);
    for (int i = 0; i < field_count; i++) {
        s->field_names[i] = NULL;
        s->fields[i] = NIL_VAL;
    }
    return s;
}

ObjVariant* adam_new_variant(VM* vm, ObjString* tag, Value payload) {
    ObjVariant* variant = ALLOCATE_OBJ(vm, ObjVariant, OBJ_VARIANT);
    variant->tag = tag;
    variant->payload = payload;
    return variant;
}

/* ── Tensor objects ────────────────────────────────────────────────── */

ObjTensor* adam_new_tensor(VM* vm, int ndim, int* shape) {
    ObjTensor* tensor = ALLOCATE_OBJ(vm, ObjTensor, OBJ_TENSOR);
    tensor->ndim = ndim;
    tensor->shape = ALLOCATE(vm, int, ndim);
    int count = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        count *= shape[i];
    }
    tensor->count = count;
    tensor->data = ALLOCATE(vm, double, count);
    for (int i = 0; i < count; i++) {
        tensor->data[i] = 0.0;
    }
    return tensor;
}

/* ── Object deallocation ───────────────────────────────────────────── */

void adam_free_object(VM* vm, Obj* object) {
#ifdef DEBUG_LOG_GC
    printf("%p free type %d\n", (void*)object, object->type);
#endif

    switch (object->type) {
    case OBJ_STRING: {
        ObjString* string = (ObjString*)object;
        /* FAM chars are part of the allocation — freed with the object. */
        adam_gc_reallocate(vm, object, sizeof(ObjString) + string->length + 1, 0);
        break;
    }
    case OBJ_FUNCTION: {
        ObjFunction* function = (ObjFunction*)object;
        adam_chunk_free(vm, &function->chunk);
        FREE(vm, ObjFunction, object);
        break;
    }
    case OBJ_CLOSURE: {
        ObjClosure* closure = (ObjClosure*)object;
        FREE_ARRAY(vm, ObjUpvalue*, closure->upvalues,
                   closure->upvalue_count);
        FREE(vm, ObjClosure, object);
        break;
    }
    case OBJ_UPVALUE:
        FREE(vm, ObjUpvalue, object);
        break;
    case OBJ_NATIVE:
        FREE(vm, ObjNative, object);
        break;
    case OBJ_ARRAY: {
        ObjArray* array = (ObjArray*)object;
        FREE_ARRAY(vm, Value, array->elements, array->capacity);
        FREE(vm, ObjArray, object);
        break;
    }
    case OBJ_STRUCT: {
        ObjStruct* s = (ObjStruct*)object;
        FREE_ARRAY(vm, ObjString*, s->field_names, s->field_count);
        FREE_ARRAY(vm, Value, s->fields, s->field_count);
        FREE(vm, ObjStruct, object);
        break;
    }
    case OBJ_VARIANT:
        FREE(vm, ObjVariant, object);
        break;
    case OBJ_TENSOR: {
        ObjTensor* tensor = (ObjTensor*)object;
        FREE_ARRAY(vm, int, tensor->shape, tensor->ndim);
        FREE_ARRAY(vm, double, tensor->data, tensor->count);
        FREE(vm, ObjTensor, object);
        break;
    }
    }
}

/* ── Object printing ───────────────────────────────────────────────── */

void adam_print_object(Value value) {
    switch (OBJ_TYPE(value)) {
    case OBJ_STRING:
        printf("%s", AS_CSTRING(value));
        break;
    case OBJ_FUNCTION: {
        ObjFunction* fn = AS_FUNCTION(value);
        if (fn->name == NULL) {
            printf("<script>");
        } else {
            printf("<fn %s>", fn->name->chars);
        }
        break;
    }
    case OBJ_CLOSURE: {
        ObjFunction* fn = AS_CLOSURE(value)->function;
        if (fn->name == NULL) {
            printf("<script>");
        } else {
            printf("<fn %s>", fn->name->chars);
        }
        break;
    }
    case OBJ_UPVALUE:
        printf("<upvalue>");
        break;
    case OBJ_NATIVE:
        printf("<native fn %s>", AS_NATIVE_OBJ(value)->name->chars);
        break;
    case OBJ_ARRAY: {
        ObjArray* array = AS_ARRAY(value);
        printf("[");
        for (int i = 0; i < array->count; i++) {
            if (i > 0) printf(", ");
            adam_print_value(array->elements[i]);
        }
        printf("]");
        break;
    }
    case OBJ_STRUCT: {
        ObjStruct* s = AS_STRUCT(value);
        printf("%s{", s->name->chars);
        for (int i = 0; i < s->field_count; i++) {
            if (i > 0) printf(", ");
            if (s->field_names[i]) printf("%s: ", s->field_names[i]->chars);
            adam_print_value(s->fields[i]);
        }
        printf("}");
        break;
    }
    case OBJ_VARIANT: {
        ObjVariant* v = AS_VARIANT(value);
        printf("%s(", v->tag->chars);
        adam_print_value(v->payload);
        printf(")");
        break;
    }
    case OBJ_TENSOR: {
        ObjTensor* t = AS_TENSOR(value);
        printf("Tensor<[");
        for (int i = 0; i < t->ndim; i++) {
            if (i > 0) printf(", ");
            printf("%d", t->shape[i]);
        }
        printf("]>(");
        /* Print first few elements for brevity */
        int limit = t->count < 8 ? t->count : 8;
        for (int i = 0; i < limit; i++) {
            if (i > 0) printf(", ");
            printf("%g", t->data[i]);
        }
        if (t->count > 8) printf(", ...");
        printf(")");
        break;
    }
    }
}
