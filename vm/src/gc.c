/*
 * gc.c — Tri-color mark-and-sweep garbage collector
 *
 * Overview:
 *   The GC reclaims heap memory occupied by objects that are no longer
 *   reachable from any root. It runs in three phases:
 *
 *   1. MARK: Starting from roots (stack, globals, open upvalues, call
 *      frames), recursively mark all reachable objects.
 *   2. TRACE: Process the gray worklist — for each gray object, mark
 *      all objects it references, turning it black.
 *   3. SWEEP: Walk the full object list, freeing unmarked (white) objects
 *      and resetting marks on survivors for the next cycle.
 *
 * Tri-color invariant:
 *   WHITE = unmarked (is_marked=false, not on gray stack)
 *   GRAY  = marked but children not yet scanned (on gray stack)
 *   BLACK = marked and all children scanned (is_marked=true, off gray stack)
 *
 *   Invariant: no black object references a white object. This ensures
 *   completeness — once we drain the gray worklist, all reachable objects
 *   are black, and all white objects are garbage.
 *
 * String intern table:
 *   Treated as weak references. During sweep, we remove entries for
 *   unmarked strings BEFORE freeing the objects, so the table doesn't
 *   hold dead pointers.
 *
 * Scheduling:
 *   GC triggers when bytes_allocated exceeds next_gc. After collection,
 *   next_gc = bytes_allocated * GROW_FACTOR. This adaptive approach means
 *   the more live data, the less frequent collections.
 */

#include "adam/common.h"
#include "adam/gc.h"
#include "adam/vm.h"
#include "adam/value.h"
#include "adam/object.h"
#include "adam/table.h"
#include "adam/chunk.h"

/* ── Core allocator ────────────────────────────────────────────────── */

void* adam_gc_reallocate(VM* vm, void* pointer, size_t old_size,
                          size_t new_size) {
    vm->bytes_allocated += new_size;
    vm->bytes_allocated -= old_size;

    if (new_size > old_size) {
#ifdef DEBUG_STRESS_GC
        adam_gc_collect(vm);
#endif
        if (vm->bytes_allocated > vm->next_gc) {
            adam_gc_collect(vm);
        }
    }

    if (new_size == 0) {
        free(pointer);
        return NULL;
    }

    /* realloc(NULL, size) == malloc(size) per C99. */
    void* result = realloc(pointer, new_size);
    if (result == NULL) {
        fprintf(stderr, "adam: out of memory\n");
        exit(1);
    }
    return result;
}

/* ── Marking ───────────────────────────────────────────────────────── */

void adam_gc_mark_value(VM* vm, Value value) {
    if (IS_OBJ(value)) adam_gc_mark_object(vm, AS_OBJ(value));
}

void adam_gc_mark_object(VM* vm, Obj* object) {
    if (object == NULL) return;
    if (object->is_marked) return;

#ifdef DEBUG_LOG_GC
    printf("%p mark ", (void*)object);
    adam_print_value(OBJ_VAL(object));
    printf("\n");
#endif

    object->is_marked = true;

    /* Add to gray worklist. Grow with raw realloc to avoid recursive GC. */
    if (vm->gray_count >= vm->gray_capacity) {
        vm->gray_capacity = GROW_CAPACITY(vm->gray_capacity);
        vm->gray_stack = (Obj**)realloc(vm->gray_stack,
                                         sizeof(Obj*) * vm->gray_capacity);
        if (vm->gray_stack == NULL) {
            fprintf(stderr, "adam: out of memory (gray stack)\n");
            exit(1);
        }
    }
    vm->gray_stack[vm->gray_count++] = object;
}

static void mark_array(VM* vm, Value* values, int count) {
    for (int i = 0; i < count; i++) {
        adam_gc_mark_value(vm, values[i]);
    }
}

/* ── Root marking ──────────────────────────────────────────────────── */

static void mark_roots(VM* vm) {
    /* Stack values */
    for (Value* slot = vm->stack; slot < vm->stack_top; slot++) {
        adam_gc_mark_value(vm, *slot);
    }

    /* Call frame closures */
    for (int i = 0; i < vm->frame_count; i++) {
        adam_gc_mark_object(vm, (Obj*)vm->frames[i].closure);
    }

    /* Open upvalues */
    for (ObjUpvalue* upvalue = vm->open_upvalues;
         upvalue != NULL;
         upvalue = upvalue->next) {
        adam_gc_mark_object(vm, (Obj*)upvalue);
    }

    /* Global variables */
    adam_table_mark(vm, &vm->globals);
}

/* ── Tracing (blackening) ──────────────────────────────────────────── */

static void blacken_object(VM* vm, Obj* object) {
#ifdef DEBUG_LOG_GC
    printf("%p blacken ", (void*)object);
    adam_print_value(OBJ_VAL(object));
    printf("\n");
#endif

    switch (object->type) {
    case OBJ_CLOSURE: {
        ObjClosure* closure = (ObjClosure*)object;
        adam_gc_mark_object(vm, (Obj*)closure->function);
        for (int i = 0; i < closure->upvalue_count; i++) {
            adam_gc_mark_object(vm, (Obj*)closure->upvalues[i]);
        }
        break;
    }
    case OBJ_FUNCTION: {
        ObjFunction* function = (ObjFunction*)object;
        adam_gc_mark_object(vm, (Obj*)function->name);
        mark_array(vm, function->chunk.constants,
                   function->chunk.constant_count);
        break;
    }
    case OBJ_UPVALUE:
        adam_gc_mark_value(vm, ((ObjUpvalue*)object)->closed);
        break;
    case OBJ_ARRAY: {
        ObjArray* array = (ObjArray*)object;
        mark_array(vm, array->elements, array->count);
        break;
    }
    case OBJ_STRUCT: {
        ObjStruct* s = (ObjStruct*)object;
        adam_gc_mark_object(vm, (Obj*)s->name);
        for (int i = 0; i < s->field_count; i++) {
            if (s->field_names[i]) {
                adam_gc_mark_object(vm, (Obj*)s->field_names[i]);
            }
        }
        mark_array(vm, s->fields, s->field_count);
        break;
    }
    case OBJ_VARIANT: {
        ObjVariant* v = (ObjVariant*)object;
        adam_gc_mark_object(vm, (Obj*)v->tag);
        adam_gc_mark_value(vm, v->payload);
        break;
    }
    case OBJ_NATIVE:
        adam_gc_mark_object(vm, (Obj*)((ObjNative*)object)->name);
        break;
    case OBJ_STRING:
        /* Strings have no outgoing references. */
        break;
    }
}

static void trace_references(VM* vm) {
    while (vm->gray_count > 0) {
        Obj* object = vm->gray_stack[--vm->gray_count];
        blacken_object(vm, object);
    }
}

/* ── Sweep ─────────────────────────────────────────────────────────── */

static void sweep(VM* vm) {
    Obj* previous = NULL;
    Obj* object = vm->objects;

    while (object != NULL) {
        if (object->is_marked) {
            object->is_marked = false;
            previous = object;
            object = object->next;
        } else {
            Obj* unreached = object;
            object = object->next;
            if (previous != NULL) {
                previous->next = object;
            } else {
                vm->objects = object;
            }
            adam_free_object(vm, unreached);
        }
    }
}

/* ── Full GC cycle ─────────────────────────────────────────────────── */

void adam_gc_collect(VM* vm) {
#ifdef DEBUG_LOG_GC
    printf("-- gc begin\n");
    size_t before = vm->bytes_allocated;
#endif

    mark_roots(vm);
    trace_references(vm);
    adam_table_remove_white(&vm->strings);
    sweep(vm);

    vm->next_gc = vm->bytes_allocated * ADAM_GC_HEAP_GROW_FACTOR;

#ifdef DEBUG_LOG_GC
    printf("-- gc end\n");
    printf("   collected %zu bytes (from %zu to %zu) next at %zu\n",
           before - vm->bytes_allocated, before,
           vm->bytes_allocated, vm->next_gc);
#endif
}
