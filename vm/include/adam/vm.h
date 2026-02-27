/*
 * vm.h — Adam virtual machine state and public API
 *
 * The Adam VM is a stack-based bytecode interpreter. Execution state
 * consists of:
 *   - A value stack shared across all call frames
 *   - A call stack (array of CallFrames), each tracking a closure,
 *     instruction pointer, and a "window" into the value stack
 *   - Global variables in a hash table
 *   - A string intern table for O(1) string equality
 *   - GC bookkeeping (object list, gray worklist, byte counters)
 *
 * Stack-based vs register-based:
 *   Stack machines produce smaller bytecode (operands are implicit on
 *   the stack, not named in the instruction) at the cost of more stack
 *   manipulation instructions. For a showcase implementation, the simpler
 *   encoding and dispatch logic of a stack machine is preferred.
 */

#pragma once

#include "adam/common.h"
#include "adam/value.h"
#include "adam/table.h"
#include "adam/chunk.h"

/* Forward declarations */
typedef struct ObjClosure ObjClosure;
typedef struct ObjUpvalue ObjUpvalue;
typedef struct Obj Obj;

/* ── Call frame ────────────────────────────────────────────────────── */

/*
 * Each function invocation gets a CallFrame. The frame stores:
 *   - closure: the ObjClosure being executed (holds function + upvalues)
 *   - ip: instruction pointer into the closure's chunk's code array
 *   - slots: pointer into the VM's value stack where this frame's
 *     local variables start (slot 0 = the function itself)
 */
typedef struct {
    ObjClosure* closure;
    uint8_t* ip;
    Value* slots;
} CallFrame;

/* ── VM state ──────────────────────────────────────────────────────── */

typedef struct VM {
    /* Call stack */
    CallFrame frames[ADAM_FRAMES_MAX];
    int frame_count;

    /* Value stack */
    Value stack[ADAM_STACK_MAX];
    Value* stack_top;

    /* Global variables: name (ObjString*) → value */
    Table globals;

    /* String interning table: all live strings are keys here.
     * This enables O(1) string equality via pointer comparison.
     * Treated as weak references during GC (dead strings are removed). */
    Table strings;

    /* Open upvalues: linked list of ObjUpvalues that still point into
     * the stack. Sorted by stack slot address (highest first) so we
     * can efficiently close upvalues when a scope ends. */
    ObjUpvalue* open_upvalues;

    /* GC state */
    Obj* objects;           /* Head of intrusive linked list of ALL objects */
    int gray_count;         /* Number of objects on the gray worklist */
    int gray_capacity;      /* Allocated size of gray_stack */
    Obj** gray_stack;       /* Worklist of gray objects to process */
    size_t bytes_allocated; /* Total bytes currently allocated */
    size_t next_gc;         /* Threshold for next GC trigger */
} VM;

/* ── Interpret result ──────────────────────────────────────────────── */

typedef enum {
    INTERPRET_OK,
    INTERPRET_COMPILE_ERROR,
    INTERPRET_RUNTIME_ERROR,
} InterpretResult;

/* ── Public API ────────────────────────────────────────────────────── */

void            adam_vm_init(VM* vm);
void            adam_vm_free(VM* vm);
InterpretResult adam_vm_interpret(VM* vm, ObjClosure* closure);

/* Stack operations (used by native functions and the compiler) */
void  adam_vm_push(VM* vm, Value value);
Value adam_vm_pop(VM* vm);
