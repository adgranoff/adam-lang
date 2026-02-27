/*
 * vm.c — Main dispatch loop (the heart of the Adam virtual machine)
 *
 * Dispatch strategy:
 *   On GCC/Clang, we use computed goto (the &&label extension) for the
 *   dispatch loop. Each opcode has a label, and a dispatch table maps
 *   opcode bytes to label addresses. After executing an instruction, we
 *   jump directly to the next opcode's handler via the table.
 *
 *   Why computed goto beats switch-case:
 *     - Switch compiles to a single indirect branch. The CPU's branch
 *       predictor has ONE prediction slot for this branch, shared across
 *       all opcodes. With computed goto, each opcode ends with its OWN
 *       indirect branch, so the predictor can learn per-opcode patterns.
 *     - Measured improvement: 15-25% faster dispatch on modern x86.
 *
 *   On MSVC (which doesn't support computed goto), we fall back to a
 *   standard switch-case. Performance is still good — the switch is the
 *   bottleneck only for very tight loops.
 *
 * Stack-based execution:
 *   Operands are implicit on the value stack. OP_ADD pops two values,
 *   pushes the result. This makes bytecode compact (no register operands
 *   in the instruction encoding) at the cost of more push/pop traffic.
 *   For a teaching/showcase VM, simplicity wins over raw throughput.
 *
 * Upvalue capture:
 *   When a closure captures a local variable, we create an ObjUpvalue
 *   that initially points into the stack. If the local goes out of scope
 *   before the closure is done with it, we "close" the upvalue: copy the
 *   value from the stack into the upvalue's `closed` field and redirect
 *   the pointer. This is the standard technique from Lua's implementation.
 */

#include "adam/common.h"
#include "adam/vm.h"
#include "adam/value.h"
#include "adam/object.h"
#include "adam/chunk.h"
#include "adam/table.h"
#include "adam/gc.h"
#include "adam/debug.h"
#include "adam/native.h"

/* ── VM lifecycle ──────────────────────────────────────────────────── */

void adam_vm_init(VM* vm) {
    vm->stack_top = vm->stack;
    vm->frame_count = 0;
    vm->objects = NULL;
    vm->open_upvalues = NULL;

    vm->gray_count = 0;
    vm->gray_capacity = 0;
    vm->gray_stack = NULL;
    vm->bytes_allocated = 0;
    vm->next_gc = 1024 * 1024; /* First GC at 1 MB */

    adam_table_init(&vm->globals);
    adam_table_init(&vm->strings);

    adam_register_natives(vm);
}

void adam_vm_free(VM* vm) {
    adam_table_free(vm, &vm->globals);
    adam_table_free(vm, &vm->strings);

    /* Free all objects. */
    Obj* object = vm->objects;
    while (object != NULL) {
        Obj* next = object->next;
        adam_free_object(vm, object);
        object = next;
    }

    free(vm->gray_stack);
    vm->gray_stack = NULL;
}

/* ── Stack operations ──────────────────────────────────────────────── */

void adam_vm_push(VM* vm, Value value) {
    *vm->stack_top = value;
    vm->stack_top++;
}

Value adam_vm_pop(VM* vm) {
    vm->stack_top--;
    return *vm->stack_top;
}

static Value peek(VM* vm, int distance) {
    return vm->stack_top[-1 - distance];
}

static void reset_stack(VM* vm) {
    vm->stack_top = vm->stack;
    vm->frame_count = 0;
    vm->open_upvalues = NULL;
}

/* ── Runtime errors ────────────────────────────────────────────────── */

static void runtime_error(VM* vm, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputs("\n", stderr);

    /* Print stack trace. */
    for (int i = vm->frame_count - 1; i >= 0; i--) {
        CallFrame* frame = &vm->frames[i];
        ObjFunction* function = frame->closure->function;
        size_t instruction = frame->ip - function->chunk.code - 1;
        fprintf(stderr, "[line %d] in ", function->chunk.lines[instruction]);
        if (function->name == NULL) {
            fprintf(stderr, "script\n");
        } else {
            fprintf(stderr, "%s()\n", function->name->chars);
        }
    }

    reset_stack(vm);
}

/* ── Truthiness ────────────────────────────────────────────────────── */

static bool is_falsey(Value value) {
    return IS_NIL(value) || (IS_BOOL(value) && !AS_BOOL(value));
}

/* ── String concatenation ──────────────────────────────────────────── */

static void concatenate(VM* vm) {
    ObjString* b = AS_STRING(peek(vm, 0));
    ObjString* a = AS_STRING(peek(vm, 1));

    int length = a->length + b->length;
    char* chars = ALLOCATE(vm, char, length + 1);
    memcpy(chars, a->chars, a->length);
    memcpy(chars + a->length, b->chars, b->length);
    chars[length] = '\0';

    ObjString* result = adam_take_string(vm, chars, length);
    adam_vm_pop(vm);
    adam_vm_pop(vm);
    adam_vm_push(vm, OBJ_VAL(result));
}

/* ── Function calls ────────────────────────────────────────────────── */

static bool call(VM* vm, ObjClosure* closure, int arg_count) {
    if (arg_count != closure->function->arity) {
        runtime_error(vm, "Expected %d arguments but got %d.",
                     closure->function->arity, arg_count);
        return false;
    }
    if (vm->frame_count == ADAM_FRAMES_MAX) {
        runtime_error(vm, "Stack overflow.");
        return false;
    }

    CallFrame* frame = &vm->frames[vm->frame_count++];
    frame->closure = closure;
    frame->ip = closure->function->chunk.code;
    frame->slots = vm->stack_top - arg_count - 1;
    return true;
}

static bool call_value(VM* vm, Value callee, int arg_count) {
    if (IS_OBJ(callee)) {
        switch (OBJ_TYPE(callee)) {
        case OBJ_CLOSURE:
            return call(vm, AS_CLOSURE(callee), arg_count);
        case OBJ_NATIVE: {
            NativeFn native = AS_NATIVE(callee);
            Value result = native(vm, arg_count,
                                  vm->stack_top - arg_count);
            vm->stack_top -= arg_count + 1;
            adam_vm_push(vm, result);
            return true;
        }
        default:
            break;
        }
    }
    runtime_error(vm, "Can only call functions and closures.");
    return false;
}

/* ── Upvalue management ────────────────────────────────────────────── */

/*
 * Find or create an upvalue for a stack slot.
 *
 * The open_upvalues list is sorted by slot address (highest first).
 * We walk until we find a matching slot or pass it, then either return
 * the existing upvalue or insert a new one.
 */
static ObjUpvalue* capture_upvalue(VM* vm, Value* local) {
    ObjUpvalue* prev_upvalue = NULL;
    ObjUpvalue* upvalue = vm->open_upvalues;

    while (upvalue != NULL && upvalue->location > local) {
        prev_upvalue = upvalue;
        upvalue = upvalue->next;
    }

    if (upvalue != NULL && upvalue->location == local) {
        return upvalue;
    }

    ObjUpvalue* created = adam_new_upvalue(vm, local);
    created->next = upvalue;

    if (prev_upvalue == NULL) {
        vm->open_upvalues = created;
    } else {
        prev_upvalue->next = created;
    }

    return created;
}

/*
 * Close all upvalues that point at stack slots at or above `last`.
 * "Closing" means copying the stack value into the upvalue's `closed`
 * field and redirecting `location` to point at `closed`.
 */
static void close_upvalues(VM* vm, Value* last) {
    while (vm->open_upvalues != NULL &&
           vm->open_upvalues->location >= last) {
        ObjUpvalue* upvalue = vm->open_upvalues;
        upvalue->closed = *upvalue->location;
        upvalue->location = &upvalue->closed;
        vm->open_upvalues = upvalue->next;
    }
}

/* ── Main dispatch loop ────────────────────────────────────────────── */

static InterpretResult run(VM* vm) {
    CallFrame* frame = &vm->frames[vm->frame_count - 1];

    /* Macros for reading bytecode. */
#define READ_BYTE()     (*frame->ip++)
#define READ_SHORT()    \
    (frame->ip += 2, (uint16_t)((frame->ip[-2] << 8) | frame->ip[-1]))
#define READ_CONSTANT() \
    (frame->closure->function->chunk.constants[READ_BYTE()])
#define READ_STRING()   AS_STRING(READ_CONSTANT())

    /*
     * Computed goto dispatch (GCC/Clang) vs switch fallback (MSVC).
     *
     * With computed goto, each opcode handler ends with DISPATCH(), which
     * reads the next byte and jumps directly to the handler via the table.
     * The CPU branch predictor can learn patterns per-opcode.
     */
#ifdef ADAM_COMPUTED_GOTO
    static void* dispatch_table[] = {
        &&op_CONST, &&op_NIL, &&op_TRUE, &&op_FALSE,
        &&op_ADD, &&op_SUB, &&op_MUL, &&op_DIV, &&op_MOD, &&op_POW, &&op_NEG,
        &&op_EQ, &&op_NEQ, &&op_LT, &&op_GT, &&op_LTE, &&op_GTE,
        &&op_NOT,
        &&op_LOAD_LOCAL, &&op_STORE_LOCAL,
        &&op_LOAD_GLOBAL, &&op_STORE_GLOBAL,
        &&op_LOAD_UPVALUE, &&op_STORE_UPVALUE, &&op_CLOSE_UPVALUE,
        &&op_JUMP, &&op_JUMP_IF_FALSE, &&op_LOOP,
        &&op_CALL, &&op_CLOSURE, &&op_RETURN,
        &&op_ARRAY_NEW, &&op_ARRAY_GET, &&op_ARRAY_SET, &&op_ARRAY_LEN,
        &&op_STRUCT_NEW, &&op_STRUCT_GET, &&op_STRUCT_SET,
        &&op_MATCH,
        &&op_PRINT, &&op_POP,
        &&op_TENSOR_MATMUL, &&op_TENSOR_ADD, &&op_TENSOR_SUB,
        &&op_TENSOR_MUL, &&op_TENSOR_NEG,
    };
    #define DISPATCH() do { goto *dispatch_table[READ_BYTE()]; } while (0)
    #define CASE(name) op_##name
    DISPATCH();
#else
    #define DISPATCH() break
    #define CASE(name) case OP_##name
    for (;;) { switch (READ_BYTE()) {
#endif

    CASE(CONST): {
        Value constant = READ_CONSTANT();
        adam_vm_push(vm, constant);
        DISPATCH();
    }

    CASE(NIL):   adam_vm_push(vm, NIL_VAL);   DISPATCH();
    CASE(TRUE):  adam_vm_push(vm, TRUE_VAL);  DISPATCH();
    CASE(FALSE): adam_vm_push(vm, FALSE_VAL); DISPATCH();

    CASE(ADD): {
        if (IS_STRING(peek(vm, 0)) && IS_STRING(peek(vm, 1))) {
            concatenate(vm);
        } else if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, INT_VAL(a + b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(a + b));
        } else {
            runtime_error(vm, "Operands must be two numbers or two strings.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(SUB): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, INT_VAL(a - b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(a - b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(MUL): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, INT_VAL(a * b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(a * b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(DIV): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            if (b == 0) {
                runtime_error(vm, "Division by zero.");
                return INTERPRET_RUNTIME_ERROR;
            }
            adam_vm_push(vm, INT_VAL(a / b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(a / b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(MOD): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            if (b == 0) {
                runtime_error(vm, "Division by zero.");
                return INTERPRET_RUNTIME_ERROR;
            }
            adam_vm_push(vm, INT_VAL(a % b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(fmod(a, b)));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(POW): {
        if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, FLOAT_VAL(pow(a, b)));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(NEG): {
        if (IS_INT(peek(vm, 0))) {
            adam_vm_push(vm, INT_VAL(-AS_INT(adam_vm_pop(vm))));
        } else if (IS_FLOAT(peek(vm, 0))) {
            adam_vm_push(vm, FLOAT_VAL(-AS_FLOAT(adam_vm_pop(vm))));
        } else {
            runtime_error(vm, "Operand must be a number.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(EQ): {
        Value b = adam_vm_pop(vm);
        Value a = adam_vm_pop(vm);
        adam_vm_push(vm, BOOL_VAL(adam_values_equal(a, b)));
        DISPATCH();
    }

    CASE(NEQ): {
        Value b = adam_vm_pop(vm);
        Value a = adam_vm_pop(vm);
        adam_vm_push(vm, BOOL_VAL(!adam_values_equal(a, b)));
        DISPATCH();
    }

    CASE(LT): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a < b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a < b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(GT): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a > b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a > b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(LTE): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a <= b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a <= b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(GTE): {
        if (IS_INT(peek(vm, 0)) && IS_INT(peek(vm, 1))) {
            int32_t b = AS_INT(adam_vm_pop(vm));
            int32_t a = AS_INT(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a >= b));
        } else if (IS_NUMBER(peek(vm, 0)) && IS_NUMBER(peek(vm, 1))) {
            double b = adam_as_number(adam_vm_pop(vm));
            double a = adam_as_number(adam_vm_pop(vm));
            adam_vm_push(vm, BOOL_VAL(a >= b));
        } else {
            runtime_error(vm, "Operands must be numbers.");
            return INTERPRET_RUNTIME_ERROR;
        }
        DISPATCH();
    }

    CASE(NOT): {
        adam_vm_push(vm, BOOL_VAL(is_falsey(adam_vm_pop(vm))));
        DISPATCH();
    }

    CASE(LOAD_LOCAL): {
        uint8_t slot = READ_BYTE();
        adam_vm_push(vm, frame->slots[slot]);
        DISPATCH();
    }

    CASE(STORE_LOCAL): {
        uint8_t slot = READ_BYTE();
        frame->slots[slot] = peek(vm, 0);
        DISPATCH();
    }

    CASE(LOAD_GLOBAL): {
        ObjString* name = READ_STRING();
        Value value;
        if (!adam_table_get(&vm->globals, name, &value)) {
            runtime_error(vm, "Undefined variable '%s'.", name->chars);
            return INTERPRET_RUNTIME_ERROR;
        }
        adam_vm_push(vm, value);
        DISPATCH();
    }

    CASE(STORE_GLOBAL): {
        ObjString* name = READ_STRING();
        adam_table_set(vm, &vm->globals, name, peek(vm, 0));
        DISPATCH();
    }

    CASE(LOAD_UPVALUE): {
        uint8_t slot = READ_BYTE();
        adam_vm_push(vm, *frame->closure->upvalues[slot]->location);
        DISPATCH();
    }

    CASE(STORE_UPVALUE): {
        uint8_t slot = READ_BYTE();
        *frame->closure->upvalues[slot]->location = peek(vm, 0);
        DISPATCH();
    }

    CASE(CLOSE_UPVALUE): {
        close_upvalues(vm, vm->stack_top - 1);
        adam_vm_pop(vm);
        DISPATCH();
    }

    CASE(JUMP): {
        uint16_t offset = READ_SHORT();
        frame->ip += offset;
        DISPATCH();
    }

    CASE(JUMP_IF_FALSE): {
        uint16_t offset = READ_SHORT();
        if (is_falsey(peek(vm, 0))) frame->ip += offset;
        DISPATCH();
    }

    CASE(LOOP): {
        uint16_t offset = READ_SHORT();
        frame->ip -= offset;
        DISPATCH();
    }

    CASE(CALL): {
        int arg_count = READ_BYTE();
        if (!call_value(vm, peek(vm, arg_count), arg_count)) {
            return INTERPRET_RUNTIME_ERROR;
        }
        frame = &vm->frames[vm->frame_count - 1];
        DISPATCH();
    }

    CASE(CLOSURE): {
        ObjFunction* function = AS_FUNCTION(READ_CONSTANT());
        ObjClosure* closure = adam_new_closure(vm, function);
        adam_vm_push(vm, OBJ_VAL(closure));
        for (int i = 0; i < closure->upvalue_count; i++) {
            uint8_t is_local = READ_BYTE();
            uint8_t index = READ_BYTE();
            if (is_local) {
                closure->upvalues[i] = capture_upvalue(vm,
                    frame->slots + index);
            } else {
                closure->upvalues[i] = frame->closure->upvalues[index];
            }
        }
        DISPATCH();
    }

    CASE(RETURN): {
        Value result = adam_vm_pop(vm);
        close_upvalues(vm, frame->slots);
        vm->frame_count--;
        if (vm->frame_count == 0) {
            adam_vm_pop(vm);
            return INTERPRET_OK;
        }
        vm->stack_top = frame->slots;
        adam_vm_push(vm, result);
        frame = &vm->frames[vm->frame_count - 1];
        DISPATCH();
    }

    CASE(ARRAY_NEW): {
        int count = READ_BYTE();
        ObjArray* array = adam_new_array(vm);
        /* Push array to protect from GC during element push. */
        adam_vm_push(vm, OBJ_VAL(array));
        /* Elements are on the stack in order, below the array. */
        for (int i = count; i > 0; i--) {
            adam_array_push(vm, array, vm->stack_top[-1 - i]);
        }
        /* Remove elements and the temporary array push. */
        vm->stack_top -= count + 1;
        adam_vm_push(vm, OBJ_VAL(array));
        DISPATCH();
    }

    CASE(ARRAY_GET): {
        if (!IS_INT(peek(vm, 0)) || !IS_ARRAY(peek(vm, 1))) {
            runtime_error(vm, "Array index must be an integer.");
            return INTERPRET_RUNTIME_ERROR;
        }
        int32_t index = AS_INT(adam_vm_pop(vm));
        ObjArray* array = AS_ARRAY(adam_vm_pop(vm));
        if (index < 0 || index >= array->count) {
            runtime_error(vm, "Array index %d out of bounds (length %d).",
                         index, array->count);
            return INTERPRET_RUNTIME_ERROR;
        }
        adam_vm_push(vm, array->elements[index]);
        DISPATCH();
    }

    CASE(ARRAY_SET): {
        Value value = adam_vm_pop(vm);
        if (!IS_INT(peek(vm, 0)) || !IS_ARRAY(peek(vm, 1))) {
            runtime_error(vm, "Array index must be an integer.");
            return INTERPRET_RUNTIME_ERROR;
        }
        int32_t index = AS_INT(adam_vm_pop(vm));
        ObjArray* array = AS_ARRAY(adam_vm_pop(vm));
        if (index < 0 || index >= array->count) {
            runtime_error(vm, "Array index %d out of bounds (length %d).",
                         index, array->count);
            return INTERPRET_RUNTIME_ERROR;
        }
        array->elements[index] = value;
        DISPATCH();
    }

    CASE(ARRAY_LEN): {
        if (!IS_ARRAY(peek(vm, 0))) {
            runtime_error(vm, "Operand must be an array.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjArray* array = AS_ARRAY(adam_vm_pop(vm));
        adam_vm_push(vm, INT_VAL(array->count));
        DISPATCH();
    }

    CASE(STRUCT_NEW): {
        ObjString* name = READ_STRING();
        int field_count = READ_BYTE();
        /* Read field name constants from bytecode before allocating
         * (allocation may trigger GC, but these are interned strings
         * already reachable from the constant pool). */
        ObjString* fnames[256];
        for (int i = 0; i < field_count; i++) {
            fnames[i] = READ_STRING();
        }
        ObjStruct* s = adam_new_struct(vm, name, field_count);
        adam_vm_push(vm, OBJ_VAL(s)); /* Protect from GC */
        for (int i = 0; i < field_count; i++) {
            s->field_names[i] = fnames[i];
        }
        /* Field values are on the stack below the struct. */
        for (int i = field_count - 1; i >= 0; i--) {
            s->fields[i] = vm->stack_top[-2 - (field_count - 1 - i)];
        }
        adam_vm_pop(vm);             /* Pop temporary struct */
        vm->stack_top -= field_count; /* Remove field values */
        adam_vm_push(vm, OBJ_VAL(s)); /* Push struct as result */
        DISPATCH();
    }

    CASE(STRUCT_GET): {
        ObjString* field_name = READ_STRING();
        if (!IS_STRUCT(peek(vm, 0))) {
            runtime_error(vm, "Only structs have fields.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjStruct* s = AS_STRUCT(adam_vm_pop(vm));
        for (int i = 0; i < s->field_count; i++) {
            if (s->field_names[i] == field_name) {
                adam_vm_push(vm, s->fields[i]);
                DISPATCH();
            }
        }
        runtime_error(vm, "Struct '%s' has no field '%s'.",
                      s->name->chars, field_name->chars);
        return INTERPRET_RUNTIME_ERROR;
    }

    CASE(STRUCT_SET): {
        ObjString* field_name = READ_STRING();
        Value value = adam_vm_pop(vm);
        if (!IS_STRUCT(peek(vm, 0))) {
            runtime_error(vm, "Only structs have fields.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjStruct* s = AS_STRUCT(peek(vm, 0));
        for (int i = 0; i < s->field_count; i++) {
            if (s->field_names[i] == field_name) {
                s->fields[i] = value;
                DISPATCH();
            }
        }
        runtime_error(vm, "Struct '%s' has no field '%s'.",
                      s->name->chars, field_name->chars);
        return INTERPRET_RUNTIME_ERROR;
    }

    CASE(MATCH): {
        uint8_t tag_idx = READ_BYTE();
        uint16_t offset = READ_SHORT();
        ObjString* expected_tag = AS_STRING(
            frame->closure->function->chunk.constants[tag_idx]);
        Value top = peek(vm, 0);
        if (!IS_VARIANT(top) || AS_VARIANT(top)->tag != expected_tag) {
            frame->ip += offset;
        }
        DISPATCH();
    }

    CASE(PRINT): {
        adam_print_value(adam_vm_pop(vm));
        printf("\n");
        DISPATCH();
    }

    CASE(POP): {
        adam_vm_pop(vm);
        DISPATCH();
    }

    CASE(TENSOR_MATMUL): {
        if (!IS_TENSOR(peek(vm, 0)) || !IS_TENSOR(peek(vm, 1))) {
            runtime_error(vm, "Operands to @@ must be tensors.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* b = AS_TENSOR(peek(vm, 0));
        ObjTensor* a = AS_TENSOR(peek(vm, 1));
        if (a->ndim < 2 || b->ndim < 2) {
            runtime_error(vm, "Matrix multiply requires at least 2D tensors.");
            return INTERPRET_RUNTIME_ERROR;
        }
        int a_rows = a->shape[a->ndim - 2];
        int a_cols = a->shape[a->ndim - 1];
        int b_rows = b->shape[b->ndim - 2];
        int b_cols = b->shape[b->ndim - 1];
        if (a_cols != b_rows) {
            runtime_error(vm, "Shape mismatch in @@: [..,%d,%d] @@ [%d,%d].",
                         a_rows, a_cols, b_rows, b_cols);
            return INTERPRET_RUNTIME_ERROR;
        }
        /* Compute batch dimensions (all dims except last two in a) */
        int batch = 1;
        for (int i = 0; i < a->ndim - 2; i++) batch *= a->shape[i];
        /* Result shape: a's batch dims + [a_rows, b_cols] */
        int result_ndim = a->ndim;
        int result_shape_buf[16]; /* Max 16 dims */
        for (int i = 0; i < a->ndim - 2; i++) result_shape_buf[i] = a->shape[i];
        result_shape_buf[result_ndim - 2] = a_rows;
        result_shape_buf[result_ndim - 1] = b_cols;
        ObjTensor* result = adam_new_tensor(vm, result_ndim, result_shape_buf);
        adam_vm_push(vm, OBJ_VAL(result)); /* GC protection */
        /* Naive batched matmul: for each batch, triple-loop */
        for (int bi = 0; bi < batch; bi++) {
            double* a_data = a->data + bi * a_rows * a_cols;
            double* r_data = result->data + bi * a_rows * b_cols;
            for (int i = 0; i < a_rows; i++) {
                for (int j = 0; j < b_cols; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < a_cols; k++) {
                        sum += a_data[i * a_cols + k] * b->data[k * b_cols + j];
                    }
                    r_data[i * b_cols + j] = sum;
                }
            }
        }
        adam_vm_pop(vm);  /* Pop GC protection */
        adam_vm_pop(vm);  /* Pop b */
        adam_vm_pop(vm);  /* Pop a */
        adam_vm_push(vm, OBJ_VAL(result));
        DISPATCH();
    }

    CASE(TENSOR_ADD): {
        if (!IS_TENSOR(peek(vm, 0)) || !IS_TENSOR(peek(vm, 1))) {
            runtime_error(vm, "Operands to tensor add must be tensors.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* b = AS_TENSOR(peek(vm, 0));
        ObjTensor* a = AS_TENSOR(peek(vm, 1));
        if (a->count != b->count) {
            runtime_error(vm, "Shape mismatch in tensor add.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* result = adam_new_tensor(vm, a->ndim, a->shape);
        adam_vm_push(vm, OBJ_VAL(result));
        for (int i = 0; i < a->count; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_push(vm, OBJ_VAL(result));
        DISPATCH();
    }

    CASE(TENSOR_SUB): {
        if (!IS_TENSOR(peek(vm, 0)) || !IS_TENSOR(peek(vm, 1))) {
            runtime_error(vm, "Operands to tensor sub must be tensors.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* b = AS_TENSOR(peek(vm, 0));
        ObjTensor* a = AS_TENSOR(peek(vm, 1));
        if (a->count != b->count) {
            runtime_error(vm, "Shape mismatch in tensor sub.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* result = adam_new_tensor(vm, a->ndim, a->shape);
        adam_vm_push(vm, OBJ_VAL(result));
        for (int i = 0; i < a->count; i++) {
            result->data[i] = a->data[i] - b->data[i];
        }
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_push(vm, OBJ_VAL(result));
        DISPATCH();
    }

    CASE(TENSOR_MUL): {
        if (!IS_TENSOR(peek(vm, 0)) || !IS_TENSOR(peek(vm, 1))) {
            runtime_error(vm, "Operands to tensor mul must be tensors.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* b = AS_TENSOR(peek(vm, 0));
        ObjTensor* a = AS_TENSOR(peek(vm, 1));
        if (a->count != b->count) {
            runtime_error(vm, "Shape mismatch in tensor mul.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* result = adam_new_tensor(vm, a->ndim, a->shape);
        adam_vm_push(vm, OBJ_VAL(result));
        for (int i = 0; i < a->count; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_push(vm, OBJ_VAL(result));
        DISPATCH();
    }

    CASE(TENSOR_NEG): {
        if (!IS_TENSOR(peek(vm, 0))) {
            runtime_error(vm, "Operand to tensor neg must be a tensor.");
            return INTERPRET_RUNTIME_ERROR;
        }
        ObjTensor* a = AS_TENSOR(peek(vm, 0));
        ObjTensor* result = adam_new_tensor(vm, a->ndim, a->shape);
        adam_vm_push(vm, OBJ_VAL(result));
        for (int i = 0; i < a->count; i++) {
            result->data[i] = -a->data[i];
        }
        adam_vm_pop(vm);
        adam_vm_pop(vm);
        adam_vm_push(vm, OBJ_VAL(result));
        DISPATCH();
    }

#ifndef ADAM_COMPUTED_GOTO
    default:
        runtime_error(vm, "Unknown opcode %d.", frame->ip[-1]);
        return INTERPRET_RUNTIME_ERROR;
    } } /* end switch, end for */
#endif

#undef READ_BYTE
#undef READ_SHORT
#undef READ_CONSTANT
#undef READ_STRING
#undef DISPATCH
#undef CASE
}

/* ── Public API ────────────────────────────────────────────────────── */

InterpretResult adam_vm_interpret(VM* vm, ObjClosure* closure) {
    adam_vm_push(vm, OBJ_VAL(closure));
    call(vm, closure, 0);
    return run(vm);
}
