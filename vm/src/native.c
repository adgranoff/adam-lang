/*
 * native.c — Native function bridge
 *
 * Provides built-in functions available to Adam programs: clock(), print(),
 * len(), type_of(), and math utilities. Each native is a C function with
 * the signature: Value fn(VM* vm, int arg_count, Value* args).
 *
 * Registration pushes the name and function onto the VM stack (to protect
 * from GC during allocation), then stores them in vm->globals.
 */

#include "adam/common.h"
#include "adam/native.h"
#include "adam/vm.h"
#include "adam/value.h"
#include "adam/object.h"
#include "adam/gc.h"

/* ── Helper ────────────────────────────────────────────────────────── */

static void define_native(VM* vm, const char* name, NativeFn function) {
    /* Push name and native to stack to protect from GC.
     * The allocations below may trigger GC, so both objects must be
     * reachable from the stack before we store them in the table. */
    ObjString* fn_name = adam_copy_string(vm, name, (int)strlen(name));
    adam_vm_push(vm, OBJ_VAL(fn_name));
    ObjNative* native = adam_new_native(vm, function, fn_name);
    adam_vm_push(vm, OBJ_VAL(native));
    adam_table_set(vm, &vm->globals,
                  AS_STRING(vm->stack_top[-2]),  /* name */
                  vm->stack_top[-1]);            /* native fn */
    adam_vm_pop(vm);
    adam_vm_pop(vm);
}

/* ── Native functions ──────────────────────────────────────────────── */

static Value clock_native(VM* vm, int arg_count, Value* args) {
    (void)vm; (void)arg_count; (void)args;
    return FLOAT_VAL((double)clock() / CLOCKS_PER_SEC);
}

static Value println_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    for (int i = 0; i < arg_count; i++) {
        if (i > 0) printf(" ");
        adam_print_value(args[i]);
    }
    printf("\n");
    return NIL_VAL;
}

static Value print_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    for (int i = 0; i < arg_count; i++) {
        if (i > 0) printf(" ");
        adam_print_value(args[i]);
    }
    return NIL_VAL;
}

static Value len_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1) return NIL_VAL;
    if (IS_STRING(args[0])) return INT_VAL(AS_STRING(args[0])->length);
    if (IS_ARRAY(args[0]))  return INT_VAL(AS_ARRAY(args[0])->count);
    return NIL_VAL;
}

static Value type_of_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1) return OBJ_VAL(adam_copy_string(vm, "unknown", 7));
    Value val = args[0];
    if (IS_NIL(val))    return OBJ_VAL(adam_copy_string(vm, "nil", 3));
    if (IS_BOOL(val))   return OBJ_VAL(adam_copy_string(vm, "bool", 4));
    if (IS_INT(val))    return OBJ_VAL(adam_copy_string(vm, "int", 3));
    if (IS_FLOAT(val))  return OBJ_VAL(adam_copy_string(vm, "float", 5));
    if (IS_OBJ(val)) {
        switch (OBJ_TYPE(val)) {
        case OBJ_STRING:   return OBJ_VAL(adam_copy_string(vm, "string", 6));
        case OBJ_FUNCTION:
        case OBJ_CLOSURE:  return OBJ_VAL(adam_copy_string(vm, "function", 8));
        case OBJ_NATIVE:   return OBJ_VAL(adam_copy_string(vm, "function", 8));
        case OBJ_ARRAY:    return OBJ_VAL(adam_copy_string(vm, "array", 5));
        case OBJ_STRUCT:   return OBJ_VAL(adam_copy_string(vm, "struct", 6));
        case OBJ_VARIANT:  return OBJ_VAL(adam_copy_string(vm, "variant", 7));
        case OBJ_TENSOR:   return OBJ_VAL(adam_copy_string(vm, "tensor", 6));
        default: break;
        }
    }
    return OBJ_VAL(adam_copy_string(vm, "unknown", 7));
}

static Value to_int_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1) return NIL_VAL;
    if (IS_INT(args[0]))   return args[0];
    if (IS_FLOAT(args[0])) return INT_VAL((int32_t)AS_FLOAT(args[0]));
    if (IS_BOOL(args[0]))  return INT_VAL(AS_BOOL(args[0]) ? 1 : 0);
    return NIL_VAL;
}

static Value to_float_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1) return NIL_VAL;
    if (IS_FLOAT(args[0])) return args[0];
    if (IS_INT(args[0]))   return FLOAT_VAL((double)AS_INT(args[0]));
    return NIL_VAL;
}

static Value sqrt_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1) return NIL_VAL;
    double val = IS_INT(args[0]) ? (double)AS_INT(args[0]) : AS_FLOAT(args[0]);
    return FLOAT_VAL(sqrt(val));
}

static Value abs_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1) return NIL_VAL;
    if (IS_INT(args[0]))   return INT_VAL(abs(AS_INT(args[0])));
    if (IS_FLOAT(args[0])) return FLOAT_VAL(fabs(AS_FLOAT(args[0])));
    return NIL_VAL;
}

/* ── Tensor native functions ───────────────────────────────────────── */

/* Helper: extract an array of ints from an Adam array value for shapes */
static bool extract_shape(Value val, int* out_shape, int* out_ndim) {
    if (!IS_ARRAY(val)) return false;
    ObjArray* arr = AS_ARRAY(val);
    *out_ndim = arr->count;
    for (int i = 0; i < arr->count; i++) {
        if (!IS_INT(arr->elements[i])) return false;
        out_shape[i] = AS_INT(arr->elements[i]);
    }
    return true;
}

static Value tensor_zeros_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1) {
        /* Return nil on bad arity — type checker prevents this in typed code */
        return NIL_VAL;
    }
    int shape[16];
    int ndim;
    if (!extract_shape(args[0], shape, &ndim)) return NIL_VAL;
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    /* adam_new_tensor already zeroes data */
    return OBJ_VAL(tensor);
}

static Value tensor_ones_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1) return NIL_VAL;
    int shape[16];
    int ndim;
    if (!extract_shape(args[0], shape, &ndim)) return NIL_VAL;
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    for (int i = 0; i < tensor->count; i++) {
        tensor->data[i] = 1.0;
    }
    return OBJ_VAL(tensor);
}

static Value tensor_randn_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1) return NIL_VAL;
    int shape[16];
    int ndim;
    if (!extract_shape(args[0], shape, &ndim)) return NIL_VAL;
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    /* Box-Muller transform for approximate normal distribution */
    for (int i = 0; i < tensor->count; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
        tensor->data[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
    return OBJ_VAL(tensor);
}

static Value tensor_from_array_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 2) return NIL_VAL;
    if (!IS_ARRAY(args[0])) return NIL_VAL;
    int shape[16];
    int ndim;
    if (!extract_shape(args[1], shape, &ndim)) return NIL_VAL;
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    ObjArray* data_arr = AS_ARRAY(args[0]);
    int count = data_arr->count < tensor->count ? data_arr->count : tensor->count;
    for (int i = 0; i < count; i++) {
        if (IS_INT(data_arr->elements[i])) {
            tensor->data[i] = (double)AS_INT(data_arr->elements[i]);
        } else if (IS_FLOAT(data_arr->elements[i])) {
            tensor->data[i] = AS_FLOAT(data_arr->elements[i]);
        }
    }
    return OBJ_VAL(tensor);
}

static Value tensor_shape_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* tensor = AS_TENSOR(args[0]);
    ObjArray* arr = adam_new_array(vm);
    adam_vm_push(vm, OBJ_VAL(arr)); /* GC protect */
    for (int i = 0; i < tensor->ndim; i++) {
        adam_array_push(vm, arr, INT_VAL(tensor->shape[i]));
    }
    adam_vm_pop(vm);
    return OBJ_VAL(arr);
}

static Value tensor_reshape_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* src = AS_TENSOR(args[0]);
    int shape[16];
    int ndim;
    if (!extract_shape(args[1], shape, &ndim)) return NIL_VAL;
    /* Verify element count matches */
    int count = 1;
    for (int i = 0; i < ndim; i++) count *= shape[i];
    if (count != src->count) return NIL_VAL;
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    memcpy(tensor->data, src->data, sizeof(double) * count);
    return OBJ_VAL(tensor);
}

static Value tensor_sum_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* tensor = AS_TENSOR(args[0]);
    double sum = 0.0;
    for (int i = 0; i < tensor->count; i++) {
        sum += tensor->data[i];
    }
    return FLOAT_VAL(sum);
}

static Value tensor_transpose_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* src = AS_TENSOR(args[0]);
    if (src->ndim != 2) return NIL_VAL; /* Only 2D transpose for now */
    int new_shape[2] = { src->shape[1], src->shape[0] };
    ObjTensor* result = adam_new_tensor(vm, 2, new_shape);
    int rows = src->shape[0];
    int cols = src->shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[j * rows + i] = src->data[i * cols + j];
        }
    }
    return OBJ_VAL(result);
}

/* ── Registration ──────────────────────────────────────────────────── */

void adam_register_natives(VM* vm) {
    define_native(vm, "clock",   clock_native);
    define_native(vm, "print",   print_native);
    define_native(vm, "println", println_native);
    define_native(vm, "len",     len_native);
    define_native(vm, "type_of", type_of_native);
    define_native(vm, "to_int",  to_int_native);
    define_native(vm, "to_float",to_float_native);
    define_native(vm, "sqrt",    sqrt_native);
    define_native(vm, "abs",     abs_native);

    /* Tensor natives */
    define_native(vm, "tensor_zeros",     tensor_zeros_native);
    define_native(vm, "tensor_ones",      tensor_ones_native);
    define_native(vm, "tensor_randn",     tensor_randn_native);
    define_native(vm, "tensor_from_array",tensor_from_array_native);
    define_native(vm, "tensor_shape",     tensor_shape_native);
    define_native(vm, "tensor_reshape",   tensor_reshape_native);
    define_native(vm, "tensor_sum",       tensor_sum_native);
    define_native(vm, "tensor_transpose", tensor_transpose_native);
}
