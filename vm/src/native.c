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

/* ── MNIST tensor natives ──────────────────────────────────────────── */

static Value tensor_exp_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* src = AS_TENSOR(args[0]);
    ObjTensor* result = adam_new_tensor(vm, src->ndim, src->shape);
    for (int i = 0; i < src->count; i++) {
        result->data[i] = exp(src->data[i]);
    }
    return OBJ_VAL(result);
}

static Value tensor_log_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* src = AS_TENSOR(args[0]);
    ObjTensor* result = adam_new_tensor(vm, src->ndim, src->shape);
    for (int i = 0; i < src->count; i++) {
        result->data[i] = log(src->data[i]);
    }
    return OBJ_VAL(result);
}

static Value tensor_relu_native(VM* vm, int arg_count, Value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* src = AS_TENSOR(args[0]);
    ObjTensor* result = adam_new_tensor(vm, src->ndim, src->shape);
    for (int i = 0; i < src->count; i++) {
        result->data[i] = src->data[i] > 0.0 ? src->data[i] : 0.0;
    }
    return OBJ_VAL(result);
}

static Value tensor_relu_backward_native(VM* vm, int arg_count, Value* args) {
    /* tensor_relu_backward(pre_activation, upstream_grad) → Tensor
     * Gradient is upstream_grad where pre_activation > 0, else 0. */
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL;
    ObjTensor* z = AS_TENSOR(args[0]);
    ObjTensor* dout = AS_TENSOR(args[1]);
    if (z->count != dout->count) return NIL_VAL;
    ObjTensor* result = adam_new_tensor(vm, z->ndim, z->shape);
    for (int i = 0; i < z->count; i++) {
        result->data[i] = z->data[i] > 0.0 ? dout->data[i] : 0.0;
    }
    return OBJ_VAL(result);
}

static Value tensor_max_native(VM* vm, int arg_count, Value* args) {
    (void)vm;
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* t = AS_TENSOR(args[0]);
    if (t->count == 0) return FLOAT_VAL(0.0);
    double max_val = t->data[0];
    for (int i = 1; i < t->count; i++) {
        if (t->data[i] > max_val) max_val = t->data[i];
    }
    return FLOAT_VAL(max_val);
}

static Value tensor_sum_axis_native(VM* vm, int arg_count, Value* args) {
    /* tensor_sum_axis(tensor, axis) → Tensor
     * For 2D: axis=0 sums columns (result shape [1, cols]),
     *         axis=1 sums rows (result shape [rows, 1]). */
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_INT(args[1])) return NIL_VAL;
    ObjTensor* t = AS_TENSOR(args[0]);
    int axis = AS_INT(args[1]);
    if (t->ndim != 2 || axis < 0 || axis >= t->ndim) return NIL_VAL;
    int rows = t->shape[0];
    int cols = t->shape[1];
    if (axis == 0) {
        /* Sum along rows → result shape [1, cols] */
        int shape[2] = {1, cols};
        ObjTensor* result = adam_new_tensor(vm, 2, shape);
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += t->data[i * cols + j];
            }
            result->data[j] = sum;
        }
        return OBJ_VAL(result);
    } else {
        /* Sum along cols → result shape [rows, 1] */
        int shape[2] = {rows, 1};
        ObjTensor* result = adam_new_tensor(vm, 2, shape);
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += t->data[i * cols + j];
            }
            result->data[i] = sum;
        }
        return OBJ_VAL(result);
    }
}

static Value tensor_slice_native(VM* vm, int arg_count, Value* args) {
    /* tensor_slice(tensor, start_row, count) → Tensor
     * Extract rows [start, start+count) from a 2D tensor. */
    if (arg_count != 3 || !IS_TENSOR(args[0])) return NIL_VAL;
    ObjTensor* t = AS_TENSOR(args[0]);
    int start = IS_INT(args[1]) ? AS_INT(args[1]) : (int)adam_as_number(args[1]);
    int count = IS_INT(args[2]) ? AS_INT(args[2]) : (int)adam_as_number(args[2]);
    if (t->ndim == 1) {
        /* 1D tensor: slice elements */
        if (start < 0 || start + count > t->count) return NIL_VAL;
        int shape[1] = {count};
        ObjTensor* result = adam_new_tensor(vm, 1, shape);
        memcpy(result->data, t->data + start, sizeof(double) * count);
        return OBJ_VAL(result);
    }
    if (t->ndim != 2) return NIL_VAL;
    int cols = t->shape[1];
    if (start < 0 || start + count > t->shape[0]) return NIL_VAL;
    int shape[2] = {count, cols};
    ObjTensor* result = adam_new_tensor(vm, 2, shape);
    memcpy(result->data, t->data + start * cols, sizeof(double) * count * cols);
    return OBJ_VAL(result);
}

static Value tensor_one_hot_native(VM* vm, int arg_count, Value* args) {
    /* tensor_one_hot(labels, num_classes) → Tensor
     * labels is a 1D tensor of integer class indices, result is [N, num_classes]. */
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_INT(args[1])) return NIL_VAL;
    ObjTensor* labels = AS_TENSOR(args[0]);
    int num_classes = AS_INT(args[1]);
    int n = labels->count;
    int shape[2] = {n, num_classes};
    ObjTensor* result = adam_new_tensor(vm, 2, shape);
    /* adam_new_tensor zeroes data */
    for (int i = 0; i < n; i++) {
        int cls = (int)labels->data[i];
        if (cls >= 0 && cls < num_classes) {
            result->data[i * num_classes + cls] = 1.0;
        }
    }
    return OBJ_VAL(result);
}

static Value tensor_load_native(VM* vm, int arg_count, Value* args) {
    /* tensor_load(path) → Tensor
     * Binary format: [ndim:i32][shape[0]:i32]...[shape[n]:i32][data:f64*] */
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    const char* path = AS_CSTRING(args[0]);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tensor_load: cannot open '%s'\n", path);
        fprintf(stderr, "Run 'just prepare-mnist' to download MNIST data first.\n");
        exit(1);
    }
    int32_t ndim;
    if (fread(&ndim, sizeof(int32_t), 1, f) != 1 || ndim <= 0 || ndim > 16) {
        fclose(f);
        return NIL_VAL;
    }
    int shape[16];
    int count = 1;
    for (int i = 0; i < ndim; i++) {
        int32_t dim;
        if (fread(&dim, sizeof(int32_t), 1, f) != 1) { fclose(f); return NIL_VAL; }
        shape[i] = dim;
        count *= dim;
    }
    ObjTensor* tensor = adam_new_tensor(vm, ndim, shape);
    adam_vm_push(vm, OBJ_VAL(tensor)); /* GC protect */
    if (fread(tensor->data, sizeof(double), count, f) != (size_t)count) {
        adam_vm_pop(vm);
        fclose(f);
        return NIL_VAL;
    }
    fclose(f);
    adam_vm_pop(vm);
    return OBJ_VAL(tensor);
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

    /* MNIST tensor natives */
    define_native(vm, "tensor_exp",           tensor_exp_native);
    define_native(vm, "tensor_log",           tensor_log_native);
    define_native(vm, "tensor_relu",          tensor_relu_native);
    define_native(vm, "tensor_relu_backward", tensor_relu_backward_native);
    define_native(vm, "tensor_max",           tensor_max_native);
    define_native(vm, "tensor_sum_axis",      tensor_sum_axis_native);
    define_native(vm, "tensor_slice",         tensor_slice_native);
    define_native(vm, "tensor_one_hot",       tensor_one_hot_native);
    define_native(vm, "tensor_load",          tensor_load_native);
}
