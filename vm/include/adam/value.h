/*
 * value.h — NaN-boxed value representation
 *
 * Every value in Adam (integers, floats, booleans, nil, and heap object
 * pointers) is encoded in a single 64-bit word using NaN boxing. This
 * technique exploits the IEEE 754 floating-point specification: a "quiet
 * NaN" has a specific bit pattern with many unused bits in the mantissa
 * that we repurpose as type tags and payloads.
 *
 * Why NaN boxing?
 *   1. Cache-friendly: every value is exactly 8 bytes, no indirection.
 *   2. No type-tag branch for float operations: if the bits aren't a
 *      quiet NaN, the value IS a valid double — zero-cost float check.
 *   3. Pointer extraction is a single mask operation.
 *   4. Comparison of non-float types is bitwise equality.
 *
 * Bit layout (64 bits, MSB first):
 *
 *   [S][EEEEEEEEEEE][1][Q][TTTTTTTTTTTT...PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
 *    ^  exponent=7FF  ^  ^  mantissa (51 bits)
 *    sign bit         |  quiet bit (1 = quiet NaN)
 *                     |
 *              Intel FP indefinite bit (must be 1 for quiet NaN)
 *
 *   Float:   Any valid double that is NOT a quiet NaN pattern.
 *   Nil:     QNAN | 0x01
 *   False:   QNAN | 0x02
 *   True:    QNAN | 0x03
 *   Int32:   QNAN | TAG_INT | (uint32_t value in lower 32 bits)
 *   Object*: SIGN_BIT | QNAN | (48-bit pointer in lower 48 bits)
 *
 * The sign bit distinguishes objects (sign=1) from special values (sign=0).
 * On x86-64, user-space pointers fit in 48 bits, so no bits are lost.
 */

#pragma once

#include "adam/common.h"

/* Forward declarations */
typedef struct Obj Obj;
typedef struct ObjString ObjString;

/* Quiet NaN: exponent all 1s, quiet bit set, Intel indefinite bit set.
 * We use 0x7ffc... (bits 50 and 51 of mantissa set) to avoid
 * colliding with hardware NaN values. */
#define QNAN     ((uint64_t)0x7ffc000000000000)

/* Sign bit — used to tag object pointers */
#define SIGN_BIT ((uint64_t)0x8000000000000000)

/* Type tags for non-float, non-object values (stored in mantissa bits) */
#define TAG_NIL   1
#define TAG_FALSE 2
#define TAG_TRUE  3

/* Integer tag: bit 49 set. Lower 32 bits hold the int32_t value. */
#define TAG_INT   ((uint64_t)0x0002000000000000)

/* The Value type — a single 64-bit word */
typedef uint64_t Value;

/* ── Constructors ──────────────────────────────────────────────────── */

#define NIL_VAL       ((Value)(QNAN | TAG_NIL))
#define FALSE_VAL     ((Value)(QNAN | TAG_FALSE))
#define TRUE_VAL      ((Value)(QNAN | TAG_TRUE))
#define BOOL_VAL(b)   ((b) ? TRUE_VAL : FALSE_VAL)
#define INT_VAL(i)    ((Value)(QNAN | TAG_INT | (uint32_t)(i)))
#define OBJ_VAL(obj)  ((Value)(SIGN_BIT | QNAN | (uint64_t)(uintptr_t)(obj)))

/* Float ↔ Value conversion uses union-based type punning (C99 §6.5.2.3).
 * This is well-defined in C (unlike reinterpret_cast in C++). */
static inline Value adam_float_to_value(double d) {
    union { double d; Value v; } u;
    u.d = d;
    return u.v;
}

static inline double adam_value_to_float(Value v) {
    union { Value v; double d; } u;
    u.v = v;
    return u.d;
}

#define FLOAT_VAL(f)  (adam_float_to_value(f))

/* ── Type checks ───────────────────────────────────────────────────── */

#define IS_NIL(v)    ((v) == NIL_VAL)
#define IS_BOOL(v)   (((v) | 1) == TRUE_VAL)  /* TRUE and FALSE differ only in bit 0 */
#define IS_INT(v)    (((v) & (QNAN | TAG_INT)) == (QNAN | TAG_INT))
#define IS_FLOAT(v)  (((v) & QNAN) != QNAN)   /* If not a NaN pattern, it's a real double */
#define IS_NUMBER(v) (IS_INT(v) || IS_FLOAT(v))
#define IS_OBJ(v)    (((v) & (SIGN_BIT | QNAN)) == (SIGN_BIT | QNAN))

/* ── Extractors ────────────────────────────────────────────────────── */

#define AS_BOOL(v)   ((v) == TRUE_VAL)
#define AS_INT(v)    ((int32_t)((v) & 0xFFFFFFFF))
#define AS_FLOAT(v)  (adam_value_to_float(v))
#define AS_OBJ(v)    ((Obj*)(uintptr_t)((v) & ~(SIGN_BIT | QNAN)))

/* Convert any numeric value to double for mixed arithmetic */
static inline double adam_as_number(Value v) {
    if (IS_INT(v)) return (double)AS_INT(v);
    return AS_FLOAT(v);
}

/* ── Operations ────────────────────────────────────────────────────── */

void adam_print_value(Value value);
bool adam_values_equal(Value a, Value b);
