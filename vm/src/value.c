/*
 * value.c — NaN-boxed value operations
 *
 * NaN boxing packs every Adam value into 8 bytes by exploiting the IEEE 754
 * quiet NaN representation. A quiet NaN has exponent bits all set and the
 * quiet bit set, leaving ~51 bits of mantissa unused. We use those bits
 * (plus the sign bit) to encode type tags and payloads:
 *
 *   - If the value is NOT a quiet NaN → it's a real double (zero-cost check)
 *   - QNAN | 0x01 = nil
 *   - QNAN | 0x02 = false, QNAN | 0x03 = true
 *   - QNAN | TAG_INT | lower32 = 32-bit signed integer
 *   - SIGN_BIT | QNAN | lower48 = heap object pointer
 *
 * This is the same technique used by LuaJIT, SpiderMonkey (Firefox), and
 * JavaScriptCore (Safari) in production. It's fast because:
 *   - Float check is a single bitwise AND + compare (no branch)
 *   - Pointer extraction is a single AND mask
 *   - Equality for non-floats is bitwise comparison
 *   - Cache-friendly: all values are exactly 8 bytes, no indirection
 */

#include "adam/common.h"
#include "adam/value.h"
#include "adam/object.h"

void adam_print_value(Value value) {
    if (IS_NIL(value)) {
        printf("nil");
    } else if (IS_BOOL(value)) {
        printf(AS_BOOL(value) ? "true" : "false");
    } else if (IS_INT(value)) {
        printf("%d", AS_INT(value));
    } else if (IS_FLOAT(value)) {
        printf("%g", AS_FLOAT(value));
    } else if (IS_OBJ(value)) {
        adam_print_object(value);
    }
}

bool adam_values_equal(Value a, Value b) {
    /* Fast path: bitwise equality handles nil, bool, int, and interned
     * strings (which share the same pointer for equal content). */
    if (a == b) return true;

    /* If both are floats, compare as doubles (handles -0.0 == 0.0 and
     * NaN != NaN correctly, though our NaN tag occupies the NaN space). */
    if (IS_FLOAT(a) && IS_FLOAT(b)) {
        return AS_FLOAT(a) == AS_FLOAT(b);
    }

    /* Mixed int/float comparison */
    if (IS_INT(a) && IS_FLOAT(b)) return (double)AS_INT(a) == AS_FLOAT(b);
    if (IS_FLOAT(a) && IS_INT(b)) return AS_FLOAT(a) == (double)AS_INT(b);

    return false;
}
