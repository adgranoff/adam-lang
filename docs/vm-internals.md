# VM Internals

The Adam VM is a stack-based bytecode interpreter written in C. It executes `.adamb` files produced by the Rust compiler.

## NaN Boxing

Every value in Adam is represented as a single 64-bit word using NaN boxing. This technique exploits unused bits in IEEE 754 quiet NaN patterns to encode type tags and payloads.

### Bit Layout

```
64 bits, MSB first:

[S][EEEEEEEEEEE][1][Q][TTTTTTTTTTTT...PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
 ^  exponent=7FF  ^  ^  mantissa (51 bits)
 sign bit         |  quiet bit (1 = quiet NaN)
                  Intel FP indefinite bit (must be 1)
```

### Value Encoding

| Type | Encoding |
|------|----------|
| Float | Any valid double that is NOT a quiet NaN pattern |
| Nil | `QNAN \| 0x01` |
| False | `QNAN \| 0x02` |
| True | `QNAN \| 0x03` |
| Int32 | `QNAN \| TAG_INT \| (uint32_t in lower 32 bits)` |
| Object* | `SIGN_BIT \| QNAN \| (48-bit pointer in lower bits)` |

Where:
- `QNAN = 0x7ffc000000000000` (exponent all 1s, quiet bit set, Intel indefinite bit set)
- `SIGN_BIT = 0x8000000000000000`
- `TAG_INT = 0x0002000000000000`

The sign bit distinguishes heap objects (sign=1) from special values (sign=0). On x86-64, user-space pointers fit in 48 bits.

### Type Checking

Checking a value's type is a bitmask operation:

```c
#define IS_FLOAT(v)  (((v) & QNAN) != QNAN)
#define IS_NIL(v)    ((v) == (QNAN | 0x01))
#define IS_BOOL(v)   (((v) & ~1ULL) == (QNAN | 0x02))
#define IS_INT(v)    (((v) & (QNAN | TAG_INT)) == (QNAN | TAG_INT))
#define IS_OBJ(v)    (((v) & (SIGN_BIT | QNAN)) == (SIGN_BIT | QNAN))
```

Float operations require no type check at all -- if the bits don't match QNAN, the value IS a valid double.

## Dispatch Loop

The VM's main loop reads opcodes and dispatches to handlers:

### Computed Goto (GCC/Clang)

```c
static void *dispatch_table[] = {
    &&op_const, &&op_nil, &&op_true, &&op_false,
    &&op_add, &&op_sub, ...
};

#define DISPATCH() goto *dispatch_table[READ_BYTE()]

op_add:
    // ... handle add
    DISPATCH();
```

Each handler ends with `DISPATCH()`, which jumps directly to the next opcode's handler. This avoids the branch prediction overhead of a central switch statement.

### Switch Fallback (MSVC)

```c
for (;;) {
    switch (READ_BYTE()) {
        case OP_ADD: ... break;
        case OP_SUB: ... break;
        ...
    }
}
```

## Garbage Collector

The GC uses a tri-color mark-and-sweep algorithm:

### Colors

| Color | Meaning |
|-------|---------|
| White | Not yet reached by the marker. Will be freed if still white after marking. |
| Gray | Reached but children not yet visited. Sits on the gray worklist. |
| Black | Fully traced. All references from this object have been visited. |

### Collection Cycle

1. **Root scanning**: Push all roots (stack values, global table, open upvalues) onto the gray worklist.
2. **Mark phase**: Pop objects from the gray worklist. For each, mark it black and push any white objects it references onto the gray worklist. Repeat until the worklist is empty.
3. **Sweep phase**: Walk the full object list. Free any white objects. Reset surviving objects to white for the next cycle.

### Triggers

Collection runs when total allocated bytes exceed `next_gc` (initially 1 MB). After each collection, `next_gc` is set to `2 * bytes_allocated`, providing an adaptive growth factor.

## Hash Table

The global variable table and string interning table use open addressing with Robin Hood hashing:

### Robin Hood Insertion

When inserting a key that hashes to an already-occupied slot, compare the "probe distance" (how far each entry is from its ideal slot). If the new entry has traveled farther than the existing entry, swap them and continue inserting the displaced entry. This bounds the variance of probe lengths.

### Hash Function

FNV-1a (Fowler-Noll-Vo) is used for string hashing:

```c
static uint32_t fnv1a(const char *key, int length) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < length; i++) {
        hash ^= (uint8_t)key[i];
        hash *= 16777619;
    }
    return hash;
}
```

### Load Factor

The table grows (doubles capacity) when the load factor exceeds 75%.

## Object Types

All heap-allocated objects share a common header:

```c
struct Obj {
    ObjType type;       // discriminant
    bool is_marked;     // GC mark bit
    struct Obj *next;   // intrusive linked list for GC
};
```

| Type | Fields | Description |
|------|--------|-------------|
| ObjString | chars, length, hash | Interned immutable string |
| ObjClosure | function, upvalues[] | Function + captured environment |
| ObjUpvalue | location, closed, next | Captured variable (open or closed) |
| ObjArray | elements[], count, capacity | Dynamic array |

### String Interning

All strings are interned: before allocating a new ObjString, the VM checks the string table. If an identical string exists, it reuses the pointer. This makes string equality O(1) (pointer comparison).

## Bytecode Format (.adamb)

```
Offset  Size    Field
0       4       Magic bytes: "ADAM"
4       1       Version (currently 1)
5       4       Constant pool length (N)
9       N       Constant pool entries
9+N     4       Code length (M)
13+N    M       Bytecode instructions
```

## Instruction Set

All instructions are stack-based. Operands are pushed onto the stack; instructions consume and produce stack values.

| Opcode | Operand | Stack Effect | Description |
|--------|---------|--------------|-------------|
| `CONST` | u16 idx | +1 | Push constant from pool |
| `NIL` | - | +1 | Push nil |
| `TRUE` | - | +1 | Push true |
| `FALSE` | - | +1 | Push false |
| `ADD` | - | -1 | Pop two, push sum |
| `SUB` | - | -1 | Pop two, push difference |
| `MUL` | - | -1 | Pop two, push product |
| `DIV` | - | -1 | Pop two, push quotient |
| `MOD` | - | -1 | Pop two, push remainder |
| `POW` | - | -1 | Pop two, push power |
| `NEG` | - | 0 | Negate top of stack |
| `EQ` | - | -1 | Equality comparison |
| `NEQ` | - | -1 | Inequality comparison |
| `LT` | - | -1 | Less than |
| `GT` | - | -1 | Greater than |
| `LTE` | - | -1 | Less than or equal |
| `GTE` | - | -1 | Greater than or equal |
| `NOT` | - | 0 | Logical not |
| `LOAD_LOCAL` | u16 slot | +1 | Push local variable |
| `STORE_LOCAL` | u16 slot | 0 | Store to local variable |
| `LOAD_GLOBAL` | u16 idx | +1 | Push global variable |
| `STORE_GLOBAL` | u16 idx | 0 | Store to global variable |
| `LOAD_UPVALUE` | u16 idx | +1 | Push captured upvalue |
| `STORE_UPVALUE` | u16 idx | 0 | Store to captured upvalue |
| `CLOSE_UPVALUE` | - | -1 | Close an upvalue (move to heap) |
| `JUMP` | u16 offset | 0 | Unconditional jump |
| `JUMP_IF_FALSE` | u16 offset | -1 | Conditional jump |
| `LOOP` | u16 offset | 0 | Backward jump |
| `CALL` | u8 argc | -(argc) | Call function |
| `CLOSURE` | u16 idx, upvalue descriptors | +1 | Create closure |
| `RETURN` | - | -1 | Return from function |
| `ARRAY_NEW` | u16 count | -(count-1) | Create array from stack values |
| `ARRAY_GET` | - | -1 | Index into array |
| `ARRAY_SET` | - | -2 | Store at array index |
| `ARRAY_LEN` | - | 0 | Push array length |
| `POP` | - | -1 | Discard top of stack |
| `PRINT` | - | -1 | Print and pop |
| `CONCAT` | - | -1 | String concatenation |
