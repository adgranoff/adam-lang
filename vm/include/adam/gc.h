/*
 * gc.h — Tri-color mark-and-sweep garbage collector
 *
 * The Adam VM uses a precise, stop-the-world, tri-color mark-and-sweep
 * garbage collector. All heap allocations go through adam_gc_reallocate(),
 * which tracks total bytes allocated and triggers collection when the
 * threshold (next_gc) is exceeded.
 *
 * Tri-color invariant:
 *   - WHITE: Not yet reached by the marker. After the mark phase, white
 *     objects are garbage and will be freed during sweep.
 *   - GRAY: Reachable from roots, but the object's own references haven't
 *     been scanned yet. Gray objects sit on a worklist (gray_stack).
 *   - BLACK: Reachable, and all references from this object have been
 *     marked. (is_marked=true AND not on the gray worklist.)
 *
 * The invariant maintained during marking: no black object directly
 * references a white object. This ensures that once an object turns
 * black, we won't miss any of its children.
 *
 * GC scheduling:
 *   After each collection, next_gc = bytes_allocated * GROW_FACTOR.
 *   This adaptive approach means: the more live data, the less frequent
 *   collections (since there's more headroom before the threshold).
 */

#pragma once

#include "adam/common.h"
#include "adam/value.h"

/* Forward declarations */
typedef struct Obj Obj;
typedef struct VM VM;

/* ── Core GC interface ─────────────────────────────────────────────── */

/* Run a full mark-and-sweep collection cycle. */
void adam_gc_collect(VM* vm);

/* Mark a Value as reachable (no-op for non-object values). */
void adam_gc_mark_value(VM* vm, Value value);

/* Mark an Obj as reachable. Adds it to the gray worklist. */
void adam_gc_mark_object(VM* vm, Obj* object);

/* ── GC-aware memory allocator ─────────────────────────────────────── */

/*
 * Central allocator for all VM heap memory. Semantics:
 *   pointer=NULL, old_size=0, new_size>0  →  malloc (allocate new block)
 *   pointer!=NULL, old_size>0, new_size=0 →  free (release block)
 *   pointer!=NULL, old_size>0, new_size>0 →  realloc (resize block)
 *
 * Triggers GC when bytes_allocated exceeds next_gc threshold.
 * Under DEBUG_STRESS_GC, triggers GC on every allocation (for testing).
 */
void* adam_gc_reallocate(VM* vm, void* pointer, size_t old_size, size_t new_size);

/* ── Convenience macros ────────────────────────────────────────────── */

#define ALLOCATE(vm, type, count) \
    (type*)adam_gc_reallocate(vm, NULL, 0, sizeof(type) * (count))

#define FREE(vm, type, pointer) \
    adam_gc_reallocate(vm, pointer, sizeof(type), 0)

#define GROW_CAPACITY(capacity) \
    ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(vm, type, pointer, old_count, new_count) \
    (type*)adam_gc_reallocate(vm, pointer, sizeof(type) * (old_count), \
                              sizeof(type) * (new_count))

#define FREE_ARRAY(vm, type, pointer, old_count) \
    adam_gc_reallocate(vm, pointer, sizeof(type) * (old_count), 0)
