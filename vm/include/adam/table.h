/*
 * table.h — Robin Hood open-addressing hash table
 *
 * This hash table uses open addressing with linear probing, enhanced by
 * the Robin Hood hashing technique. The key insight of Robin Hood hashing:
 *
 *   When inserting, if the new entry has probed farther from its ideal
 *   slot than the existing entry at the current slot, we SWAP them and
 *   continue inserting the displaced entry. This "steals from the rich
 *   (short probe) and gives to the poor (long probe)," dramatically
 *   reducing the variance in probe sequence lengths.
 *
 * Benefits over plain linear probing:
 *   - Expected maximum probe length is O(log log n) instead of O(log n)
 *   - More predictable cache behavior under high load
 *   - Lookups can terminate early: if we find an entry with a shorter
 *     PSL than what we'd have at that position, the key doesn't exist
 *
 * Why open addressing instead of chaining?
 *   - No per-entry allocation (entries live contiguously in one array)
 *   - Cache-friendly: linear probing touches adjacent memory
 *   - Lower memory overhead (no next pointers)
 *
 * Deletion uses tombstones (key=NULL, value=TRUE_VAL as sentinel) rather
 * than the backward-shift technique, for simplicity.
 *
 * Keys are always interned ObjString pointers, enabling O(1) key
 * comparison via pointer equality during normal lookups.
 */

#pragma once

#include "adam/common.h"
#include "adam/value.h"

/* Forward declarations */
typedef struct ObjString ObjString;
typedef struct VM VM;

typedef struct {
    ObjString* key;     /* NULL = empty or tombstone */
    Value value;        /* TRUE_VAL when key=NULL means tombstone */
    uint32_t psl;       /* Probe Sequence Length (distance from ideal slot) */
} Entry;

typedef struct {
    int count;          /* Number of live entries (excludes tombstones) */
    int capacity;       /* Total slots in entries array (always power of 2) */
    Entry* entries;
} Table;

void       adam_table_init(Table* table);
void       adam_table_free(VM* vm, Table* table);
bool       adam_table_get(Table* table, ObjString* key, Value* value);
bool       adam_table_set(VM* vm, Table* table, ObjString* key, Value value);
bool       adam_table_delete(Table* table, ObjString* key);
ObjString* adam_table_find_string(Table* table, const char* chars,
                                  int length, uint32_t hash);
void       adam_table_mark(VM* vm, Table* table);
void       adam_table_remove_white(Table* table);
