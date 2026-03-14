/*
 * table.c — Robin Hood open-addressing hash table
 *
 * Open addressing stores all entries in a single contiguous array. When a
 * collision occurs, we probe linearly to the next slot. This is extremely
 * cache-friendly: probing touches adjacent cache lines.
 *
 * Robin Hood hashing enhancement:
 *   Each entry tracks its PSL (Probe Sequence Length) — how far it is from
 *   its ideal slot. During insertion, if the new entry's PSL exceeds the
 *   existing entry's PSL at a slot, we SWAP them. The displaced entry
 *   continues probing. This "steal from the rich, give to the poor"
 *   strategy equalizes probe lengths, giving:
 *     - O(log log n) expected maximum probe length (vs O(log n) for plain)
 *     - Early termination for lookups: if our PSL exceeds the stored PSL,
 *       the key doesn't exist (no need to probe further)
 *
 * Deletion via tombstones:
 *   Deleting an entry leaves a tombstone (key=NULL, value=TRUE_VAL).
 *   Tombstones are treated as occupied during probing (so chains aren't
 *   broken) but can be overwritten during insertion. The count does NOT
 *   include tombstones, but capacity calculations do (tombstones occupy
 *   slots that affect load factor).
 *
 * Load factor: 0.75
 *   Above this threshold, we grow the table (double capacity). This keeps
 *   average probe length around 1.5 for successful lookups.
 */

#include "adam/common.h"
#include "adam/table.h"
#include "adam/object.h"
#include "adam/value.h"
#include "adam/gc.h"
#include "adam/vm.h"

void adam_table_init(Table* table) {
    table->count = 0;
    table->capacity = 0;
    table->entries = NULL;
}

void adam_table_free(VM* vm, Table* table) {
    FREE_ARRAY(vm, Entry, table->entries, table->capacity);
    adam_table_init(table);
}

/*
 * Find the entry for a key, or the first empty/tombstone slot where
 * it would go. Uses Robin Hood probing with early termination.
 */
static Entry* find_entry(Entry* entries, int capacity, ObjString* key) {
    uint32_t index = key->hash & (capacity - 1); /* capacity is power of 2 */
    Entry* tombstone = NULL;
    uint32_t psl = 0;

    for (;;) {
        Entry* entry = &entries[index];

        if (entry->key == NULL) {
            if (IS_NIL(entry->value)) {
                /* Empty slot — return tombstone if we passed one, else this. */
                return tombstone != NULL ? tombstone : entry;
            } else {
                /* Tombstone. Remember it but keep probing. */
                if (tombstone == NULL) tombstone = entry;
            }
        } else if (entry->key == key) {
            /* Found the key (pointer equality — strings are interned). */
            return entry;
        }

        /* Robin Hood early termination: if the entry at this slot has a
         * shorter probe distance than we've traveled, our key can't be
         * further along. It would have displaced this entry during insert. */
        if (entry->key != NULL && entry->psl < psl) {
            return tombstone != NULL ? tombstone : entry;
        }

        psl++;
        index = (index + 1) & (capacity - 1);
    }
}

static void adjust_capacity(VM* vm, Table* table, int capacity) {
    Entry* entries = ALLOCATE(vm, Entry, capacity);
    for (int i = 0; i < capacity; i++) {
        entries[i].key = NULL;
        entries[i].value = NIL_VAL;
        entries[i].psl = 0;
    }

    /* Re-insert all existing entries (skip tombstones). */
    table->count = 0;
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (entry->key == NULL) continue;

        /* Find new slot using Robin Hood insertion. */
        uint32_t index = entry->key->hash & (capacity - 1);
        uint32_t psl = 0;
        ObjString* key = entry->key;
        Value value = entry->value;

        for (;;) {
            Entry* dest = &entries[index];
            if (dest->key == NULL) {
                dest->key = key;
                dest->value = value;
                dest->psl = psl;
                table->count++;
                break;
            }
            /* Robin Hood: swap if new entry has longer probe distance. */
            if (psl > dest->psl) {
                ObjString* tmp_key = dest->key;
                Value tmp_val = dest->value;
                uint32_t tmp_psl = dest->psl;
                dest->key = key;
                dest->value = value;
                dest->psl = psl;
                key = tmp_key;
                value = tmp_val;
                psl = tmp_psl;
            }
            psl++;
            index = (index + 1) & (capacity - 1);
        }
    }

    FREE_ARRAY(vm, Entry, table->entries, table->capacity);
    table->entries = entries;
    table->capacity = capacity;
}

bool adam_table_get(Table* table, ObjString* key, Value* value) {
    if (table->count == 0) return false;

    Entry* entry = find_entry(table->entries, table->capacity, key);
    if (entry->key == NULL) return false;

    *value = entry->value;
    return true;
}

bool adam_table_set(VM* vm, Table* table, ObjString* key, Value value) {
    if (table->count + 1 > table->capacity * ADAM_TABLE_MAX_LOAD) {
        int capacity = GROW_CAPACITY(table->capacity);
        adjust_capacity(vm, table, capacity);
    }

    /* Robin Hood insertion into the live table. */
    uint32_t index = key->hash & (table->capacity - 1);
    uint32_t psl = 0;
    bool is_new_key = true;

    for (;;) {
        Entry* entry = &table->entries[index];

        if (entry->key == NULL) {
            /* Empty or tombstone — insert here. */
            bool was_tombstone = !IS_NIL(entry->value);
            entry->key = key;
            entry->value = value;
            entry->psl = psl;
            if (!was_tombstone) table->count++;
            return is_new_key;
        }

        if (entry->key == key) {
            /* Key exists — update value. */
            entry->value = value;
            return false;
        }

        /* Robin Hood: if our PSL exceeds the existing entry's, swap. */
        if (psl > entry->psl) {
            /* Swap the new entry with the existing one. */
            ObjString* tmp_key = entry->key;
            Value tmp_val = entry->value;
            uint32_t tmp_psl = entry->psl;
            entry->key = key;
            entry->value = value;
            entry->psl = psl;
            key = tmp_key;
            value = tmp_val;
            psl = tmp_psl;
            is_new_key = true; /* The displaced entry needs a new home. */
        }

        psl++;
        index = (index + 1) & (table->capacity - 1);
    }
}

bool adam_table_delete(Table* table, ObjString* key) {
    if (table->count == 0) return false;

    Entry* entry = find_entry(table->entries, table->capacity, key);
    if (entry->key == NULL) return false;

    /* Place a tombstone: key=NULL, value=TRUE_VAL (sentinel). */
    entry->key = NULL;
    entry->value = TRUE_VAL;
    entry->psl = 0;
    return true;
}

/*
 * Find a string in the table by raw characters (not by pointer).
 * Used during string interning: we need to check if an identical string
 * exists BEFORE creating the ObjString, so we can't use pointer equality.
 */
ObjString* adam_table_find_string(Table* table, const char* chars,
                                   int length, uint32_t hash) {
    if (table->count == 0) return NULL;

    uint32_t index = hash & (table->capacity - 1);
    uint32_t psl = 0;

    for (;;) {
        Entry* entry = &table->entries[index];

        if (entry->key == NULL) {
            /* Empty (non-tombstone) slot means the string isn't here. */
            if (IS_NIL(entry->value)) return NULL;
        } else if (entry->key->length == length &&
                   entry->key->hash == hash &&
                   memcmp(entry->key->chars, chars, length) == 0) {
            /* Found it: same hash, same length, same bytes. */
            return entry->key;
        }

        /* Robin Hood early termination. */
        if (entry->key != NULL && entry->psl < psl) return NULL;

        psl++;
        index = (index + 1) & (table->capacity - 1);
    }
}

/* ── GC support ────────────────────────────────────────────────────── */

void adam_table_mark(VM* vm, Table* table) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        adam_gc_mark_object(vm, (Obj*)entry->key);
        adam_gc_mark_value(vm, entry->value);
    }
}

/*
 * Remove entries whose key is an unmarked (white) string. Used for the
 * string intern table, which holds weak references: if no one else
 * references a string, it should be collected.
 */
void adam_table_remove_white(Table* table) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (entry->key != NULL && !entry->key->obj.is_marked) {
            adam_table_delete(table, entry->key);
        }
    }
}
