/*
 * common.h — Shared definitions for the Adam virtual machine
 *
 * This header provides the common includes, constants, and configuration
 * macros used throughout the VM implementation. Every .c file in the VM
 * includes this first.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>

/* Maximum depth of the value stack. Each function call uses a portion
 * of this shared stack (its "window" starts at frame->slots). 256 slots
 * is generous for non-pathological programs. */
#define ADAM_STACK_MAX 256

/* Maximum call depth. Prevents infinite recursion from eating all memory
 * before we can report a useful error. */
#define ADAM_FRAMES_MAX 64

/* After a GC cycle, set next_gc = bytes_allocated * this factor.
 * Higher = fewer collections but more peak memory. 2 is a common default. */
#define ADAM_GC_HEAP_GROW_FACTOR 2

/* Hash table maximum load factor. Above this, we grow the table.
 * 0.75 balances memory usage vs probe length. */
#define ADAM_TABLE_MAX_LOAD 0.75

/* Number of possible uint8_t values — used for local variable slots
 * and upvalue indices, which are encoded as single bytes. */
#define UINT8_COUNT (UINT8_MAX + 1)
