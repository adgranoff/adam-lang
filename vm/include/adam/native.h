/*
 * native.h — Native function bridge
 *
 * Registers built-in functions (clock, print, len, etc.) into the
 * VM's global variable table so they're available to Adam programs.
 */

#pragma once

#include "adam/vm.h"

/* Register all native functions into vm->globals. */
void adam_register_natives(VM* vm);
