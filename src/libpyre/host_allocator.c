// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

_Static_assert(sizeof(pyre_host_allocator_t) == sizeof(iree_allocator_t),
               "host allocator layout mismatch");
_Static_assert(_Alignof(pyre_host_allocator_t) == _Alignof(iree_allocator_t),
               "host allocator alignment mismatch");

// Initialized before main() via constructor. iree_allocator_system() is
// safe to call at load time (it's an inline returning two constants).
pyre_host_allocator_t pyre_host_allocator_system_value;

__attribute__((constructor))
static void pyre_host_allocator_init(void) {
  iree_allocator_t sys = iree_allocator_system();
  pyre_host_allocator_system_value.self = sys.self;
  pyre_host_allocator_system_value.ctl = (void*)sys.ctl;
}
