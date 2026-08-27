// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_MEM_POOL_H_
#define LIBHRX_SRC_LIBHRX_MEM_POOL_H_

#include <stddef.h>

#include "hrx_runtime.h"

// Releases the bounded-pool budget reservation associated with a buffer.
// The buffer lifetime retains |pool| until this call completes.
void hrx_mem_pool_release_allocation_budget(hrx_mem_pool_t pool, size_t size);

#endif  // LIBHRX_SRC_LIBHRX_MEM_POOL_H_
