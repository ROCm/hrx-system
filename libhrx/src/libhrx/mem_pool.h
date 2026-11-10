// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_MEM_POOL_H_
#define LIBHRX_SRC_LIBHRX_MEM_POOL_H_

#include "hrx_memory.h"
#include "iree/base/api.h"
#include "iree/hal/pool.h"

// Explicit native allocation strategy. No HIP policy or execution ownership.
typedef struct hrx_mem_pool_s {
  // Public handle references; native children retain the native pool directly.
  iree_atomic_ref_count_t ref_count;
  // Cached immutable memory type for buffer wrapper initialization.
  hrx_memory_type_t memory_type;
  // Owned native pool containing access and allocation strategy.
  iree_hal_pool_t* hal_pool;
  // Retained group providing topology and native progress lifetime.
  hrx_device_group_t group;
  // Optional retained finite-range owner, propagated through descendants.
  // Preserves HRX final-release obligations beyond native buffer ownership.
  hrx_buffer_t backing_buffer;
} hrx_mem_pool_s;

#endif  // LIBHRX_SRC_LIBHRX_MEM_POOL_H_
