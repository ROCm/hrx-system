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

#ifdef __cplusplus
extern "C" {
#endif

// Translates a validated public array into caller-owned construction scratch.
// Each device/family must belong to group. out_families has family_count
// entries and remains live for every native use of out_scope. This acquires no
// owners.
iree_status_t hrx_mem_pool_prepare_scope(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    iree_hal_pool_family_access_t* out_families,
    iree_hal_pool_scope_t* out_scope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_MEM_POOL_H_
