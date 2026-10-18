// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_UTIL_POOL_TEST_UTIL_H_
#define IREE_HAL_CTS_UTIL_POOL_TEST_UTIL_H_

#include <future>

#include "iree/hal/device_group.h"
#include "iree/hal/memory/fixed_block_pool.h"

namespace iree::hal::cts {

// Waits for native committed capacity to reach |expected| through the pool's
// public capacity notification. Callers must ensure no new allocations race the
// observation. Returns false if the notification wait fails.
bool WaitPoolCommittedBytes(iree_hal_pool_t* pool, iree_device_size_t expected);

// Facts captured while releasing an imported host allocation.
struct ImportedHostAllocationReleaseResult {
  // True when the callback received the expected external storage class.
  bool storage_class_matches = false;
  // True when the callback received the original host pointer.
  bool payload_matches = false;
};

// Heap-owned state retained until an asynchronous imported host allocation is
// released. After a successful import, ownership transfers to
// ReleaseImportedHostAllocation.
struct ImportedHostAllocation {
  ~ImportedHostAllocation();

  // Allocator that owns |pointer|.
  iree_allocator_t host_allocator = iree_allocator_null();
  // Host allocation transferred to the external storage owner.
  void* pointer = nullptr;
  // External storage class expected by the release callback.
  iree_hal_external_storage_class_t storage_class = {};
  // Result fulfilled after the callback releases |pointer|.
  std::promise<ImportedHostAllocationReleaseResult> release_result;
};

// Takes ownership of a heap-allocated ImportedHostAllocation, releases its host
// pointer, and publishes the observed transport payload. A future obtained
// from |release_result| is the completion edge for asynchronous native
// retirement.
void ReleaseImportedHostAllocation(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload);

// Creates a native slab pool scoped to one queue family.
iree_status_t CreateQueueSlabPool(iree_hal_device_group_t* device_group,
                                  const iree_hal_queue_family_t* queue_family,
                                  iree_hal_buffer_usage_t usage,
                                  iree_hal_pool_host_access_t host_access,
                                  iree_allocator_t host_allocator,
                                  iree_hal_pool_t** out_pool);

// Allocates one range from |backing_pool| and uses it as a finite block arena.
// The requested blocks_per_slab determines its initial capacity. The caller
// owns |backing_pool| and must release the finite pool before it.
iree_status_t CreateFiniteBlockPool(iree_hal_pool_t* backing_pool,
                                    iree_hal_fixed_block_pool_options_t options,
                                    iree_allocator_t host_allocator,
                                    iree_hal_pool_t** out_pool);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_POOL_TEST_UTIL_H_
