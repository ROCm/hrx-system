// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_UTIL_ATOMIC_TEST_UTIL_H_
#define IREE_HAL_CTS_UTIL_ATOMIC_TEST_UTIL_H_

#include "iree/hal/api.h"

namespace iree::hal::cts {

// Atomic behavior required by a CTS test.
struct AtomicTestRequirements {
  // Atomic operations that must be supported.
  iree_hal_atomic_operation_flags_t operation_flags =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  // Wait conditions that must be supported when testing atomic waits.
  iree_hal_atomic_wait_condition_flags_t wait_condition_flags =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE;
  // Buffer usage bits required by the test and its readback path.
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_NONE;
  // Width of each atomic operation.
  iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32;
  // Ordering and coherence domain requested by the test.
  iree_hal_atomic_flags_t atomic_flags = IREE_HAL_ATOMIC_FLAG_NONE;
};

// Queue-family access and allocation geometry for an atomic test requirement.
struct AtomicTestConfiguration {
  // Queue family providing the requested atomic operations.
  iree_hal_queue_family_ordinal_t queue_family_ordinal = 0;
  // Exact family contract used to select a compatible native pool.
  iree_hal_pool_family_access_t family_access = {};
  // Natural alignment required by the selected atomic width.
  iree_device_size_t min_alignment = 1;
};

// Selects one provisioned queue family satisfying |requirements|.
//
// The returned family access carries the matching memory requirements. Callers
// must construct a pool from it; pool construction selects and qualifies the
// native memory route independently from queue selection.
bool SelectAtomicTestConfiguration(iree_hal_device_t* device,
                                   const AtomicTestRequirements& requirements,
                                   AtomicTestConfiguration* out_configuration);

// Constructs an owned native pool for a selected atomic test configuration.
iree_status_t CreateAtomicTestPool(iree_hal_device_group_t* device_group,
                                   const AtomicTestConfiguration& configuration,
                                   iree_hal_pool_t** out_pool);

// Constructs an import-only host-pointer pool for testing target validation.
// Import admission requires the same atomic operations as the owned test pool.
iree_status_t CreateAtomicTestHostImportPool(
    iree_hal_device_group_t* device_group,
    const AtomicTestConfiguration& configuration, iree_hal_pool_t** out_pool);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_ATOMIC_TEST_UTIL_H_
