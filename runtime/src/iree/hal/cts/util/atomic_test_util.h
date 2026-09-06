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
  // Memory type bits that must be available on the target allocation.
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  // Buffer usage bits required by the test and its readback path.
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_NONE;
  // Memory access bits required by the tested operations.
  iree_hal_memory_access_t memory_access = IREE_HAL_MEMORY_ACCESS_NONE;
  // Width of each atomic operation.
  iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32;
  // Ordering and coherence domain requested by the test.
  iree_hal_atomic_flags_t atomic_flags = IREE_HAL_ATOMIC_FLAG_NONE;
};

// Queue and allocation parameters satisfying an atomic test requirement.
struct AtomicTestConfiguration {
  // Queue family providing the requested atomic operations.
  iree_hal_queue_family_ordinal_t queue_family_ordinal = 0;

  // Allocation parameters selecting a matching memory type.
  iree_hal_buffer_params_t buffer_params = {};
};

// Selects one queue family and memory type satisfying |requirements|.
//
// Queue and memory atomic capabilities are independent and both must contain
// the requested coherence-domain cell. The selected heap must also be visible
// to the physical devices serviced by the queue family.
bool SelectAtomicTestConfiguration(const iree_hal_device_spec_t* device_spec,
                                   const AtomicTestRequirements& requirements,
                                   AtomicTestConfiguration* out_configuration);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_UTIL_ATOMIC_TEST_UTIL_H_
