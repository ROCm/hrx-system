// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_DEVICE_SELECTION_H_
#define LIBHRX_SRC_BINDING_HIP_DEVICE_SELECTION_H_

#include "iree/hal/driver.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed environment values captured before HIP initialization. NULL means
// unset; an empty ROCr value selects no devices, while an empty HIP value
// permits the CUDA fallback. No value is retained after initialization.
typedef struct iree_hip_device_selection_t {
  // Optional first-stage ROCr visibility filter.
  const char* rocr_visible_devices;
  // Preferred second-stage HIP filter when nonempty.
  const char* hip_visible_devices;
  // Second-stage fallback when the HIP filter is unset or empty.
  const char* cuda_visible_devices;
} iree_hip_device_selection_t;

// Reads the initialization-only compatibility environment without modifying it.
iree_hip_device_selection_t iree_hip_device_selection_from_environment(void);

// Recognizes explicit no-device requests without requiring native discovery.
bool iree_hip_device_selection_selects_none(
    const iree_hip_device_selection_t* selection);

// HRX initialization selector. |self| borrows an iree_hip_device_selection_t.
// Reorders and compacts the physical-device inventory in place. Driver-owned
// opaque IDs and borrowed strings are preserved. Invalid selection tokens
// terminate their stage normally; only scratch allocation can fail.
iree_status_t iree_hip_device_selection_apply(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_host_size_t* device_count, iree_hal_device_info_t* device_infos);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_DEVICE_SELECTION_H_
