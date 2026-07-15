// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_EXECUTABLE_TARGET_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_EXECUTABLE_TARGET_H_

#include "iree/hal/drivers/amdgpu/util/target_id.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Adds canonical exact and compatible generic AMDGPU executable targets for
// |exact_target_id| to |builder|.
//
// The generic target is omitted when the target map requires exact code
// objects. Repeated targets are merged by the device spec builder.
iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_id_t* exact_target_id,
    iree_hal_physical_device_affinity_t physical_device_affinity);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_EXECUTABLE_TARGET_H_
