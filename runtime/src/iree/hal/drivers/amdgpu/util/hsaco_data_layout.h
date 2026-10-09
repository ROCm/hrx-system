// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_DATA_LAYOUT_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_DATA_LAYOUT_H_

#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Populates an explicitly advertised code-object data layout from ELF dynamic
// symbols. Unmarked code objects retain the default opaque layout.
iree_status_t iree_hal_amdgpu_hsaco_data_layout_populate(
    iree_hal_amdgpu_hsaco_metadata_t* metadata);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_DATA_LAYOUT_H_
