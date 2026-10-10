// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_DRIVER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_DRIVER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the native libamdf XDNA driver. Discovery is a passive snapshot;
// device creation opens the selected endpoint and an exact-width context.
// Device paths are zero-based discovery ordinals. The `columns` URI parameter
// sets logical context width (default 1), which must match the compiled image.
iree_status_t iree_hal_amd_xdna_driver_create(iree_allocator_t host_allocator,
                                              iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_DRIVER_H_
