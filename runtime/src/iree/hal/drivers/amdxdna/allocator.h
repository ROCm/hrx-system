// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Executable creation (active != 0) counts a live executable and resumes
// reuse. Executable destruction (active == 0) drops that count and, at zero,
// frees the reuse cache. Other allocator types are ignored.
// Returns 1 when this release drops the last live executable.
int iree_hal_amdxdna_allocator_set_model_reuse(iree_hal_allocator_t* allocator,
                                               int active);

// Creates a buffer allocator used for persistent allocations.
iree_status_t iree_hal_amdxdna_allocator_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_ALLOCATOR_H_
