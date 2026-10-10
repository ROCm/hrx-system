// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_BARRIER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_BARRIER_H_

#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies that validated direct-operation boundaries are executable by the
// XDNA queue family. Descriptors are consumed synchronously and need not remain
// live after this call returns.
iree_status_t iree_hal_amd_xdna_queue_barriers_validate(
    const iree_hal_queue_barriers_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_BARRIER_H_
