// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_ATOMIC_H_
#define IREE_HAL_LOCAL_ATOMIC_H_

#include "iree/base/api.h"
#include "iree/hal/atomic.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when |width| is backed by lock-free machine atomics.
bool iree_hal_local_atomic_width_is_lock_free(iree_hal_atomic_width_t width);

// Builds memory operation capabilities for lock-free CPU atomic widths.
iree_hal_atomic_operation_capabilities_t
iree_hal_local_atomic_operation_capabilities(
    iree_hal_atomic_operation_flags_t allowed_operations);

// Builds queue operation and wait-predicate capabilities for lock-free CPU
// atomic widths.
iree_hal_atomic_capabilities_t iree_hal_local_atomic_capabilities(
    iree_hal_atomic_operation_flags_t allowed_operations);

// Waits until |params| is satisfied by the naturally aligned |target|.
//
// The parameters must have passed iree_hal_atomic_wait_params_validate() and
// their width must be lock-free. This function occupies the calling thread and
// is only suitable for explicitly synchronous execution paths.
void iree_hal_local_atomic_wait(void* target,
                                iree_hal_atomic_wait_params_t params);

// Stores |params.value| to the naturally aligned |target|.
//
// The parameters must have passed iree_hal_atomic_store_params_validate() and
// their width must be lock-free.
void iree_hal_local_atomic_store(void* target,
                                 iree_hal_atomic_store_params_t params);

// Applies the no-result read-modify-write |params| to the naturally aligned
// |target|.
//
// The parameters must have passed iree_hal_atomic_rmw_params_validate() and
// their width must be lock-free.
void iree_hal_local_atomic_rmw(void* target,
                               iree_hal_atomic_rmw_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_ATOMIC_H_
