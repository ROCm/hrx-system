// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_TASK_ATOMIC_H_
#define IREE_HAL_DRIVER_TASK_ATOMIC_H_

#include "iree/base/api.h"
#include "iree/hal/atomic.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when |width| is backed by lock-free machine atomics.
bool iree_hal_task_atomic_width_is_lock_free(iree_hal_atomic_width_t width);

// Validates a public operation's execution requirements. CPU atomics execute
// on workers and cannot satisfy ZERO_COMPUTE; accepted widths are lock-free.
iree_status_t iree_hal_task_atomic_validate(iree_hal_atomic_width_t width,
                                            iree_hal_atomic_flags_t flags);

// Builds queue operation and wait-predicate capabilities for lock-free CPU
// atomic widths.
iree_hal_atomic_capabilities_t iree_hal_task_atomic_capabilities(
    iree_hal_atomic_operation_flags_t allowed_operations);

// Resolves one validated HAL barrier to the ordering required in the task
// driver's coherent CPU memory domain. Ranged queue actions are conservatively
// promoted to CPU-wide fences. Host cache recipes require separate maintenance
// in addition to the ordering resolved here.
iree_hal_atomic_flags_t iree_hal_task_barrier_resolve_atomic_flags(
    const iree_hal_barrier_t* barrier);

// Applies acquire/release ordering at a coherent CPU execution boundary.
// Empty flags require no fence. System and device scope share the CPU's
// coherent memory domain; non-coherent mappings require their own maintenance.
void iree_hal_task_atomic_fence(iree_hal_atomic_flags_t flags);

// Waits until |params| is satisfied by the naturally aligned |target|.
//
// The parameters must have passed iree_hal_atomic_wait_params_validate() and
// their width must be lock-free. This function occupies the calling thread and
// is only suitable for explicitly synchronous execution paths.
void iree_hal_task_atomic_wait(void* target,
                               iree_hal_atomic_wait_params_t params);

// Stores |params.value| to the naturally aligned |target|.
//
// The parameters must have passed iree_hal_atomic_store_params_validate() and
// their width must be lock-free.
void iree_hal_task_atomic_store(void* target,
                                iree_hal_atomic_store_params_t params);

// Applies the no-result read-modify-write |params| to the naturally aligned
// |target|.
//
// The parameters must have passed iree_hal_atomic_rmw_params_validate() and
// their width must be lock-free.
void iree_hal_task_atomic_rmw(void* target,
                              iree_hal_atomic_rmw_params_t params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_TASK_ATOMIC_H_
