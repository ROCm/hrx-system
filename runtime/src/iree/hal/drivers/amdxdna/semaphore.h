// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

struct iree_async_proactor_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Software timeline semaphore (same behavior as IREE local_sync). Required for
// correct host-side wait/signal with the async HAL.
iree_status_t iree_hal_amdxdna_semaphore_create(
    iree_async_proactor_t* proactor, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_SEMAPHORE_H_
