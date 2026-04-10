// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_HSA_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_HSA_HSA_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_async_proactor_t iree_async_proactor_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Creates a semaphore backed by iree_async_semaphore_t.
// The |proactor| is used for async I/O operations (import/export fences).
iree_status_t iree_hal_hsa_semaphore_create(
    iree_async_proactor_t *proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t **out_semaphore);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_HSA_HSA_SEMAPHORE_H_
