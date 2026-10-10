// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_HOST_SEMAPHORE_H_
#define IREE_HAL_UTILS_HOST_SEMAPHORE_H_

#include <stdint.h>

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a host timeline semaphore with asynchronous wait integration.
// Timeline advancement publishes satisfied timepoints through |proactor|.
// |proactor| is borrowed from the device's proactor pool and must outlive the
// semaphore.
iree_status_t iree_hal_host_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_HOST_SEMAPHORE_H_
