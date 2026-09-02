// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TRANSFER_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Captures and submits one exact-queue transfer transaction.
iree_status_t iree_hal_amdgpu_host_queue_transfer(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TRANSFER_H_
