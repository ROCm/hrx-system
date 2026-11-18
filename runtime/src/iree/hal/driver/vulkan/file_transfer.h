// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_VULKAN_FILE_TRANSFER_H_
#define IREE_HAL_DRIVER_VULKAN_FILE_TRANSFER_H_

#include "iree/hal/driver/vulkan/buffer.h"
#include "iree/hal/driver/vulkan/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Captures a finite file group and schedules it after its producer
// dependencies. File/buffer tables are retained once. Indirect metadata is
// acquired only for the live prefix; device-private metadata uses bounded
// staging downloads. Native files access original imported host backing
// directly when admitted. Other paths share the queue's existing staging slots
// and native copy batches.
iree_status_t iree_hal_vulkan_queue_submit_file_transfer(
    iree_hal_vulkan_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access,
    const iree_hal_vulkan_queue_barriers_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_VULKAN_FILE_TRANSFER_H_
