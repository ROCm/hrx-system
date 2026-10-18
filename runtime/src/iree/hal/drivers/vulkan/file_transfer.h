// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_FILE_TRANSFER_H_
#define IREE_HAL_DRIVERS_VULKAN_FILE_TRANSFER_H_

#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transfers a validated native file range through |buffer| after its queue
// dependencies retire. |file_access| is READ (file to buffer) or WRITE (buffer
// to file). The operation captures buffer/file lifetimes and terminal
// semaphores. Backing is resolved at readiness: imported host backing receives
// native I/O directly; device-only storage uses the queue's staging ring.
// Direct I/O uses bounded request metadata with no payload allocation, mapping,
// or staging copy.
iree_status_t iree_hal_vulkan_queue_submit_file_transfer(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_file_t* file,
    iree_hal_memory_access_t file_access, uint64_t file_offset,
    iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
    iree_device_size_t length,
    const iree_hal_vulkan_queue_barriers_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_VULKAN_FILE_TRANSFER_H_
