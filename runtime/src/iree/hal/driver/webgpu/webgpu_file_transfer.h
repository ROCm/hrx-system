// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_WEBGPU_WEBGPU_FILE_TRANSFER_H_
#define IREE_HAL_DRIVER_WEBGPU_WEBGPU_FILE_TRANSFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_webgpu_queue_t iree_hal_webgpu_queue_t;

// Captures one finite file group. Native windows and indirect metadata pages
// have bounded storage; the group retains its operands through accepted work
// and releases them before publishing terminal signals. Queue and proactor are
// borrowed. READ moves file bytes into buffers; WRITE moves buffers into files.
iree_status_t iree_hal_webgpu_queue_submit_file_transfer(
    iree_hal_webgpu_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_WEBGPU_WEBGPU_FILE_TRANSFER_H_
