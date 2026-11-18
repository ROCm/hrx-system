// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_FILE_TRANSFER_H_
#define IREE_HAL_REPLAY_RECORDER_FILE_TRANSFER_H_

#include "iree/hal/replay/recorder_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Captures one finite group and forwards it intact to the wrapped native queue.
// Deferred metadata is referenced without reading its contents. Range-captured
// files require immediately readable HOST descriptions without producer waits.
iree_status_t iree_hal_replay_recorder_queue_file_transfer(
    iree_hal_replay_recorder_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access,
    const iree_hal_queue_barriers_t* barriers, uint64_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_REPLAY_RECORDER_FILE_TRANSFER_H_
