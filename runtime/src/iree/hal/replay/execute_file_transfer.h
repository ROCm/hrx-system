// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_EXECUTE_FILE_TRANSFER_H_
#define IREE_HAL_REPLAY_EXECUTE_FILE_TRANSFER_H_

#include "iree/hal/replay/execute_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// Restores one finite file group, including deferred metadata references and
// compact range-captured files, using the executor's queue completion policy.
iree_status_t iree_hal_replay_executor_queue_file_transfer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_queue_barriers_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_REPLAY_EXECUTE_FILE_TRANSFER_H_
