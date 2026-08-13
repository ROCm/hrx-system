// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_EXECUTE_OPERATION_H_
#define IREE_HAL_REPLAY_EXECUTE_OPERATION_H_

#include "iree/hal/replay/execute_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_REPLAY_INLINE_MEMORY_BARRIER_LIST_CAPACITY 8
#define IREE_HAL_REPLAY_INLINE_BUFFER_BARRIER_LIST_CAPACITY 8

// Executes a successful generic operation record.
iree_status_t iree_hal_replay_executor_replay_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record);

// Reports an unsupported captured operation when it originally succeeded.
iree_status_t iree_hal_replay_executor_replay_unsupported(
    const iree_hal_replay_file_record_t* record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_EXECUTE_OPERATION_H_
