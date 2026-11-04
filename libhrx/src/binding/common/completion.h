// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_COMPLETION_H_
#define LIBHRX_SRC_BINDING_COMMON_COMPLETION_H_

// Completion state of work covered by a successful non-blocking query.
typedef enum iree_hal_streaming_completion_state_e {
  // At least one accepted operation covered by the query is still pending.
  IREE_HAL_STREAMING_COMPLETION_STATE_PENDING = 0,
  // Every accepted operation covered by the query has completed.
  IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE = 1,
} iree_hal_streaming_completion_state_t;

#endif  // LIBHRX_SRC_BINDING_COMMON_COMPLETION_H_
