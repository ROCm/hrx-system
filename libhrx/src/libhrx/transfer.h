// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_TRANSFER_H_
#define HRX_TRANSFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Submits a transfer transaction to the caller's selected |queue| and waits for
// terminal completion. The queue, its device, the operation array, and all
// borrowed host ranges remain live until the function returns. The caller
// supplies its host scheduling policy and owns queue placement and priority.
iree_status_t hrx_hal_queue_transfer_and_wait(
    iree_hal_queue_t* queue, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations,
    iree_async_wait_flags_t wait_flags);

#ifdef __cplusplus
}
#endif

#endif  // HRX_TRANSFER_H_
