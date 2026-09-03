// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_TRANSFER_H_
#define HRX_TRANSFER_H_

#include "hrx_runtime.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Submits a transfer transaction to |device|'s provisioned transfer queue and
// waits for terminal completion. The operation array and all borrowed host
// ranges remain live until the function returns.
iree_status_t hrx_hal_queue_transfer_and_wait(
    hrx_device_t device, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations);

#ifdef __cplusplus
}
#endif

#endif  // HRX_TRANSFER_H_
