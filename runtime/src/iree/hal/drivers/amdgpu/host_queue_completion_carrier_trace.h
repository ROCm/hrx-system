// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMPLETION_CARRIER_TRACE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMPLETION_CARRIER_TRACE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_host_queue_t iree_hal_amdgpu_host_queue_t;

// Installs the bounded completion-carrier experiment when selected by the test
// environment. Inactive calls are no-ops.
iree_status_t iree_hal_amdgpu_host_queue_completion_carrier_trace_initialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Records and exports a stable hardware-completion overrun. The caller must
// have quiesced submission before calling.
void iree_hal_amdgpu_host_queue_completion_carrier_trace_record_overrun(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t completed_epoch,
    uint64_t published_epoch, uint64_t submitted_epoch, uint64_t drained_epoch);

// Exports a clean trace if no overrun was observed and releases trace storage.
// The queue must be idle and its hardware ring must still be live.
void iree_hal_amdgpu_host_queue_completion_carrier_trace_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMPLETION_CARRIER_TRACE_H_
