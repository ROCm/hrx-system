// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_COMPLETION_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_COMPLETION_H_

#include "amdf/amdf.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif

// One presecured error handoff from native copy ownership to the file's poll
// owner. Early failure and final retirement may arrive before or during one
// callback. They coalesce without resubmitting a still-owned NOP. Only the
// retired callback permits reclaiming this state and its borrowed resources.
// Healthy copies bypass this handoff entirely.
typedef struct iree_hal_amd_gpu_file_completion_t {
  // Stable operation reused only after its proactor callback returns ownership.
  iree_async_nop_operation_t publication;
  // Borrowed file poll owner, live through the final retired callback.
  iree_async_proactor_t* proactor;
  // Consumes status; retired=false never relinquishes the native operand use.
  void (*notify)(void* user_data, iree_status_t status, bool retired);
  // Borrowed file slot, live through final notification.
  void* user_data;
  // First native error, independent of asynchronous callback delivery.
  iree_atomic_int64_t status;
  // Coalesced callback ownership and monotonic native retirement proof.
  iree_atomic_int32_t state;
} iree_hal_amd_gpu_file_completion_t;

// Initializes an embedded handoff before one native copy becomes visible.
// The preceding copy and callback have joined; no allocation is performed.
void iree_hal_amd_gpu_file_completion_initialize(
    iree_async_proactor_t* proactor, void (*notify)(void*, iree_status_t, bool),
    void* user_data, iree_hal_amd_gpu_file_completion_t* completion);

// Publishes native failure and, optionally, final access cessation. At most one
// early failure precedes one retired notification. This ends the producer's
// access to completion; the retired callback may reclaim it before return.
void iree_hal_amd_gpu_file_completion_notify(
    iree_hal_amd_gpu_file_completion_t* completion, amdf_status_t status,
    bool retired);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_COMPLETION_H_
