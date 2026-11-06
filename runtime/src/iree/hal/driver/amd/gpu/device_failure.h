// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_DEVICE_FAILURE_H_
#define IREE_HAL_DRIVER_AMD_GPU_DEVICE_FAILURE_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/driver/amd/gpu/queue/aql/code.h"
#include "iree/hal/driver/amd/gpu/queue/membership.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-owned terminal coordination. The existing completion worker joins
// native access before failed captures and joins CPU callback tails even on
// unsuccessful cleanup. The existing publication worker independently reports
// a blocked join or native removal. All pointers are borrowed and remain live
// through both worker joins. No child retains the device, and this owner
// creates no thread or background destruction task.
typedef struct iree_hal_amd_gpu_device_failure_t {
  // First native fault, absolute software deadline, and cached diagnostics.
  iree_hal_amd_gpu_failure_t state;
  // Serializes first-error recording, monitor arming, and host stop
  // publication. Never held across a native call, diagnostic sink, or ownership
  // join.
  iree_slim_mutex_t mutex;
  // Native API and execution-failure callbacks installed before workers start.
  iree_hal_amd_gpu_context_t* context;
  // Existing independent publisher; stays live through terminal cleanup.
  iree_hal_amd_gpu_publication_t* publication;
  // Sole native observation/retirement owner executing the terminal join.
  iree_hal_amd_gpu_completion_t* completion;
  // Stable compute owners, including dynamically acquired queues.
  iree_hal_amd_gpu_queue_membership_t* membership;
  // Internal transfer owner, or NULL when no native copy service exists.
  iree_hal_amd_gpu_sdma_queue_t* transfers;
  // Private instruction-cache queue, or NULL for PM4 execution.
  iree_hal_amd_gpu_aql_code_t* code;
} iree_hal_amd_gpu_device_failure_t;

// Installs the terminal callbacks after native owners exist, before starting
// either worker or exposing the device. Uses caller storage only.
void iree_hal_amd_gpu_device_failure_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    uint32_t timeout_ms, iree_hal_amd_gpu_publication_t* publication,
    iree_hal_amd_gpu_completion_t* completion,
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_sdma_queue_t* transfers, iree_hal_amd_gpu_aql_code_t* code,
    iree_hal_amd_gpu_device_failure_t* failure);

// Called only after both workers and native callbacks have joined. Clears
// their borrowed context callbacks before releasing diagnostic storage.
void iree_hal_amd_gpu_device_failure_deinitialize(
    iree_hal_amd_gpu_device_failure_t* failure);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_DEVICE_FAILURE_H_
