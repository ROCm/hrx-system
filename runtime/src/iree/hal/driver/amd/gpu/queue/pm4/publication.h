// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PUBLICATION_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PUBLICATION_H_

#include "iree/hal/driver/amd/gpu/queue/pm4/submission.h"

#ifdef __cplusplus
extern "C" {
#endif

struct iree_hal_amd_gpu_pm4_queue_t;

// A publication either transfers its admitted ownership or retains a precise
// unmet resource demand. Only native acceptance grants retirement ownership.
typedef enum iree_hal_amd_gpu_pm4_publication_result_e {
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_FINISHED,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_MORE,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_CAPACITY,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_ARGUMENTS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_TRANSFERS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SAMPLES,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SIGNALS,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_PROGRAM,
  IREE_HAL_AMD_GPU_PM4_PUBLICATION_GROW_SCRATCH,
} iree_hal_amd_gpu_pm4_publication_result_t;

// Borrowed inputs to one dispatch/barrier publication attempt. Existing
// captures and inline API calls use the same packet writer. The writer creates
// retirement ownership only after dependencies, addresses, and capacity permit
// publishing a complete piece.
typedef struct iree_hal_amd_gpu_pm4_dispatch_publication_t {
  // Borrowed argument/launch inputs, valid until this attempt returns.
  const iree_hal_amd_gpu_pm4_dispatch_request_t* dispatch;
  // Admitted dependencies; native readers transfer only on successful capture.
  const iree_hal_amd_gpu_pm4_submission_wait_t* waits;
  // Number of entries in waits, including already-completed dependencies.
  iree_host_size_t wait_count;
  // Number of native readers requiring a device wait.
  uint32_t native_wait_count;
  // Borrowed output timelines and values, retained before native acceptance.
  iree_hal_semaphore_list_t signals;
  // Prepared payload acquisition boundary.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared terminal payload release boundary.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // Optional attribution, copied into retirement ownership before emission.
  iree_hal_amd_gpu_profile_operation_t* profile;
} iree_hal_amd_gpu_pm4_dispatch_publication_t;

// Publishes an admitted dispatch or barrier. Inputs remain borrowed until this
// call returns; successful capture transfers them to native retirement.
// Capacity and growth results publish nothing. A non-OK out_status leaves any
// captured submission with the caller for rejection.
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_dispatch(
    struct iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t position,
    uint32_t scratch_required,
    const iree_hal_amd_gpu_pm4_dispatch_publication_t* request,
    uint64_t* addresses, iree_hal_amd_gpu_pm4_submission_t** inout_submission,
    iree_status_t* out_status);

// Publish one admitted capture under exclusive publisher ownership. A terminal
// accepted piece transfers its final reference to retirement; MORE retains an
// unpublished suffix. Immediate retirement may free a terminal capture before
// these calls return. Capacity/growth retains it for the queue's next attempt.
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_transfer(
    struct iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position);
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_execute(
    struct iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position);
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_control(
    struct iree_hal_amd_gpu_pm4_queue_t* queue,
    iree_hal_amd_gpu_pm4_submission_t* submission, uint64_t position);

// Retries exactly the prepared KERNEL piece. BUSY preserves its bytes and
// borrows; rejection rolls back unpublished storage and consumes the capture.
iree_hal_amd_gpu_pm4_publication_result_t iree_hal_amd_gpu_pm4_publish_kernel(
    struct iree_hal_amd_gpu_pm4_queue_t* queue, uint64_t position);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_PUBLICATION_H_
