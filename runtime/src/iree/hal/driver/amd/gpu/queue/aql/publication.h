// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PUBLICATION_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PUBLICATION_H_

#include "iree/hal/driver/amd/gpu/queue/aql/submission.h"

#ifdef __cplusplus
extern "C" {
#endif

struct iree_hal_amd_gpu_aql_queue_t;

// Every fallible preparation precedes native reservation. Native release can
// only fail the queue terminally after acceptance. Once reserved, all
// ownership is published before firmware can observe the first valid header.
typedef enum iree_hal_amd_gpu_aql_publish_result_e {
  IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_MORE,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_PROGRAM,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_ARGUMENTS,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_TRANSFERS,
  IREE_HAL_AMD_GPU_AQL_PUBLISH_SIGNALS,
} iree_hal_amd_gpu_aql_publish_result_t;

// At most nine scalar DWORDs plus the two-DWORD virtual-XCC predicate.
enum { IREE_HAL_AMD_GPU_AQL_CONTROL_BYTE_LENGTH = 48 };

// Exact argument extent needed before ring reservation, including optional
// cooperative control data and profiling samples.
uint64_t iree_hal_amd_gpu_aql_argument_length(
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    uint32_t grid_length, bool sampled);

// Exact upload extent for the current recorded-command chunk.
uint64_t iree_hal_amd_gpu_aql_execute_argument_length(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation, bool sampled);

// Publishes a prepared direct dispatch or empty barrier with its retained
// dependency ownership. Only successful acceptance advances ring and argument
// credit. On acceptance, the retirement record takes the returned capture.
// counts.retained_count sizes a new inline capture; a supplied capture already
// owns its readers and needs only counts.native_count for packet reservation.
iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish(
    struct iree_hal_amd_gpu_aql_queue_t* queue,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, iree_hal_amd_gpu_aql_wait_t* waits,
    iree_hal_amd_gpu_aql_wait_counts_t counts,
    iree_hal_semaphore_list_t signals, uint32_t before, uint32_t after,
    uint64_t* addresses, const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_aql_submission_t** inout_submission,
    iree_status_t* out_status);

// Publishes one admitted scalar operation and its cache boundaries.
iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_control(
    struct iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status);

// Publishes the next recorded-command chunk; MORE retains the pending suffix.
iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_execute(
    struct iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status);

// Publishes the next transfer chunk, retaining host staging until its final
// use.
iree_hal_amd_gpu_aql_publish_result_t iree_hal_amd_gpu_aql_publish_transfer(
    struct iree_hal_amd_gpu_aql_queue_t* queue,
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t* out_status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_PUBLICATION_H_
