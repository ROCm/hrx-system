// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SCRATCH_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SCRATCH_H_

#include "iree/hal/drivers/amd/gpu/queue/aql/growth.h"

#ifdef __cplusplus
extern "C" {
#endif

// Firmware service owns query -> surrender -> release -> acquire -> resume.
// No speculative scratch is allocated. The phase transfers request/backing
// ownership between completion, the typed allocator, and publication.
typedef struct iree_hal_amd_gpu_aql_scratch_t {
  // Stable typed result consumed only by the exact publisher.
  iree_hal_amd_gpu_aql_entry_t entry;
  // Borrowed queue publication/acquisition domain.
  iree_hal_amd_gpu_aql_growth_context_t* growth;
  // Exact native queue, borrowed through service and final detach.
  amdf_user_queue_t* queue;
  // Qualified local storage policy; no host mapping is required.
  const iree_hal_amd_fabric_memory_options_t* options;
  // IDLE, ACQUIRING, READY, RELEASING, or FAILED; transfers fields below.
  iree_atomic_int32_t phase;
  // Current binding, or surrendered/new backing while service is pending.
  iree_hal_amd_fabric_memory_t memory;
  // Authoritative request published by the completion observer.
  amdf_gpu_aql_scratch_request_t request;
  // Native allocation geometry borrowed through completion.
  iree_hal_amd_fabric_memory_request_t allocation;
  // Embedded acquire descriptor; release never needs fresh scratch storage.
  iree_hal_amd_gpu_memory_acquire_t acquire;
  // Embedded native release descriptor, reused only after its callback.
  iree_hal_amd_gpu_memory_release_t release;
  // Owned native acquisition result, consumed by the publisher.
  iree_status_t status;
} iree_hal_amd_gpu_aql_scratch_t;

void iree_hal_amd_gpu_aql_scratch_initialize(
    iree_hal_amd_gpu_aql_growth_context_t* growth, amdf_user_queue_t* queue,
    const iree_hal_amd_fabric_memory_options_t* options,
    iree_hal_amd_gpu_aql_scratch_t* out_scratch);

// Completion-owner observation after a native wake. Empty/pending service
// performs no allocation. A genuine request surrenders existing backing and
// submits only typed memory operations; the result returns via publication.
amdf_status_t iree_hal_amd_gpu_aql_scratch_poll(
    iree_hal_amd_gpu_aql_scratch_t* scratch);

// Publisher consumes a completed acquisition and resumes firmware without a
// kernel call. Failure leaves service terminal and backing owned for teardown.
iree_status_t iree_hal_amd_gpu_aql_scratch_resume(
    iree_hal_amd_gpu_aql_scratch_t* scratch);

// Cold predicate after accepted work and service publication have joined.
bool iree_hal_amd_gpu_aql_scratch_is_pending(
    const iree_hal_amd_gpu_aql_scratch_t* scratch);

// Cold explicit trim requires idle native execution/consumption and exclusive
// publication/service ownership. It may enter the kernel. Successful detach
// releases backing through the native allocation owner; no host wait is added.
amdf_status_t iree_hal_amd_gpu_aql_scratch_trim(
    iree_hal_amd_gpu_aql_scratch_t* scratch);

// Called only after successful native queue destruction and callback join.
void iree_hal_amd_gpu_aql_scratch_deinitialize(
    iree_hal_amd_gpu_aql_scratch_t* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SCRATCH_H_
