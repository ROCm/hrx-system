// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SUBMISSION_H_

#include "iree/hal/drivers/amd/gpu/queue/aql/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/publisher.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_aql_wait_t {
  // Borrowed during inline admission; retained by a deferred capture.
  iree_hal_semaphore_t* semaphore;
  // Required HAL timeline value.
  uint64_t value;
  // Independent native reader acquired during admission, or NULL.
  iree_hal_amd_gpu_signal_t* signal;
} iree_hal_amd_gpu_aql_wait_t;

// Borrowed call arguments; native packing consumes these directly when ready.
typedef struct iree_hal_amd_gpu_aql_dispatch_request_t {
  // Executable dominating the immutable function, or NULL for a barrier.
  iree_hal_executable_t* executable;
  // Prepared native descriptor and shared argument layout.
  const iree_hal_amd_gpu_aql_function_t* function;
  // Admitted launch dimensions and local storage.
  iree_hal_amd_gpu_dispatch_geometry_t geometry;
  // Device-time arguments/geometry as selected by flags.
  iree_hal_buffer_ref_t parameters;
  // Dense or native host argument bytes.
  iree_const_byte_span_t constants;
  // Shader buffer references; addresses resolve after dependencies.
  iree_hal_buffer_ref_list_t bindings;
  // Argument representation and optional resource-lifetime borrow.
  iree_hal_dispatch_flags_t flags;
} iree_hal_amd_gpu_aql_dispatch_request_t;

typedef enum iree_hal_amd_gpu_aql_entry_kind_e {
  IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION,
  IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS,
  IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS,
  IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_SIGNALS,
  IREE_HAL_AMD_GPU_AQL_ENTRY_CAPACITY,
  IREE_HAL_AMD_GPU_AQL_ENTRY_SCRATCH,
  IREE_HAL_AMD_GPU_AQL_ENTRY_TRIM,
} iree_hal_amd_gpu_aql_entry_kind_t;

typedef struct iree_hal_amd_gpu_aql_entry_t {
  // Single-publisher FIFO membership.
  iree_hal_amd_gpu_publication_entry_t publication;
  // Exact enclosing owner, immutable through FIFO membership.
  iree_hal_amd_gpu_aql_entry_kind_t kind;
} iree_hal_amd_gpu_aql_entry_t;

// One direct dispatch or barrier's asynchronous ownership. Ready calls use the
// reserved retirement slot; deferred calls capture their arguments in a pool.
typedef struct iree_hal_amd_gpu_aql_submission_t {
  // Typed pending-publication record.
  iree_hal_amd_gpu_aql_entry_t entry;
  // Capture storage owner, or NULL for an embedded retirement capture.
  iree_hal_amd_gpu_capture_pool_t* pool;
  // Complete provided block extent in bytes.
  iree_host_size_t allocation_size;
  // Initialized capture extent; unused tail can hold a deferred-wait sidecar.
  iree_host_size_t capture_size;
  // Number of captured dependency records.
  iree_host_size_t wait_count;
  // Inline dependency records, including independently retained native readers.
  iree_hal_amd_gpu_aql_wait_t* waits;
  // Native reader count, or UINT32_MAX before dependency admission.
  uint32_t native_wait_count;
  // Native ordering and cache effects before the payload.
  uint32_t before;
  // Native ordering and cache effects after the payload.
  uint32_t after;
  // Queue acceptance owns a host count until publication or rejection.
  bool has_host_owner;
  // Retained HAL signal timelines and copied values in inline storage.
  iree_hal_semaphore_list_t signals;
  // Native execution point owned until all payload/public effects retire.
  iree_hal_amd_gpu_signal_t* signal;
  // Captured launch and retained payload resources.
  iree_hal_amd_gpu_aql_dispatch_request_t dispatch;
} iree_hal_amd_gpu_aql_submission_t;

// Captures a deferred call in one reusable block. Retains all external owners
// unless flags explicitly borrow payload lifetimes. Failure retains nothing.
iree_status_t iree_hal_amd_gpu_aql_submission_capture(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    uint32_t before, uint32_t after,
    iree_hal_amd_gpu_aql_submission_t** out_submission);

// Captures only retirement ownership for an already prepared direct call.
// Transfers native readers on success, retains signal timelines and payloads,
// and does not copy constants or retain completed wait semaphores. Failure
// transfers nothing. Prefers inline_storage, falling back to a pool block.
iree_status_t iree_hal_amd_gpu_aql_submission_retirement(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, const iree_hal_amd_gpu_aql_wait_t* waits,
    uint32_t native_wait_count, iree_hal_semaphore_list_t signals,
    iree_hal_amd_gpu_aql_submission_t** out_submission);

// Native completion is necessary but producer payload retirement is separate.
// True joins every retained dependency and returns the first failure, if any.
bool iree_hal_amd_gpu_aql_submission_retirement_ready(
    const iree_hal_amd_gpu_aql_submission_t* submission,
    iree_status_code_t* out_status_code);

// Returns payload resources before publishing public completion. Idempotent.
void iree_hal_amd_gpu_aql_submission_release_payload(
    iree_hal_amd_gpu_aql_submission_t* submission);

// Returns all retained ownership after native users join or before publication.
// Embedded storage is left to its retirement owner; pooled storage is returned.
void iree_hal_amd_gpu_aql_submission_release(
    iree_hal_amd_gpu_aql_submission_t* submission);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_SUBMISSION_H_
