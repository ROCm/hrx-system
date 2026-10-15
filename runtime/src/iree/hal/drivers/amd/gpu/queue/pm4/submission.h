// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_

#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/publisher.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One captured timeline edge. The semaphore is retained independently of an
// optional native reader. A reader never borrows its producer's capture.
typedef struct iree_hal_amd_gpu_pm4_submission_wait_t {
  // Owned HAL timeline, live through admission and terminal completion.
  iree_hal_semaphore_t* semaphore;
  // Requested timeline value.
  uint64_t value;
  // Owned eligible native reader, or NULL for a completed/host dependency.
  iree_hal_amd_gpu_pm4_signal_t* signal;
} iree_hal_amd_gpu_pm4_submission_wait_t;

typedef enum iree_hal_amd_gpu_pm4_submission_kind_e {
  IREE_HAL_AMD_GPU_PM4_SUBMISSION_BARRIER,
  IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH,
  IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER,
} iree_hal_amd_gpu_pm4_submission_kind_t;

// Captured public operation shared by publication and native retirement.
// Variable arrays and captured bytes occupy the same CPU allocation. Each
// published piece owns separate retirement credit; this capture also owns any
// unpublished suffix and never stores mutable per-publication completion data.
typedef struct iree_hal_amd_gpu_pm4_submission_t {
  // Host publication and native retirement may finish in either order.
  iree_atomic_ref_count_t ref_count;
  // Number of retained native readers, or UINT32_MAX before admission.
  uint32_t native_wait_count;
  // Captured-work link; the publisher parks unresolved operations separately.
  iree_hal_amd_gpu_publication_entry_t publication;
  // Borrowed allocation owner, live through the capture's final release.
  iree_hal_amd_gpu_capture_pool_t* pool;
  // Complete owned extent, including unused tail storage, in bytes.
  iree_host_size_t allocation_size;
  // Number of captured wait edges.
  iree_host_size_t wait_count;
  // Inline wait edges; their native readers are acquired during admission.
  iree_hal_amd_gpu_pm4_submission_wait_t* waits;
  // Owned signal timelines and copied values in inline storage.
  iree_hal_semaphore_list_t signals;
  // Prepared pre-operation execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared terminal execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // Owned native producer point, if any signal can use native acceleration.
  iree_hal_amd_gpu_pm4_signal_t* signal;
  // Active payload; barriers have no payload.
  iree_hal_amd_gpu_pm4_submission_kind_t kind;
  // Exact initialized extent, computed once by the capture constructor.
  iree_host_size_t capture_size;
  union {
    // Admitted dispatch with privately captured arguments and resources.
    struct {
      // Owned executable dominating the prepared function.
      iree_hal_executable_t* executable;
      // Borrowed prepared function dominated by executable.
      const iree_hal_amd_gpu_pm4_function_t* function;
      // Copied buffer references; each non-NULL buffer is retained.
      iree_hal_buffer_ref_list_t bindings;
      // Copied caller constants, in dense HAL or explicit native ABI layout.
      iree_const_byte_span_t constants;
      // Admitted interpretation of constants, bindings, and launch geometry.
      iree_hal_dispatch_flags_t flags;
      // Admitted direct geometry.
      iree_hal_amd_gpu_pm4_dispatch_t geometry;
    } dispatch;
    // Captured transfer descriptors and the exclusively owned publish cursor.
    iree_hal_amd_gpu_pm4_transfer_t transfer;
  };
} iree_hal_amd_gpu_pm4_submission_t;

// Creates a refcounted capture of an admitted operation before the public API
// returns. The pool is borrowed and outlives every retained reference.
// Reuses one block for ordinary calls; oversized calls use the pool allocator.
// A NULL executable/function captures a barrier and requires empty arguments.
// Failure retains nothing and leaves out_submission NULL. Arrays and bytes
// cease borrowing the caller on success. Buffer addresses remain unresolved
// until queue-ordered allocation and dependency admission permit reading them.
iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Captures the complete public transfer transaction, including update bytes
// and fill patterns. Native address slots are qualified now; queue-ordered
// addresses are resolved only after dependencies permit execution. Zero-length
// operations are omitted; an empty transaction becomes a barrier capture.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Borrows the unused, naturally aligned tail of a reusable capture block.
// Oversized captures have no spare storage. Admission may place sidecars here
// without another allocation; their users must join before the capture's final
// release. The capture owns these bytes and this query transfers no ownership.
iree_byte_span_t iree_hal_amd_gpu_pm4_submission_tail_storage(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Pins publication bookkeeping independently of a possibly earlier GPU
// completion. Existing ownership is required to acquire a reference.
void iree_hal_amd_gpu_pm4_submission_retain(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Releases payload resources after terminal native completion, before public
// completion signals. Publication may still retain the capture for its signal
// bookkeeping, but never accesses the payload after publishing the final piece.
// Only terminal retirement calls this while another capture reference exists.
void iree_hal_amd_gpu_pm4_submission_retire(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Releases captured resources and storage on the last reference. Native uses
// must already have completed or never have been published; failure alone does
// not suffice. NULL is ignored.
void iree_hal_amd_gpu_pm4_submission_release(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Resolves a prepared direct buffer binding at the point execution is ready.
// Dispatch admission has checked the captured range. This checks its native
// permissions, exact family domain, and queue-ordered backing commitment.
// It never maps memory, retains a buffer, or inspects pointer-valued bytes.
iree_status_t iree_hal_amd_gpu_pm4_submission_resolve_address(
    const iree_hal_queue_family_t* family, iree_hal_buffer_ref_t reference,
    iree_hal_memory_access_t access, uint64_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
