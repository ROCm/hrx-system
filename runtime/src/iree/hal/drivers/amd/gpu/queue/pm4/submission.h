// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"
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

// Admitted direct-dispatch geometry, independent of packet publication mode.
typedef struct iree_hal_amd_gpu_pm4_dispatch_t {
  // Work-items in each complete workgroup.
  uint32_t workgroup_size[3];
  // Number of dispatched workgroups, including a possible partial group.
  uint32_t workgroup_count[3];
  // Final group dimensions, all zero when every group is complete.
  uint32_t partial_size[3];
  // Fixed and dynamic LDS allocation in bytes.
  uint32_t group_segment_byte_length;
} iree_hal_amd_gpu_pm4_dispatch_t;

// Captured public operation shared by caller publication and native retirement.
// Variable arrays and argument bytes occupy the same CPU allocation. Native
// argument storage is reserved separately in publication order. A NULL function
// represents a semaphore/memory barrier without a dispatch.
typedef struct iree_hal_amd_gpu_pm4_submission_t {
  // Host publication and native retirement may finish in either order.
  iree_atomic_ref_count_t ref_count;
  // Ready-list link; unresolved operations remain outside the publisher.
  iree_hal_amd_gpu_publication_entry_t publication;
  // Borrowed allocation owner, live through the capture's final release.
  iree_arena_block_pool_t* pool;
  // Non-NULL when this allocation came from the reusable capture block pool.
  iree_arena_block_t* block;
  // Owned executable, or NULL for a barrier.
  iree_hal_executable_t* executable;
  // Borrowed prepared function dominated by executable.
  const iree_hal_amd_gpu_pm4_function_t* function;
  // Number of captured wait edges.
  iree_host_size_t wait_count;
  // Inline wait edges; their native readers are acquired during admission.
  iree_hal_amd_gpu_pm4_submission_wait_t* waits;
  // Owned signal timelines and copied values in inline storage.
  iree_hal_semaphore_list_t signals;
  // Copied buffer references; each non-NULL buffer is retained.
  iree_hal_buffer_ref_list_t bindings;
  // Copied caller constants, in dense HAL or explicit native ABI layout.
  iree_const_byte_span_t constants;
  // Admitted interpretation of constants, bindings, and launch geometry.
  iree_hal_dispatch_flags_t flags;
  // Admitted geometry; unused for a barrier.
  iree_hal_amd_gpu_pm4_dispatch_t dispatch;
  // Prepared pre-operation execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared terminal execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // Owned native producer point, if any signal can use native acceleration.
  iree_hal_amd_gpu_pm4_signal_t* signal;
  // Exact primary-ring prefix proven by this operation's completion record.
  uint64_t consumed_words;
  // Exclusive argument-byte position proven by the same completion record.
  uint64_t consumed_arguments;
} iree_hal_amd_gpu_pm4_submission_t;

// Admits direct geometry and argument representation at the backend's public
// dispatch boundary. The public HAL wrapper has validated list storage, flags,
// and exact-family identity. No caller bytes or native addresses are captured.
iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch);

// Creates a refcounted capture of an admitted operation before the public API
// returns. The pool is borrowed and outlives every retained reference.
// Reuses one block for ordinary calls; oversized calls use the pool allocator.
// A NULL executable/function captures a barrier and requires empty arguments.
// Failure retains nothing and leaves out_submission NULL. Arrays and bytes
// cease borrowing the caller on success. Buffer addresses remain unresolved
// until queue-ordered allocation and dependency admission permit reading them.
iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_arena_block_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Pins publication bookkeeping independently of a possibly earlier GPU
// completion. Existing ownership is required to acquire a reference.
void iree_hal_amd_gpu_pm4_submission_retain(
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

// Writes the admitted native kernarg segment after native addresses have been
// resolved. binding_addresses contains one address for each dense HAL binding;
// native direct constants do not consume it. Indirect arguments bypass this.
void iree_hal_amd_gpu_pm4_submission_emplace_arguments(
    const iree_hal_amd_gpu_pm4_submission_t* submission,
    const uint64_t* binding_addresses, void* target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
