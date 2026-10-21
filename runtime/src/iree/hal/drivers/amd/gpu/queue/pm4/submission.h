// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_

#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/executable.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/control.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"
#include "iree/hal/drivers/amd/gpu/queue/publisher.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/hal/drivers/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One admission/retirement edge. A deferred capture retains its semaphore;
// ready retirement needs only an exact native reader. A reader never borrows
// its producer's capture.
typedef struct iree_hal_amd_gpu_pm4_submission_wait_t {
  // HAL timeline borrowed during inline admission, retained by a deferred
  // capture, or NULL once only its native retirement result is needed.
  iree_hal_semaphore_t* semaphore;
  // Requested timeline value.
  uint64_t value;
  // Owned eligible native reader, or NULL for a completed/host dependency.
  iree_hal_amd_gpu_signal_t* signal;
} iree_hal_amd_gpu_pm4_submission_wait_t;

// Borrowed dispatch inputs consumed by one publication attempt. Native
// argument packing reads caller storage directly; only a deferred attempt
// needs a capture. No pointer to this view survives the publishing call.
typedef struct iree_hal_amd_gpu_pm4_dispatch_request_t {
  // Borrowed executable dominating function until retirement takes ownership.
  iree_hal_executable_t* executable;
  // Prepared immutable launch and argument representation.
  const iree_hal_amd_gpu_pm4_function_t* function;
  // Admitted geometry and private-storage requirements.
  iree_hal_amd_gpu_pm4_dispatch_t launch;
  // Borrowed device-time geometry, when flags select indirect parameters.
  iree_hal_buffer_ref_t parameters;
  // Borrowed dense or native argument bytes, copied directly to native storage.
  iree_const_byte_span_t constants;
  // Borrowed shader binding references, resolved after dependency admission.
  iree_hal_buffer_ref_list_t bindings;
  // Admitted argument and launch interpretation.
  iree_hal_dispatch_flags_t flags;
} iree_hal_amd_gpu_pm4_dispatch_request_t;

typedef enum iree_hal_amd_gpu_pm4_entry_kind_e {
  IREE_HAL_AMD_GPU_PM4_ENTRY_BARRIER,
  IREE_HAL_AMD_GPU_PM4_ENTRY_DISPATCH,
  IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER,
  IREE_HAL_AMD_GPU_PM4_ENTRY_CONTROL,
  IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE,
  IREE_HAL_AMD_GPU_PM4_ENTRY_TRIM,
  IREE_HAL_AMD_GPU_PM4_ENTRY_FRONTIERS,
  IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS,
  IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_PROGRAM,
  IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS,
  IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS,
  IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_PROGRAM,
  IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SCRATCH,
} iree_hal_amd_gpu_pm4_entry_kind_t;

// Typed publication record. Commands and completed native acquisitions use the
// same FIFO handoff; no allocation callback can invoke a command consumer.
typedef struct iree_hal_amd_gpu_pm4_entry_t {
  // Detached before the publisher accesses the enclosing owner.
  iree_hal_amd_gpu_publication_entry_t publication;
  // Exact enclosing record and payload interpretation.
  iree_hal_amd_gpu_pm4_entry_kind_t kind;
} iree_hal_amd_gpu_pm4_entry_t;

// Asynchronous ownership shared by publication and native retirement. Deferred
// operations also capture caller inputs in the same CPU allocation. Each
// published piece owns separate retirement credit; this object also owns any
// unpublished suffix and never stores mutable per-publication completion data.
typedef struct iree_hal_amd_gpu_pm4_submission_t {
  // Unpublished work and nonterminal native pieces hold independent references.
  iree_atomic_ref_count_t ref_count;
  // Number of retained native readers, or UINT32_MAX before admission.
  uint32_t native_wait_count;
  // Captured-work link; the publisher parks unresolved operations separately.
  iree_hal_amd_gpu_pm4_entry_t entry;
  // Borrowed allocation owner, or NULL for storage borrowed from a retirement
  // slot. Either owner keeps the bytes live through the final release.
  iree_hal_amd_gpu_capture_pool_t* pool;
  // Complete provided extent, including unused tail storage, in bytes.
  iree_host_size_t allocation_size;
  // Number of captured wait edges.
  iree_host_size_t wait_count;
  // Inline dependencies, with native readers acquired during admission.
  iree_hal_amd_gpu_pm4_submission_wait_t* waits;
  // Owned signal timelines and copied values in inline storage.
  iree_hal_semaphore_list_t signals;
  // Prepared pre-operation execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t before;
  // Prepared terminal execution/cache boundary.
  iree_hal_amd_gpu_pm4_barrier_t after;
  // Owned native producer point, if any signal can use native acceleration.
  iree_hal_amd_gpu_signal_t* signal;
  // Optional profile operation occupies the bytes immediately after this
  // header.
  bool profiled;
  // File owner callback occupies the end of the initialized capture extent.
  bool has_retirement_callback;
  // Deferred acceptance owns one queue host count until terminal publication
  // or abandonment. Only the publisher or detached admission owner accesses
  // this bit; native retirement uses its published prefix instead.
  bool has_host_owner;
  // First failure, retained until all accepted pieces end their host accesses.
  iree_atomic_int32_t failure_code;
  // Exact initialized extent, computed once by the capture constructor.
  iree_host_size_t capture_size;
  union {
    // Scalar operation and target retained through terminal native retirement.
    struct {
      // Prepared address slot, resolved only after allocation dependencies.
      iree_hal_amd_gpu_buffer_reference_t target;
      // Immutable validated scalar operation.
      iree_hal_amd_gpu_control_t command;
    } control;
    // Admitted dispatch with privately captured arguments and resources.
    struct {
      // Executable dominating function, retained unless flags borrow lifetime.
      iree_hal_executable_t* executable;
      // Borrowed prepared function dominated by executable.
      const iree_hal_amd_gpu_pm4_function_t* function;
      // Copied shader references, retained unless flags borrow lifetimes.
      // Indirect parameters append one additional reference after count. A
      // deferred capture also reserves address scratch after the references.
      // The public shader binding count is unchanged.
      iree_hal_buffer_ref_list_t bindings;
      // Copied deferred constants; empty once an inline call packs native
      // bytes.
      iree_const_byte_span_t constants;
      // Admitted interpretation of constants, bindings, and launch geometry.
      iree_hal_dispatch_flags_t flags;
      // Admitted geometry and private-storage requirements.
      iree_hal_amd_gpu_pm4_dispatch_t launch;
    } dispatch;
    // Captured transfer descriptors and the exclusively owned publish cursor.
    iree_hal_amd_gpu_transfer_t transfer;
    // Immutable recording and private table/cursor, owned through completion.
    iree_hal_amd_gpu_pm4_command_invocation_t execute;
  };
} iree_hal_amd_gpu_pm4_submission_t;

// Reads retained admission facts in constant time; empty dispatches need no
// private backing. Recorded programs already aggregate all their dispatches.
static inline uint32_t iree_hal_amd_gpu_pm4_submission_scratch_required(
    const iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    return submission->execute.program->scratch_wave_byte_length;
  }
  if (submission->entry.kind != IREE_HAL_AMD_GPU_PM4_ENTRY_DISPATCH) {
    return 0;
  }
  const iree_hal_amd_gpu_pm4_dispatch_t* launch = &submission->dispatch.launch;
  return iree_hal_dispatch_uses_indirect_parameters(
             submission->dispatch.flags) ||
                 (launch->geometry.workgroup_count[0] &&
                  launch->geometry.workgroup_count[1] &&
                  launch->geometry.workgroup_count[2])
             ? launch->scratch_wave_byte_length
             : 0;
}

// Borrows optional metadata owned by the capture. Its publication owner is the
// sole writer; completion never modifies it.
static inline iree_hal_amd_gpu_profile_operation_t*
iree_hal_amd_gpu_pm4_submission_profile(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  return submission->profiled
             ? (iree_hal_amd_gpu_profile_operation_t*)(submission + 1)
             : NULL;
}

// Creates a refcounted capture of an admitted operation before the public API
// returns. The pool is borrowed and outlives every retained reference.
// Reuses one block for ordinary calls; oversized calls use the pool allocator.
// A NULL executable/function captures a barrier and requires empty arguments.
// Failure retains nothing and leaves out_submission NULL. Arrays and bytes
// cease borrowing the caller on success. Buffer addresses remain unresolved
// until queue-ordered allocation and dependency admission permit reading them.
iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_hal_buffer_ref_t parameters,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Creates only the asynchronous ownership for a dispatch already prepared for
// publication. Constants and wait semaphores are not captured. On success,
// transfers the native readers in waits, retains signals, and retains resources
// unless flags borrow their lifetimes. No borrow of request or waits survives.
// Failure transfers nothing.
// Uses inline_storage when the complete capture fits, otherwise acquires pool
// storage. The span must be aligned for submission_t and remain exclusively
// owned through final release. Final release returns resources but does not
// free the borrowed span; its owner reclaims those bytes afterward.
// The prepared native command bytes, including arguments, must survive any
// KERNEL capacity deferral; this object cannot reconstruct those bytes.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_retirement(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    const iree_hal_amd_gpu_pm4_dispatch_request_t* request,
    iree_host_size_t wait_count,
    const iree_hal_amd_gpu_pm4_submission_wait_t* waits,
    uint32_t native_wait_count, iree_hal_semaphore_list_t signals,
    iree_hal_amd_gpu_pm4_barrier_t before, iree_hal_amd_gpu_pm4_barrier_t after,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Captures a qualified scalar control target and its dependency edges. Native
// address resolution waits for admission; the buffer is retained through
// terminal retirement. Failure retains nothing.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_control(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_amd_gpu_control_t command,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Captures the complete public transfer transaction, including update bytes
// and fill patterns. Native address slots are qualified now; queue-ordered
// addresses are resolved only after dependencies permit execution. Zero-length
// operations are omitted; an empty transaction becomes a barrier capture.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Captures one nonempty file UPLOAD/DOWNLOAD, or a barrier when operation is
// NULL. The file owner has resolved dependencies and supplies already-prepared
// boundaries.
// The callback replaces private signal semaphores and owns the file's borrowed
// bytes until the capture's last publication/native reference retires. Failure
// retains nothing and invokes no callback. Success may complete immediately.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_file(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_queue_family_t* family,
    const iree_hal_transfer_operation_t* operation,
    const iree_hal_amd_gpu_pm4_barrier_t* before,
    const iree_hal_amd_gpu_pm4_barrier_t* after,
    iree_hal_amd_gpu_retirement_callback_t callback,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Records rejection of an unpublished suffix. Public failure waits
// for the final capture reference, after all accepted host accesses have ended.
// Each reporter retains its reference through this call.
void iree_hal_amd_gpu_pm4_submission_reject(
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_code_t code);

// Records a borrowed terminal failure and notifies the optional file owner
// immediately without releasing captured resources. Future host accesses must
// already be stopped. Concurrent reporters deliver only the first failure.
// Each reporter owns a capture reference through the callback's final access.
void iree_hal_amd_gpu_pm4_submission_fail(
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status);

// Records the completed queue epoch for the final file callback. Only the
// terminal native retirement owner writes this, before dropping its reference.
// NULL records no frontier. Ordinary submissions have no callback storage.
void iree_hal_amd_gpu_pm4_submission_set_frontier(
    iree_hal_amd_gpu_pm4_submission_t* submission,
    const iree_async_single_frontier_t* frontier);

// Captures a recording and its invocation table in the same reusable block as
// semaphore ownership. Recorded barriers own payload visibility; execution adds
// no implicit system-scope before/after boundary. Failure retains nothing.
iree_status_t iree_hal_amd_gpu_pm4_submission_create_execute(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission);

// Summarizes the already-captured nonempty transfer payload for profiling.
// Several transfers form one TRANSFER event; a single transfer retains its
// specific COPY/FILL/UPDATE type. Empty transactions are barriers.
void iree_hal_amd_gpu_pm4_submission_profile_transfer(
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal);

// Borrows the unused, naturally aligned tail of a reusable capture block.
// Oversized captures have no spare storage. Admission may place sidecars here
// without another allocation; their users must join before the capture's final
// release. The capture owns these bytes and this query transfers no ownership.
iree_byte_span_t iree_hal_amd_gpu_pm4_submission_tail_storage(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Pins one nonterminal native piece independently of its unpublished suffix.
// The final publication transfers the suffix reference directly to retirement.
void iree_hal_amd_gpu_pm4_submission_retain(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Tests the host-retirement facts of already captured native dependencies.
// Native execution may complete before the fixed-order completion observer
// retires a producer's payload. All signals here belong to the same device
// completion owner; a producer's progress prompts another source round.
// A true result publishes the first failed dependency's status code, or OK.
// A failed dependency never substitutes for joining the other native users.
// out_status_code is unspecified while any dependency remains unretired.
bool iree_hal_amd_gpu_pm4_submission_retirement_ready(
    const iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_status_code_t* out_status_code);

// Releases payload resources after terminal native completion, before public
// completion signals. The final publisher has transferred its reference and
// no longer accesses this capture; earlier native pieces have already retired.
// Does not publish signal retirement: the completion owner must first finish
// its public signal effects and then publish their outcome. Idempotent for
// terminal retirement and final capture release.
void iree_hal_amd_gpu_pm4_submission_release_payload(
    iree_hal_amd_gpu_pm4_submission_t* submission);

// Releases captured resources and storage on the last reference. Native uses
// must already have completed or never have been published; failure alone does
// not suffice. NULL is ignored.
void iree_hal_amd_gpu_pm4_submission_release(
    iree_hal_amd_gpu_pm4_submission_t* submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SUBMISSION_H_
