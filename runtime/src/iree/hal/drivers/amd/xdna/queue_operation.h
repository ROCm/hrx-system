// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/amd/fabric/file_transfer.h"
#include "iree/hal/drivers/amd/xdna/command_buffer.h"
#include "iree/hal/drivers/amd/xdna/queue_frontier.h"
#include "iree/hal/drivers/amd/xdna/queue_producer_index.h"
#include "iree/hal/drivers/amd/xdna/queue_service.h"
#include "iree/hal/drivers/amd/xdna/queue_storage.h"
#include "iree/hal/pool_wait.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_xdna_queue_t iree_hal_amd_xdna_queue_t;
typedef struct iree_hal_amd_xdna_operation_t iree_hal_amd_xdna_operation_t;

// One locally visible signal owned by an unaccepted dispatch.
typedef struct iree_hal_amd_xdna_producer_entry_t {
  // Intrusive queue-local lookup node.
  iree_hal_amd_xdna_queue_producer_node_t index_node;
  // Next signal owned by the same queue operation, or NULL.
  struct iree_hal_amd_xdna_producer_entry_t* next;
} iree_hal_amd_xdna_producer_entry_t;

typedef enum iree_hal_amd_xdna_memory_wait_kind_e {
  IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE = 0,
  IREE_HAL_AMD_XDNA_MEMORY_WAIT_FRONTIER,
  IREE_HAL_AMD_XDNA_MEMORY_WAIT_CAPACITY,
  IREE_HAL_AMD_XDNA_MEMORY_WAIT_CAPACITY_PREPARED,
} iree_hal_amd_xdna_memory_wait_kind_t;

// Cold alloca state used only when the selected pool cannot return immediately
// usable bytes. Storage comes from the operation capture arena.
typedef struct iree_hal_amd_xdna_memory_wait_t {
  // Active asynchronous wait source.
  iree_hal_amd_xdna_memory_wait_kind_t kind;
  // Joined prerequisite for every held reservation.
  iree_async_frontier_t* frontier;
  // Tracker-owned waiter storage while |kind| is FRONTIER.
  iree_async_frontier_waiter_t frontier_waiter;
  // Reusable observe-check-wait helper for pool capacity.
  iree_hal_pool_wait_t* capacity_wait;
} iree_hal_amd_xdna_memory_wait_t;

typedef enum iree_hal_amd_xdna_operation_kind_e {
  IREE_HAL_AMD_XDNA_OPERATION_BARRIER,
  IREE_HAL_AMD_XDNA_OPERATION_ALLOCA,
  IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA,
  IREE_HAL_AMD_XDNA_OPERATION_TRANSFER,
  IREE_HAL_AMD_XDNA_OPERATION_FILE,
  IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL,
  IREE_HAL_AMD_XDNA_OPERATION_DISPATCH,
  IREE_HAL_AMD_XDNA_OPERATION_EXECUTE,
} iree_hal_amd_xdna_operation_kind_t;

// Result of one native publication attempt, distinct from terminal status.
typedef enum iree_hal_amd_xdna_publication_result_e {
  // Preparation or native publication rejected the operation terminally.
  IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_REJECTED = 0,
  // Native capacity was temporarily unavailable and progress must be checked.
  IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_RETRY,
  // Native publication accepted the command and returned its opaque point.
  IREE_HAL_AMD_XDNA_PUBLICATION_RESULT_ACCEPTED,
} iree_hal_amd_xdna_publication_result_t;

// Queue-state ownership of a native operation's accepted prefix and suffix.
typedef enum iree_hal_amd_xdna_native_phase_e {
  // A publisher or ready-list entry owns the unissued suffix.
  IREE_HAL_AMD_XDNA_NATIVE_PHASE_ISSUING = 0,
  // The accepted prefix must retire before the next host phase can run.
  IREE_HAL_AMD_XDNA_NATIVE_PHASE_DRAINING,
  // No more commands will issue; pending native references still own lifetime.
  IREE_HAL_AMD_XDNA_NATIVE_PHASE_COMPLETE,
  // Retirement cannot be observed; native-reachable ownership is retained.
  IREE_HAL_AMD_XDNA_NATIVE_PHASE_ABANDONED,
} iree_hal_amd_xdna_native_phase_t;

// One captured host transfer descriptor.
typedef struct iree_hal_amd_xdna_transfer_t {
  // Next descriptor in transaction capture order, or NULL.
  struct iree_hal_amd_xdna_transfer_t* next;
  // Public descriptor with retained logical buffers.
  iree_hal_transfer_operation_t operation;
  // Inline storage for the active FILL pattern.
  uint8_t fill_pattern[4];
  // Chunked captured bytes for the active UPDATE operation.
  iree_hal_amd_xdna_queue_payload_t update_payload;
} iree_hal_amd_xdna_transfer_t;

typedef uint32_t iree_hal_amd_xdna_operation_profile_flags_t;
enum iree_hal_amd_xdna_operation_profile_flag_bits_e {
  // The queue event has been appended or dropped by a full ring.
  IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_RECORDED = 1u << 0,
  // The current execution phase runs on a progress service.
  IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_DEFERRED = 1u << 1,
  // Current native array has already selected optional measurement commands.
  IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_NATIVE_PREPARED = 1u << 2,
};

// Optional operation-arena sidecar, accessed by the current execution owner.
typedef struct iree_hal_amd_xdna_operation_profile_t {
  // Public submission identity and readiness, captured once per transaction.
  iree_hal_profile_queue_event_t event;
  // Capture and execution-owner state for this operation.
  iree_hal_amd_xdna_operation_profile_flags_t flags;
} iree_hal_amd_xdna_operation_profile_t;

// Captured queue transaction shared by submission and progress owners.
struct iree_hal_amd_xdna_operation_t {
  // Placed admission callback that resolves causal state on the queue owner.
  iree_async_operation_t admission;
  // Placed callback returning resolved software waits to the queue owner.
  iree_async_operation_t wait_completion;
  // Placed callback returning file I/O to the queue owner.
  iree_async_operation_t continuation;
  // Intrusive progress-service admission and placed result handoff.
  iree_hal_amd_xdna_queue_service_item_t service_item;
  // Registered timepoints plus one registration sentinel.
  iree_atomic_int32_t wait_count;
  // First failure transferred from a resolved timepoint callback.
  iree_atomic_intptr_t wait_status;
  // File-I/O status transferred to the proactor owner.
  iree_atomic_intptr_t continuation_status;
  // Borrowed queue kept live by |device|.
  iree_hal_amd_xdna_queue_t* queue;
  // Retained device dominating the queue through arena return.
  iree_hal_device_t* device;
  // Borrowed reusable block pool kept live by |device|.
  iree_arena_block_pool_t* metadata_block_pool;
  // Queue-owned arenas containing this operation and captured payloads.
  iree_hal_amd_xdna_queue_capture_t capture;
  // Intrusive readiness linkage, independent of proactor linkage.
  iree_hal_amd_xdna_operation_t* next;
  // Captured wait semaphores, retained through readiness.
  iree_hal_semaphore_list_t waits;
  // First wait not yet proven reached or native FIFO ordered.
  iree_host_size_t wait_resolution_offset;
  // Captured signal semaphores, retained through final publication.
  iree_hal_semaphore_list_t signals;
  // Optional profiling state placed in the operation capture arena.
  iree_hal_amd_xdna_operation_profile_t* profile;
  // Owning terminal operation status.
  iree_status_t status;
  // Causal lower bound retained through final publication.
  iree_hal_amd_xdna_frontier_state_t frontier;
  // Native execution ownership shared by direct and recorded dispatches.
  struct {
    // Accepted invocations still reachable by the device.
    uint32_t pending_count;
    // Suffix owner; protected by the queue state mutex.
    iree_hal_amd_xdna_native_phase_t phase;
    // Exclusive invocation prepared for the next publication or retry.
    iree_hal_amd_xdna_invocation_t* invocation;
    // Opaque command borrowing |invocation| until accepted or rejected.
    amdf_xdna_kernel_command_t command;
    // Producer registrations retained until the terminal native tail is
    // accepted.
    iree_hal_amd_xdna_producer_entry_t* producer_entries;
    // True while producer entries are linked into the queue index.
    bool producers_indexed;
    // First consumer awaiting this operation's terminal native acceptance.
    iree_hal_amd_xdna_operation_t* acceptance_waiters_head;
    // Last consumer awaiting this operation's terminal native acceptance.
    iree_hal_amd_xdna_operation_t* acceptance_waiters_tail;
    // One publication batch owned exclusively by the publisher claim.
    struct {
      // First reserved ring slot, outside the visible pending prefix.
      uint32_t start;
      // Maximum accepted commands fitting in the reserved free suffix.
      uint32_t capacity;
      // Number of commands accepted into reserved slots.
      uint32_t count;
      // Native capacity or terminal outcome of the most recent batch.
      iree_hal_amd_xdna_publication_result_t result;
      // Publisher-owned status, transferred to the queue owner at commit.
      iree_status_t status;
      // True when a BUSY attempt observed older native progress to wait for.
      bool had_pending;
    } publication;
  } native;
  // Active operation payload.
  iree_hal_amd_xdna_operation_kind_t kind;
  union {
    // Queue-ordered allocation transaction.
    struct {
      // Borrowed exact source pool selected by the caller.
      iree_hal_pool_t* pool;
      // Number of allocation rows in every trailing array.
      iree_host_size_t request_count;
      // Canonical requests copied during public capture.
      iree_hal_pool_reservation_request_t* requests;
      // Stable allocation roots returned before physical commitment.
      iree_hal_buffer_t** transient_buffers;
      // Pool tokens owned until attached or released.
      iree_hal_pool_reservation_t* reservations;
      // Per-reservation reuse prerequisites owned by the pool.
      iree_hal_pool_acquire_info_t* acquire_infos;
      // Borrowed prepared views awaiting wrapper staging.
      iree_hal_pool_reservation_view_t* reservation_views;
      // Materialized fallback buffers, or all NULL when the pool exposes
      // reservation views directly.
      iree_hal_buffer_t** materialized_buffers;
      // Cold memory readiness sidecar.
      iree_hal_amd_xdna_memory_wait_t* memory_wait;
      // Result from the most recent pool acquisition attempt.
      iree_hal_pool_acquire_result_t acquire_result;
      // Submission and callback claims protecting the borrowed pool arrays.
      iree_atomic_int32_t acquire_pending;
      // Callback-owned result transferred after the final acquisition claim.
      iree_status_t acquire_status;
      // True while this operation owns |reservations|.
      bool reservations_held;
    } alloca;
    // Queue-ordered allocation retirement transaction.
    struct {
      // Borrowed common source pool discovered during capture.
      iree_hal_pool_t* pool;
      // Number of allocation roots in the transaction.
      iree_host_size_t buffer_count;
      // Retained transient allocation roots.
      iree_hal_buffer_t** transient_buffers;
      // Detached tokens returned to |pool| after decommit.
      iree_hal_pool_reservation_t* reservations;
      // True while captured deallocation marks must be aborted on failure.
      bool marks_owned;
    } dealloca;
    // Finite native function invocation.
    struct {
      // Retained executable owning mutable command backing.
      iree_hal_executable_t* executable;
      // Borrowed function owned by executable.
      iree_hal_amd_xdna_function_t* function;
      // Number of captured binding rows.
      iree_host_size_t binding_count;
      // Stable native slots with retained logical buffers in trailing storage.
      iree_hal_amd_xdna_executable_binding_t* bindings;
    } dispatch;
    // Reusable recording with an exclusive execution frame.
    iree_hal_amd_xdna_execution_t* execution;
    // Host-mapped transfer transaction.
    struct {
      // First captured descriptor in transaction order, or NULL.
      iree_hal_amd_xdna_transfer_t* head;
      // Final captured descriptor used while appending, or NULL.
      iree_hal_amd_xdna_transfer_t* tail;
    } transfer;
    // File transaction using a prepared host execution binding.
    struct {
      // Arena-owned state retaining the file and logical buffer.
      iree_hal_amd_fabric_file_transfer_t* transfer;
      // True after a chunk retires until host publication prepares another.
      bool io_completed;
    } file;
    // Captured host callback invoked after semaphore dependencies complete.
    struct {
      // Function and optional retained callback-state resource.
      iree_hal_host_call_t call;
      // Arguments copied before submission returns.
      uint64_t args[4];
      // Signal-publication policy and execution hints.
      iree_hal_host_call_flags_t flags;
    } host_call;
  };
};

// Returns whether the operation's next phase can enter native FIFO order.
// The cursor is accessed only by its execution owner or after a handoff.
static inline bool iree_hal_amd_xdna_operation_has_dispatch(
    const iree_hal_amd_xdna_operation_t* operation) {
  return operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DISPATCH ||
         (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE &&
          operation->execution->cursor &&
          operation->execution->cursor->kind ==
              IREE_HAL_AMD_XDNA_RECORD_DISPATCH);
}

// Captures profiling metadata before ownership transfers to queue execution.
iree_status_t iree_hal_amd_xdna_operation_profile_capture(
    iree_hal_amd_xdna_operation_t* operation);

// Prepares optional measurement once before native publication; BUSY retains
// it.
void iree_hal_amd_xdna_operation_profile_prepare_native(
    iree_hal_amd_xdna_operation_t* operation, uint32_t count,
    iree_hal_amd_xdna_invocation_t* const* invocations,
    amdf_xdna_kernel_command_t* commands);

// Commits native acceptance attribution before advancing the replay cursor.
void iree_hal_amd_xdna_operation_profile_accept_native(
    iree_hal_amd_xdna_operation_t* operation, uint32_t count,
    iree_hal_amd_xdna_invocation_t* const* invocations,
    const amdf_xdna_kernel_command_t* commands);

// Records the first observed execution readiness; subsequent phases are no-ops.
void iree_hal_amd_xdna_operation_profile_ready(
    iree_hal_amd_xdna_operation_t* operation);

// Emits any submission that failed before becoming ready.
void iree_hal_amd_xdna_operation_profile_complete(
    iree_hal_amd_xdna_operation_t* operation);

// Appends a complete host-work span, excluding asynchronous native execution.
void iree_hal_amd_xdna_operation_profile_host_end(
    iree_hal_amd_xdna_operation_t* operation, iree_time_t start,
    iree_status_code_t status_code, uint32_t operation_count);

// Creates an operation in queue-owned storage. The caller must submit or
// discard the returned operation.
iree_status_t iree_hal_amd_xdna_operation_allocate(
    iree_hal_amd_xdna_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_amd_xdna_operation_kind_t kind,
    iree_hal_amd_xdna_operation_t** out_operation);

// Releases a captured operation that has not been submitted.
void iree_hal_amd_xdna_operation_discard(
    iree_hal_amd_xdna_operation_t* operation);

// Submits and consumes |operation| on both success and failure.
iree_status_t iree_hal_amd_xdna_operation_submit(
    iree_hal_amd_xdna_operation_t* operation);

// Releases all kind-specific resources retained by |operation|.
void iree_hal_amd_xdna_operation_release_resources(
    iree_hal_amd_xdna_operation_t* operation);

// Returns held allocation reservations with their original reuse frontiers.
void iree_hal_amd_xdna_operation_release_alloca_reservations(
    iree_hal_amd_xdna_operation_t* operation);

// Consumes a ready memory operation directly, without a queue or pool lock.
// Pending pool acquisition resumes through its captured completion callback.
void iree_hal_amd_xdna_operation_issue_memory(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status);

// Executes transfers, callbacks, or file preparation
// and publication on the host service and returns the represented operation
// count. Disk I/O runs on its file's proactor.
uint32_t iree_hal_amd_xdna_operation_execute_host(
    iree_hal_amd_xdna_operation_t* operation);

// Captures a callback for the existing host service, including optional state
// retention. Semaphore dependencies govern invocation and completion.
iree_status_t iree_hal_amd_xdna_queue_host_call(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags);

// Captures a queue-ordered allocation transaction.
iree_status_t iree_hal_amd_xdna_queue_alloca(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers);

// Captures a queue-ordered deallocation transaction.
iree_status_t iree_hal_amd_xdna_queue_dealloca(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers);

// Captures a host-mapped transfer transaction.
iree_status_t iree_hal_amd_xdna_queue_transfer(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers);

// Captures a file read through the buffer's private host execution binding.
iree_status_t iree_hal_amd_xdna_queue_read(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags);

// Captures a file write through the buffer's private host execution binding.
iree_status_t iree_hal_amd_xdna_queue_write(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_
