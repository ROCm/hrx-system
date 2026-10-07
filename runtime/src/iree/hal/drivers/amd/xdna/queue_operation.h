// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_

#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/amd/xdna/executable.h"
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
  IREE_HAL_AMD_XDNA_OPERATION_DISPATCH,
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

// Captured queue transaction shared by submission and progress owners.
struct iree_hal_amd_xdna_operation_t {
  // Placed admission callback that resolves causal state on the queue owner.
  iree_async_operation_t admission;
  // Placed callback returning resolved software waits to the queue owner.
  iree_async_operation_t wait_completion;
  // Placed callback returning pool readiness to the queue owner.
  iree_async_operation_t memory_completion;
  // Intrusive progress-service admission and placed result handoff.
  iree_hal_amd_xdna_queue_service_item_t service_item;
  // Registered timepoints plus one registration sentinel.
  iree_atomic_int32_t wait_count;
  // First failure transferred from a resolved timepoint callback.
  iree_atomic_intptr_t wait_status;
  // Terminal hidden-memory-wait status transferred to the proactor owner.
  iree_atomic_intptr_t memory_status;
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
  // Owning terminal operation status.
  iree_status_t status;
  // Causal lower bound retained through final publication.
  iree_hal_amd_xdna_frontier_state_t frontier;
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
      // Exclusive prepared storage returned only after rejection or retirement.
      iree_hal_amd_xdna_invocation_t* invocation;
      // Native command borrowing |invocation| across publication retries.
      amdf_xdna_kernel_command_t command;
      // Opaque point returned by successful native acceptance.
      uint64_t submission;
      // Outcome of the most recent native publication attempt.
      iree_hal_amd_xdna_publication_result_t publication_result;
      // True when the most recent attempt observed older accepted work.
      bool publication_had_pending;
      // Number of captured binding rows.
      iree_host_size_t binding_count;
      // Stable native slots with retained logical buffers in trailing storage.
      iree_hal_amd_xdna_executable_binding_t* bindings;
      // Intrusive index entries for locally visible signal semaphores.
      iree_hal_amd_xdna_producer_entry_t* producer_entries;
      // True while |producer_entries| are present in the queue index.
      bool producers_indexed;
      // First consumer waiting for this operation's native acceptance.
      iree_hal_amd_xdna_operation_t* acceptance_waiters_head;
      // Final consumer waiting for this operation's native acceptance.
      iree_hal_amd_xdna_operation_t* acceptance_waiters_tail;
    } dispatch;
    // Host-mapped transfer transaction.
    struct {
      // First captured descriptor in transaction order, or NULL.
      iree_hal_amd_xdna_transfer_t* head;
      // Final captured descriptor used while appending, or NULL.
      iree_hal_amd_xdna_transfer_t* tail;
    } transfer;
  };
};

// Creates an operation in queue-owned storage. The caller must submit or
// discard the returned operation.
iree_status_t iree_hal_amd_xdna_operation_create(
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

// Merges all reservation reuse prerequisites into the operation frontier.
void iree_hal_amd_xdna_operation_merge_memory_frontiers(
    iree_hal_amd_xdna_operation_t* operation);

// Builds the joined frontier for reservations that require asynchronous reuse.
iree_status_t iree_hal_amd_xdna_operation_prepare_frontier_wait(
    iree_hal_amd_xdna_operation_t* operation);

// Executes one allocation, deallocation, or transfer transaction on the
// blocking host service.
void iree_hal_amd_xdna_operation_execute_host(
    iree_hal_amd_xdna_operation_t* operation);

// Captures a queue-ordered allocation transaction.
iree_status_t iree_hal_amd_xdna_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers);

// Captures a queue-ordered deallocation transaction.
iree_status_t iree_hal_amd_xdna_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers);

// Captures a host-mapped transfer transaction.
iree_status_t iree_hal_amd_xdna_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_OPERATION_H_
