// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Task queue: two persistent processes, a wake_budget == 1 control process and
// a wake_budget > 1 compute process, that cooperatively execute queue
// operations.
//
// Submissions flow through semaphore waits into the ready list. The
// wake_budget == 1 control process pops operations and handles them by type:
// barriers, host calls, and allocations are handled inline; command buffers are
// filled into recording items and pushed to the compute process's pending list.
// Mapping batches run as finite processes on the existing executor workers.
//
// The wake_budget > 1 compute process drains recording items cooperatively
// across all workers via the block processor. It occupies a single compute slot
// for the queue's lifetime. Per-recording two-phase completion ensures
// semaphores are signaled eagerly while resources stay alive until all
// workers have exited drain.
//
// Operations are arena-allocated at submit time and freed by the completion
// callback. No per-submission task allocations at issue time.
//
// Shutdown first retires the control process, the sole compute-item producer,
// and only its release callback requests compute-process termination. This
// orders full-pool cleanup after producer quiescence without blocking a worker.

#ifndef IREE_HAL_DRIVERS_TASK_QUEUE_QUEUE_H_
#define IREE_HAL_DRIVERS_TASK_QUEUE_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/command/block_processor.h"
#include "iree/hal/drivers/task/profile.h"
#include "iree/hal/drivers/task/queue/memory_barrier.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/scope.h"
#include "iree/task/tuning.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A single command buffer submitted to a device queue.
// All of the wait semaphores must reach or exceed the given payload values
// prior to the batch beginning execution. Only after all commands have
// completed will the signal semaphores be updated to the provided payload
// values.
typedef struct iree_hal_task_submission_batch_t {
  // Semaphores to wait on prior to executing any command buffer.
  iree_hal_semaphore_list_t wait_semaphores;

  // Command buffer to execute and optional binding table.
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;

  // Semaphores to signal once all command buffers have completed execution.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_task_submission_batch_t;

typedef struct iree_async_file_t iree_async_file_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_hal_resource_set_t iree_hal_resource_set_t;
typedef struct iree_hal_task_queue_t iree_hal_task_queue_t;

//===----------------------------------------------------------------------===//
// Queue operation (ready list entry)
//===----------------------------------------------------------------------===//

// Type of operation queued for execution.
typedef enum iree_hal_task_queue_op_type_e {
  IREE_HAL_TASK_QUEUE_OP_COMMANDS,
  IREE_HAL_TASK_QUEUE_OP_BARRIER,
  IREE_HAL_TASK_QUEUE_OP_HOST_CALL,
  IREE_HAL_TASK_QUEUE_OP_ALLOCA,
  IREE_HAL_TASK_QUEUE_OP_DEALLOCA,
  IREE_HAL_TASK_QUEUE_OP_READ,
  IREE_HAL_TASK_QUEUE_OP_WRITE,
  IREE_HAL_TASK_QUEUE_OP_TRANSFER,
  IREE_HAL_TASK_QUEUE_OP_ATOMIC_WAIT,
  IREE_HAL_TASK_QUEUE_OP_ATOMIC_STORE,
  IREE_HAL_TASK_QUEUE_OP_ATOMIC_RMW,
  IREE_HAL_TASK_QUEUE_OP_DISPATCH,
  IREE_HAL_TASK_QUEUE_OP_MAP,
} iree_hal_task_queue_op_type_t;

typedef uint32_t iree_hal_task_queue_op_flags_t;
enum iree_hal_task_queue_op_flag_bits_e {
  IREE_HAL_TASK_QUEUE_OP_FLAG_NONE = 0u,
  IREE_HAL_TASK_QUEUE_OP_FLAG_HAS_PROFILE_OPERATION = 1u << 0,
  // Payload acquisition after the operation's semaphore waits.
  IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_ACQUIRE = 1u << 1,
  // Payload publication before the operation (including atomic flag stores).
  IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_RELEASE = 1u << 2,
  // Payload acquisition after the operation (including atomic flag waits).
  IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_ACQUIRE = 1u << 3,
  // Payload publication before terminal completion signals.
  IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_RELEASE = 1u << 4,
  // Conservative acquisition of the one-shot operation's read operands.
  IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_DEFAULT = 1u << 5,
  // Conservative publication of the one-shot operation's written operands.
  IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_DEFAULT = 1u << 6,
  // Successful file I/O returned from the proactor for cache operations.
  IREE_HAL_TASK_QUEUE_OP_FLAG_COMPLETING_IO = 1u << 7,
};

typedef struct iree_hal_task_queue_op_t iree_hal_task_queue_op_t;
typedef struct iree_hal_task_queue_alloca_memory_wait_t
    iree_hal_task_queue_alloca_memory_wait_t;

#if !defined(NDEBUG)
// Debug-only metadata attached to one queue operation.
typedef struct iree_hal_task_queue_op_debug_state_t {
  // Monotonic ordinal assigned at allocation time. Used only for queue dumps
  // when diagnosing lost-submit/lost-completion bugs.
  uint64_t ordinal;
} iree_hal_task_queue_op_debug_state_t;
#endif  // !defined(NDEBUG)

// A single queued operation. Arena-allocated; the arena is freed when the
// operation completes (barriers/host calls in drain, command buffers in
// the CB process completion callback).
//
// The slist_next field must be at a stable offset for the MPSC ready list.
struct iree_hal_task_queue_op_t {
  // Intrusive slist node for the ready list.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Operation type.
  iree_hal_task_queue_op_type_t type;

  // Bitfield of iree_hal_task_queue_op_flag_bits_e.
  iree_hal_task_queue_op_flags_t flags;

  // Arena owning this operation and all transient allocations (semaphore
  // list storage, wait entries, binding table copies). The arena is
  // deinitialized when the operation completes.
  iree_arena_allocator_t arena;

  // Scope for begin/end lifecycle tracking. The matching scope_begin is
  // called at submit time; scope_end is called when the operation completes.
  iree_task_scope_t* scope;

  // Semaphores to signal on completion.
  iree_hal_semaphore_list_t signal_semaphores;

  // Resources retained until all have retired (command buffers, binding
  // table buffers, etc.). Allocated from the small block pool.
  iree_hal_resource_set_t* resource_set;

  // Arena-owned host recipes, NULL when only fences/default operands suffice.
  iree_hal_task_queue_memory_barriers_t* memory_barriers;

  // Frontier tracker advanced when the operation completes.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Frontier axis advanced when the operation completes.
  iree_async_axis_t axis;

  // Monotonic epoch counter incremented to generate completion epochs.
  iree_atomic_int64_t* epoch_counter;

  // Outstanding semaphore wait count. Atomically decremented by semaphore
  // timepoint callbacks. When this reaches zero, the operation is pushed
  // to the ready list. Initialized to the number of unsatisfied waits.
  iree_atomic_int32_t wait_count;

  // Back-pointer to the owning queue. Used by semaphore wait callbacks to
  // push to the ready list and schedule the queue process when all waits
  // are satisfied. The queue outlives all operations (deinitialize waits
  // for scope idle).
  iree_hal_task_queue_t* queue;

#if !defined(NDEBUG)
  // Debug-only metadata omitted from release builds.
  iree_hal_task_queue_op_debug_state_t debug;
#endif  // !defined(NDEBUG)

  // First error encountered from a failed semaphore wait. Set via CAS —
  // only the first error wins. Checked on the last wait_count decrement
  // to decide between pushing to the ready list or failing the operation.
  iree_atomic_intptr_t error_status;

  // Type-specific data.
  union {
    struct {
      iree_hal_command_buffer_t* command_buffer;
      iree_hal_buffer_binding_table_t binding_table;
    } commands;
    struct {
      // Exact hardware queue reported to the callback.
      iree_hal_queue_t* queue;

      // User callback and state.
      iree_hal_host_call_t call;

      // User-defined callback arguments.
      uint64_t args[4];

      // Flags controlling callback ordering and completion.
      iree_hal_host_call_flags_t flags;
    } host_call;
    // Arena-owned native mapping process, independent of the control process.
    struct iree_hal_task_queue_mapping_t* mapping;
    struct {
      // Borrowed pool used for the complete allocation transaction.
      iree_hal_pool_t* pool;

      // Number of allocation requests in the transaction.
      iree_host_size_t request_count;

      // Arena-owned requests captured after allocator canonicalization.
      iree_hal_pool_reservation_request_t* requests;

      // Arena-owned transient wrapper pointers retained by |resource_set|.
      iree_hal_buffer_t** transient_buffers;

      // Arena-owned reservations populated by the pool acquisition.
      iree_hal_pool_reservation_t* reservations;

      // Arena-owned per-request pool acquisition information.
      iree_hal_pool_acquire_info_t* acquire_infos;

      // Arena-owned materialized backing buffer outputs.
      iree_hal_buffer_t** backing_buffers;

      // Pool reservation flags used when probing the selected pool.
      iree_hal_pool_reserve_flags_t reserve_flags;

      // Native acquisition completion, joined with its submitting call before
      // any continuation may consume the operation or its borrowed arrays.
      struct {
        // Submission and callback each relinquish one claim.
        iree_atomic_int32_t pending;
        // Terminal pool result, published by completion.
        iree_hal_pool_acquire_result_t result;
        // Owned terminal failure, consumed by the continuation.
        iree_status_t status;
      } acquire;

      // True while this operation owns every entry in |reservations|.
      bool reservations_held;

      // Cold-path memory-readiness wait state. NULL on the immediate path.
      iree_hal_task_queue_alloca_memory_wait_t* memory_wait;
    } alloca;
    struct {
      // Borrowed common source pool for every buffer in the transaction.
      iree_hal_pool_t* pool;

      // Number of transient buffers in the transaction.
      iree_host_size_t buffer_count;

      // Arena-owned transient wrapper pointers retained by |resource_set|.
      iree_hal_buffer_t** transient_buffers;

      // Arena-owned reservation storage populated when the operation runs.
      iree_hal_pool_reservation_t* reservations;

      // True while this operation owns each buffer's deallocation mark.
      bool marks_owned;
    } dealloca;
    struct {
      iree_hal_file_t* hal_file;
      iree_async_file_t* async_file;  // NULL = sync fallback via hal_file.
      uint64_t file_offset;
      iree_hal_buffer_t* buffer;
      iree_device_size_t buffer_offset;
      iree_device_size_t length;
    } read;
    struct {
      iree_hal_file_t* hal_file;
      iree_async_file_t* async_file;  // NULL = sync fallback via hal_file.
      uint64_t file_offset;
      iree_hal_buffer_t* buffer;
      iree_device_size_t buffer_offset;
      iree_device_size_t length;
    } write;
    struct {
      // Number of operations in the captured transaction.
      iree_host_size_t operation_count;
      // Arena-owned operation storage with captured fill and update data.
      iree_hal_transfer_operation_t* operations;
      // Saturating sum of active operation lengths used for strategy selection.
      iree_device_size_t total_length;
    } transfer;
    struct {
      // Retained buffer containing the target location.
      iree_hal_buffer_t* target_buffer;
      // Byte offset of the target location in |target_buffer|.
      iree_device_size_t target_offset;
      // Width, predicate, ordering, and scope of the wait.
      iree_hal_atomic_wait_params_t params;
    } atomic_wait;
    struct {
      // Retained buffer containing the target location.
      iree_hal_buffer_t* target_buffer;
      // Byte offset of the target location in |target_buffer|.
      iree_device_size_t target_offset;
      // Width, value, ordering, and scope of the store.
      iree_hal_atomic_store_params_t params;
    } atomic_store;
    struct {
      // Retained buffer containing the target location.
      iree_hal_buffer_t* target_buffer;
      // Byte offset of the target location in |target_buffer|.
      iree_device_size_t target_offset;
      // Width, operand, operation, ordering, and scope of the update.
      iree_hal_atomic_rmw_params_t params;
    } atomic_rmw;
    struct {
      iree_hal_executable_t* executable;
      iree_hal_executable_function_t export_ordinal;
      iree_hal_dispatch_config_t config;
      // Constants arena-allocated in the operation arena.
      iree_const_byte_span_t constants;
      // Bindings arena-allocated (pointer into operation arena).
      const iree_hal_buffer_ref_t* bindings;
      iree_host_size_t binding_count;
      iree_hal_dispatch_flags_t flags;
    } dispatch;
  };
};

// Typed MPSC slist for the ready list.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_task_queue_op,
                                iree_hal_task_queue_op_t,
                                offsetof(iree_hal_task_queue_op_t, slist_next));

//===----------------------------------------------------------------------===//
// Compute recording items (pool-managed)
//===----------------------------------------------------------------------===//

// Initial pool size for compute recording items. The pool grows dynamically
// when all items are in-flight. 16 covers typical pipeline depths (10+
// concurrent recordings between submission and final release).
#define IREE_HAL_TASK_QUEUE_COMPUTE_INITIAL_POOL_SIZE 16

typedef struct iree_hal_task_queue_compute_item_t
    iree_hal_task_queue_compute_item_t;

// A single recording being drained by the compute process. Arena-allocated
// with a trailing worker_states[] FAM. Items cycle through:
//   free_pool → (wake_budget == 1 fills) → pending → (compute installs) →
//   current → (completion + release) → free_pool
//
// Each item holds the block processor context and per-worker drain state for
// one command buffer recording. The low-bit drainer count in item->drainers is
// per-recording (separate from the compute slot's per-process active_drainers):
// the compute process occupies one slot persistently, but recordings flow
// through it sequentially.
//
// Two-phase lifecycle per recording:
//   Eager close:       First worker to observe completed=true sets CLOSED and
//                      installs the next pending recording (or NULL) in
//                      compute_current so unrelated work can start
//                      immediately.
//   Final release:     Last worker to leave after CLOSED is set claims
//                      RELEASE_CLAIMED, consumes the processor result exactly
//                      once, completes/fails the operation, tears down the
//                      processor context and any item-owned recording blocks,
//                      bumps the generation, and returns the item to the free
//                      pool.
struct iree_hal_task_queue_compute_item_t {
  // Intrusive slist node for the pending list and free pool.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Linked list of all allocated items (for shutdown cleanup enumeration).
  // Written once at allocation, read only during shutdown.
  iree_hal_task_queue_compute_item_t* next_allocated;

  // Combined generation + drainer count + release/closed flags in one 64-bit
  // atomic. This replaces three separate fields (generation, active_drainers,
  // release_pending) to eliminate the TOCTOU race between checking the
  // drainer count and setting the completion flag.
  //
  //   bits 63-32: generation (monotonic, ABA prevention for recycled items)
  //   bit 31:     CLOSED flag (set when recording completes)
  //   bit 30:     RELEASE_CLAIMED flag (one worker owns final cleanup)
  //   bits 29-0:  active drainer count
  //
  // Protocol (mirrors the compute slot protocol in worker.c):
  //   Entry:  fetch_add(1). If (int32_t)prev < 0 → CLOSED, bail.
  //           Snapshot the item's generation from item->drainers before
  //           registering, then verify fetch_add observed that same
  //           generation. Re-check queue->compute_current_revision and
  //           queue->compute_current after registering to reject recycled
  //           same-address items.
  //   Close:  fetch_or(CLOSED_BIT). First to set publishes the next item.
  //   Exit:   fetch_sub(1). If prev == (gen | CLOSED | 1), try to CAS
  //           gen|CLOSED → gen|CLOSED|RELEASE_CLAIMED. The winner performs
  //           final cleanup and operation completion/failure.
  //   Reset:  CAS(gen|CLOSED|RELEASE_CLAIMED → next_gen|0) during final
  //           cleanup, waiting for late CLOSED bailers to drain first.
  iree_atomic_int64_t drainers;
#define IREE_HAL_TASK_QUEUE_ITEM_CLOSED_BIT ((int64_t)(uint32_t)INT32_MIN)
#define IREE_HAL_TASK_QUEUE_ITEM_RELEASE_CLAIMED_BIT \
  ((int64_t)(uint32_t)(1u << 30))
#define IREE_HAL_TASK_QUEUE_ITEM_GEN_INCREMENT ((int64_t)1 << 32)

  // Block processor execution context. Separately allocated with cache-line
  // alignment by context_allocate; freed in the final release path. NULL for
  // no-block recordings, which complete immediately through the null-safe
  // processor drain path.
  iree_hal_cmd_block_processor_context_t* processor_context;

  // Number of workers participating in draining this recording.
  uint32_t worker_count;

  // Back-pointer to the queue operation that submitted this recording.
  // Owned by the item from fill time until the final release worker consumes
  // the processor result and completes/fails the operation. Keeping the
  // operation alive until the last drainer exits preserves its resource_set
  // and scope without a separate ownership handoff.
  iree_hal_task_queue_op_t* operation;

  // Allocator used to free the processor context. Snapshotted at fill time so
  // the final release path doesn't chase through queue pointers that
  // may be destroyed during shutdown.
  iree_allocator_t host_allocator;

  // For queue-built recordings (not from a command buffer): the recording
  // whose blocks must be released in the final release path. For command
  // buffer recordings, first_block is NULL and the command buffer's recording
  // remains live through item->operation->resource_set until final release.
  iree_hal_cmd_block_recording_t recording;

  // Per-worker state for block processor drain calls. Each worker maintains
  // a block_sequence counter to detect block transitions. Zero-initialized
  // when the item is filled. Trailing FAM sized to worker_count at allocation.
  iree_hal_cmd_block_processor_worker_state_t worker_states[];
};

// Typed MPSC slist for the compute pending and free pool lists.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_task_queue_compute_item,
                                iree_hal_task_queue_compute_item_t,
                                offsetof(iree_hal_task_queue_compute_item_t,
                                         slist_next));

//===----------------------------------------------------------------------===//
// iree_hal_task_queue_t
//===----------------------------------------------------------------------===//

#if !defined(NDEBUG)
// Debug-only queue operation counters. These are best-effort monotonic
// snapshots for queue dumps and are not part of queue correctness.
typedef struct iree_hal_task_queue_debug_state_t {
  iree_atomic_int64_t next_operation_ordinal;
  iree_atomic_int64_t submitted_operation_count;
  iree_atomic_int64_t completed_operation_count;
  iree_atomic_int64_t failed_operation_count;
  iree_atomic_int64_t destroyed_ok_operation_count;
  iree_atomic_int64_t destroyed_failed_operation_count;
  iree_atomic_int64_t last_submitted_operation_ordinal;
  iree_atomic_int64_t last_completed_operation_ordinal;
  iree_atomic_int64_t last_failed_operation_ordinal;
  iree_atomic_int64_t last_destroyed_operation_ordinal;
} iree_hal_task_queue_debug_state_t;
#endif  // !defined(NDEBUG)

// Ordered queue shutdown phases.
typedef enum iree_hal_task_queue_shutdown_phase_e {
  IREE_HAL_TASK_QUEUE_SHUTDOWN_PHASE_RUNNING = 0,
  IREE_HAL_TASK_QUEUE_SHUTDOWN_PHASE_CONTROL = 1,
  IREE_HAL_TASK_QUEUE_SHUTDOWN_PHASE_COMPUTE = 2,
} iree_hal_task_queue_shutdown_phase_t;

// Cold-path construction parameters shared by provisioned and dynamically
// acquired task queues.
typedef struct iree_hal_task_queue_create_params_t {
  // Human-readable queue identifier borrowed for the duration of creation.
  iree_string_view_t identifier;

  // Parent device used to validate submitted resources. Borrowed for the queue
  // lifetime under the HAL device/queue lifetime contract.
  iree_hal_device_t* device;

  // Exact queue family identity borrowed from |device|.
  const iree_hal_queue_family_t* queue_family;

  // Exact immutable queue properties. Execution-resource storage is borrowed
  // for initialization and copied by dynamic allocation.
  iree_hal_queue_params_t queue_params;

  // Flags used to initialize the queue task scope.
  iree_task_scope_flags_t scope_flags;

  // Executor retained by the queue.
  iree_task_executor_t* executor;

  // Local or explicitly unplaced proactor borrowed from the device's pool.
  iree_async_proactor_t* proactor;

  // Aggregate transfer payload length selecting direct or recorded execution.
  iree_device_size_t inline_transfer_threshold;

  // Shared block pool for submission transients.
  iree_arena_block_pool_t* small_block_pool;

  // Shared block pool for command recordings and compute items.
  iree_arena_block_pool_t* large_block_pool;

  // Device allocator retained by the queue.
  iree_hal_allocator_t* device_allocator;
} iree_hal_task_queue_create_params_t;

// Called after a separately allocated queue is quiesced and its frontier axis
// is retired. The parent device remains live by contract.
typedef void(IREE_API_PTR* iree_hal_task_queue_release_slot_fn_t)(
    void* user_data, uint8_t queue_index);

// Callback returning a dynamic queue identity slot to its parent device.
typedef struct iree_hal_task_queue_release_slot_callback_t {
  // Callback function invoked exactly once during queue destruction.
  iree_hal_task_queue_release_slot_fn_t fn;

  // Opaque parent-device state passed to |fn|.
  void* user_data;

  // Device-local queue identity slot returned to |fn|.
  uint8_t queue_index;
} iree_hal_task_queue_release_slot_callback_t;

// Storage ownership present only on separately allocated queues.
typedef struct iree_hal_task_queue_storage_t {
  // Allocator used to reclaim the queue after it has quiesced.
  iree_allocator_t allocator;

  // Callback returning the queue identity slot before storage reclamation.
  iree_hal_task_queue_release_slot_callback_t release_slot;
} iree_hal_task_queue_storage_t;

struct iree_hal_task_queue_t {
  // Base HAL queue resource. Must be at offset zero.
  iree_hal_queue_t base;

  // Dynamic queue storage and identity-slot ownership. Zeroed for provisioned
  // queues embedded in their parent device.
  iree_hal_task_queue_storage_t storage;

  // Parent device used to validate queue operation resources. Borrowed.
  iree_hal_device_t* device;

  // Shared executor that the queue submits processes to.
  iree_task_executor_t* executor;

  // Proactor for async I/O operations on this queue. Borrowed from the
  // device's proactor pool. Device construction selects a local or explicitly
  // unplaced service; dynamic queues reuse their executor's selected service.
  // Valid as long as the device (which retains the proactor pool) is alive.
  iree_async_proactor_t* proactor;

  // Shared frontier tracker for cross-device causal ordering.
  // Borrowed from the device after topology assignment.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This queue's axis identity in the frontier system.
  // Derived from the device's base_axis + queue_index at initialization.
  iree_async_axis_t axis;

  // Monotonic epoch counter for frontier advancement. Incremented at
  // COMPLETION time (not submit time) because the task queue is out-of-order:
  // operations complete on arbitrary threads in arbitrary order. Completion-
  // time assignment guarantees monotonic epoch progression without head-of-
  // line blocking. The epoch means "this many completions have occurred on
  // this queue."
  iree_atomic_int64_t epoch;

  // Aggregate payload length at which queue transfer transactions route
  // through the multi-worker block processor. See
  // iree_hal_task_device_params_t for details.
  iree_device_size_t inline_transfer_threshold;

  // Shared block pool for allocating submission transients.
  iree_arena_block_pool_t* small_block_pool;
  // Shared block pool for large allocations (command buffers/etc).
  iree_arena_block_pool_t* large_block_pool;

  // Device allocator used for transient allocations/tracking.
  iree_hal_allocator_t* device_allocator;

  // Active local CPU profile recorder, or NULL when profiling is disabled.
  iree_hal_task_profile_recorder_t* profile_recorder;

  // Queue metadata identity used by records during the active profile session.
  iree_hal_task_profile_queue_scope_t profile_scope;

  // Device-owned submission id counter shared by all queues in the session.
  iree_atomic_int64_t* profile_submission_counter;

  // Scope tracking all accepted queue operations through terminal completion.
  iree_task_scope_t operation_scope;

  // Scope tracking the persistent control and compute processes until their
  // release callbacks have run during queue shutdown.
  iree_task_scope_t process_scope;

  // Persistent process release callbacks that may still access queue fields
  // after the process scope reaches idle.
  iree_atomic_int32_t pending_process_release_count;

#if !defined(NDEBUG)
  // Debug-only operation counters omitted from release builds.
  iree_hal_task_queue_debug_state_t debug;
#endif  // !defined(NDEBUG)

  // MPSC ready list of operations with all semaphore waits satisfied.
  // Populated by semaphore timepoint callbacks (slow path) or directly
  // by the submitting thread (fast path when all waits are satisfied).
  iree_hal_task_queue_op_slist_t ready_list;

  // Monotonic shutdown phase. RUNNING accepts work, CONTROL rejects new work
  // and terminates the control process, and COMPUTE terminates the compute
  // process after the control process has released its producer ownership.
  iree_atomic_int32_t shutdown_phase;

  // The queue's persistent control process. Uses wake_budget == 1 and drains
  // operations from the ready list sequentially. When the ready list is empty,
  // the process returns did_work=false and the executor's sleeping protocol
  // parks it. New submissions wake it via schedule_process. Completes during
  // the CONTROL shutdown phase started by deinitialize.
  //
  // For COMMANDS operations, this process fills a recording item and pushes
  // it to compute_pending (delegating actual execution to the compute
  // process below). For all other operation types, this process handles
  // them inline, except mappings scheduled as finite processes on the executor.
  //
  // The process participates in the process scope: scope_begin at
  // initialization, scope_end in the release callback. This ensures
  // scope_wait_idle blocks until the process has fully completed and its
  // worker has exited the drain stack.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_process_t process;

  // Persistent wake_budget > 1 compute process for executing command buffer
  // recordings. Placed in a compute slot on first recording; stays there
  // for the queue's lifetime. Workers cooperatively drain recordings from
  // this process via the block processor.
  //
  // The wake_budget == 1 control process above is the control plane: it pops
  // operations from the ready list, fills recording items, and pushes them
  // to compute_pending. This compute process is the data plane: its drain
  // function loads the current recording item and delegates to
  // processor_drain for cooperative tile execution.
  //
  // schedule_process behavior for a persistent process: the first call
  // CAS(IDLE->DRAINING) and places it in a compute slot. Subsequent calls
  // fail the CAS (already DRAINING) but still wake workers. The process
  // stays in its slot until shutdown.
  //
  // Participates in the process scope: scope_begin at initialization,
  // scope_end in the release callback after the last drainer exits. Completes
  // only after the control process release callback advances shutdown to the
  // COMPUTE phase, ensuring no producer can still touch the item pool.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_task_process_t compute_process;

  // MPSC list of filled recording items ready to drain. The wake_budget == 1
  // process pushes items after filling; the compute drain function's
  // completer pops the next item when the current recording finishes.
  iree_hal_task_queue_compute_item_slist_t compute_pending;

  // Free pool of recording items. All items start here at initialization.
  // wake_budget == 1 process pops to acquire; final release pushes to return.
  iree_hal_task_queue_compute_item_slist_t compute_free_pool;

  // Current recording being drained by workers. Pointer to the active item,
  // or NULL when no recording is active.
  //
  // compute_current_revision is a seqlock-style publication sequence for this
  // pointer. Writers transition even->odd before changing compute_current and
  // odd->next-even afterwards. Workers snapshot the revision, load this
  // pointer, register on item->drainers, and re-check both revision and
  // pointer before entering the block processor.
  //
  // The publication revision matters because items are recycled at stable
  // addresses. Pointer identity alone does not distinguish "same item, next
  // lifecycle," and a stale worker must not attach itself to a recycled item
  // whose payload has already been reset.
  iree_atomic_intptr_t compute_current;
  iree_atomic_int64_t compute_current_revision;

  // Arena for recording item allocation. Items are bump-allocated with
  // cache-line alignment from blocks acquired from the large block pool.
  // Items are never individually freed — they cycle through the free slist.
  // The arena is deinitialized at queue shutdown, returning all blocks.
  iree_arena_allocator_t compute_item_arena;

  // Head of the all-items linked list for shutdown cleanup enumeration.
  iree_hal_task_queue_compute_item_t* compute_item_head;

  // Number of workers for this queue (cached from executor at init time).
  // Used to size the worker_states FAM in newly allocated items.
  uint32_t compute_worker_count;
};

// Retains the queue through its HAL resource lifetime. NULL is allowed.
static inline void iree_hal_task_queue_retain(iree_hal_task_queue_t* queue) {
  iree_hal_queue_retain((iree_hal_queue_t*)queue);
}

// Releases a queue reference. NULL is allowed.
static inline void iree_hal_task_queue_release(iree_hal_task_queue_t* queue) {
  iree_hal_queue_release((iree_hal_queue_t*)queue);
}

iree_status_t iree_hal_task_queue_initialize(
    const iree_hal_task_queue_create_params_t* params,
    iree_hal_task_queue_t* out_queue);

// Creates a dynamic task queue.
//
// Execution-resource ordinals in |params| are copied into queue-owned storage.
// |release_slot| is captured only on success; callers retain responsibility for
// returning the slot when allocation fails. |out_queue| is unchanged on
// failure.
iree_status_t iree_hal_task_queue_create(
    const iree_hal_task_queue_create_params_t* params,
    iree_hal_task_queue_release_slot_callback_t release_slot,
    iree_allocator_t host_allocator, iree_hal_task_queue_t** out_queue);

iree_status_t iree_hal_task_queue_assign_frontier(
    iree_hal_task_queue_t* queue,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis);

void iree_hal_task_queue_retire_frontier(iree_hal_task_queue_t* queue);

void iree_hal_task_queue_trim(iree_hal_task_queue_t* queue);

// Sets the active profile recorder snapshot used by future queue submissions.
//
// The owning device must call this only under the HAL profiling scope contract:
// no concurrent submissions and no in-flight queue work while profiling is
// enabled or disabled. Existing operations therefore never observe the recorder
// pointer changing underneath them.
void iree_hal_task_queue_set_profile_recorder(
    iree_hal_task_queue_t* queue,
    iree_hal_task_profile_recorder_t* profile_recorder,
    iree_hal_task_profile_queue_scope_t profile_scope,
    iree_atomic_int64_t* submission_counter);

iree_status_t iree_hal_task_queue_submit_barrier(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_commands(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_task_submission_batch_t* batches);

iree_status_t iree_hal_task_queue_submit_host_call(
    iree_hal_task_queue_t* queue, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags);

// Captures and submits one mapping vector with one dependency boundary.
iree_status_t iree_hal_task_queue_submit_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

// Schedules a ready native mapping process on the existing executor.
// The operation may complete and free itself before this call returns.
void iree_hal_task_queue_issue_map(iree_hal_task_queue_op_t* operation);

// Allocates one operation and begins its accepted-work scope. On success the
// caller fills its payload and passes it to submit_op_finish exactly once.
iree_status_t iree_hal_task_queue_submit_op_begin(
    iree_hal_task_queue_t* queue, iree_hal_task_queue_op_type_t type,
    const iree_hal_semaphore_list_t* signals,
    iree_hal_task_queue_op_t** out_operation);

// Retains resources and registers waits. Consumes the operation on all paths.
iree_status_t iree_hal_task_queue_submit_op_finish(
    iree_hal_task_queue_op_t* operation, iree_hal_semaphore_list_t waits,
    iree_host_size_t resource_count, iree_hal_resource_t* const* resources);

// Returns whether new queue progress should complete with cancellation.
bool iree_hal_task_queue_is_shutting_down(iree_hal_task_queue_t* queue);

// Constructs the terminal error for work cancelled during queue shutdown.
iree_status_t iree_hal_task_queue_make_shutdown_status(void);

// Transfers one ready operation to the existing queue process.
void iree_hal_task_queue_schedule_ready(iree_hal_task_queue_op_t* operation);

// Discards rejected capture without publishing signals. Consumes the operation.
void iree_hal_task_queue_op_discard(iree_hal_task_queue_op_t* operation);

// Retires an operation and fails its signals on error, without advancing the
// queue frontier. Consumes the operation and status.
void iree_hal_task_queue_op_destroy(iree_hal_task_queue_op_t* operation,
                                    iree_status_t failure_status);

// Acquires and publishes allocation backing after user waits resolve. Success
// transfers ownership to completion; failure leaves cleanup with the caller.
iree_status_t iree_hal_task_queue_issue_alloca(
    iree_hal_task_queue_op_t* operation);

// Returns allocation ownership and completes its accepted queue operation.
void iree_hal_task_queue_issue_dealloca(iree_hal_task_queue_op_t* operation);

// Returns staged allocation reservations and destroys completed wait state.
void iree_hal_task_queue_alloca_cleanup(iree_hal_task_queue_op_t* operation);

// Marks a captured operation as requiring asynchronous dependency progress.
void iree_hal_task_queue_profile_force_software_defer(
    iree_hal_task_queue_op_t* operation);

// Records one allocation lifecycle event when memory profiling is enabled.
void iree_hal_task_queue_profile_record_memory_event(
    iree_hal_task_queue_op_t* operation,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result,
    const iree_hal_pool_reservation_t* reservations,
    uint64_t frontier_entry_count);

// Records the operation's transition from submitted to ready exactly once.
void iree_hal_task_queue_profile_record_queue_event(
    iree_hal_task_queue_op_t* operation, iree_time_t ready_host_time_ns);

// Records terminal failure before the operation became ready to execute.
void iree_hal_task_queue_profile_record_failed_before_ready(
    iree_hal_task_queue_op_t* operation);

// Begins the bounded host-execution interval when profiling is enabled.
void iree_hal_task_queue_profile_start_host_execution(
    iree_hal_task_queue_op_t* operation);

// Publishes terminal success and retires the operation's frontier and scope.
// Consumes the operation.
void iree_hal_task_queue_op_complete(iree_hal_task_queue_op_t* operation);

// Publishes terminal failure to the signals and frontier, then retires the
// operation's scope. Consumes both the operation and its non-OK status.
void iree_hal_task_queue_op_fail(iree_hal_task_queue_op_t* operation,
                                 iree_status_t status);

iree_status_t iree_hal_task_queue_submit_alloca(
    iree_hal_task_queue_t* queue, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t* const* transient_buffers,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_dealloca(
    iree_hal_task_queue_t* queue, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* transient_buffers,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_read(
    iree_hal_task_queue_t* queue, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_write(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_fill(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_copy(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_update(
    iree_hal_task_queue_t* queue, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

iree_status_t iree_hal_task_queue_submit_atomic_wait(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_wait_params_t params,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_atomic_store(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_store_params_t params,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_atomic_rmw(
    iree_hal_task_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_atomic_rmw_params_t params,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

iree_status_t iree_hal_task_queue_submit_dispatch(
    iree_hal_task_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags, iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    const iree_hal_queue_barriers_t* barriers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TASK_QUEUE_QUEUE_H_
