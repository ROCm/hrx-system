// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_

#include "iree/async/frontier.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/profile.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/queue_scope.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/block_pool.h"
#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/notification_ring.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/queue_upload_ring.h"
#include "iree/hal/pool.h"
#include "iree/hal/profile_schema.h"
#include "iree/hal/profile_sink.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_pending_op_t iree_hal_amdgpu_pending_op_t;
typedef struct iree_hal_amdgpu_pm4_ib_slot_t iree_hal_amdgpu_pm4_ib_slot_t;
typedef struct iree_hal_amdgpu_host_queue_command_buffer_scratch_t
    iree_hal_amdgpu_host_queue_command_buffer_scratch_t;
typedef struct iree_hal_amdgpu_profile_counter_sample_slot_t
    iree_hal_amdgpu_profile_counter_sample_slot_t;
typedef struct iree_hal_amdgpu_profile_counter_range_slot_t
    iree_hal_amdgpu_profile_counter_range_slot_t;
typedef struct iree_hal_amdgpu_profile_counter_session_t
    iree_hal_amdgpu_profile_counter_session_t;
typedef struct iree_hal_amdgpu_profile_trace_session_t
    iree_hal_amdgpu_profile_trace_session_t;
typedef struct iree_hal_amdgpu_profile_trace_slot_t
    iree_hal_amdgpu_profile_trace_slot_t;
typedef struct iree_hal_amdgpu_feedback_state_t
    iree_hal_amdgpu_feedback_state_t;
typedef struct iree_hal_amdgpu_staging_pool_t iree_hal_amdgpu_staging_pool_t;
typedef struct iree_hal_amdgpu_tsan_memory_policy_t
    iree_hal_amdgpu_tsan_memory_policy_t;
typedef struct iree_hal_amdgpu_transient_buffer_pool_t
    iree_hal_amdgpu_transient_buffer_pool_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;

// Queue-local reservation of dispatch profiling event records.
typedef struct iree_hal_amdgpu_profile_dispatch_event_reservation_t {
  // Logical ring position of the first reserved dispatch event.
  uint64_t first_event_position;
  // Number of reserved dispatch events.
  uint32_t event_count;
  // Reserved padding.
  uint32_t reserved0;
} iree_hal_amdgpu_profile_dispatch_event_reservation_t;

// Queue-local reservation of device-timestamped queue operation records.
typedef struct iree_hal_amdgpu_profile_queue_device_event_reservation_t {
  // Logical ring position of the first reserved queue device event.
  uint64_t first_event_position;
  // Number of reserved queue device events.
  uint32_t event_count;
  // Reserved padding.
  uint32_t reserved0;
} iree_hal_amdgpu_profile_queue_device_event_reservation_t;

typedef struct iree_hal_amdgpu_host_queue_post_drain_action_t
    iree_hal_amdgpu_host_queue_post_drain_action_t;

// Memory policy used for queue profiling records.
typedef struct iree_hal_amdgpu_host_queue_profiling_memory_t {
  // HSA memory pool used for raw completion-signal timestamp records.
  hsa_amd_memory_pool_t signal_memory_pool;
  // HSA memory pool used for host-readable, device-writable event records.
  hsa_amd_memory_pool_t event_memory_pool;
  // Agents granted explicit access to event records after allocation.
  const hsa_agent_t* event_access_agents;
  // Number of entries in |event_access_agents|.
  iree_host_size_t event_access_agent_count;
  // Publication required after CPU writes event metadata.
  iree_hal_amdgpu_kernarg_ring_publication_t event_host_write_publication;
} iree_hal_amdgpu_host_queue_profiling_memory_t;

// Callback run after notification-ring drain has published completed entries
// and reclaimed queue-owned ring state.
typedef void(IREE_API_PTR* iree_hal_amdgpu_host_queue_post_drain_fn_t)(
    void* user_data);

// Intrusive continuation queued by pre-signal reclaim actions.
//
// Pre-signal actions run while notification-ring drain is still publishing a
// completion entry. Work that may submit additional AQL packets must instead
// queue one of these actions so it runs after drain has released all completed
// notification/kernarg state.
struct iree_hal_amdgpu_host_queue_post_drain_action_t {
  // Next action in the queue-owned pending list.
  iree_hal_amdgpu_host_queue_post_drain_action_t* next;
  // Callback invoked exactly once after the action is dequeued.
  iree_hal_amdgpu_host_queue_post_drain_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
};

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

// Maximum number of frontier entries the queue's accumulated frontier can
// track. Each entry is one (axis, epoch) pair representing a causal
// dependency on another queue or device. 64 entries covers rack-scale
// systems (8 machines x 8 GPUs x 4 queues = 256 theoretical axes, but a
// single queue only waits on its collective peers — typically 8-16 axes).
// Overflow is handled gracefully (frontier merge returns false, wait elision
// degrades but correctness is preserved).
//
// Transition snapshots serialize this frontier verbatim, so the queue's
// frontier capacity is tied to the notification ring's snapshot entry limit.
#define IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY \
  IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT

IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_amdgpu_host_queue_frontier_t,
                               IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY);

// Maximum number of direct buffer bindings accepted by queue_dispatch.
//
// Command buffers support large binding tables through their own lifetime
// tracking path. Direct dispatch keeps the submission path bounded and uses
// queue-local scratch storage under submission_mutex.
#define IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_BINDING_CAPACITY 256u

// Maximum number of operation resources retained by one direct queue_dispatch:
// the executable, one optional indirect-parameter buffer, plus one resource per
// direct buffer binding.
#define IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_RESOURCE_CAPACITY \
  (2u + IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_BINDING_CAPACITY)

// Host-driven queue with per-queue epoch signal and wait-backed
// notification ring. Embeds iree_hal_queue_t at offset zero.
//
// The epoch signal (owned by the notification ring) is a single hsa_signal_t
// set as completion_signal on each submission's last AQL packet. The CP
// decrements it by 1 on completion. The notification ring maps epochs to
// semaphore signals that serialized completion drain publishes when the epoch
// advances.
//
// Every operation targets this exact queue and shares its submission, progress,
// failure, notification, and reclaim state.
typedef struct iree_hal_amdgpu_host_queue_t {
  // Base HAL queue resource. Must be at offset zero.
  iree_hal_queue_t base;

  // HSA API handle for queue operations. Not retained.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Logical device owning this queue. Not retained.
  iree_hal_device_t* logical_device;
  // Stable opaque hostcall device address for this queue, or NULL.
  void* hostcall_buffer;
  // Proactor used to arm async semaphore/timepoint waits. Borrowed from the
  // logical device.
  iree_async_proactor_t* proactor;
  // Shared frontier tracker for this queue's axis. Borrowed from the logical
  // device.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Allocator used for host-side queue resources.
  iree_allocator_t host_allocator;

  // Sticky terminal failure for this queue. Non-zero means the queue will never
  // make progress again: an unrecoverable GPU error, or an HSA wait that
  // returned a result the queue cannot act on.
  //
  // Written through iree_hal_amdgpu_host_queue_record_failure by any of the HSA
  // queue error callback, the process-wide HSA system event callback, and this
  // driver's own wait paths, so writers are not restricted to HSA runtime
  // threads. First-error-wins CAS; acquire-loaded by the completion drain path
  // to fail pending semaphores instead of signaling them.
  //
  // Owned by the queue. Deinitialization takes the status out of the slot in
  // the same step that frees it, so a non-zero slot always names a live status
  // and every reader may clone from it without further coordination.
  iree_atomic_intptr_t error_status;

  // Hardware AQL queue created via hsa_queue_create. Owned by this queue.
  hsa_queue_t* hardware_queue;

  // Cached AQL ring state for zero-indirection packet submission.
  // Initialized from hardware_queue at init time.
  iree_hal_amdgpu_aql_ring_t aql_ring;

  // Per-queue kernarg bump allocator backed by HSA kernarg-init memory.
  iree_hal_amdgpu_kernarg_ring_t kernarg_ring;

  // Per-queue upload ring for device-visible control records.
  // Submission paths reserve from this only when they have queue-ordered
  // metadata such as device-side fixup inputs.
  iree_hal_amdgpu_queue_upload_ring_t queue_upload_ring;

  // Optional per-AQL-slot PM4 IB buffer used by PM4-backed wait, transfer, and
  // profiling snippets. This is not an independent scheduling ring: each slot
  // is indexed by the matching AQL packet id and inherits the AQL ring's
  // lifetime/backpressure.
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slots;

  // Queue-owned TSAN state used by instrumented command buffers.
  struct {
    // Base pointer returned by HSA for the combined TSAN allocation.
    IREE_AMDGPU_DEVICE_PTR void* allocation_base;
    // Total byte length of |allocation_base|.
    iree_device_size_t allocation_size;
    // Host-owned mirror of the device queue state header.
    iree_hal_amdgpu_tsan_queue_state_t host_state;
    // Device-visible queue state header inside |allocation_base|.
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_tsan_queue_state_t* queue_state;
    // Queue-local dispatch shadow storage inside |allocation_base|.
    IREE_AMDGPU_DEVICE_PTR void* shadow_base;
    // Byte length of |shadow_base|.
    iree_device_size_t shadow_size;
  } tsan;

  // Epoch-driven notification ring mapping submission completions to
  // semaphore signals. Serialized completion drain consumes this ring.
  iree_hal_amdgpu_notification_ring_t notification_ring;

  // Completion-thread state for queue epoch drain and teardown/error wakeups.
  struct {
    // Host thread blocked on the queue epoch signal and draining completed
    // notification-ring entries.
    iree_thread_t* thread;
    // HSA signal used to wake the completion thread during teardown or after
    // an unrecoverable HSA queue error. Value 0 means the thread should
    // continue waiting for completions; any other value requests exit after a
    // final drain.
    hsa_signal_t stop_signal;
  } completion;

  //--- Submission pipeline state -------------------------------------------//
  //
  // Threading model: the queue has three execution contexts.
  //
  //   Submission (any thread, serialized by submission_mutex):
  //     AQL slot reservation, packet fill, notification ring push, frontier
  //     snapshot push, queue frontier mutation, last_signal update, kernarg
  //     allocation. Multiple threads may submit to the same queue; the mutex
  //     serializes them. Independent queues do not synchronize.
  //
  //   Completion drain (single serialized consumer):
  //     Waits on the notification ring epoch signal with
  //     hsa_amd_signal_wait_any from the queue-owned completion thread, or is
  //     entered by a direct host waiter after it independently observes a
  //     producer epoch. Drains completed entries, checks error_status, reclaims
  //     kernargs, and advances the queue frontier tracker. Reads the
  //     notification ring and the atomic error_status under
  //     completion_drain_mutex, preserving the ring's single-consumer
  //     representation even though there are multiple possible callers. The
  //     queue-owned completion thread also runs post-drain continuations after
  //     the serialized drain; direct host waiters do not. Never writes to
  //     submission-path fields.
  //
  //   Terminal failure (any thread that can prove the queue is initialized):
  //     Writes error_status via atomic CAS and signals completion.stop_signal
  //     so the completion thread wakes, closes admission and fails outstanding
  //     notifications. Reached from the queue's own HSA error callback, from
  //     the process-wide HSA system event callback, and from this driver's own
  //     wait paths, so the writer is not always an HSA runtime thread.
  //
  // Wait-resolution fast-path contract:
  //   - Same-queue signal-before-wait is elided directly from the semaphore's
  //     last_signal cache when the cached producer axis matches queue->axis
  //     under this strategy's current all-BARRIER AQL policy.
  //   - Local cross-queue waits use one producer epoch barrier when the
  //     semaphore cache marks that producer frontier as exact and this queue's
  //     frontier does not already dominate that producer axis/epoch.
  //   - The full semaphore-frontier mutex/copy path is reserved for unresolved
  //     waits whose cached producer frontier is not exact (for example, true
  //     multi-producer fan-in) or for conservative fallback after
  //     cache/frontier overflow.
  //   - Wait-before-signal, remote/non-queue-domain axes, and queue teardown
  //     use software deferral.
  //
  // Signal-commit fast-path contract:
  //   - Each successful AQL submission advances this queue's epoch, merges this
  //     queue axis into queue->frontier, reserves one queue-private reclaim
  //     slot, pushes one notification-ring entry per user-visible signal
  //     semaphore, and records enough signal metadata for completion drain.
  //     Zero-signal submissions still consume one queue epoch and reclaim slot
  //     so kernel resources retire through the same mechanism.
  //   - Public/multi-producer semaphores publish queue->frontier under the
  //     semaphore mutex so later waits can prove transitive dependencies.
  //   - Private single-producer AMDGPU stream semaphores skip that mutex/copy
  //     path and publish only the producer queue axis/epoch/value to the
  //     seqlock-protected last_signal cache. Waiting on that producer epoch is
  //     sufficient because all transitive waits are encoded before the
  //     producer queue epoch can complete.

  // Queue-local locks. Keep the 4-byte slim mutexes packed before pointer-sized
  // continuation state.
  struct {
    // Serializes the submission path. All queue operations (dispatch, copy,
    // fill, execute, etc.) acquire this before touching submission state and
    // release after signal commit. The proactor thread does not acquire this.
    iree_slim_mutex_t submission_mutex;
    // Serializes notification-ring drain between the completion thread and
    // direct host waiters observing a producer epoch.
    iree_slim_mutex_t completion_drain_mutex;
    // Serializes the post-drain continuation list.
    iree_slim_mutex_t post_drain_mutex;
  } locks;

  // Post-drain continuation queue for work that cannot run while notification
  // drain is still publishing or reclaiming ring state.
  struct {
    // First queued post-drain continuation.
    iree_hal_amdgpu_host_queue_post_drain_action_t* head;
    // Tail pointer for appending post-drain continuations.
    iree_hal_amdgpu_host_queue_post_drain_action_t* tail;
  } post_drain;

  // Queue-local scratch used by queue_dispatch under submission_mutex.
  struct {
    // Operation resources copied into the notification reclaim entry.
    iree_hal_resource_t* operation_resources
        [IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_RESOURCE_CAPACITY];
    // Resolved device pointers written into final dispatch kernargs.
    uint64_t binding_ptrs
        [IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_BINDING_CAPACITY];
  } dispatch_scratch;

  // Lazily allocated queue_execute scratch storage. Kept out of the host queue
  // object so direct-dispatch hot state does not carry command-buffer sideband
  // arrays.
  iree_hal_amdgpu_host_queue_command_buffer_scratch_t* command_buffer_scratch;

  // Set under submission_mutex when the queue permanently closes admission,
  // either for teardown or after a fatal queue failure. Never cleared. Every
  // submission entry point rejects work with CANCELLED once it is set, and
  // deferred ops whose waits race to completion after this point are failed
  // with CANCELLED instead of issuing new AQL packets.
  //
  // Because notification epochs are published under submission_mutex, closing
  // admission under that mutex also establishes that no epoch published later
  // can escape the failure drain that follows.
  bool is_shutting_down;

  // Profiling data-family state for this queue. Mutated only by device
  // profiling begin/end while the profiling API's idle-device precondition is
  // held.
  struct {
    // True when ROCR should populate dispatch completion signal timestamps.
    uint32_t hsa_queue_timestamps_enabled : 1;
    // True when host-side queue operation events should be recorded.
    uint32_t queue_events_enabled : 1;
    // True when device-timestamped queue operation events should be recorded.
    uint32_t queue_device_events_enabled : 1;
    // True when selected dispatches may receive profile packet augmentation.
    uint32_t dispatch_profiling_enabled : 1;
    // Memory pools and publication policy for queue-local profiling storage.
    iree_hal_amdgpu_host_queue_profiling_memory_t memory;
    // Serializes profile event ring mutation and flush.
    iree_slim_mutex_t event_mutex;
    // Queue-owned raw completion-signal records paired with dispatch slots.
    iree_amd_signal_t* completion_signals;
    // Shared host-readable, device-writable allocation backing event rings.
    void* event_storage;
    // Device-visible dispatch event ring waiting for sink flush.
    struct {
      // Dispatch event record storage in the shared event allocation.
      iree_hal_amdgpu_profile_dispatch_event_t* values;
      // Power-of-two capacity of |values| in records.
      uint32_t capacity;
      // Capacity minus one, for mapping logical positions to physical slots.
      uint32_t mask;
      // Logical ring position of the next event to write to the sink.
      uint64_t read_position;
      // Logical ring position one past the last event ready to write.
      uint64_t ready_position;
      // Logical ring position one past the last reserved event.
      uint64_t write_position;
      // Next queue-local dispatch event id assigned during submission.
      uint64_t next_event_id;
    } dispatch_events;
    // Device-visible queue operation event ring waiting for sink flush.
    struct {
      // Queue device event record storage in the shared event allocation.
      iree_hal_amdgpu_profile_queue_device_event_t* values;
      // Power-of-two capacity of |values| in records.
      uint32_t capacity;
      // Capacity minus one, for mapping logical positions to physical slots.
      uint32_t mask;
      // Logical ring position of the next event to write to the sink.
      uint64_t read_position;
      // Logical ring position one past the last event ready to write.
      uint64_t ready_position;
      // Logical ring position one past the last reserved event.
      uint64_t write_position;
      // Next queue-local queue-device event id assigned during submission.
      uint64_t next_event_id;
    } queue_device_events;
    // Queue-local hardware counter profile resources.
    struct {
      // Borrowed hardware counter session active for this queue, or NULL.
      iree_hal_amdgpu_profile_counter_session_t* session;
      // Number of selected counter sets in |session|.
      uint32_t set_count;
      // Dispatch-attributed counter sample storage.
      struct {
        // Host-side slot table pairing dispatch event slots with aqlprofile
        // handles.
        iree_hal_amdgpu_profile_counter_sample_slot_t* slots;
      } dispatch_samples;
      // Queue-range counter sample storage.
      struct {
        // Host-side slot table pairing range banks with aqlprofile handles.
        iree_hal_amdgpu_profile_counter_range_slot_t* slots;
        // Host-readable, device-written start/end ticks for each range bank.
        uint64_t* ticks;
        // Byte length of |ticks|.
        iree_host_size_t tick_storage_size;
        // Bank currently capturing queue work.
        uint32_t active_bank;
        // Number of reusable range banks in |slots| and |ticks|.
        uint32_t bank_count;
        // True when a range bank has been started and must be stopped.
        bool is_active;
      } ranges;
    } counters;
    // Queue-local executable trace profile resources.
    struct {
      // Borrowed executable trace session active for this queue, or NULL.
      iree_hal_amdgpu_profile_trace_session_t* session;
      // Host-side slot table pairing dispatch event slots with ATT handles.
      iree_hal_amdgpu_profile_trace_slot_t* slots;
    } traces;
  } profiling;

  // False once this queue's accumulated frontier overflows while merging waited
  // axes. After that, the frontier remains a safe lower bound for resolving
  // this queue's own waits, but it is no longer a conservative summary that can
  // be published to public/multi-producer signal semaphores. Those signal
  // commits therefore clear last_signal, skip semaphore-frontier merges, and
  // stop pushing transition snapshots, forcing downstream not-yet-complete
  // waits onto the software path instead of under-barriering.
  bool can_publish_frontier;

  // This queue's axis in the causal graph. The device index is assigned by the
  // HAL device-group topology and the queue index is the flattened logical
  // queue ordinal, preserving uniqueness for both separate logical devices and
  // composite devices. Used in frontier entries and epoch signal lookups.
  // Immutable after initialization.
  iree_async_axis_t axis;

  // Device-side wait strategy selected once from the GPU ISA at initialization.
  iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy;

  // AMD vendor-packet capabilities selected from the GPU ISA.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;

  // Queue-local PM4 timestamp strategy initialized from the owning physical
  // device's ISA selection. Submission reads it under submission_mutex and
  // profiling enable reads it unlocked, so writes must precede both.
  iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy;

  // One-bit logical queue affinity routing device operations to this queue.
  iree_hal_queue_affinity_t queue_affinity;

  // Flattened logical queue ordinal in the owning HAL device.
  iree_host_size_t queue_ordinal;

  // Queue ordinal relative to |device_ordinal|.
  iree_host_size_t physical_queue_ordinal;

  // Logical-device epoch signal table for cross-queue barrier emission (tier 2
  // wait resolution). Maps flattened queue indices to hsa_signal_t values for
  // queues in this logical device. Used to look up peer queues' epoch signals
  // when emitting AQL barrier-value packets for multi-axis dependencies.
  //
  // Borrowed from the logical device and valid for the lifetime of the queue.
  // This queue's own epoch signal is registered at init and deregistered at
  // deinit. Read-only during normal operation.
  iree_hal_amdgpu_epoch_signal_table_t* epoch_table;

  // Last semaphore pushed to the notification ring and its epoch. Used to
  // detect semaphore transitions for frontier snapshot recording: when a
  // push targets a different semaphore than last_signal.semaphore, the
  // signal commit path writes a frontier snapshot at last_signal.epoch
  // before starting the new span.
  //
  // Protected by submission_mutex (submission-context-only).
  //
  // ABA safety: the semaphore pointer is only compared for identity (not
  // dereferenced) during transition detection. ABA can occur if a semaphore
  // is released and a new one is allocated at the same address between two
  // submissions. This is benign:
  //   - The old semaphore's notification entries must have been drained
  //     before release (notification ring lifetime contract), so no
  //     undrained entries for the old semaphore remain.
  //   - A missed transition causes the new semaphore's entries to be
  //     coalesced with a span that has no pending entries — the drain
  //     produces the correct signal for the new semaphore.
  //   - The frontier snapshot at the end of the coalesced span (when the
  //     next transition occurs, or the fallback frontier at drain end)
  //     captures the queue's accumulated frontier, which is an upper bound
  //     on the actual causal context. Over-attribution (conservative), never
  //     under-attribution (unsafe).
  struct {
    // Most recent semaphore pushed to the notification ring.
    iree_async_semaphore_t* semaphore;
    // Queue epoch associated with the most recent semaphore push.
    uint64_t epoch;
    // True when the current same-semaphore span requires a frontier snapshot
    // if the next signal targets a different semaphore.
    bool needs_frontier_snapshot;
    // Reserved padding for stable layout.
    uint8_t reserved[7];
  } last_signal;

  // Block pool for arena-allocating deferred operations. NUMA-pinned to the
  // physical device. Borrowed from the physical device; valid for the
  // lifetime of the queue.
  iree_arena_block_pool_t* block_pool;

  // Ordinal of this queue's physical device within the topology. Used by
  // device-specific executable metadata paths and queue-local helpers.
  iree_host_size_t device_ordinal;

  // Builtin blit kernel table for this queue's physical device. Borrowed from
  // the physical device and immutable for the queue's lifetime.
  const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context;

  // Borrowed logical-device feedback state drained during queue retirement.
  // NULL when feedback is disabled.
  iree_hal_amdgpu_feedback_state_t* feedback_state;

  // Borrowed default pool set for this queue's physical device.
  const iree_hal_pool_set_t* default_pool_set;

  // Borrowed TLSF default pool for this queue's physical device.
  iree_hal_pool_t* default_pool;

  // Borrowed transient wrapper pool for queue_alloca results.
  iree_hal_amdgpu_transient_buffer_pool_t* transient_buffer_pool;

  // Borrowed fixed-size staging pool used by queue_read/queue_write for
  // non-mappable file transfers.
  iree_hal_amdgpu_staging_pool_t* staging_pool;

  // Intrusive singly-linked list of pending (deferred) operations. Used for
  // cleanup on shutdown and GPU fault propagation. Operations add themselves
  // on deferral and remove themselves on issue/fail/cancel. Protected by
  // submission_mutex.
  iree_hal_amdgpu_pending_op_t* pending_head;

  // Accumulated frontier. Advances on each AQL submission: the queue's own
  // axis entry is set to the current epoch, and cross-queue wait dependencies
  // are merged in. Used for:
  //   - Queue-order wait elision (tier 1): queue->frontier dominates the wait
  //     semaphore's frontier → no additional barrier packet is needed.
  //   - Submission-time causal merge: merged into signal semaphores' frontiers
  //     at AQL submission time so same-queue and already-dominated cross-queue
  //     waits can resolve before GPU completion under the current all-barrier
  //     AQL queue policy.
  //   - Frontier snapshot recording: snapshotted to the notification ring's
  //     frontier byte ring at semaphore transitions.
  //
  // Fixed-capacity storage for the accumulated frontier.
  iree_hal_amdgpu_host_queue_frontier_t frontier;
} iree_hal_amdgpu_host_queue_t;

// Loads the queue-owned terminal error as an opaque raw value. Zero means the
// queue has not failed. A nonzero value remains owned by |queue| and must be
// cloned before crossing an ownership boundary.
static inline intptr_t iree_hal_amdgpu_host_queue_load_error_status_raw(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return iree_atomic_load(&queue->error_status, iree_memory_order_acquire);
}

// Returns true when the queue has recorded a terminal error.
static inline bool iree_hal_amdgpu_host_queue_has_error(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return iree_hal_amdgpu_host_queue_load_error_status_raw(queue) != 0;
}

// Returns an owned clone of the queue's terminal error, or OK when the queue
// has not failed.
static inline iree_status_t iree_hal_amdgpu_host_queue_clone_error_status(
    const iree_hal_amdgpu_host_queue_t* queue) {
  const intptr_t error_status =
      iree_hal_amdgpu_host_queue_load_error_status_raw(queue);
  return IREE_LIKELY(error_status == 0)
             ? iree_ok_status()
             : iree_status_clone((iree_status_t)error_status);
}

// Returns a pointer to the queue's accumulated frontier. The returned pointer
// is layout-compatible with iree_async_frontier_t and valid for all frontier
// APIs (compare, merge, etc.). Valid for the lifetime of the queue.
static inline iree_async_frontier_t* iree_hal_amdgpu_host_queue_frontier(
    iree_hal_amdgpu_host_queue_t* queue) {
  return iree_async_fixed_frontier_as_frontier(&queue->frontier);
}

// Returns a const pointer to the queue's accumulated frontier.
static inline const iree_async_frontier_t*
iree_hal_amdgpu_host_queue_const_frontier(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return iree_async_fixed_frontier_as_const_frontier(&queue->frontier);
}

// Submits a buffer-copy payload through the queue with the requested queue
// profiling event type.
iree_status_t iree_hal_amdgpu_host_queue_copy_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type);

// Enqueues a driver-owned host action ordered after |wait_semaphore_list|.
// |action| uses the reclaim-action status ownership contract: OK means the
// ordering barrier completed, while non-OK is a borrowed queue/device failure
// status that must be cloned before any async propagation.
// |operation_resources| are retained before this returns and released after the
// action has executed or failed; callers keep ownership of their references.
iree_status_t iree_hal_amdgpu_host_queue_enqueue_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count);

// Enqueues |action| to run after the current or next notification-ring drain
// has fully published completed entries. The action storage must remain valid
// until |action->fn| is invoked.
void iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_post_drain_action_t* action,
    iree_hal_amdgpu_host_queue_post_drain_fn_t fn, void* user_data);

// Initializes a host queue in caller-provided memory.
// The caller must allocate at least sizeof(iree_hal_amdgpu_host_queue_t).
//
// Creates an HSA hardware queue on |gpu_agent|, initializes the AQL ring from
// it, allocates a kernarg ring from |kernarg_memory|, creates the epoch signal
// and notification ring, and starts the completion thread.
//
// |axis| is this queue's identity in the causal graph. Its device index is the
// topology-assigned HAL logical device index and its queue index is the
// flattened logical queue ordinal within that device.
//
// |epoch_table| is the logical-device epoch signal table for cross-queue
// barrier emission. This queue registers its epoch signal in the table at init
// and deregisters at deinit. The table must outlive the queue.
//
// |feedback_state| is borrowed from the logical device. When enabled, queue
// retirement drains the physical-device feedback channel before releasing
// queue-owned resources or publishing signal semaphores.
//
// |hostcall_buffer| is the optional stable opaque device address copied into
// every implicit-argument suffix emitted by this queue.
//
// |completion_thread_affinity| pins the completion thread near the host CPU
// agent associated with the GPU. The platform may ignore the request, but on
// NUMA-aware systems this keeps blocked-wait wakeups and notification-ring
// drains close to the GPU's nearest CPU node.
//
// |aql_queue_capacity| is the power-of-two hardware AQL queue size in packets.
// |notification_capacity| is the power-of-two notification ring size.
// |kernarg_capacity_in_blocks| is the power-of-two kernarg ring size in
// 64-byte blocks, at least 2x |aql_queue_capacity| to cover one tail-padding
// gap at wrap. Submission admission proves space in both the AQL and kernarg
// rings before publishing packets.
// |upload_capacity| is the byte capacity of the device-visible control upload
// ring used for queue-ordered submission metadata. Zero disables the optional
// upload ring; non-zero values must be powers of two.
//
// |aql_queue_execution_mode| identifies whether the GPU or a ROCr host worker
// consumes AQL packets and selects the matching doorbell notification path.
//
// |vendor_packet_capabilities| describes the AQL/PM4 vendor-packet support
// selected from the physical device ISA. Queues allocate dynamic PM4 IB slots
// when AQL_PM4_IB is available so BARRIER_VALUE-based CDNA queues can still use
// PM4 snippets for profiling or tiny operations.
//
// |pm4_timestamp_strategy| describes the PM4 packet sequence used for
// queue-device timestamp records. NONE disables profiling paths that need
// queue-local timestamp ranges.
//
// |profiling_memory| provides memory for queue-local profiling event rings and
// raw iree_amd_signal_t records. Raw signals may use device-only memory because
// packets only use them as CP-written timestamp targets and never as host HSA
// waits or interrupts. They are still initialized into the armed user-signal
// ABI shape before being placed in packet completion_signal fields. Profile
// event records must be host-readable because the profiling sink serializes
// them on the CPU.
iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_device_t* logical_device,
    void* hostcall_buffer, iree_async_proactor_t* proactor,
    hsa_agent_t gpu_agent,
    const iree_hal_amdgpu_kernarg_ring_memory_t* kernarg_memory,
    hsa_amd_memory_pool_t pm4_ib_pool,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t queue_ordinal,
    iree_host_size_t physical_queue_ordinal,
    iree_thread_affinity_t completion_thread_affinity,
    iree_hal_amdgpu_aql_queue_execution_mode_t aql_queue_execution_mode,
    iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_hal_amdgpu_feedback_state_t* feedback_state,
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_host_queue_profiling_memory_t profiling_memory,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    const iree_hal_pool_set_t* default_pool_set, iree_hal_pool_t* default_pool,
    iree_hal_amdgpu_transient_buffer_pool_t* transient_buffer_pool,
    iree_hal_amdgpu_staging_pool_t* staging_pool,
    iree_host_size_t device_ordinal, uint32_t aql_queue_capacity,
    uint32_t notification_capacity, uint32_t kernarg_capacity_in_blocks,
    uint32_t upload_capacity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_queue_t* out_queue);

// Trims transient resources retained by |queue|.
void iree_hal_amdgpu_host_queue_trim(iree_hal_amdgpu_host_queue_t* queue);

// Enqueues a buffer fill on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

// Enqueues a host-to-buffer update on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

// Enqueues a buffer copy on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

// Enqueues a file-to-buffer read on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_submit_read(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Enqueues a buffer-to-file write on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_submit_write(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// Enqueues a host callback on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);

// Enqueues a direct executable dispatch on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

// Enqueues an atomic wait on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_atomic_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params);

// Enqueues an atomic store on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_atomic_store(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params);

// Enqueues an atomic read-modify-write operation on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_atomic_rmw(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params);

// Enqueues a device timestamp write on |queue|.
iree_status_t iree_hal_amdgpu_host_queue_timestamp(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags);

// Initializes queue-owned TSAN state.
//
// This is called after queue creation when logical TSAN is enabled. The queue
// owns the HSA allocation and releases it during queue deinitialization.
iree_status_t iree_hal_amdgpu_host_queue_initialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_tsan_memory_policy_t* memory_policy,
    iree_host_size_t queue_ordinal, iree_host_size_t physical_queue_ordinal,
    iree_device_size_t workgroup_shadow_stride,
    iree_device_size_t dispatch_shadow_stride, uint32_t workgroup_capacity,
    uint32_t shadow_entry_size, uint32_t memory_granule_shift,
    uint32_t shadow_slot_count);

// Deinitializes queue-owned TSAN state if present.
void iree_hal_amdgpu_host_queue_deinitialize_tsan_state(
    iree_hal_amdgpu_host_queue_t* queue);

// Begins queue teardown by permanently closing submission admission.
//
// After this returns every submission entry point rejects work and no new
// notification epoch can be published. Deferred operations already linked on
// the queue are not settled here.
//
// Callers tearing down several queues should call this on all of them before
// waiting on any so teardown latency is not serialized across queues.
void iree_hal_amdgpu_host_queue_begin_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Waits until no queue work remains that can depend on GPU progress: either
// hardware retires the last submitted epoch or the queue records a failure.
// Requires that admission has already been closed.
//
// Returns immediately when the queue already failed or never submitted
// anything. A failure recorded while this is waiting wakes it through the queue
// stop signal, which is why external failure delivery is a precondition of this
// call rather than something the caller may retire first.
//
// This wait is unbounded and a GPU that can no longer advance the submitted
// epoch has only the recorded failure to release it. Delivery of that failure
// is not guaranteed: the HSA runtime delivers a fatal event at most once per
// runtime instance and delivers none at all when its interrupt path is off
// (see system_event.h), so a queue stranded by a second fault in one runtime
// instance - or by any fault at all where delivery is disabled - leaves this
// call blocked for the life of the process.
void iree_hal_amdgpu_host_queue_wait_idle_before_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Deinitializes the queue. Destroys all owned resources and stops the
// completion thread.
//
// All in-flight work must have completed or failed: the caller must have run
// begin_deinitialize and wait_idle_before_deinitialize first, must ensure no
// concurrent access to the queue, and must have retired any external failure
// delivery that can still reach it.
void iree_hal_amdgpu_host_queue_finish_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Runs the full begin/wait/finish teardown sequence for one queue.
// Callers tearing down several queues should drive the phases themselves.
void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue);

// Records a permanent terminal failure on the queue from an unrecoverable
// asynchronous error and wakes the completion service so all submitted and
// deferred operations fail promptly. Consumes |status|; the first failure wins
// and later ones are freed.
//
// The completion thread then runs the queue's terminal transition and exits,
// after which the queue rejects all further submissions and never accepts work
// again. The only remaining valid operation on the queue is deinitialization.
//
// Callable from any thread, including an HSA runtime callback thread, as long
// as the caller holds a guarantee the queue is still initialized: the whole
// call is a first-error-wins compare-exchange on the failure slot, a free of
// the status that loses, and a store to the queue's stop signal when the queue
// has one. That store is the only HSA entry point it reaches, and the signal
// behind it is what the guarantee has to cover.
void iree_hal_amdgpu_host_queue_record_failure(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_t status);

// Populates |out_scope| with immutable queue identity and AQL ring facts.
void iree_hal_amdgpu_host_queue_query_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_scope_t* out_scope);

// Drains completed notification entries, retires queue-owned resources, runs
// post-drain continuations, and advances the queue frontier tracker.
//
// The queue completion thread calls this after epoch-signal wakeups. Internal
// serialization keeps notification-ring consumption single-reader.
iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions(
    iree_hal_amdgpu_host_queue_t* queue);

// Drains completed notification entries and retires queue-owned resources after
// a direct host waiter has independently observed a producer epoch.
//
// This intentionally does not run post-drain continuations. Those continuations
// may submit more AQL, resume command-buffer replay, or drive capacity retries;
// they belong on the queue-owned completion service thread where device/NUMA
// affinity policy can be controlled.
iree_host_size_t iree_hal_amdgpu_host_queue_drain_completions_for_waiter(
    iree_hal_amdgpu_host_queue_t* queue);

// Waits until a queue-owned setup submission reaches |epoch|. Setup
// submissions must not carry user-visible post-drain work; this direct waiter
// drains only the completed notification entries it observes.
iree_status_t iree_hal_amdgpu_host_queue_wait_for_setup_epoch(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t epoch);

// Enables or disables HSA dispatch timestamp population for this queue.
//
// This toggles the ROCR queue profiler bit. It is a cold profiling-session
// operation and must only be called while the device is idle, matching the HAL
// profiling API contract.
iree_status_t iree_hal_amdgpu_host_queue_set_hsa_profiling_enabled(
    iree_hal_amdgpu_host_queue_t* queue, bool enabled);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_H_
