// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for io_uring proactor implementation.
//
// This header exposes the proactor struct and internal helpers for use by
// io_uring-specific modules (socket.c, etc.). External users should include
// api.h instead.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_
#define IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_

#include "iree/async/platform/io_uring/api.h"
#include "iree/async/platform/io_uring/event_source.h"
#include "iree/async/platform/io_uring/sparse_table.h"
#include "iree/async/platform/io_uring/uring.h"
#include "iree/async/platform/linux/signal.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/semaphore_wait.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/async/util/signal.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_wait_operation_t
    iree_async_semaphore_wait_operation_t;

//===----------------------------------------------------------------------===//
// Proactor implementation struct
//===----------------------------------------------------------------------===//

// Lifecycle states for the singleton fixed-buffer table on pre-5.19 kernels.
typedef enum iree_async_io_uring_legacy_buffer_table_state_e {
  // No fixed-buffer table is registered or being changed.
  IREE_ASYNC_IO_URING_LEGACY_BUFFER_TABLE_STATE_FREE = 0,

  // A task has claimed the table and is registering fixed buffers.
  IREE_ASYNC_IO_URING_LEGACY_BUFFER_TABLE_STATE_REGISTERING = 1,

  // A fixed-buffer table is registered with the kernel.
  IREE_ASYNC_IO_URING_LEGACY_BUFFER_TABLE_STATE_ACTIVE = 2,

  // The active fixed-buffer table is being unregistered.
  IREE_ASYNC_IO_URING_LEGACY_BUFFER_TABLE_STATE_UNREGISTERING = 3,
} iree_async_io_uring_legacy_buffer_table_state_t;

// io_uring-specific proactor state.
typedef struct iree_async_proactor_io_uring_t {
  // Base proactor (must be first for safe casting).
  iree_async_proactor_t base;

  // io_uring ring buffer wrapper.
  iree_io_uring_ring_t ring;

  // Eventfd for cross-thread wake().
  int wake_eventfd;

  // Set when wake poll is armed to avoid duplicate submissions.
  bool wake_poll_armed;

  // Poll ownership and submission dispatch. Native cancellation requires the
  // permanent owner identity even outside the inline submission window.
  struct {
    // TID of the lifetime poll owner, or zero before the first poll. Only this
    // task may enter the ring, including cold-path cancellation submission.
    iree_atomic_int32_t owner_tid;

    // Owner TID during CQE processing and its drain loop, zero otherwise.
    // submit() flushes inline on this task or skips waking from other tasks.
    // Cleared before the final software drain so callbacks in that drain wake
    // the next poll if they enqueue more software work.
    // Cross-thread submitters publish work with an RMW before suppressing a
    // wake; the owner's exchange to zero acquires it before the final drain.
    iree_atomic_int32_t dispatch_tid;
  } polling;

  // Detected kernel capabilities before applying caller policy. Internal
  // registration mechanisms remain usable when optional behavior is disabled.
  iree_async_proactor_capabilities_t kernel_capabilities;

  // Enabled capabilities after applying the caller's allowed mask.
  iree_async_proactor_capabilities_t capabilities;

  // True when a valid provided-buffer ring registration succeeds. Independent
  // of opcode/version probes and the caller's allowed capability mask.
  bool supports_buffer_ring;

  // Preferred PBUF group ID for the next slab registration. The kernel is the
  // source of truth for live IDs; collisions scan for the next free ID.
  iree_atomic_int32_t next_buffer_group_id;

  // Sparse buffer table for dynamic buffer registration (kernel 5.19+).
  // NULL on pre-5.19 kernels where the legacy singleton
  // IORING_REGISTER_BUFFERS path is used instead.
  iree_io_uring_sparse_table_t* buffer_table;

  // One iree_async_io_uring_legacy_buffer_table_state_t value.
  iree_atomic_int32_t legacy_buffer_table_state;

  // Cross-proactor messaging state.
  // The pool is used by the fallback path (pre-5.18 kernels without MSG_RING).
  // MSG_RING delivers messages entirely through the kernel and does not use
  // the pool. The callback is invoked for both paths during poll().
  iree_async_message_pool_t message_pool;
  iree_async_proactor_message_callback_t message_callback;

  // Sequence emulator for IREE_ASYNC_OPERATION_TYPE_SEQUENCE operations.
  // Drives step-by-step execution when step_fn is set. When step_fn is NULL,
  // the LINK path is used instead (no emulator involvement).
  iree_async_sequence_emulator_t sequence_emulator;

  // MPSC queue of software work owned by the poll thread. Most entries carry
  // completed software operations awaiting callback delivery. SEQUENCE entries
  // carry admitted operations whose first step must start on the poll owner.
  //
  // Operations use base.next (offset 0) as the slist entry and
  // base.pending_status to carry owned completion status through the queue.
  iree_atomic_slist_t pending_software_operations;

  // Coalesced notification-local admission, readiness, and cancellation work.
  // Entries exist only while accepted consumers own the notification.
  iree_atomic_slist_t pending_notifications;

  // MPSC queue of semaphore wait operations ready to complete.
  // Timepoint callbacks push trackers here, poll() drains and completes them.
  iree_atomic_slist_t pending_semaphore_waits;

  // Serializes wait operation association with cancellation and completion.
  iree_async_semaphore_wait_context_t semaphore_wait_context;

  // Linked list of registered event sources. Proactor-owned.
  // Uses doubly-linked list for O(1) removal during unregister.
  iree_async_event_source_t* event_sources;

  // Linked list of registered relays. Proactor-owned.
  // Uses doubly-linked list for O(1) removal during unregister.
  struct iree_async_relay_t* relays;

  // Signal handling state (lazy-initialized on first subscribe).
  struct {
    // Whether signal handling has been initialized (signalfd poll registered).
    bool initialized;

    // Linux signalfd state (shared with potential future epoll backend).
    iree_async_linux_signal_state_t linux_state;

    // Dispatch state for deferred unsubscribe during callback iteration.
    iree_async_signal_dispatch_state_t dispatch_state;

    // Per-signal subscription lists. Each is a doubly-linked list.
    iree_async_signal_subscription_t* subscriptions[IREE_ASYNC_SIGNAL_COUNT];

    // Per-signal subscriber counts. Used to track when to add/remove signals
    // from the signalfd mask (only block signals that have subscribers).
    int subscriber_count[IREE_ASYNC_SIGNAL_COUNT];

    // Event source for monitoring the signalfd.
    iree_async_event_source_t* event_source;
  } signal;
} iree_async_proactor_io_uring_t;

// Upcast from base proactor to io_uring implementation.
static inline iree_async_proactor_io_uring_t* iree_async_proactor_io_uring_cast(
    iree_async_proactor_t* proactor) {
  return (iree_async_proactor_io_uring_t*)proactor;
}

//===----------------------------------------------------------------------===//
// Internal user data tagging
//===----------------------------------------------------------------------===//

// Internal operations use tagged user_data values to distinguish them from
// caller operations (which use the operation pointer directly).
//
// Encoding (LA57-safe, supports 5-level paging):
//   Bit 63:     INTERNAL_MARKER (1 = internal, 0 = user operation pointer)
//   Bits 56-62: Tag identifying the internal operation type (7 bits = 128 tags)
//   Bits 0-55:  Payload (56 bits - covers all LA57 userspace addresses)
//
// This encoding is safe for 5-level paging (LA57) because userspace addresses
// on x86-64 always have bit 56 = 0 (bit 56 distinguishes user/kernel space).
// The payload is stored unshifted, so pointers are preserved exactly.
//
// All internal tags are mutually exclusive and checked with equality.
#define IREE_IO_URING_INTERNAL_MARKER ((uint64_t)1 << 63)
#define IREE_IO_URING_INTERNAL_TAG_SHIFT 56
#define IREE_IO_URING_INTERNAL_TAG_MASK ((uint64_t)0x7F)
#define IREE_IO_URING_INTERNAL_PAYLOAD_MASK ((uint64_t)0x00FFFFFFFFFFFFFFULL)

// Internal operation tags (mutually exclusive, 7 bits = 0-127).
typedef enum iree_io_uring_internal_tag_e {
  IREE_IO_URING_TAG_WAKE = 1,             // Wake eventfd poll completion.
  IREE_IO_URING_TAG_CANCEL = 2,           // Async cancel completion.
  IREE_IO_URING_TAG_FENCE_IMPORT = 3,     // Fence import poll completion.
  IREE_IO_URING_TAG_MESSAGE_RECEIVE = 4,  // MSG_RING from another proactor.
  IREE_IO_URING_TAG_MESSAGE_SOURCE =
      5,  // MSG_RING source completion (no user callback).
  IREE_IO_URING_TAG_EVENT_SOURCE = 6,  // Event source multishot poll.
  IREE_IO_URING_TAG_RELAY = 7,         // Relay source completion.
  IREE_IO_URING_TAG_SIGNAL = 8,        // Signal fd multishot poll.
  // Linked POLL_ADD head for EVENT_WAIT.
  // The POLL_ADD CQE is always ignored; the linked READ CQE handles
  // resource release and user callback dispatch for both success and failure.
  IREE_IO_URING_TAG_LINKED_POLL = 9,
  // Event-source cancellation receipt, independent of the final poll CQE.
  IREE_IO_URING_TAG_EVENT_SOURCE_CANCEL = 10,
  // Primitive relay cancellation key retirement, independent of its poll.
  IREE_IO_URING_TAG_RELAY_CANCEL = 11,
} iree_io_uring_internal_tag_t;

// Helpers for encoding/decoding internal user_data.
// Payload is stored in low 56 bits (unshifted), tag in bits 56-62.
#define iree_io_uring_internal_encode(tag, payload)        \
  (IREE_IO_URING_INTERNAL_MARKER |                         \
   ((uint64_t)(tag) << IREE_IO_URING_INTERNAL_TAG_SHIFT) | \
   ((uint64_t)(uintptr_t)(payload) & IREE_IO_URING_INTERNAL_PAYLOAD_MASK))
#define iree_io_uring_internal_tag(user_data)          \
  (((user_data) >> IREE_IO_URING_INTERNAL_TAG_SHIFT) & \
   IREE_IO_URING_INTERNAL_TAG_MASK)
#define iree_io_uring_internal_payload(user_data) \
  ((user_data) & IREE_IO_URING_INTERNAL_PAYLOAD_MASK)

#define iree_io_uring_is_internal_cqe(cqe) \
  iree_any_bit_set((cqe)->user_data, IREE_IO_URING_INTERNAL_MARKER)

//===----------------------------------------------------------------------===//
// Internal tracker types
//===----------------------------------------------------------------------===//

// Tracks a pending fence import (POLL_ADD on an external fd).
// Heap-allocated per import_fence call, freed when the CQE fires.
typedef struct iree_async_io_uring_fence_import_tracker_t {
  iree_async_semaphore_t* semaphore;  // Retained.
  // Value to signal semaphore to.
  uint64_t signal_value;
  // Owned fd; closed on completion.
  int fence_fd;
  iree_allocator_t allocator;  // For freeing this tracker.
} iree_async_io_uring_fence_import_tracker_t;

// Tracks a pending fence export (semaphore timepoint -> eventfd write).
// Heap-allocated per export_fence call, freed when the timepoint callback
// fires.
typedef struct iree_async_io_uring_fence_export_tracker_t {
  // Embedded; owned by semaphore until callback.
  iree_async_semaphore_timepoint_t timepoint;
  iree_async_semaphore_t* semaphore;  // Retained.
  // Written to on signal (caller-owned, not closed).
  int eventfd;
  iree_allocator_t allocator;  // For freeing this tracker.
} iree_async_io_uring_fence_export_tracker_t;

//===----------------------------------------------------------------------===//
// Internal APIs (shared across proactor implementation files)
//===----------------------------------------------------------------------===//

// Vtable for same-backend validation in submit.
extern const iree_async_proactor_vtable_t iree_async_proactor_io_uring_vtable;

// Destruction joins borrowed native observers before closing the ring.
void iree_async_proactor_io_uring_destroy(iree_async_proactor_t* base_proactor);

// Poll-owned native progress shared with terminal observer retirement.
iree_status_t iree_async_proactor_io_uring_submit_pending_event_monitors(
    iree_async_proactor_io_uring_t* proactor);

// Dispatches one owned CQE. The caller advances its CQ slot after dispatch.
iree_host_size_t iree_async_proactor_io_uring_process_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_poll_status);

// Cancellation vtable implementation. Owner callbacks can submit a full SQ to
// admit cancellation without allocating or waiting for operation completion.
iree_status_t iree_async_proactor_io_uring_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation);

// Issues a bounded FIFO batch of owned cancellations after ready target
// callbacks had the opportunity to withdraw their unissued requests.
iree_status_t iree_async_proactor_io_uring_submit_cancel_requests(
    iree_async_proactor_io_uring_t* proactor);

// Delivers an owned key-retirement receipt, counting its callback as progress.
// Native cancellation errors return to poll independently of the receipt.
iree_status_t iree_async_proactor_io_uring_complete_cancel_request(
    const iree_io_uring_cqe_t* cqe, iree_host_size_t* out_completed_count);

// Capability probing (called from create).
// Probes the kernel for supported io_uring features and populates
// |out_capabilities| with the detected capabilities.
// Returns IREE_STATUS_UNAVAILABLE if the kernel is too old.
iree_status_t iree_async_proactor_io_uring_detect_capabilities(
    iree_io_uring_ring_t* ring, uint32_t ring_features,
    iree_async_proactor_capabilities_t* out_capabilities);

// Probes an empty, single-entry provided-buffer ring before other
// registrations. |probe_page| is a zeroed, page-aligned allocation of at least
// one page. On failure the unpublished ring must be destroyed before freeing
// that page: registration may have succeeded while unregistration failed.
iree_status_t iree_async_proactor_io_uring_probe_buffer_ring(
    iree_io_uring_ring_t* ring, void* probe_page, bool* out_supported);

// Buffer registration vtable implementations (in proactor_registration.c).
iree_status_t iree_async_proactor_io_uring_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry);
iree_status_t iree_async_proactor_io_uring_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry);
void iree_async_proactor_io_uring_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state);
iree_status_t iree_async_proactor_io_uring_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region);

// Continuation dispatch helpers (in proactor.c, used by proactor_submit.c).
void iree_async_proactor_io_uring_submit_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Pushes software work to the poll-owned MPSC queue. For SEQUENCE operations,
// an OK status requests poll-owned startup. All other entries are terminal
// completions. Takes ownership of |status| and carries it with |operation|.
void iree_async_proactor_io_uring_push_software_operation(
    iree_async_proactor_io_uring_t* proactor, iree_async_operation_t* operation,
    iree_status_t status);

// Iteratively dispatches a continuation chain containing software operations.
// Software ops execute inline (side effects only) with completions pushed to
// MPSC for poll-thread callback delivery. Kernel ops are submitted to the ring
// for CQE-driven completion. (In proactor_submit.c, used by proactor.c.)
void iree_async_proactor_io_uring_dispatch_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Cancels a continuation chain by pushing CANCELLED completions to the MPSC
// queue. Resources are retained before pushing so the drain's release is
// balanced. (In proactor_submit.c, used by proactor.c.)
void iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Builds the intrusive list of software operations selected from a prepared
// submission batch. All batch metadata is consumed before the list is
// returned; callers must capture and clear each operation's next pointer before
// publishing it.
iree_async_operation_t*
iree_async_proactor_io_uring_build_software_submission_list(
    iree_async_operation_list_t operations);

// Submit vtable implementation (in proactor_submit.c).
iree_status_t iree_async_proactor_io_uring_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_
