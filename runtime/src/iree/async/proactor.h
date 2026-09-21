// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Proactor: the central abstraction for completion-based async I/O.
//
// A proactor manages the submission of async operations and dispatches
// completion callbacks when they finish. It is vtable-dispatched for backend
// polymorphism (io_uring, kqueue, IOCP, threaded emulation).
//
// The proactor is caller-driven: it only makes progress when poll() is called.
// All callbacks fire from within poll() on the calling thread. A utility
// wrapper (iree_async_proactor_thread_t) provides optional dedicated-thread
// operation.
//
// Threading model:
//   The proactor has single-thread ownership semantics for poll(). The first
//   thread to call poll() becomes the poll owner for the proactor's lifetime.
//   Only that thread may call poll(); calling from any other thread is
//   undefined behavior. This enables lock-free fast paths in high-performance
//   backends (e.g., io_uring's DEFER_TASKRUN mode). Other operations like
//   submit(), cancel(), wake(), import_fence(), and export_fence() are
//   thread-safe and may be called from any thread.
//
// Lifecycle: backend-specific _create functions → _retain/_release.

#ifndef IREE_ASYNC_PROACTOR_H_
#define IREE_ASYNC_PROACTOR_H_

#include "iree/async/operation.h"
#include "iree/async/primitive.h"
#include "iree/async/region.h"
#include "iree/async/relay.h"
#include "iree/async/slab.h"
#include "iree/async/socket.h"
#include "iree/async/types.h"
#include "iree/async/util/intrusive_list.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_file_t iree_async_file_t;
typedef struct iree_async_event_t iree_async_event_t;
typedef struct iree_async_event_source_t iree_async_event_source_t;
typedef struct iree_async_notification_t iree_async_notification_t;
typedef struct iree_async_notification_native_t
    iree_async_notification_native_t;
typedef struct iree_async_relay_t iree_async_relay_t;
typedef struct iree_async_semaphore_t iree_async_semaphore_t;

// Notification flags type (full definition in notification.h).
typedef uint32_t iree_async_notification_flags_t;

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_proactor_vtable_t iree_async_proactor_vtable_t;
typedef struct iree_async_signal_subscription_t
    iree_async_signal_subscription_t;

//===----------------------------------------------------------------------===//
// Event source callbacks
//===----------------------------------------------------------------------===//

// Poll events passed to event source callbacks.
//
// Cross-platform abstraction over native poll events. Backends translate from
// their native representation (POLLIN/POLLERR/POLLHUP on POSIX, equivalent
// flags on other platforms) to these portable values.
//
// The callback always fires for any poll activity; use these flags to
// determine what occurred and how to respond.
typedef enum iree_async_poll_event_e {
  IREE_ASYNC_POLL_EVENT_NONE = 0,
  // Data available to read (POLLIN equivalent).
  IREE_ASYNC_POLL_EVENT_IN = 1u << 0,
  // Error condition on the fd (POLLERR equivalent).
  IREE_ASYNC_POLL_EVENT_ERR = 1u << 1,
  // Hang up / disconnected (POLLHUP/POLLRDHUP equivalent).
  IREE_ASYNC_POLL_EVENT_HUP = 1u << 2,
  // Ready for writing / send buffer available (POLLOUT equivalent).
  // Used for backpressure handling in non-blocking write loops.
  // POSIX: POLLOUT/EVFILT_WRITE. Not applicable to Windows waitable HANDLEs.
  IREE_ASYNC_POLL_EVENT_OUT = 1u << 3,
} iree_async_poll_event_t;
// Bitmask of poll events.
typedef uint32_t iree_async_poll_events_t;

// Returns true if error events are present without data available.
//
// This indicates a fatal error condition on the fd (disconnected, device
// removed, etc.). The callback should handle cleanup rather than attempting
// to read from the fd.
static inline bool iree_async_poll_has_error(iree_async_poll_events_t events) {
  return (events & (IREE_ASYNC_POLL_EVENT_ERR | IREE_ASYNC_POLL_EVENT_HUP)) &&
         !(events & IREE_ASYNC_POLL_EVENT_IN);
}

// Callback invoked when an event source is signaled.
//
// Fires from within proactor poll() on the polling thread. The source remains
// armed after the callback returns (multishot behavior). Heavy work should be
// deferred to avoid stalling completion dispatch.
//
// Parameters:
//   user_data: Value from the callback struct at registration time.
//   source: The event source that was signaled. The source remains registered
//     until explicitly unregistered. Do not unregister from within the
//     callback.
//   events: Bitmask of poll events that occurred. Check with
//     iree_async_poll_has_error() to detect error conditions. For RDMA CQ
//     channels, IREE_ASYNC_POLL_EVENT_IN indicates completions are available.
typedef void (*iree_async_event_source_callback_fn_t)(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t events);

// Callback wrapper struct for event source notifications.
// Follows the pattern of iree_async_buffer_recycle_callback_t.
typedef struct iree_async_event_source_callback_t {
  // Function dispatched by the polling thread when the handle is ready.
  iree_async_event_source_callback_fn_t fn;
  // Borrowed callback context, retained through terminal unregistration.
  void* user_data;
} iree_async_event_source_callback_t;

// Returns a null event source callback (no notification).
static inline iree_async_event_source_callback_t
iree_async_event_source_callback_null(void) {
  iree_async_event_source_callback_t callback = {NULL, NULL};
  return callback;
}

// Completion of terminal event-source unregistration. Native monitoring and
// all queued callbacks have retired and the source has been destroyed before
// this function runs. Borrowed handle and callback storage may be released.
typedef void(IREE_API_PTR* iree_async_event_source_unregistered_fn_t)(
    void* user_data);

typedef struct iree_async_event_source_unregistered_callback_t {
  // Function invoked after the event source has been destroyed.
  iree_async_event_source_unregistered_fn_t fn;
  // Opaque value passed to |fn|.
  void* user_data;
} iree_async_event_source_unregistered_callback_t;

// Returns an empty unregistration callback.
static inline iree_async_event_source_unregistered_callback_t
iree_async_event_source_unregistered_callback_none(void) {
  iree_async_event_source_unregistered_callback_t callback = {NULL, NULL};
  return callback;
}

//===----------------------------------------------------------------------===//
// Progress callbacks (inline poll-loop work)
//===----------------------------------------------------------------------===//
//
// Progress callbacks perform poll-owner work before the backend waits for
// native completions. An entry stays registered only while it has work to
// attempt on each poll; waiting for native readiness belongs in a native
// operation or event source instead.
//
// Registration/unregistration are poll-thread-only operations: no
// synchronization is needed because the same thread calls poll(). Callbacks
// fire from within poll() before the blocking wait. The backend does not block
// while any entry remains registered or after a callback reports completions.
// A callback failure returns through poll() with all dispatched completions
// counted. Failure alone neither removes an entry nor retires its native work.
//
// Callbacks may request their own removal by setting remove_requested=true.
// The proactor removes the entry from the list after the callback returns,
// avoiding list corruption during iteration.
//
// Each entry runs at most once per poll, in unspecified order. Callbacks may
// register new entries or unregister other entries. New registrations,
// including re-registration from on_remove, first run on the next poll.
//
// Intrusive singly-linked list entry for per-poll-iteration progress checks.
// Owned by the registrant (e.g., embedded in a native I/O owner). The proactor
// holds a linked list of these; all mutations happen on the poll thread only.
typedef struct iree_async_progress_entry_t {
  // Next registered entry, maintained by the proactor.
  struct iree_async_progress_entry_t* next;

  // Called each poll() iteration. Reports dispatched completions through the
  // non-NULL |out_completed_count|, initially zero, including on failure.
  // Transfers any failure status to poll(). The entry remains registered unless
  // |remove_requested| is set, independently of the returned status.
  iree_status_t (*fn)(void* user_data, iree_host_size_t* out_completed_count);
  // Borrowed context passed to fn and on_remove.
  void* user_data;

  // Set by the callback to request removal after fn returns. The proactor
  // removes the entry from the list after the callback completes, avoiding
  // list corruption during iteration.
  bool remove_requested;

  // Optional callback invoked AFTER the entry is unlinked from the proactor's
  // list (only called when remove_requested is set). This fires after fn
  // returns and the entry is safely removed, so it may trigger higher-level
  // cleanup (e.g., deactivation completion) that could free the entry. The
  // proactor does not access the entry after on_remove returns.
  void (*on_remove)(void* user_data);
} iree_async_progress_entry_t;

//===----------------------------------------------------------------------===//
// Proactor
//===----------------------------------------------------------------------===//

// Proactor capability flags.
//
// Use iree_async_proactor_query_capabilities() to discover which features
// the current backend supports. Callers can use these to select optimal code
// paths or verify that the backend supports required features.
//
// Capability matrix by backend:
//
//   Capability         generic | io_uring | IOCP | kqueue
//   ────────────────────────────────────────────────────────
//   MULTISHOT          emul    | 5.19+    | emul | emul
//   FIXED_FILES        emul    | 5.1+     | n/a  | n/a
//   REGISTERED_BUFFERS emul    | 5.1+     | reg  | emul
//   LINKED_OPERATIONS  emul    | 5.3+     | emul | emul
//   ZERO_COPY_SEND     copy    | 6.0+     | reg  | copy
//   DMABUF             n/a     | 5.7+     | n/a  | n/a
//   DEVICE_FENCE       poll    | poll     | yes  | poll
//   ABSOLUTE_TIMEOUT   emul    | 5.4+     | yes  | emul
//   FUTEX_OPERATIONS   n/a     | 6.7+     | n/a  | n/a
//
// Legend:
//   yes/X.Y+ = Native kernel support (optimal path).
//   emul     = Emulated (correct behavior, per-operation overhead).
//   copy     = Falls back to copy (flag accepted but not honored).
//   reg      = Native only with pre-registered buffers.
//   poll     = Polls on fd (correct, adds wake latency).
//   n/a      = Hardware-specific; API returns IREE_STATUS_UNAVAILABLE.
//
enum iree_async_proactor_capability_bits_e {
  IREE_ASYNC_PROACTOR_CAPABILITY_NONE = 0u,

  // Supports multishot operations (persistent accept/recv).
  // When enabled, a single submit can produce multiple completions until
  // explicitly cancelled. Reduces syscall overhead for high-throughput
  // accept loops and streaming receives.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   emul    | 5.19+    | emul | emul
  IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT = 1u << 0,

  // Supports registered/fixed file descriptors for reduced syscall overhead.
  // Pre-registers sockets/files with the kernel, eliminating per-operation
  // fd lookup costs. Most beneficial for high-frequency operations on the
  // same set of connections.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   emul    | 5.1+     | n/a  | n/a
  IREE_ASYNC_PROACTOR_CAPABILITY_FIXED_FILES = 1u << 1,

  // Supports registered memory buffers for zero-copy I/O.
  // Pre-registers memory regions with the kernel, enabling DMA directly
  // to/from application buffers without intermediate copies. Critical for
  // achieving line-rate network throughput.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   emul    | 5.1+     | reg  | emul
  IREE_ASYNC_PROACTOR_CAPABILITY_REGISTERED_BUFFERS = 1u << 2,

  // Supports linked operations (io_uring linked SQEs) for kernel-chained
  // sequences without user-space round-trips between steps. Enables patterns
  // like "recv → process → send" to execute entirely in kernel space.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   emul    | 5.3+     | emul | emul
  IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS = 1u << 3,

  // Supports zero-copy send (MSG_ZEROCOPY / io_uring SEND_ZC).
  // Avoids copying send data into kernel buffers. The application buffer
  // must remain valid until the zero-copy completion notification arrives.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   copy    | 6.0+     | reg  | copy
  IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND = 1u << 4,

  // Supports importing Linux DMA buffers as mapped I/O regions. Backends may
  // use fixed-buffer or device-direct optimizations when the mapping and
  // transport support them; callers must not infer zero-copy from this bit.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   n/a     | 5.7+     | n/a  | n/a
  IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF = 1u << 5,

  // Supports device fence import/export (sync_file ↔ semaphore bridging).
  // Enables GPU→proactor and proactor→GPU synchronization without CPU
  // involvement in the fast path.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   poll    | poll     | yes  | poll
  IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE = 1u << 6,

  // Supports absolute timeouts (e.g., io_uring IORING_TIMEOUT_ABS on 5.4+).
  // When set, timer operations use absolute deadlines with no drift.
  // When not set, the backend converts to relative at submission time,
  // which introduces potential drift under scheduling delays.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   emul    | 5.4+     | yes  | emul
  IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT = 1u << 7,

  // Supports kernel-side futex operations (io_uring FUTEX_WAIT/WAKE on 6.7+).
  // When set, futex wait/wake can be submitted as io_uring operations,
  // enabling LINK chains like POLL_ADD → FUTEX_WAKE for efficient
  // semaphore-to-futex bridging without userspace round-trips.
  // When not set, callers must use direct syscalls for futex operations
  // (iree/base/threading/futex.h).
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   n/a     | 6.7+     | n/a  | n/a
  IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS = 1u << 8,

  // Supports native kernel-mediated cross-proactor messaging via io_uring
  // MSG_RING (5.18+). When absent, messaging-capable backends use a bounded
  // software message pool plus a poll wake. LINKED MESSAGE operations remain
  // available through userspace continuation dispatch.
  //
  // Availability:
  //   generic | io_uring | IOCP | kqueue
  //   n/a     | 5.18+    | n/a  | n/a
  IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING = 1u << 9,

  // Supports kernel wait completion packets for direct Event-to-IOCP delivery
  // (NtAssociateWaitCompletionPacket on Windows 8.1+). When set, one-shot
  // event waits and shared notification wake monitoring bypass the
  // RegisterWaitForSingleObject threadpool, and persistent event sources are
  // available. When not set, one-shot waits and shared notification wakes use
  // RegisterWaitForSingleObject while event source registration returns
  // IREE_STATUS_UNAVAILABLE.
  //
  // Availability:
  //   generic | io_uring | IOCP     | kqueue
  //   n/a     | n/a      | Win 8.1+ | n/a
  IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET = 1u << 10,

  // All capabilities enabled (for allowed_capabilities default).
  IREE_ASYNC_PROACTOR_CAPABILITY_ALL = ~0u,
};

// Backend capabilities reported by query_capabilities().
typedef uint32_t iree_async_proactor_capabilities_t;

// How the proactor will be used relative to threading.
typedef enum iree_async_proactor_threading_mode_e {
  // The caller will poll from the same thread that creates the proactor.
  // This is the common case for tests, benchmarks, and simple single-threaded
  // usage.
  IREE_ASYNC_PROACTOR_THREADING_SAME_THREAD = 0,

  // The caller will create the proactor on one thread and poll from a different
  // thread (e.g., a proactor pool with dedicated poll threads). The proactor
  // will defer any per-thread setup to the first poll() call.
  IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD = 1,
} iree_async_proactor_threading_mode_t;

// Options for proactor creation.
//
// All fields have sensible defaults; zero-initialize and override as needed.
// Use iree_async_proactor_options_default() for canonical initialization.
typedef struct iree_async_proactor_options_t {
  // Hint: maximum number of concurrently in-flight operations.
  // Backends use this to size internal structures (ring buffers, etc.).
  // Zero means "use a reasonable default" (typically 256-4096 depending on
  // the backend and available system resources).
  //
  // io_uring: Sets the ring size. Powers of 2 are most efficient.
  // IOCP: Ignored (IOCP is dynamically sized).
  // kqueue: Sets initial event list allocation.
  // generic: Sets internal completion queue size.
  iree_host_size_t max_concurrent_operations;

  // Debug name for tracing and diagnostics.
  // Copied into the proactor; the string view need not remain valid after
  // the create call returns.
  iree_string_view_t debug_name;

  // Capabilities to allow. The backend's detected capabilities are masked
  // with this value: `effective = detected & allowed_capabilities`.
  // Defaults to IREE_ASYNC_PROACTOR_CAPABILITY_ALL (all capabilities enabled).
  //
  // Use this to disable specific features for testing or compatibility:
  //   options.allowed_capabilities &=
  //   ~IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
  iree_async_proactor_capabilities_t allowed_capabilities;

  // Capacity of the cross-proactor message pool. Zero means use the default
  // (IREE_ASYNC_MESSAGE_POOL_DEFAULT_CAPACITY = 256). Each entry is 24 bytes,
  // so the default uses ~6KB of trailing data in the proactor allocation.
  // Increase for workloads with many concurrent in-flight messages across
  // proactors; decrease on memory-constrained embedded targets.
  iree_host_size_t message_pool_capacity;

  // Threading model. Defaults to SAME_THREAD.
  iree_async_proactor_threading_mode_t threading_mode;
} iree_async_proactor_options_t;

// Returns default proactor options.
static inline iree_async_proactor_options_t iree_async_proactor_options_default(
    void) {
  iree_async_proactor_options_t options = {0};
  options.allowed_capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_ALL;
  return options;
}

//===----------------------------------------------------------------------===//
// Cross-proactor messaging
//===----------------------------------------------------------------------===//

// Callback invoked when a message is received from another proactor.
//
// |proactor| is the receiving proactor (from within whose poll() this fires).
// |message_data| is the 64-bit payload sent by the source proactor.
// |user_data| is the value set via iree_async_proactor_set_message_callback.
//
// Callbacks fire from within iree_async_proactor_poll() on the polling thread.
// Heavy work should be deferred to avoid stalling completion dispatch.
typedef void (*iree_async_proactor_message_callback_fn_t)(
    iree_async_proactor_t* proactor, uint64_t message_data, void* user_data);

// Bundled message callback (function pointer + user data).
typedef struct iree_async_proactor_message_callback_t {
  iree_async_proactor_message_callback_fn_t fn;
  void* user_data;
} iree_async_proactor_message_callback_t;

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//
//
// Signals are process-global, so only ONE proactor per process may own signal
// subscriptions. The first proactor to subscribe claims ownership; subsequent
// subscribe calls from other proactors return IREE_STATUS_FAILED_PRECONDITION.
//
// Signal subscriptions use a doubly-linked list per signal type for O(1)
// removal. Subscribe/unsubscribe must be serialized with poll() (see below).
//
// Threading model ("serialized with poll()"):
//   Subscribe/unsubscribe are safe in these contexts:
//     - Before poll loop starts (no concurrent poll)
//     - After poll loop exits (poll has returned)
//     - From within poll() callbacks (already serialized)
//     - Via wake + message to defer to poll thread
//   The dangerous case is calling from another thread while poll() is running.
//
// Typical usage:
//   // Early init (before threads):
//   iree_async_signal_block_default();
//   iree_async_signal_ignore_broken_pipe();
//
//   // After proactor creation, before poll loop:
//   iree_async_signal_subscription_t* sub = NULL;
//   iree_async_proactor_subscribe_signal(proactor, IREE_ASYNC_SIGNAL_INTERRUPT,
//       callback, &sub);
//
//   // Poll loop: signal callbacks fire from within poll().
//
//   // After poll loop exits:
//   iree_async_proactor_unsubscribe_signal(proactor, sub);

// Platform-abstracted signal identifiers.
//
// These map to common signals that server applications handle. On Windows,
// only INTERRUPT and TERMINATE are meaningful (console control events).
// Platform-specific signals (HANGUP, QUIT, USER1, USER2) return
// IREE_STATUS_INVALID_ARGUMENT on platforms that don't support them.
typedef enum iree_async_signal_e {
  IREE_ASYNC_SIGNAL_NONE = 0,

  // Interrupt request (SIGINT / Ctrl+C / CTRL_C_EVENT).
  // Typical use: Initiate graceful shutdown.
  IREE_ASYNC_SIGNAL_INTERRUPT,

  // Termination request (SIGTERM / CTRL_CLOSE_EVENT).
  // Typical use: Forced but clean shutdown (systemd, container runtime).
  IREE_ASYNC_SIGNAL_TERMINATE,

  // Hangup / terminal disconnect (SIGHUP). Unix only.
  // Typical use: Configuration reload.
  IREE_ASYNC_SIGNAL_HANGUP,

  // Quit with core dump (SIGQUIT). Unix only.
  // Note: Subscribing to QUIT via signalfd suppresses core dump generation.
  // Typical use: Diagnostic dump (thread stacks, memory state).
  IREE_ASYNC_SIGNAL_QUIT,

  // User-defined signal 1 (SIGUSR1). Unix only.
  IREE_ASYNC_SIGNAL_USER1,

  // User-defined signal 2 (SIGUSR2). Unix only.
  IREE_ASYNC_SIGNAL_USER2,

  IREE_ASYNC_SIGNAL_COUNT,
} iree_async_signal_t;

// Returns the platform name for a signal (e.g., "SIGINT", "interrupt").
// Returns "unknown" for invalid signal values.
IREE_API_EXPORT iree_string_view_t
iree_async_signal_name(iree_async_signal_t signal);

// Callback invoked when a subscribed signal is received. The |signal|
// indicates which signal fired and |user_data| is the value provided at
// subscription time.
//
// Fires from within iree_async_proactor_poll() on the polling thread. Multiple
// signals of the same type arriving between polls may be coalesced into a
// single callback. The callback should be fast; defer heavy work to avoid
// stalling completion dispatch.
typedef void (*iree_async_signal_callback_fn_t)(void* user_data,
                                                iree_async_signal_t signal);

typedef struct iree_async_signal_callback_t {
  iree_async_signal_callback_fn_t fn;
  void* user_data;
} iree_async_signal_callback_t;

static inline iree_async_signal_callback_t iree_async_signal_callback_null(
    void) {
  iree_async_signal_callback_t callback = {NULL, NULL};
  return callback;
}

// Signal subscription handle.
//
// Subscriptions form a doubly-linked list per signal type. The proactor owns
// the list heads; subscriptions are allocated by subscribe and freed by
// unsubscribe. The subscription struct is exposed for inline access but
// should be treated as opaque by callers.
//
// Lifetime:
//   - Created by iree_async_proactor_subscribe_signal()
//   - Valid until iree_async_proactor_unsubscribe_signal() or proactor destroy
//   - Proactor destroy automatically unsubscribes all remaining subscriptions
struct iree_async_signal_subscription_t {
  // Intrusive doubly-linked list for O(1) removal.
  struct iree_async_signal_subscription_t* next;
  struct iree_async_signal_subscription_t* prev;

  // Separate link for deferred unsubscribe list. During dispatch, unsubscribes
  // are deferred to avoid corrupting the main list during iteration. This field
  // is used to build the deferred list without touching next/prev.
  struct iree_async_signal_subscription_t* pending_next;

  // Back-pointer to owning proactor (for unsubscribe validation).
  iree_async_proactor_t* proactor;

  // Which signal this subscription is for.
  iree_async_signal_t signal;

  // User callback.
  iree_async_signal_callback_t callback;
};

//===----------------------------------------------------------------------===//
// Cancellation requests
//===----------------------------------------------------------------------===//

// Notifies the owner that a cancellation no longer references its target
// identity. The target's terminal callback separately reports its outcome.
typedef struct iree_async_cancel_callback_t {
  // Invoked on the poll owner. May release the request storage.
  void (*fn)(void* user_data);
  // Borrowed until the callback returns.
  void* user_data;
} iree_async_cancel_callback_t;

// Backend-owned phases of a caller-owned cancellation request.
typedef enum iree_async_cancel_request_phase_e {
  IREE_ASYNC_CANCEL_REQUEST_PHASE_IDLE = 0,
  IREE_ASYNC_CANCEL_REQUEST_PHASE_QUEUED,
  IREE_ASYNC_CANCEL_REQUEST_PHASE_ISSUED,
} iree_async_cancel_request_phase_t;

// Storage borrowed by request_cancel until its receipt callback. The owner
// must also join the target's terminal callback before releasing or reusing
// the target storage. Fields are private to the proactor after initialization.
typedef struct iree_async_cancel_request_t {
  // Owner-thread pending queue linkage. Must be first for backend casts.
  iree_intrusive_list_entry_t pending_entry;
  // Borrowed while queued; cleared once the native cancellation is issued.
  iree_async_operation_t* target;
  // Receipt returning the request storage to its owner.
  iree_async_cancel_callback_t callback;
  // Distinguishes withdrawable software work from an issued native request.
  iree_async_cancel_request_phase_t phase;
} iree_async_cancel_request_t;

// Initializes idle caller storage. A request can be reused after its receipt.
IREE_API_EXPORT void iree_async_cancel_request_initialize(
    iree_async_cancel_callback_t callback,
    iree_async_cancel_request_t* out_request);

//===----------------------------------------------------------------------===//
// Proactor vtable
//===----------------------------------------------------------------------===//

typedef struct iree_async_proactor_vtable_t {
  // Called when ref_count reaches zero. Must release all backend resources.
  void (*destroy)(iree_async_proactor_t* proactor);

  iree_async_proactor_capabilities_t (*query_capabilities)(
      iree_async_proactor_t* proactor);

  iree_status_t (*submit)(iree_async_proactor_t* proactor,
                          iree_async_operation_list_t operations);
  iree_status_t (*poll)(iree_async_proactor_t* proactor, iree_timeout_t timeout,
                        iree_host_size_t* out_completed_count);
  void (*end_polling)(iree_async_proactor_t* proactor);
  void (*wake)(iree_async_proactor_t* proactor);
  iree_status_t (*cancel)(iree_async_proactor_t* proactor,
                          iree_async_operation_t* operation);

  iree_status_t (*create_socket)(iree_async_proactor_t* proactor,
                                 iree_async_socket_type_t type,
                                 iree_async_socket_options_t options,
                                 iree_async_socket_t** out_socket);
  iree_status_t (*import_socket)(iree_async_proactor_t* proactor,
                                 iree_async_primitive_t primitive,
                                 iree_async_socket_type_t type,
                                 iree_async_socket_flags_t flags,
                                 iree_async_socket_t** out_socket);
  void (*destroy_socket)(iree_async_proactor_t* proactor,
                         iree_async_socket_t* socket);

  iree_status_t (*import_file)(iree_async_proactor_t* proactor,
                               iree_async_primitive_t primitive,
                               iree_async_file_t** out_file);
  void (*destroy_file)(iree_async_proactor_t* proactor,
                       iree_async_file_t* file);

  iree_status_t (*create_event)(iree_async_proactor_t* proactor,
                                iree_async_event_t** out_event);
  void (*destroy_event)(iree_async_proactor_t* proactor,
                        iree_async_event_t* event);

  iree_status_t (*register_event_source)(
      iree_async_proactor_t* proactor, iree_async_primitive_t handle,
      iree_async_event_source_callback_t callback,
      iree_async_event_source_t** out_event_source);
  void (*unregister_event_source)(
      iree_async_proactor_t* proactor, iree_async_event_source_t* event_source,
      iree_async_event_source_unregistered_callback_t callback);

  iree_status_t (*create_notification)(
      iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
      iree_async_notification_t** out_notification);
  iree_status_t (*create_notification_shared)(
      iree_async_proactor_t* proactor, iree_async_notification_native_t* native,
      iree_async_notification_t** out_notification);
  void (*destroy_notification)(iree_async_proactor_t* proactor,
                               iree_async_notification_t* notification);
  void (*notification_signal)(iree_async_proactor_t* proactor,
                              iree_async_notification_t* notification,
                              int32_t wake_count);
  bool (*notification_wait)(iree_async_proactor_t* proactor,
                            iree_async_notification_t* notification,
                            uint32_t wait_token, iree_timeout_t timeout);

  iree_status_t (*register_relay)(
      iree_async_proactor_t* proactor, iree_async_relay_source_t source,
      iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
      iree_async_relay_error_callback_t error_callback,
      iree_async_relay_t** out_relay);
  void (*unregister_relay)(iree_async_proactor_t* proactor,
                           iree_async_relay_t* relay,
                           iree_async_relay_unregistered_callback_t callback);

  iree_status_t (*register_buffer)(
      iree_async_proactor_t* proactor,
      iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
      iree_async_buffer_access_flags_t access_flags,
      iree_async_buffer_registration_entry_t** out_entry);
  iree_status_t (*register_dmabuf)(
      iree_async_proactor_t* proactor,
      iree_async_buffer_registration_state_t* state, int dmabuf_fd,
      uint64_t offset, iree_host_size_t length,
      iree_async_buffer_access_flags_t access_flags,
      iree_async_buffer_registration_entry_t** out_entry);
  void (*unregister_buffer)(iree_async_proactor_t* proactor,
                            iree_async_buffer_registration_entry_t* entry,
                            iree_async_buffer_registration_state_t* state);
  iree_status_t (*register_slab)(iree_async_proactor_t* proactor,
                                 iree_async_slab_t* slab,
                                 iree_async_buffer_access_flags_t access_flags,
                                 iree_async_region_t** out_region);

  iree_status_t (*import_fence)(iree_async_proactor_t* proactor,
                                iree_async_primitive_t fence,
                                iree_async_semaphore_t* semaphore,
                                uint64_t signal_value);
  iree_status_t (*export_fence)(iree_async_proactor_t* proactor,
                                iree_async_semaphore_t* semaphore,
                                uint64_t wait_value,
                                iree_async_primitive_t* out_fence);

  // Cross-proactor messaging. NULL if backend does not support messaging.
  // io_uring may use native MSG_RING delivery. IOCP, POSIX, and the io_uring
  // fallback use a pre-allocated message pool plus a backend wake.
  void (*set_message_callback)(iree_async_proactor_t* proactor,
                               iree_async_proactor_message_callback_t callback);
  iree_status_t (*send_message)(iree_async_proactor_t* target,
                                uint64_t message_data);

  // Signal handling. NULL if platform does not support signals.
  // Implementation must claim process-wide signal ownership on first subscribe.
  iree_status_t (*subscribe_signal)(
      iree_async_proactor_t* proactor, iree_async_signal_t signal,
      iree_async_signal_callback_t callback,
      iree_async_signal_subscription_t** out_subscription);
  void (*unsubscribe_signal)(iree_async_proactor_t* proactor,
                             iree_async_signal_subscription_t* subscription);
} iree_async_proactor_vtable_t;

// Base proactor structure. Backends extend this with additional fields by
// embedding it as the first member of their implementation struct.
typedef struct iree_async_proactor_t {
  iree_atomic_ref_count_t ref_count;
  const iree_async_proactor_vtable_t* vtable;
  iree_allocator_t allocator;
  IREE_TRACE(char debug_name[32];)

  // Head of the progress callback list. Poll-thread-only; no synchronization.
  // See "Progress callbacks" section above.
  iree_async_progress_entry_t* progress_list;

  // Pending cancellation work only; not a registry of submitted operations.
  // Mutated exclusively by the poll owner.
  struct {
    // FIFO head with O(1) removal when a target completes before issue.
    iree_intrusive_list_t list;
    // FIFO tail so callbacks cannot starve older requests by enqueueing work.
    iree_intrusive_list_entry_t* tail;
  } cancellations;
} iree_async_proactor_t;

// Initializes the base proactor fields. Called by backend create functions.
IREE_API_EXPORT void iree_async_proactor_initialize(
    const iree_async_proactor_vtable_t* vtable, iree_string_view_t debug_name,
    iree_allocator_t allocator, iree_async_proactor_t* out_proactor);

//===----------------------------------------------------------------------===//
// Progress callback management
//===----------------------------------------------------------------------===//

// Registers a progress callback that runs every poll() iteration.
//
// The entry is owned by the caller and must remain valid until unregistered
// (either explicitly or via remove_requested). The entry's fn, user_data, and
// on_remove fields must be initialized before calling this function. The entry
// must not already be registered. Registration clears remove_requested.
//
// Must be called from the poll thread only (from within poll() callbacks or
// before the poll loop starts).
IREE_API_EXPORT void iree_async_proactor_register_progress(
    iree_async_proactor_t* proactor, iree_async_progress_entry_t* entry);

// Unregisters a progress callback. The entry must have been previously
// registered with this proactor. After this call the entry's fn will not be
// called again.
//
// Must be called from the poll thread only. Must NOT be called from within
// the entry's own fn callback; use remove_requested instead. Callbacks may
// unregister other entries and immediately release their storage. Explicit
// unregistration does not invoke on_remove.
IREE_API_EXPORT void iree_async_proactor_unregister_progress(
    iree_async_proactor_t* proactor, iree_async_progress_entry_t* entry);

// Runs registered progress callbacks until all have run or one fails. Reports
// all dispatched completions through non-NULL |out_completed_count|, including
// on failure. Entries with remove_requested set are removed after their
// callback returns, including on failure; all other entries remain owned and
// registered. Registrations made during this pass are deferred until the next
// pass.
//
// Called by backend poll() implementations before the blocking wait. Backends
// must force a non-blocking poll whenever progress callbacks are registered
// (progress_list is non-NULL), regardless of whether they returned progress
// this iteration. Progress callbacks exist to be polled; blocking while they
// are registered would starve them until an unrelated I/O event arrives.
IREE_API_EXPORT iree_status_t iree_async_proactor_run_progress(
    iree_async_proactor_t* proactor, iree_host_size_t* out_completed_count);

// Retains a reference to the proactor.
static inline void iree_async_proactor_retain(iree_async_proactor_t* proactor) {
  if (proactor) {
    iree_atomic_ref_count_inc(&proactor->ref_count);
  }
}

// Releases a reference to the proactor. Destroys when count reaches zero.
// Callers stop submission and drain user operations before final release.
// Once polling has begun, final release with live observers or pending
// unregistrations must occur on the poll owner. It drives remaining native
// retirement and terminal callbacks before destruction returns.
// Those terminal callbacks may release resources but must not admit new work.
// After all owner-bound work has retired, final release may occur on any
// thread.
static inline void iree_async_proactor_release(
    iree_async_proactor_t* proactor) {
  if (proactor && iree_atomic_ref_count_dec(&proactor->ref_count) == 1) {
    proactor->vtable->destroy(proactor);
  }
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

// Queries backend capabilities.
static inline iree_async_proactor_capabilities_t
iree_async_proactor_query_capabilities(iree_async_proactor_t* proactor) {
  return proactor->vtable->query_capabilities(proactor);
}

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

// Submits a batch of operations for async execution.
//
// Operations are caller-owned and must remain valid until their final
// callbacks fire. The operation list contents are consumed during this call;
// the list struct itself need not remain valid after return.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Threading model:
//   Thread-safe with respect to poll() and wake(). Multiple threads may
//   submit concurrently. However, each individual operation must be owned by
//   exactly one thread at submission time (no concurrent access to the same
//   operation struct).
//
// Batch semantics:
//   All operations in the batch are submitted atomically where possible.
//   io_uring: Single io_uring_submit() call.
//   IOCP: Individual PostQueuedCompletionStatus calls (no atomic batch).
//   kqueue: Single kevent() call with all changes.
//
// Region lifetime:
//   A successful submit acquires resources for every operation in the list,
//   including linked successors that do not execute until a predecessor
//   completes. Registered span regions remain alive through the final callback
//   and are then released. For multishot operations, callbacks carrying
//   IREE_ASYNC_COMPLETION_FLAG_MORE retain ownership until the terminal
//   callback. A synchronously rejected batch acquires no resources.
//
// Returns:
//   IREE_STATUS_OK: All operations submitted successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Submission queue full (retry after poll).
//   IREE_STATUS_INVALID_ARGUMENT: Malformed operation in the batch.
static inline iree_status_t iree_async_proactor_submit(
    iree_async_proactor_t* proactor, iree_async_operation_list_t operations) {
  return proactor->vtable->submit(proactor, operations);
}

// Submits a single operation. Convenience wrapper around submit().
static inline iree_status_t iree_async_proactor_submit_one(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation) {
  iree_async_operation_list_t list = {&operation, 1};
  return proactor->vtable->submit(proactor, list);
}

// Drains completions and invokes callbacks.
//
// Blocks until |timeout| is reached, at least one completion is available, or
// an explicit wake or internal progress event is observed.
// All callbacks fire synchronously from within this call on the calling thread.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Threading model:
//   Poll must be called from the proactor's poll owner thread. The first thread
//   to call poll() becomes the poll owner for the proactor's lifetime. Calling
//   poll() from any other thread is undefined behavior and may cause errors
//   (e.g., EEXIST on io_uring with DEFER_TASKRUN). This constraint enables
//   lock-free fast paths in high-performance backends.
//
//   NOT thread-safe with respect to other poll() calls. Thread-safe with
//   respect to submit() and wake(), which may be called from any thread.
//   Callbacks must not recursively poll the same proactor.
//   Typical pattern: dedicated I/O thread owns the proactor and calls poll()
//   in a loop while worker threads call submit().
//
// Parameters:
//   timeout: Maximum time to block. Use iree_timeout_t (not raw duration) to
//     avoid drift—absolute deadlines are converted to relative only at the
//     syscall boundary. Use iree_immediate_timeout() for non-blocking poll.
//   out_completed_count: Number of operation completions and independently
//     dispatched cancellation receipts (may be NULL). Receipts withdrawn
//     inline by target callbacks are part of that target's dispatch, not an
//     additional completion.
//
// Returns:
//   IREE_STATUS_OK: One or more completions were processed, an explicit wake
//     was observed, or an internal source or relay made progress. The completed
//     count may be zero for wake and internal progress events.
//   IREE_STATUS_DEADLINE_EXCEEDED: Timeout expired with no completions
//     (not an error—normal for polling loops).
//   IREE_STATUS_ABORTED: Proactor is shutting down.
//
// Example:
//   while (running) {
//     iree_status_t status = iree_async_proactor_poll(
//         proactor, iree_infinite_timeout(), NULL);
//     if (!iree_status_is_ok(status)) return status;
//   }
//   // The thread clearing |running| calls iree_async_proactor_wake().
static inline iree_status_t iree_async_proactor_poll(
    iree_async_proactor_t* proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  return proactor->vtable->poll(proactor, timeout, out_completed_count);
}

// Notifies the proactor that its poll owner will never call poll() again.
//
// Some backends bind kernel resources to the first polling task. Ending that
// task without this notification could strand synchronous callers waiting for
// owner-task work. Backends use this terminal boundary to reject new requests
// and complete accepted requests that the owner can no longer service.
//
// The poll owner must call this exactly when its polling loop permanently
// exits. The standard iree_async_proactor_thread_t runner does so
// automatically. Calling poll() again afterward is invalid. Repeated calls are
// permitted so owner cleanup and final proactor destruction can share paths.
//
// This does not cancel in-flight asynchronous operations or invoke callbacks.
// Those operations must be drained or cancelled before the polling loop exits.
// Registered resources whose teardown may require backend owner-task work must
// also be released before this call. Event-source and relay unregistrations
// must have reached their terminal callbacks, not merely been requested.
static inline void iree_async_proactor_end_polling(
    iree_async_proactor_t* proactor) {
  if (proactor->vtable->end_polling) {
    proactor->vtable->end_polling(proactor);
  }
}

// Wakes a blocked poll() from another thread.
//
// Causes poll() to return immediately with IREE_STATUS_OK and zero completions
// (or whatever completions were already pending). Use this to notify the poll
// thread that new work has been submitted or that shutdown is requested.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// Threading model:
//   Fully thread-safe. May be called from any thread, including signal
//   handlers (async-signal-safe on POSIX). Idempotent—multiple concurrent
//   wake() calls coalesce into a single poll() wakeup.
//
// Implementation:
//   io_uring: IORING_OP_NOP or eventfd write.
//   IOCP: PostQueuedCompletionStatus with NULL overlapped.
//   kqueue: EVFILT_USER trigger.
//   generic: eventfd/pipe write or condition variable signal.
static inline void iree_async_proactor_wake(iree_async_proactor_t* proactor) {
  proactor->vtable->wake(proactor);
}

// Attempts to admit cancellation of a pending operation.
//
// This API has no cancellation-retirement callback. The target's terminal
// callback returns the target execution, not any outstanding cancellation key.
// Callers must keep the identity from being reused while a cancellation can
// still match it. Private connection setup and readiness owners should use
// request_cancel below to explicitly join both lifetimes.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | 5.5+     | yes  | yes
//
// Threading model:
//   Thread-safe. May be called from any thread. The cancellation completion
//   is delivered via the normal poll() path on the poll thread.
//
// Callback expectations:
//   When cancellation wins, the operation's terminal callback fires with:
//     - status: IREE_STATUS_CANCELLED
//     - flags: Does NOT include IREE_ASYNC_COMPLETION_FLAG_MORE
//   For multishot operations, cancellation terminates the operation entirely;
//   no further completions are delivered after the cancelled callback.
//
// Returns:
//   IREE_STATUS_OK: Cancellation admitted. The target may still complete
//     naturally; this does not establish that it was pending or cancelled.
//   IREE_STATUS_NOT_FOUND: The backend could not find a pending target. This
//     does not return ownership; its terminal callback may already be queued.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Cancellation was not admitted. io_uring
//     callers other than the established poll owner can encounter a full
//     submission queue. After the first poll, cancellation on the poll owner
//     makes room without waiting for the target to complete. Platform
//     submission failures still propagate; no failure returns ownership.
//
// Note: Even if cancel() returns OK, the operation may complete successfully
// before the kernel processes the cancellation request. Check the callback
// status to determine the actual outcome.
static inline iree_status_t iree_async_proactor_cancel(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation) {
  return proactor->vtable->cancel(proactor, operation);
}

// Admits an owned cancellation request without allocating or issuing native
// cancellation. Links the request and wakes the proactor. Only the poll owner
// may call this (including before its first poll).
// Native admission/completion errors are returned by poll(), not by the
// receipt. Accepted requests remain owned across poll failures until their
// receipt. No callback runs inline during admission.
//
// The target must have been successfully submitted to this proactor and must
// not yet have delivered its terminal callback. Supported targets are private,
// non-pooled SOCKET_CONNECT, SOCKET_ACCEPT (including multishot), and
// HANDLE_POLL operations. Targets must not participate in LINKED chains or
// sequences. The caller owns this precondition, including for the final link of
// a chain. There must be at most one cancellation request for each target
// execution; callers coalesce duplicate intent and must not mix this with
// cancel().
//
// Until both the receipt and target terminal callback have run, the target's
// storage must not be released or reused. In its terminal callback the owner
// calls cancel_request_target_retired if the receipt is still pending. A MORE
// completion is not terminal. The receipt itself returns request storage and
// carries no operation result; cancellation may lose to natural completion.
IREE_API_EXPORT iree_status_t iree_async_proactor_request_cancel(
    iree_async_proactor_t* proactor, iree_async_operation_t* target,
    iree_async_cancel_request_t* request);

// Reports terminal target completion while the cancellation receipt is pending.
// Poll-owner only. Withdraws an unissued request and runs its receipt inline;
// an issued request remains owned through its native receipt. The callback may
// release the owner's entire state, so callers must not access it afterward.
IREE_API_EXPORT void iree_async_proactor_cancel_request_target_retired(
    iree_async_proactor_t* proactor, iree_async_cancel_request_t* request);

// Backend helpers for owner-thread native cancellation service. Issue unlinks a
// queued request before native completion can refer to it. Complete resets an
// issued request to idle and invokes its receipt; no access is permitted after
// that callback. Neither helper accesses the target operation.
void iree_async_proactor_issue_cancel_request(
    iree_async_proactor_t* proactor, iree_async_cancel_request_t* request);
void iree_async_cancel_request_complete(iree_async_cancel_request_t* request);

// Maximum requests issued by one native service pass. Remaining work keeps its
// FIFO ownership and wakes the next poll turn; this is not a lifetime bound.
#define IREE_ASYNC_CANCEL_REQUEST_SERVICE_BUDGET 64

//===----------------------------------------------------------------------===//
// Buffer registration
//===----------------------------------------------------------------------===//
//
// Ownership model:
//   The proactor ALLOCATES the entry (including any backend-specific
//   trailing data) during register_buffer/register_dmabuf. The entry is
//   linked into the caller's |state| list and returned via |out_entry|.
//
//   The entry holds a reference to its iree_async_region_t and sets a
//   cleanup_fn that releases the region reference and frees the entry's
//   memory. Cleanup happens in one of two ways:
//     - Explicit: caller calls unregister_buffer, which removes the entry
//       from |state| and invokes cleanup_fn.
//     - Automatic: buffer destruction calls
//       iree_async_buffer_registration_state_deinitialize(), which walks the
//       list and invokes each entry's cleanup_fn.
//
//   The caller MUST NOT free entry memory directly — always go through
//   unregister_buffer or state_deinitialize.
//
//   The proactor MUST outlive all registrations. If the proactor is
//   destroyed before entries are cleaned up, cleanup_fn receives a stale
//   proactor pointer (undefined behavior). In practice: unregister or
//   destroy all registered buffers before releasing the proactor.
//
// Thread safety:
//   Registration/unregistration must be serialized with respect to
//   |state| (typically single-threaded during setup). The entry's region
//   and backend handles are immutable after registration, so I/O
//   operations can read them concurrently from any thread.
//
// See iree/async/types.h for the entry and state type definitions.

// Registers host memory for zero-copy I/O.
//
// Pre-registering buffers with the kernel enables DMA directly to/from
// application memory, eliminating copy overhead. This is the critical path
// for achieving line-rate network throughput and NVMe bandwidth.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.1+     | reg  | emul
//
// Registration limits:
//   io_uring: Limited by IORING_MAX_FIXED_BUFFERS (typically 32768).
//   IOCP: No hard limit; each registration pins pages.
//   Others: Emulated; no kernel-side benefit but API is consistent.
//
// Buffer lifetime:
//   The memory referenced by |buffer| must remain valid and at a stable
//   address for the lifetime of the registration. The proactor does not
//   copy or own the memory—it just registers the address range with the
//   kernel for DMA access.
//
// Ownership:
//   The proactor allocates the entry (including backend-specific trailing
//   data), links it into |state|, and returns it via |out_entry|. The
//   entry's cleanup_fn handles deallocation when unregistered or when
//   the owning buffer is destroyed via state_cleanup().
//
// Returns:
//   IREE_STATUS_OK: Buffer registered successfully.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Registration table full.
//   IREE_STATUS_INVALID_ARGUMENT: Buffer address not suitably aligned.
//
// See also: iree_async_proactor_unregister_buffer()
static inline iree_status_t iree_async_proactor_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return proactor->vtable->register_buffer(proactor, state, buffer,
                                           access_flags, out_entry);
}

// Registers memory exported as a Linux DMA buffer for asynchronous I/O.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   n/a     | 5.7+     | n/a  | n/a
//
// The io_uring backend maps the DMA buffer for CPU access. Linux 5.19+ may
// additionally place READ regions in the sparse fixed-buffer table when the
// mapping supports long-term writable page pins. Fixed registration and
// SEND_ZC are optimizations: unsupported mappings remain usable through the
// copy path. Device-direct DMA-buf networking and storage are not implemented.
//
// Parameters:
//   dmabuf_fd: DMA-BUF file descriptor exported by the GPU driver.
//   offset: Byte offset within the dmabuf to start registration.
//   length: Number of bytes to register starting from offset.
//   access_flags: READ for send paths, WRITE for receive paths.
//
// Ownership:
//   Same model as register_buffer: the proactor allocates the entry, sets
//   cleanup_fn, and the entry is freed via unregister or state_deinitialize.
//   The proactor does NOT take ownership of dmabuf_fd—caller manages its
//   lifetime and must keep it open until unregistration.
//
// Returns:
//   IREE_STATUS_OK: dmabuf registered successfully.
//   IREE_STATUS_UNAVAILABLE: Backend does not support dmabuf (use
//     IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF to check beforehand).
//   IREE_STATUS_INVALID_ARGUMENT: Invalid fd, zero length, or no local access.
//   IREE_STATUS_OUT_OF_RANGE: The offset and length cannot be represented by
//     the host mapping API.
static inline iree_status_t iree_async_proactor_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return proactor->vtable->register_dmabuf(proactor, state, dmabuf_fd, offset,
                                           length, access_flags, out_entry);
}

// Unregisters a previously registered buffer (host or dmabuf).
//
// Removes |entry| from |state|, invokes entry->cleanup_fn (which releases
// the region reference and frees the entry memory), and releases any
// backend resources (RDMA MR deregistration, io_uring buffer table slot).
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// In-flight safety:
//   Must not be called while I/O operations referencing this entry's region
//   are in flight. While operations retain the region via span (so the memory
//   is safe), the backend may need to deregister the buffer table slot,
//   which can corrupt in-flight DMA transfers.
//
//   Safe pattern: Cancel all operations using this buffer, wait for their
//   cancellation callbacks, then unregister.
//
// Post-call state:
//   The entry pointer is invalid after this call. Do not access it.
//   The underlying memory (for host buffers) is not freed—only the
//   registration is removed. The caller may continue using the memory
//   for non-registered I/O or free it.
static inline void iree_async_proactor_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  proactor->vtable->unregister_buffer(proactor, entry, state);
}

//===----------------------------------------------------------------------===//
// Slab registration (indexed zero-copy)
//===----------------------------------------------------------------------===//
//
// Ownership:
//   The caller owns the returned region. The region holds a retained
//   reference to the slab. Deregistration happens automatically when the
//   region's ref count reaches zero (via the region's destroy_fn callback).
//   No explicit unregister call is needed.
//
// Thread safety:
//   Registration and region release are thread-safe. Backends that bind
//   registration to the poll owner synchronously dispatch the kernel operation
//   to that task. All regions must be released before the poll owner calls
//   iree_async_proactor_end_polling and before the proactor is released.

// Registers a slab for indexed zero-copy I/O.
//
// Creates a region with backend-specific handles that enable zero-copy
// operations across all buffers in the slab. This is more efficient than
// registering individual buffers when using a slab-based allocation pattern.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   emul    | 5.1+     | emul | emul
//
// Backend behavior:
//   io_uring (READ): Uses a sparse fixed-buffer table on Linux 5.19+ and a
//     singleton table on older supported kernels. SEND_ZC operations derive
//     buf_index from the span offset at fill time. Multiple READ registrations
//     may coexist when sparse tables are available.
//   io_uring (WRITE): Creates a provided buffer ring (PBUF_RING) for
//     kernel-managed buffer selection in multishot recv.
//   Others: Emulated; slab structure tracked but no kernel registration.
//
// Ownership:
//   The caller owns the returned region. The region holds a retained
//   reference to the slab. Deregistration happens automatically when the
//   region's ref count reaches zero (via the region's destroy_fn callback).
//   No explicit unregister call is needed.
//
// Returns:
//   IREE_STATUS_OK: Slab registered successfully.
//   IREE_STATUS_ALREADY_EXISTS: A READ slab already owns the singleton fixed
//     buffer table on an older io_uring kernel.
//   IREE_STATUS_RESOURCE_EXHAUSTED: The backend registration table is full.
static inline iree_status_t iree_async_proactor_register_slab(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  return proactor->vtable->register_slab(proactor, slab, access_flags,
                                         out_region);
}

//===----------------------------------------------------------------------===//
// Event source registration (persistent handle monitoring)
//===----------------------------------------------------------------------===//
//
// Event sources provide persistent monitoring of external handles (file
// descriptors, RDMA completion queue channels, etc.) with callback-based
// notification. Unlike one-shot poll operations, event sources remain armed
// after each callback and continue monitoring until explicitly unregistered.
//
// Common use cases:
//   - RDMA CQ channel fd monitoring (notify on completion queue events)
//   - External device fd monitoring (GPUs, accelerators)
//   - Integration with external event loops
//
// Ownership model:
//   The handle is NOT owned by the event source. The caller retains ownership
//   and must retain it through terminal unregistration completion. The
//   proactor allocates internal tracking structures during registration and
//   frees them during unregistration.
//
// Thread safety:
//   Registration and unregistration must be serialized with respect to
//   poll(). Callbacks fire from within poll() on the polling thread.
//   Do NOT call unregister_event_source from within the callback.

// Registers an external handle for persistent event monitoring.
//
// The handle is monitored for readability (POLLIN). When the handle becomes
// readable, the callback fires from within poll() on the polling thread. The
// event source remains armed after the callback returns (multishot behavior);
// no re-registration is needed for subsequent events.
//
// Availability:
//   generic | io_uring | IOCP     | kqueue
//   poll    | 5.19+    | Win 8.1+ | poll
//
// io_uring implementation uses multishot POLL_ADD for efficient persistent
// monitoring without per-event syscalls.
//
// Parameters:
//   handle: The external handle to monitor. POSIX backends require an fd;
//     IOCP requires a waitable Win32 HANDLE.
//   callback: Function to invoke when the handle is signaled. The callback
//     receives poll events and should drain the handle (e.g., ibv_poll_cq for
//     RDMA CQ channels) and re-arm if needed (e.g., ibv_req_notify_cq).
//     For an iree_async_event_t, use iree_async_event_consume() before checking
//     the state announced by the event; it accounts for native auto-reset.
//   out_event_source: Receives the event source handle for later
//     unregistration.
//
// Returns:
//   IREE_STATUS_OK: Event source registered successfully.
//   IREE_STATUS_INVALID_ARGUMENT: Invalid handle or NULL callback.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Too many event sources registered.
//   IREE_STATUS_UNAVAILABLE: The backend cannot monitor the handle directly.
static inline iree_status_t iree_async_proactor_register_event_source(
    iree_async_proactor_t* proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  return proactor->vtable->register_event_source(proactor, handle, callback,
                                                 out_event_source);
}

// Begins terminal event-source unregistration and stops callback admission.
//
// The ordinary event callback will not fire again after this call returns.
// |callback| fires after all native monitoring and cancellation operations have
// retired and the event source has been destroyed. The caller must retain the
// borrowed handle and ordinary callback context until this completion, which
// may fire inline. The source handle becomes invalid immediately and must not
// be used or unregistered again. A NULL source completes inline.
//
// Deferred completions run from poll() on the polling thread. Proactor
// destruction also completes any unregistration it owns before returning.
// The proactor must remain alive to drive retirement and must not be released
// from within the completion callback.
//
// Must NOT be called from within the event source's callback. If you need to
// unregister from a callback, defer the unregistration to the next poll()
// iteration (e.g., via a flag check in your poll loop).
//
// Thread safety:
//   Must be called from the proactor's poll thread (same thread that calls
//   poll()). Not thread-safe with respect to poll().
static inline void iree_async_proactor_unregister_event_source(
    iree_async_proactor_t* proactor, iree_async_event_source_t* event_source,
    iree_async_event_source_unregistered_callback_t callback) {
  if (!event_source) {
    if (callback.fn) {
      callback.fn(callback.user_data);
    }
    return;
  }
  proactor->vtable->unregister_event_source(proactor, event_source, callback);
}

//===----------------------------------------------------------------------===//
// Relay registration (event-to-event bridging)
//===----------------------------------------------------------------------===//

// Registers a relay that triggers |sink| when |source| is signaled.
//
// The relay is active immediately after registration. The proactor monitors
// the source and fires the sink from within poll() when triggered.
//
// For PERSISTENT relays, the source is re-armed after each trigger and the
// relay remains active until iree_async_proactor_unregister_relay() is called.
// For one-shot relays (no PERSISTENT flag), the relay auto-cleans up after
// firing once.
//
// Error handling:
//   If a sink transfer fails or a persistent relay cannot re-arm its source,
//   |error_callback| is invoked with the error code. Persistent relay handles
//   remain valid for terminal unregistration after poll() returns. Pass
//   iree_async_relay_error_callback_none() only when terminal relay failure is
//   intentionally unobserved.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   yes     | yes      | yes  | yes
//
// io_uring optimizations:
//   When LINK chains are supported, certain source/sink combinations can
//   execute entirely in kernel space without userspace round-trips:
//     - PRIMITIVE → SIGNAL_PRIMITIVE: POLL_ADD → LINK → WRITE
//     - PRIMITIVE → SIGNAL_NOTIFICATION (futex): POLL_ADD → LINK → FUTEX_WAKE
//     - NOTIFICATION (futex) → SIGNAL_NOTIFICATION (futex): kernel chain
//
// Parameters:
//   proactor: The proactor to register with.
//   source: Event source specification.
//   sink: Event sink specification.
//   flags: Behavioral flags (PERSISTENT, OWN_SOURCE_PRIMITIVE,
//     ERROR_SENSITIVE).
//   error_callback: Called if the relay fails to re-arm. May be _none().
//   out_relay: Receives the relay handle for later unregistration.
//
// Returns:
//   IREE_STATUS_OK: Relay registered successfully.
//   IREE_STATUS_INVALID_ARGUMENT: Invalid source or sink specification.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Too many relays or SQ full.
static inline iree_status_t iree_async_proactor_register_relay(
    iree_async_proactor_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  return proactor->vtable->register_relay(proactor, source, sink, flags,
                                          error_callback, out_relay);
}

// Begins terminal relay unregistration and stops monitoring.
//
// |callback| fires after the sink can no longer fire, all backend operations
// referencing the relay have completed, and the relay has been destroyed. If
// OWN_SOURCE_PRIMITIVE was set, the source primitive has been closed before
// the callback fires. The callback may fire synchronously from this call.
// The relay handle becomes invalid immediately and must not be used or
// unregistered again.
// Deferred callbacks normally fire from iree_async_proactor_poll() on the
// polling thread. Proactor destruction also completes any unregistration it
// owns before returning. The proactor must not be released from inside the
// callback.
//
// The caller must keep source and sink resources not owned by the relay alive
// until |callback| fires. The proactor must also remain alive so asynchronous
// backends can dispatch terminal cancellation completions.
//
// Must NOT be called from within a relay sink's callback (the sink fires
// from poll(), which holds internal state). Defer unregistration if needed.
//
// For persistent relays, this is the only way to stop monitoring.
// For one-shot relays, this can be used to cancel before the relay fires.
//
// Thread safety:
//   Must be called from the proactor's poll thread.
static inline void iree_async_proactor_unregister_relay(
    iree_async_proactor_t* proactor, iree_async_relay_t* relay,
    iree_async_relay_unregistered_callback_t callback) {
  if (!relay) {
    if (callback.fn) {
      callback.fn(callback.user_data);
    }
    return;
  }
  proactor->vtable->unregister_relay(proactor, relay, callback);
}

//===----------------------------------------------------------------------===//
// Cross-proactor messaging
//===----------------------------------------------------------------------===//

// Sets the callback invoked when a message arrives from another proactor.
//
// Cross-proactor messaging enables efficient communication between proactors
// running on different threads. Messages can arrive via backend-specific paths:
//   - io_uring MSG_RING: Kernel posts a CQE directly to the target ring.
//   - IOCP/POSIX/io_uring fallback: Bounded message pool plus target wake.
//
// All paths deliver messages through this callback during poll().
//
// Only one callback may be registered at a time; subsequent calls replace the
// previous callback. Pass a callback with NULL fn to disable message delivery.
//
// Must be called before the poll loop starts or from within a poll() callback
// (not thread-safe with poll).
static inline void iree_async_proactor_set_message_callback(
    iree_async_proactor_t* proactor,
    iree_async_proactor_message_callback_t callback) {
  proactor->vtable->set_message_callback(proactor, callback);
}

// Sends a message to a proactor from any thread.
//
// This is a thread-safe, fire-and-forget message send that can be called from
// any context (proactor thread, worker thread, signal handler). The target
// proactor's registered message callback will be invoked during its next
// poll() call.
//
// Unlike the operation-based MESSAGE submit (which supports LINK chains and
// completion callbacks), this function is optimized for simplicity:
//   - No completion callback (fire-and-forget semantics).
//   - No LINK chain support.
//   - Minimal overhead: backend-specific (kernel CQE, pool+wake, etc.).
//
// Use cases:
//   - Shutdown signals from main thread to I/O thread.
//   - Work injection from worker threads.
//   - Cross-thread event notification.
//
// Threading model:
//   Fully thread-safe. May be called from any thread, including from within
//   completion callbacks. Multiple concurrent sends are safe.
//
// Parameters:
//   target: The proactor to receive the message. Must remain valid until the
//     message is delivered (i.e., until target's poll() processes it).
//   message_data: Arbitrary 64-bit payload delivered to the callback.
//
// Returns:
//   IREE_STATUS_OK: Message queued for delivery.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Backend-specific resource limit reached.
//     Retry after the target proactor's next poll() drains pending messages.
static inline iree_status_t iree_async_proactor_send_message(
    iree_async_proactor_t* target, uint64_t message_data) {
  return target->vtable->send_message(target, message_data);
}

//===----------------------------------------------------------------------===//
// Signal subscription
//===----------------------------------------------------------------------===//

// Returns true if |proactor| supports signal handling on this platform.
// All current backends (POSIX, io_uring, IOCP) support signal handling,
// though with different signal type availability (Windows only supports
// INTERRUPT and TERMINATE; POSIX additionally supports HANGUP, QUIT, USER1,
// USER2). Use this to check if signal-based functionality will actually work
// before relying on it.
static inline bool iree_async_proactor_supports_signals(
    iree_async_proactor_t* proactor) {
  return proactor->vtable->subscribe_signal != NULL;
}

// Subscribes to |signal| on |proactor|, invoking |callback| each time the
// signal is received. The |out_subscription| handle must eventually be passed
// to iree_async_proactor_unsubscribe_signal() or released implicitly when the
// proactor is destroyed.
//
// Only one proactor per process may own signal subscriptions. The first
// proactor to call this function claims ownership; calls from other proactors
// return IREE_STATUS_FAILED_PRECONDITION. Ownership persists for the lifetime
// of the process (never released).
//
// Multiple subscriptions to the same signal are allowed; all callbacks fire in
// registration order. Callbacks fire from within poll() on the polling thread.
//
// Must be serialized with poll(). Safe to call before the poll loop starts,
// after it exits, or from within poll() callbacks. Not safe to call from
// another thread while poll() is running.
//
// Returns IREE_STATUS_FAILED_PRECONDITION if another proactor owns signals, or
// IREE_STATUS_INVALID_ARGUMENT if |signal| is out of range. If the platform
// does not support signal handling this is a no-op that returns OK with
// |*out_subscription| set to NULL.
static inline iree_status_t iree_async_proactor_subscribe_signal(
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  if (!proactor->vtable->subscribe_signal) {
    *out_subscription = NULL;
    return iree_ok_status();
  }
  return proactor->vtable->subscribe_signal(proactor, signal, callback,
                                            out_subscription);
}

// Unsubscribes |subscription| from signal delivery on |proactor|. After this
// call the callback will not fire again and the subscription handle becomes
// invalid.
//
// Safe to call with NULL |subscription| (no-op), allowing cleanup code to
// unconditionally unsubscribe without checking whether subscribe succeeded.
//
// Must be serialized with poll(). When called from within a signal callback,
// the unsubscription is deferred until dispatch completes to avoid corrupting
// the subscription list during iteration.
static inline void iree_async_proactor_unsubscribe_signal(
    iree_async_proactor_t* proactor,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) {
    return;  // NULL-safe no-op.
  }
  if (proactor->vtable->unsubscribe_signal) {
    proactor->vtable->unsubscribe_signal(proactor, subscription);
  }
}

//===----------------------------------------------------------------------===//
// Process-wide signal utilities
//===----------------------------------------------------------------------===//

// Blocks signals that will be handled via proactor subscriptions.
//
// Call BEFORE creating any threads so child threads inherit the blocked mask.
// This prevents signals from being delivered with default handlers (which
// terminate the process) before proactor signal handling is set up.
//
// Blocks: SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2.
// On Windows this is a no-op (console events don't use signal masks).
IREE_API_EXPORT iree_status_t iree_async_signal_block_default(void);

// Ignores SIGPIPE globally.
//
// Call once at startup to prevent SIGPIPE from terminating the process when
// writing to a closed socket/pipe. Essential for any network server—the
// correct way to detect broken pipes is via EPIPE return from write().
// On Windows this is a no-op (no SIGPIPE equivalent).
IREE_API_EXPORT iree_status_t iree_async_signal_ignore_broken_pipe(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PROACTOR_H_
