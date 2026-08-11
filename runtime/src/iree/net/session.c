// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/session.h"

#include <string.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/bootstrap.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/transport_factory.h"

// Maximum peer-provided REJECT reason bytes included in a local status
// message. The parser retains the complete reason; bounding only the diagnostic
// avoids amplifying an untrusted payload into a second large allocation.
#define IREE_NET_SESSION_REJECT_REASON_DIAGNOSTIC_BYTE_LIMIT 1024

//===----------------------------------------------------------------------===//
// Internal types
//===----------------------------------------------------------------------===//

// Session role determines bootstrap protocol behavior.
typedef enum iree_net_session_role_e {
  IREE_NET_SESSION_ROLE_CLIENT = 0,
  IREE_NET_SESSION_ROLE_SERVER = 1,
} iree_net_session_role_t;

// Internal bootstrap sub-phases within BOOTSTRAPPING state.
typedef enum iree_net_session_bootstrap_phase_e {
  // Client: factory.connect() in progress.
  IREE_NET_SESSION_BOOTSTRAP_CONNECTING = 0,
  // Both: connection.open_endpoint() for control channel in progress.
  IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL = 1,
  // Client: HELLO sent, waiting for HELLO_ACK or REJECT.
  IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT = 2,
  // Server: control channel active, waiting for client HELLO.
  IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO = 3,
} iree_net_session_bootstrap_phase_t;

// Retains an application endpoint-ready callback through async dispatch.
typedef struct iree_net_session_endpoint_callback_t {
  // Session retained until the endpoint callback returns.
  iree_net_session_t* session;

  // Application callback admitted before endpoint submission.
  iree_net_endpoint_ready_callback_t callback;
} iree_net_session_endpoint_callback_t;

typedef enum iree_net_session_submission_kind_e {
  IREE_NET_SESSION_SUBMISSION_ENDPOINT = 0,
  IREE_NET_SESSION_SUBMISSION_DATA_SEND = 1,
  IREE_NET_SESSION_SUBMISSION_SHUTDOWN = 2,
} iree_net_session_submission_kind_t;

//===----------------------------------------------------------------------===//
// iree_net_session_t
//===----------------------------------------------------------------------===//

struct iree_net_session_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  iree_net_session_role_t role;
  iree_net_session_bootstrap_phase_t bootstrap_phase;

  // External state (atomic for cross-thread reads).
  iree_atomic_int32_t state;

  // Application callbacks.
  // Protected by |callback_mutex|.
  iree_net_session_callbacks_t callbacks;

  // Serializes application callback admission, transport submission, and
  // connection deactivation.
  iree_slim_mutex_t callback_mutex;

  // Number of application callbacks admitted and not yet returned.
  // Protected by |callback_mutex|.
  uint32_t callback_count;

  // Number of admitted transport calls that have not returned from their
  // synchronous submission boundary.
  // Protected by |callback_mutex|.
  uint32_t transport_submission_count;

  // True after application callback detachment begins.
  // Protected by |callback_mutex|.
  bool callbacks_detached;

  // Completion pending until |callback_count| reaches zero.
  // Protected by |callback_mutex|.
  iree_net_session_callbacks_detached_callback_t detach_callback;

  // True after connection deactivation has claimed the submission gate.
  // Protected by |callback_mutex|.
  bool connection_deactivation_requested;

  // True after connection deactivation was submitted to the connection.
  // Protected by |callback_mutex|.
  bool connection_deactivation_submitted;

  // True after the connection and all endpoints have drained.
  // Protected by |callback_mutex|.
  bool connection_deactivated;

  // True when the final reference was released during connection drain.
  // Protected by |callback_mutex|.
  bool destroy_after_deactivation;

  // Application completion pending for explicit session deactivation.
  // Protected by |callback_mutex|.
  iree_net_session_deactivated_callback_t deactivated_callback;

  // Protocol version offered or accepted during bootstrap.
  uint32_t protocol_version;
  // Capability bits advertised during bootstrap.
  uint32_t capabilities;
  // Capability bits required in the negotiated peer intersection.
  uint32_t required_capabilities;
  // Number of application endpoint slots reserved during setup.
  uint32_t application_endpoint_count;
  // Bootstrap timeout in nanoseconds.
  iree_duration_t bootstrap_timeout_ns;

  // Server-assigned session identifier.
  uint64_t session_id;

  // Transport factory (retained, client path only — NULL for server).
  iree_net_transport_factory_t* transport_factory;

  // Connection (retained).
  iree_net_connection_t* connection;

  // Control channel (owned by session).
  iree_net_control_channel_t* control_channel;

  // Frontier tracker retained for remote axis registration and cleanup.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Proactor retained for bootstrap timers and cancellation.
  iree_async_proactor_t* proactor;

  // Bootstrap timeout timer. Non-NULL while the timer is in flight.
  // Allocated at session creation, freed in the timer completion callback
  // (either expiry or cancellation). NULL means no timer pending. Protected by
  // |callback_mutex| so deactivation can join timer retirement exactly.
  iree_async_timer_operation_t* bootstrap_timer;

  // Retained receive buffer pool (client path only).
  iree_async_buffer_pool_t* recv_pool;

  // Server address (copied, client path only — for factory.connect).
  char* server_address_storage;
  iree_string_view_t server_address;

  // Local topology (copied at creation).
  iree_async_axis_t* local_axes;
  uint64_t* local_epochs;
  uint32_t local_axis_count;
  uint8_t local_machine_index;
  uint8_t local_session_epoch;
  uint8_t* local_application_data;
  iree_host_size_t local_application_data_length;

  // Remote topology (allocated during bootstrap).
  iree_async_axis_t* remote_axes;
  uint64_t* remote_epochs;
  uint32_t remote_axis_count;
  uint8_t remote_machine_index;
  uint8_t remote_session_epoch;

  // Proxy semaphores for remote axes (parallel array with remote_axes).
  // Owned by the session; released on shutdown/destroy.
  iree_async_semaphore_t** proxy_semaphores;

  // Negotiated capabilities (set during bootstrap).
  iree_net_bootstrap_capabilities_t negotiated_capabilities;
};

// Claims a snapshot of the application callback set. Detachment prevents new
// claims while allowing callbacks already admitted to finish against their
// snapshot.
static bool iree_net_session_begin_callback(
    iree_net_session_t* session, iree_net_session_callbacks_t* out_callbacks) {
  iree_slim_mutex_lock(&session->callback_mutex);
  bool callback_active = !session->callbacks_detached;
  if (callback_active) {
    ++session->callback_count;
    *out_callbacks = session->callbacks;
  } else {
    memset(out_callbacks, 0, sizeof(*out_callbacks));
  }
  iree_slim_mutex_unlock(&session->callback_mutex);
  return callback_active;
}

// Returns true when explicit deactivation has retired every session-owned
// asynchronous operation and application callback. The caller must hold
// |callback_mutex|.
static bool iree_net_session_deactivation_ready_locked(
    iree_net_session_t* session) {
  return session->connection_deactivation_requested &&
         (!session->connection || session->connection_deactivated) &&
         session->callback_count == 0 && session->bootstrap_timer == NULL;
}

// Claims an explicit deactivation callback once its completion boundary has
// been reached. The caller must hold |callback_mutex|.
static iree_net_session_deactivated_callback_t
iree_net_session_take_deactivated_callback_locked(iree_net_session_t* session) {
  iree_net_session_deactivated_callback_t callback = {0};
  if (iree_net_session_deactivation_ready_locked(session) &&
      session->deactivated_callback.fn) {
    callback = session->deactivated_callback;
    memset(&session->deactivated_callback, 0,
           sizeof(session->deactivated_callback));
  }
  return callback;
}

// Releases an admitted application callback. The detach completion is taken
// under the mutex and invoked afterward because it may release the session and
// its callback target.
static void iree_net_session_end_callback(iree_net_session_t* session) {
  iree_net_session_callbacks_detached_callback_t detach_callback = {0};
  iree_net_session_deactivated_callback_t deactivated_callback = {0};
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(session->callback_count > 0);
  --session->callback_count;
  if (session->callback_count == 0) {
    if (session->callbacks_detached) {
      memset(&session->callbacks, 0, sizeof(session->callbacks));
      if (session->detach_callback.fn) {
        detach_callback = session->detach_callback;
        memset(&session->detach_callback, 0, sizeof(session->detach_callback));
      }
    }
    deactivated_callback =
        iree_net_session_take_deactivated_callback_locked(session);
    if (session->connection_deactivated && !deactivated_callback.fn &&
        !session->bootstrap_timer) {
      IREE_ASSERT(!session->destroy_after_deactivation,
                  "implicit session teardown completed before its final "
                  "application callback returned");
    }
  }
  iree_slim_mutex_unlock(&session->callback_mutex);
  if (deactivated_callback.fn) {
    deactivated_callback.fn(deactivated_callback.user_data);
  }
  if (detach_callback.fn) {
    detach_callback.fn(detach_callback.user_data);
  }
}

static void iree_net_session_on_application_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_net_session_endpoint_callback_t* context =
      (iree_net_session_endpoint_callback_t*)user_data;
  iree_net_session_t* session = context->session;
  iree_net_endpoint_ready_callback_t callback = context->callback;
  iree_allocator_free(session->host_allocator, context);

  callback.fn(callback.user_data, status, endpoint);
  iree_net_session_end_callback(session);
  iree_net_session_release(session);
}

static void iree_net_session_cleanup_remote_axes(iree_net_session_t* session);

static iree_status_t iree_net_session_validate_endpoint_capacity(
    iree_net_session_t* session, iree_net_connection_t* connection) {
  const uint32_t required_endpoint_count =
      session->application_endpoint_count + 1u;
  const uint32_t max_endpoint_count =
      iree_net_connection_max_endpoint_count(connection);
  if (max_endpoint_count < required_endpoint_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "connection has %u endpoint slots but session requires %u "
        "(1 control + %u application)",
        max_endpoint_count, required_endpoint_count,
        session->application_endpoint_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_session_validate_required_capabilities(
    const iree_net_session_t* session,
    iree_net_bootstrap_capabilities_t negotiated_capabilities) {
  iree_net_bootstrap_capabilities_t missing_capabilities =
      session->required_capabilities & ~negotiated_capabilities;
  if (missing_capabilities == IREE_NET_BOOTSTRAP_CAPABILITY_NONE) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "required session capabilities were not negotiated: required=0x%08x "
      "negotiated=0x%08x missing=0x%08x",
      session->required_capabilities, negotiated_capabilities,
      missing_capabilities);
}

//===----------------------------------------------------------------------===//
// State helpers
//===----------------------------------------------------------------------===//

static iree_net_session_state_t iree_net_session_load_state(
    const iree_net_session_t* session) {
  return (iree_net_session_state_t)iree_atomic_load(
      &((iree_net_session_t*)session)->state, iree_memory_order_acquire);
}

static void iree_net_session_set_state(iree_net_session_t* session,
                                       iree_net_session_state_t new_state) {
  iree_atomic_store(&session->state, (int32_t)new_state,
                    iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// Bootstrap timeout timer
//===----------------------------------------------------------------------===//

// Timer completion callback. Fires on either expiry or cancellation.
static void iree_net_session_bootstrap_timer_completion(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;

  if (!iree_status_is_ok(status)) iree_status_abort(status);

  // Retire the operation before processing its outcome. This prevents an
  // expiry from trying to cancel itself through the generic failure path and
  // allows deactivation to observe the exact timer boundary.
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(session->bootstrap_timer ==
              (iree_async_timer_operation_t*)operation);
  session->bootstrap_timer = NULL;
  iree_net_session_deactivated_callback_t deactivated_callback =
      iree_net_session_take_deactivated_callback_locked(session);
  iree_slim_mutex_unlock(&session->callback_mutex);
  iree_allocator_free(session->host_allocator, operation);

  if (!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED) &&
      iree_net_session_load_state(session) ==
          IREE_NET_SESSION_STATE_BOOTSTRAPPING) {
    // Timer expired before bootstrap reached a terminal state.
    iree_net_session_fail(
        session,
        iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED, "bootstrap timed out"));
  }

  if (deactivated_callback.fn) {
    deactivated_callback.fn(deactivated_callback.user_data);
  }
  iree_net_session_release(session);
}

// Starts the bootstrap timeout timer. Must be called after session->proactor
// is set. The timer retains the session to prevent use-after-free if the
// session is released while the timer is in flight.
static iree_status_t iree_net_session_start_bootstrap_timer(
    iree_net_session_t* session, iree_duration_t timeout_ns) {
  if (timeout_ns == 0)
    timeout_ns = IREE_NET_SESSION_DEFAULT_BOOTSTRAP_TIMEOUT_NS;

  iree_async_timer_operation_t* timer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(session->host_allocator,
                                             sizeof(*timer), (void**)&timer));
  memset(timer, 0, sizeof(*timer));

  iree_async_operation_initialize(
      &timer->base, IREE_ASYNC_OPERATION_TYPE_TIMER,
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
      iree_net_session_bootstrap_timer_completion, session);
  timer->deadline_ns = iree_relative_timeout_to_deadline_ns(timeout_ns);

  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(!session->bootstrap_timer);
  session->bootstrap_timer = timer;
  iree_slim_mutex_unlock(&session->callback_mutex);
  iree_net_session_retain(session);

  iree_status_t status =
      iree_async_proactor_submit_one(session->proactor, &timer->base);
  if (!iree_status_is_ok(status)) {
    // Submit failed; retire the operation synchronously because its callback
    // will not fire.
    iree_slim_mutex_lock(&session->callback_mutex);
    IREE_ASSERT(session->bootstrap_timer == timer);
    session->bootstrap_timer = NULL;
    iree_net_session_deactivated_callback_t deactivated_callback =
        iree_net_session_take_deactivated_callback_locked(session);
    iree_slim_mutex_unlock(&session->callback_mutex);
    iree_allocator_free(session->host_allocator, timer);
    iree_net_session_release(session);
    if (deactivated_callback.fn) {
      deactivated_callback.fn(deactivated_callback.user_data);
    }
    return status;
  }

  return iree_ok_status();
}

// Cancels the bootstrap timer if pending. Safe to call when no timer is active.
static void iree_net_session_cancel_bootstrap_timer(
    iree_net_session_t* session) {
  iree_slim_mutex_lock(&session->callback_mutex);
  iree_async_timer_operation_t* timer = session->bootstrap_timer;
  if (!timer) {
    iree_slim_mutex_unlock(&session->callback_mutex);
    return;
  }
  iree_status_t status =
      iree_async_proactor_cancel(session->proactor, &timer->base);
  iree_slim_mutex_unlock(&session->callback_mutex);
  if (iree_status_is_ok(status)) return;
  if (iree_status_is_not_found(status)) {
    // The timer completion won the race and owns the operation until its
    // callback runs.
    iree_status_free(status);
    return;
  }
  // The timer owns a session reference until its callback fires. There is no
  // safe path that can forget an operation after cancellation itself fails.
  iree_status_abort(status);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_net_session_fail(iree_net_session_t* session,
                                           iree_status_t status) {
  // Atomically transition to ERROR. The CAS loop handles state progression
  // (e.g., BOOTSTRAPPING → OPERATIONAL between our load and the exchange).
  int32_t expected =
      iree_atomic_load(&session->state, iree_memory_order_acquire);
  do {
    if (expected == (int32_t)IREE_NET_SESSION_STATE_ERROR ||
        expected == (int32_t)IREE_NET_SESSION_STATE_CLOSED) {
      // Another thread already moved to a terminal state.
      iree_status_free(status);
      return;
    }
  } while (!iree_atomic_compare_exchange_weak(
      &session->state, &expected, (int32_t)IREE_NET_SESSION_STATE_ERROR,
      iree_memory_order_acq_rel, iree_memory_order_acquire));

  // CAS succeeded — we own the ERROR transition. Cancel the bootstrap timer
  // if it's still in flight. The timer callback will fire with CANCELLED and
  // release its session ref.
  iree_net_session_cancel_bootstrap_timer(session);

  // Fail remote axes in the tracker and release proxy semaphores immediately
  // so that waiters depending on remote axes are woken with errors rather
  // than hanging indefinitely.
  iree_net_session_cleanup_remote_axes(session);

  iree_net_session_callbacks_t callbacks;
  if (iree_net_session_begin_callback(session, &callbacks)) {
    if (callbacks.on_error) {
      callbacks.on_error(callbacks.user_data, session, status);
    } else {
      iree_status_free(status);
    }
    iree_net_session_end_callback(session);
  } else {
    iree_status_free(status);
  }
}

//===----------------------------------------------------------------------===//
// Proxy semaphore management
//===----------------------------------------------------------------------===//

// Creates proxy semaphores for remote axes and registers them in the
// frontier_tracker. Called during bootstrap after receiving the peer's
// topology.
static iree_status_t iree_net_session_register_remote_axes(
    iree_net_session_t* session,
    const iree_net_bootstrap_axis_list_t* axis_list) {
  uint32_t axis_count = axis_list->count;
  if (axis_count == 0) return iree_ok_status();

  // Allocate parallel arrays for remote topology and proxy semaphores.
  iree_host_size_t axes_size = axis_count * sizeof(iree_async_axis_t);
  iree_host_size_t epochs_size = axis_count * sizeof(uint64_t);
  iree_host_size_t semaphores_size =
      axis_count * sizeof(iree_async_semaphore_t*);
  iree_host_size_t total_size = axes_size + epochs_size + semaphores_size;

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(session->host_allocator,
                                             total_size, (void**)&storage));
  memset(storage, 0, total_size);

  session->remote_axes = (iree_async_axis_t*)storage;
  session->remote_epochs = (uint64_t*)(storage + axes_size);
  session->proxy_semaphores =
      (iree_async_semaphore_t**)(storage + axes_size + epochs_size);
  session->remote_axis_count = axis_count;

  // Create proxy semaphores and register axes.
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < axis_count && iree_status_is_ok(status); ++i) {
    iree_net_bootstrap_axis_entry_t entry =
        iree_net_bootstrap_axis_list_get(axis_list, i);
    session->remote_axes[i] = (iree_async_axis_t)entry.axis;
    session->remote_epochs[i] = entry.current_epoch;

    // Create a software proxy semaphore for the remote axis. The tracker
    // advance below synchronizes its epoch with the remote's bootstrap epoch
    // through the same bridge path used by later ADVANCE frames.
    iree_async_semaphore_t* proxy_semaphore = NULL;
    status = iree_async_semaphore_create(
        session->proactor, /*initial_value=*/0,
        IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, session->host_allocator,
        &proxy_semaphore);
    if (!iree_status_is_ok(status)) break;

    status = iree_async_frontier_tracker_register_axis(
        session->frontier_tracker, session->remote_axes[i], proxy_semaphore);
    if (iree_status_is_ok(status)) {
      session->proxy_semaphores[i] = proxy_semaphore;
      if (entry.current_epoch > 0) {
        iree_async_frontier_tracker_advance(session->frontier_tracker,
                                            session->remote_axes[i],
                                            entry.current_epoch);
      }
    } else {
      iree_async_semaphore_release(proxy_semaphore);
    }
  }

  if (!iree_status_is_ok(status)) {
    // Full cleanup: fail registered axes in the tracker (so waiters see
    // errors instead of hanging), release all created semaphores, and free the
    // combined allocation. cleanup_remote_axes is idempotent — the later call
    // from destroy() is a no-op since all pointers are NULLed.
    iree_net_session_cleanup_remote_axes(session);
  }

  return status;
}

// Fails all remote axes in the frontier tracker and releases proxy semaphores.
static void iree_net_session_cleanup_remote_axes(iree_net_session_t* session) {
  if (!session->proxy_semaphores) return;

  for (uint32_t i = 0; i < session->remote_axis_count; ++i) {
    if (session->proxy_semaphores[i]) {
      // Retire the axis before releasing the borrowed proxy semaphore.
      if (session->frontier_tracker) {
        iree_async_frontier_tracker_retire_axis(
            session->frontier_tracker, session->remote_axes[i],
            iree_make_status(IREE_STATUS_UNAVAILABLE,
                             "remote session disconnected"));
      }
      iree_async_semaphore_release(session->proxy_semaphores[i]);
      session->proxy_semaphores[i] = NULL;
    }
  }

  // Free the combined allocation (remote_axes is the base pointer).
  iree_allocator_free(session->host_allocator, session->remote_axes);
  session->remote_axes = NULL;
  session->remote_epochs = NULL;
  session->proxy_semaphores = NULL;
  session->remote_axis_count = 0;
}

//===----------------------------------------------------------------------===//
// Bootstrap message construction and parsing
//===----------------------------------------------------------------------===//

static void iree_net_session_on_bootstrap_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  void* buffer = (void*)(uintptr_t)operation_user_data;
  iree_allocator_free(session->host_allocator, buffer);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session,
        iree_status_annotate(status, IREE_SV("bootstrap send failed")));
  }
}

// Builds and sends a HELLO message on the control channel.
static iree_status_t iree_net_session_send_hello(iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_bootstrap_topology_layout_t layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_bootstrap_topology_layout_calculate(
              IREE_NET_BOOTSTRAP_TYPE_HELLO, session->local_axis_count,
              session->local_application_data_length, &layout));
  uint8_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(session->host_allocator, layout.payload_length,
                                (void**)&buffer));
  memset(buffer, 0, layout.payload_length);

  iree_net_bootstrap_hello_t* hello = (iree_net_bootstrap_hello_t*)buffer;
  hello->header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO;
  hello->protocol_version = session->protocol_version;
  hello->capabilities = session->capabilities;
  hello->machine_index = session->local_machine_index;
  hello->session_epoch = session->local_session_epoch;
  hello->axis_count = (uint16_t)session->local_axis_count;
  hello->application_data_length =
      (uint64_t)session->local_application_data_length;

  iree_net_bootstrap_axis_entry_t* entries =
      (iree_net_bootstrap_axis_entry_t*)(buffer + layout.axis_entries_offset);
  for (uint32_t i = 0; i < session->local_axis_count; ++i) {
    entries[i].axis = (uint64_t)session->local_axes[i];
    entries[i].current_epoch = session->local_epochs[i];
  }
  if (session->local_application_data_length > 0) {
    memcpy(buffer + layout.application_data_offset,
           session->local_application_data,
           session->local_application_data_length);
  }

  // Send with zero-copy payload and a protocol-owned completion. Application
  // DATA may begin as soon as the peer's response arrives, before this send's
  // completion, so bootstrap traffic must not share the application callback.
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer, layout.payload_length);
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  iree_net_control_channel_send_completion_t completion = {
      .fn = iree_net_session_on_bootstrap_send_complete,
      .user_data = session,
  };
  iree_status_t status = iree_net_control_channel_send_data_with_completion(
      session->control_channel, IREE_NET_CONTROL_DATA_FLAG_NONE, span_list,
      (uint64_t)(uintptr_t)buffer, completion);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(session->host_allocator, buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Builds and sends a HELLO_ACK message on the control channel.
static iree_status_t iree_net_session_send_hello_ack(
    iree_net_session_t* session,
    iree_net_bootstrap_capabilities_t negotiated_capabilities) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_bootstrap_topology_layout_t layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_bootstrap_topology_layout_calculate(
              IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK, session->local_axis_count,
              session->local_application_data_length, &layout));
  uint8_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(session->host_allocator, layout.payload_length,
                                (void**)&buffer));
  memset(buffer, 0, layout.payload_length);

  iree_net_bootstrap_hello_ack_t* ack = (iree_net_bootstrap_hello_ack_t*)buffer;
  ack->header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK;
  ack->session_id = session->session_id;
  ack->negotiated_capabilities = negotiated_capabilities;
  ack->machine_index = session->local_machine_index;
  ack->session_epoch = session->local_session_epoch;
  ack->axis_count = (uint16_t)session->local_axis_count;
  ack->application_data_length =
      (uint64_t)session->local_application_data_length;

  iree_net_bootstrap_axis_entry_t* entries =
      (iree_net_bootstrap_axis_entry_t*)(buffer + layout.axis_entries_offset);
  for (uint32_t i = 0; i < session->local_axis_count; ++i) {
    entries[i].axis = (uint64_t)session->local_axes[i];
    entries[i].current_epoch = session->local_epochs[i];
  }
  if (session->local_application_data_length > 0) {
    memcpy(buffer + layout.application_data_offset,
           session->local_application_data,
           session->local_application_data_length);
  }

  // Send with zero-copy payload and a protocol-owned completion.
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer, layout.payload_length);
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  iree_net_control_channel_send_completion_t completion = {
      .fn = iree_net_session_on_bootstrap_send_complete,
      .user_data = session,
  };
  iree_status_t status = iree_net_control_channel_send_data_with_completion(
      session->control_channel, IREE_NET_CONTROL_DATA_FLAG_NONE, span_list,
      (uint64_t)(uintptr_t)buffer, completion);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(session->host_allocator, buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Sends a bootstrap rejection carrying the stable status code. Detailed local
// diagnostics may contain host paths or other private data, so the wire reason
// is limited to the canonical status-code name.
static iree_status_t iree_net_session_send_reject(
    iree_net_session_t* session, iree_status_code_t reason_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_view_t reason =
      iree_make_cstring_view(iree_status_code_string(reason_code));
  iree_host_size_t payload_size =
      sizeof(iree_net_bootstrap_reject_t) + reason.size;

  uint8_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(session->host_allocator, payload_size,
                                (void**)&buffer));
  memset(buffer, 0, payload_size);

  iree_net_bootstrap_reject_t* reject = (iree_net_bootstrap_reject_t*)buffer;
  reject->header.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  reject->reason_code = (uint32_t)reason_code;
  memcpy(buffer + sizeof(*reject), reason.data, reason.size);

  iree_async_span_t span = iree_async_span_from_ptr(buffer, payload_size);
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  iree_net_control_channel_send_completion_t completion = {
      .fn = iree_net_session_on_bootstrap_send_complete,
      .user_data = session,
  };
  iree_status_t status = iree_net_control_channel_send_data_with_completion(
      session->control_channel, IREE_NET_CONTROL_DATA_FLAG_NONE, span_list,
      (uint64_t)(uintptr_t)buffer, completion);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(session->host_allocator, buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Validates peer topology against local capacity, cancels the bootstrap timer,
// and registers all remote axes. No acceptance message or application callback
// may run until this succeeds.
static iree_status_t iree_net_session_prepare_bootstrap_completion(
    iree_net_session_t* session,
    const iree_net_bootstrap_axis_list_t* axis_list) {
  if (iree_net_session_load_state(session) !=
      IREE_NET_SESSION_STATE_BOOTSTRAPPING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session left bootstrapping before peer topology "
                            "could be registered");
  }

  uint32_t axis_capacity =
      iree_async_frontier_tracker_axis_capacity(session->frontier_tracker);
  if (axis_list->count > axis_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "peer topology has %u axes but the frontier tracker capacity is %u",
        axis_list->count, axis_capacity);
  }

  iree_net_session_cancel_bootstrap_timer(session);
  return iree_net_session_register_remote_axes(session, axis_list);
}

// Commits validated peer state and transitions the session to OPERATIONAL.
static void iree_net_session_complete_bootstrap(
    iree_net_session_t* session, uint8_t remote_machine_index,
    uint8_t remote_session_epoch, iree_const_byte_span_t application_data) {
  session->remote_machine_index = remote_machine_index;
  session->remote_session_epoch = remote_session_epoch;

  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_OPERATIONAL);

  // Build topology descriptor for the on_ready callback.
  iree_net_session_topology_t remote_topology;
  memset(&remote_topology, 0, sizeof(remote_topology));
  remote_topology.axes = session->remote_axes;
  remote_topology.current_epochs = session->remote_epochs;
  remote_topology.axis_count = session->remote_axis_count;
  remote_topology.application_data = application_data;
  remote_topology.machine_index = session->remote_machine_index;
  remote_topology.session_epoch = session->remote_session_epoch;

  iree_net_session_callbacks_t callbacks;
  if (iree_net_session_begin_callback(session, &callbacks)) {
    callbacks.on_ready(callbacks.user_data, session, &remote_topology);
    iree_net_session_end_callback(session);
  }
}

// Handles a received HELLO (server side).
static iree_status_t iree_net_session_handle_hello(
    iree_net_session_t* session, const iree_net_bootstrap_hello_view_t* hello) {
  if (session->role != IREE_NET_SESSION_ROLE_SERVER ||
      session->bootstrap_phase != IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unexpected HELLO (role=%d, phase=%d)",
                            (int)session->role, (int)session->bootstrap_phase);
  }

  iree_net_bootstrap_capabilities_t negotiated_capabilities =
      hello->fixed.capabilities & session->capabilities;
  IREE_RETURN_IF_ERROR(iree_net_session_validate_required_capabilities(
      session, negotiated_capabilities));
  IREE_RETURN_IF_ERROR(
      iree_net_session_prepare_bootstrap_completion(session, &hello->axes));

  // Submit acceptance only after every local resource required by the peer
  // topology has been acquired successfully.
  IREE_RETURN_IF_ERROR(
      iree_net_session_send_hello_ack(session, negotiated_capabilities));

  session->negotiated_capabilities = negotiated_capabilities;
  iree_net_session_complete_bootstrap(session, hello->fixed.machine_index,
                                      hello->fixed.session_epoch,
                                      hello->application_data);
  return iree_ok_status();
}

// Handles a received HELLO_ACK (client side).
static iree_status_t iree_net_session_handle_hello_ack(
    iree_net_session_t* session,
    const iree_net_bootstrap_hello_ack_view_t* hello_ack) {
  if (session->role != IREE_NET_SESSION_ROLE_CLIENT ||
      session->bootstrap_phase != IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unexpected HELLO_ACK (role=%d, phase=%d)",
                            (int)session->role, (int)session->bootstrap_phase);
  }

  iree_net_bootstrap_capabilities_t unoffered_capabilities =
      hello_ack->fixed.negotiated_capabilities & ~session->capabilities;
  if (unoffered_capabilities != IREE_NET_BOOTSTRAP_CAPABILITY_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HELLO_ACK negotiated capabilities not offered by the client: "
        "offered=0x%08x negotiated=0x%08x unoffered=0x%08x",
        session->capabilities, hello_ack->fixed.negotiated_capabilities,
        unoffered_capabilities);
  }
  IREE_RETURN_IF_ERROR(iree_net_session_validate_required_capabilities(
      session, hello_ack->fixed.negotiated_capabilities));
  IREE_RETURN_IF_ERROR(
      iree_net_session_prepare_bootstrap_completion(session, &hello_ack->axes));

  session->session_id = hello_ack->fixed.session_id;
  session->negotiated_capabilities = hello_ack->fixed.negotiated_capabilities;
  iree_net_session_complete_bootstrap(session, hello_ack->fixed.machine_index,
                                      hello_ack->fixed.session_epoch,
                                      hello_ack->application_data);
  return iree_ok_status();
}

// Handles a received REJECT (client side).
static iree_status_t iree_net_session_handle_reject(
    iree_net_session_t* session,
    const iree_net_bootstrap_reject_view_t* reject) {
  if (session->role != IREE_NET_SESSION_ROLE_CLIENT ||
      session->bootstrap_phase != IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unexpected REJECT (role=%d, phase=%d)",
                            (int)session->role, (int)session->bootstrap_phase);
  }

  int reason_length = (int)iree_min(
      reject->reason.size,
      (iree_host_size_t)IREE_NET_SESSION_REJECT_REASON_DIAGNOSTIC_BYTE_LIMIT);
  return iree_make_status((iree_status_code_t)reject->fixed.reason_code,
                          "session rejected: %.*s", reason_length,
                          reject->reason.data);
}

static iree_status_t iree_net_session_apply_bootstrap_message(
    iree_net_session_t* session,
    const iree_net_bootstrap_message_view_t* message) {
  switch (message->type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      return iree_net_session_handle_hello(session, &message->value.hello);
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      return iree_net_session_handle_hello_ack(session,
                                               &message->value.hello_ack);
    case IREE_NET_BOOTSTRAP_TYPE_REJECT:
      return iree_net_session_handle_reject(session, &message->value.reject);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "parsed bootstrap message type %u is invalid",
                              (unsigned)message->type);
  }
}

//===----------------------------------------------------------------------===//
// Control channel callbacks
//===----------------------------------------------------------------------===//

// Control channel DATA handler. During bootstrap, parses bootstrap messages.
// After bootstrap, forwards to the application's on_control_data handler.
//
// The connection drain keeps the session allocation alive through callbacks
// accepted before teardown, including callbacks that begin after the owning
// reference reaches zero.
static iree_status_t iree_net_session_on_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;

  iree_status_t status = iree_ok_status();
  iree_net_session_state_t state = iree_net_session_load_state(session);

  if (state == IREE_NET_SESSION_STATE_OPERATIONAL ||
      state == IREE_NET_SESSION_STATE_DRAINING) {
    iree_net_session_callbacks_t callbacks;
    if (iree_net_session_begin_callback(session, &callbacks)) {
      // on_control_data is required and validated at session creation.
      status =
          callbacks.on_control_data(callbacks.user_data, flags, payload, lease);
      iree_net_session_end_callback(session);
    } else {
      status =
          iree_make_status(IREE_STATUS_CANCELLED, "session callbacks detached");
    }
    return status;
  }

  if (state != IREE_NET_SESSION_STATE_BOOTSTRAPPING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session in terminal state %d", (int)state);
  }

  iree_net_bootstrap_message_view_t message;
  memset(&message, 0, sizeof(message));
  if (flags != IREE_NET_CONTROL_DATA_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bootstrap DATA flags must be zero, got 0x%02x",
                              flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_bootstrap_message_parse(payload, &message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_session_apply_bootstrap_message(session, &message);
  }

  // Route bootstrap failures through fail() so the session transitions to
  // ERROR with the specific local diagnostic. A server also sends the peer a
  // REJECT before entering ERROR; otherwise the client has no protocol edge to
  // distinguish rejection from a peer that has stopped making progress.
  if (!iree_status_is_ok(status)) {
    if (session->role == IREE_NET_SESSION_ROLE_SERVER) {
      iree_status_code_t reason_code = iree_status_code(status);
      iree_status_t reject_status =
          iree_net_session_send_reject(session, reason_code);
      if (!iree_status_is_ok(reject_status)) {
        status = iree_status_join(
            status,
            iree_status_annotate(
                reject_status, IREE_SV("failed to send bootstrap rejection")));
      }
    }
    iree_net_session_fail(session, status);
    status = iree_ok_status();
  }

  return status;
}

// Control channel GOAWAY handler.
static void iree_net_session_on_goaway(void* user_data, uint32_t reason_code,
                                       iree_string_view_t message) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;

  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_DRAINING);

  // Clean up remote axes (fail them in tracker, release proxy semaphores).
  iree_net_session_cleanup_remote_axes(session);

  iree_net_session_callbacks_t callbacks;
  if (iree_net_session_begin_callback(session, &callbacks)) {
    if (callbacks.on_goaway) {
      callbacks.on_goaway(callbacks.user_data, session, reason_code, message);
    }
    iree_net_session_end_callback(session);
  }
}

// Control channel ERROR handler.
static void iree_net_session_on_control_error(void* user_data,
                                              iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_fail(session, status);
}

// Control channel application DATA send completion handler.
static void iree_net_session_on_send_complete(void* user_data,
                                              uint64_t operation_user_data,
                                              iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(session->callback_count > 0,
              "unadmitted application send completed");
  iree_net_session_callbacks_t callbacks = session->callbacks;
  IREE_ASSERT(callbacks.on_send_complete,
              "accepted DATA send has no completion callback");
  iree_slim_mutex_unlock(&session->callback_mutex);

  callbacks.on_send_complete(callbacks.user_data, operation_user_data, status);
  iree_net_session_end_callback(session);
}

static void iree_net_session_on_send_ready(void* user_data) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_callbacks_t callbacks;
  if (iree_net_session_begin_callback(session, &callbacks)) {
    if (callbacks.on_send_ready) {
      callbacks.on_send_ready(callbacks.user_data, session);
    }
    iree_net_session_end_callback(session);
  }
}

// Control channel transport error handler.
static void iree_net_session_on_transport_error(void* user_data,
                                                iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_fail(
      session,
      iree_status_annotate(status, IREE_SV("control channel transport error")));
}

//===----------------------------------------------------------------------===//
// Connection callbacks (async bootstrap chain)
//===----------------------------------------------------------------------===//

// Called when the control endpoint is ready. Creates the control channel,
// activates it, and begins the bootstrap protocol.
// Consumes the session reference transferred to the endpoint callback.
static void iree_net_session_on_control_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to open control endpoint")));
    iree_net_session_release(session);
    return;
  }

  // Guard: if the session was failed while endpoint open was in flight (e.g.,
  // bootstrap timeout expired), skip further bootstrap progression.
  if (iree_net_session_load_state(session) == IREE_NET_SESSION_STATE_ERROR) {
    iree_net_session_release(session);
    return;
  }

  // Create control channel with session as the callback target.
  // The control channel routes sends through the endpoint (not directly to the
  // carrier), so transport-specific framing (e.g., TCP mux stream headers) is
  // applied automatically. max_send_spans uses FRAME_SENDER_MAX_SPANS as a
  // conservative default — the endpoint's send path may add transport headers
  // that consume spans, but FRAME_SENDER_MAX_SPANS (8) is well within the
  // typical carrier max_iov (16+) even after overhead.
  iree_net_control_channel_callbacks_t channel_callbacks = {
      .on_data = iree_net_session_on_data,
      .on_goaway = iree_net_session_on_goaway,
      .on_error = iree_net_session_on_control_error,
      .on_pong = NULL,  // Session doesn't use PONG directly.
      .on_transport_error = iree_net_session_on_transport_error,
      .on_send_complete = iree_net_session_on_send_complete,
      .on_send_ready = iree_net_session_on_send_ready,
      .user_data = session,
  };

  status = iree_net_control_channel_create(
      endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS,
      iree_net_control_channel_options_default(), channel_callbacks,
      session->host_allocator, &session->control_channel);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to create control channel")));
    iree_net_session_release(session);
    return;
  }

  if (session->role == IREE_NET_SESSION_ROLE_SERVER) {
    // Server activation may synchronously drain a transport receive queue that
    // already contains the client's HELLO.
    session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO;
  }

  // Activate the control channel (start receiving).
  status = iree_net_control_channel_activate(session->control_channel);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to activate control channel")));
    iree_net_session_release(session);
    return;
  }

  if (session->role == IREE_NET_SESSION_ROLE_CLIENT) {
    // Client: send HELLO and wait for HELLO_ACK.
    session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT;
    status = iree_net_session_send_hello(session);
    if (!iree_status_is_ok(status)) {
      iree_net_session_fail(
          session,
          iree_status_annotate(status, IREE_SV("failed to send HELLO")));
      iree_net_session_release(session);
      return;
    }
  }

  iree_net_session_release(session);
}

// Called when factory.connect() completes (client path only).
// Consumes the session reference transferred to the connect callback.
static void iree_net_session_on_connect(void* user_data, iree_status_t status,
                                        iree_net_connection_t* connection) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session,
        iree_status_annotate(status, IREE_SV("failed to connect to server")));
    iree_net_session_release(session);
    return;
  }

  // Guard: if the session was failed while the connect was in flight (e.g.,
  // bootstrap timeout expired), skip further bootstrap progression. The
  // connection is released since we never adopted it.
  if (iree_net_session_load_state(session) == IREE_NET_SESSION_STATE_ERROR) {
    iree_net_connection_release(connection);
    iree_net_session_release(session);
    return;
  }

  // Adopt the connection (the factory callback transfers ownership).
  session->connection = connection;

  iree_status_t capacity_status =
      iree_net_session_validate_endpoint_capacity(session, connection);
  if (!iree_status_is_ok(capacity_status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     capacity_status,
                     IREE_SV("connection endpoint capacity is insufficient")));
    iree_net_session_release(session);
    return;
  }

  // Open the control endpoint.
  session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL;
  iree_net_session_t* endpoint_session = session;
  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_net_session_on_control_endpoint_ready,
      .user_data = endpoint_session,
  };
  iree_net_session_retain(endpoint_session);
  status = iree_net_connection_open_endpoint(connection, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    // Synchronous submission failure means the endpoint callback will not
    // consume its transferred reference.
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to open control endpoint")));
    iree_net_session_release(endpoint_session);
  }

  iree_net_session_release(session);
}

//===----------------------------------------------------------------------===//
// Common initialization
//===----------------------------------------------------------------------===//

// Allocates and initializes a session with the given options.
static iree_status_t iree_net_session_create_common(
    iree_net_session_role_t role, const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;

  // Validate required callbacks.
  if (!callbacks.on_ready) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_ready callback is required");
  }
  if (!callbacks.on_control_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_control_data callback is required");
  }
  if (options->application_endpoint_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "application endpoint count overflows required "
                            "endpoint count");
  }
  iree_net_bootstrap_capabilities_t unrecognized_capabilities =
      options->capabilities & ~IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED;
  if (unrecognized_capabilities != IREE_NET_BOOTSTRAP_CAPABILITY_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "session capabilities contain unrecognized bits 0x%08x",
        unrecognized_capabilities);
  }
  if ((options->required_capabilities & ~options->capabilities) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "required session capabilities must be advertised: capabilities=0x%08x "
        "required=0x%08x",
        options->capabilities, options->required_capabilities);
  }

  // Validate axis count fits in the wire format (uint16_t in HELLO/HELLO_ACK).
  uint32_t local_axis_count = options->local_topology.axis_count;
  if (local_axis_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local axis count %u exceeds wire format maximum %u", local_axis_count,
        (uint32_t)UINT16_MAX);
  }
  if (local_axis_count > 0 && !options->local_topology.axes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local topology axes must have storage when axis_count is nonzero");
  }
  if (local_axis_count > 0 && !options->local_topology.current_epochs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local topology epochs must have storage when axis_count is nonzero");
  }
  if (options->local_topology.reserved[0] != 0 ||
      options->local_topology.reserved[1] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local topology reserved bytes must be zero");
  }
  iree_host_size_t local_application_data_length =
      options->local_topology.application_data.data_length;
  if (local_application_data_length > 0 &&
      !options->local_topology.application_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local application data must have storage when "
                            "data_length is nonzero");
  }

  // Compute allocation size for session + local topology arrays.
  iree_host_size_t local_axes_offset = 0;
  iree_host_size_t local_epochs_offset = 0;
  iree_host_size_t local_application_data_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_net_session_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(local_axis_count, iree_async_axis_t,
                                iree_alignof(iree_async_axis_t),
                                &local_axes_offset),
      IREE_STRUCT_FIELD(local_axis_count, uint64_t, &local_epochs_offset),
      IREE_STRUCT_FIELD(local_application_data_length, uint8_t,
                        &local_application_data_offset)));

  iree_net_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&session));
  memset(session, 0, total_size);

  iree_atomic_ref_count_init(&session->ref_count);
  session->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&session->callback_mutex);
  session->role = role;
  session->bootstrap_phase = (role == IREE_NET_SESSION_ROLE_CLIENT)
                                 ? IREE_NET_SESSION_BOOTSTRAP_CONNECTING
                                 : IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL;
  iree_atomic_store(&session->state,
                    (int32_t)IREE_NET_SESSION_STATE_BOOTSTRAPPING,
                    iree_memory_order_release);
  session->callbacks = callbacks;

  // Copy configuration.
  session->protocol_version = options->protocol_version
                                  ? options->protocol_version
                                  : IREE_NET_BOOTSTRAP_PROTOCOL_VERSION;
  session->capabilities = options->capabilities;
  session->required_capabilities = options->required_capabilities;
  session->application_endpoint_count = options->application_endpoint_count;
  session->bootstrap_timeout_ns = options->bootstrap_timeout_ns;

  // Copy local topology into trailing storage.
  session->local_axes =
      (iree_async_axis_t*)((uint8_t*)session + local_axes_offset);
  session->local_epochs = (uint64_t*)((uint8_t*)session + local_epochs_offset);
  session->local_axis_count = local_axis_count;
  session->local_machine_index = options->local_topology.machine_index;
  session->local_session_epoch = options->local_topology.session_epoch;
  session->local_application_data =
      (uint8_t*)session + local_application_data_offset;
  session->local_application_data_length = local_application_data_length;

  if (local_axis_count > 0) {
    memcpy(session->local_axes, options->local_topology.axes,
           local_axis_count * sizeof(iree_async_axis_t));
    memcpy(session->local_epochs, options->local_topology.current_epochs,
           local_axis_count * sizeof(uint64_t));
  }
  if (local_application_data_length > 0) {
    memcpy(session->local_application_data,
           options->local_topology.application_data.data,
           local_application_data_length);
  }

  *out_session = session;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Destroy (two-phase: begin_teardown → deactivation callback → complete)
//===----------------------------------------------------------------------===//

// Frees all session-owned resources and the session itself. Called after the
// connection's carriers have been deactivated (or immediately if there is no
// connection or no active carriers).
static void iree_net_session_complete_teardown(iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release control channel. All carrier completions have drained, so the
  // frame_sender's sends_in_flight is 0 and deinitialize is safe. Must happen
  // before connection release because carrier memory is still valid.
  iree_net_control_channel_release(session->control_channel);

  // Release connection. Safe now — all carriers are deactivated, so releasing
  // cannot trigger use-after-free on pending proactor operations.
  iree_net_connection_release(session->connection);

  // Release the client receive pool after connection teardown. Carriers borrow
  // this pool for recv operations and may still hold leases until connection
  // deactivation completes.
  iree_async_buffer_pool_release(session->recv_pool);

  // Release transport factory (client path only).
  iree_net_transport_factory_release(session->transport_factory);

  // Release async infrastructure used by in-flight session operations.
  iree_async_frontier_tracker_release(session->frontier_tracker);
  iree_async_proactor_release(session->proactor);

  IREE_ASSERT(session->callback_count == 0);
  IREE_ASSERT(session->transport_submission_count == 0);
  IREE_ASSERT(!session->detach_callback.fn);
  IREE_ASSERT(!session->deactivated_callback.fn);
  iree_slim_mutex_deinitialize(&session->callback_mutex);

  // Free server address storage (client path only).
  if (session->server_address_storage) {
    iree_allocator_free(session->host_allocator,
                        session->server_address_storage);
  }

  iree_allocator_t host_allocator = session->host_allocator;
  iree_allocator_free(host_allocator, session);
  IREE_TRACE_ZONE_END(z0);
}

// Callback from iree_net_connection_deactivate(). All carriers are now drained
// and in the DEACTIVATED state. Safe to release the connection and free the
// session.
static void iree_net_session_on_connection_deactivated(void* user_data) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_deactivated_callback_t callback = {0};
  bool complete_teardown = false;
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(session->connection_deactivation_requested);
  IREE_ASSERT(session->connection_deactivation_submitted);
  IREE_ASSERT(!session->connection_deactivated);
  session->connection_deactivated = true;
  if (iree_net_session_deactivation_ready_locked(session)) {
    callback = iree_net_session_take_deactivated_callback_locked(session);
    if (!callback.fn && session->destroy_after_deactivation) {
      complete_teardown = true;
    }
  } else {
    IREE_ASSERT(
        !session->destroy_after_deactivation || session->callback_count > 0,
        "implicit session teardown completed with a pending bootstrap "
        "timer");
  }
  iree_slim_mutex_unlock(&session->callback_mutex);

  if (callback.fn) {
    callback.fn(callback.user_data);
  } else if (complete_teardown) {
    iree_net_session_complete_teardown(session);
  }
}

// Claims the connection deactivation submission once all calls admitted before
// the request have crossed their synchronous transport submission boundary.
// The caller must hold |callback_mutex|.
static bool iree_net_session_prepare_connection_deactivation_locked(
    iree_net_session_t* session) {
  if (!session->connection_deactivation_requested ||
      session->connection_deactivation_submitted ||
      session->transport_submission_count != 0) {
    return false;
  }
  IREE_ASSERT(session->connection);
  session->connection_deactivation_submitted = true;
  return true;
}

// Submits a previously claimed connection drain. The callback may run
// synchronously and release |session|, so callers must not touch it afterward.
static void iree_net_session_submit_connection_deactivation(
    iree_net_session_t* session) {
  iree_net_connection_deactivate_callback_t callback = {
      .fn = iree_net_session_on_connection_deactivated,
      .user_data = session,
  };
  iree_net_connection_deactivate(session->connection, callback);
}

static iree_status_t iree_net_session_begin_transport_submission(
    iree_net_session_t* session,
    iree_net_session_submission_kind_t submission_kind) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&session->callback_mutex);
  iree_net_session_state_t state = iree_net_session_load_state(session);
  if (state != IREE_NET_SESSION_STATE_OPERATIONAL) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "session state %d does not accept application submissions", (int)state);
  } else if (submission_kind != IREE_NET_SESSION_SUBMISSION_SHUTDOWN &&
             session->callbacks_detached) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "session callbacks are detached");
  } else if (session->connection_deactivation_requested) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "session connection is deactivating");
  } else if (submission_kind == IREE_NET_SESSION_SUBMISSION_DATA_SEND &&
             !session->callbacks.on_send_complete) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "DATA sends require an on_send_complete callback");
  } else {
    if (submission_kind == IREE_NET_SESSION_SUBMISSION_SHUTDOWN) {
      iree_net_session_set_state(session, IREE_NET_SESSION_STATE_DRAINING);
    }
    if (submission_kind != IREE_NET_SESSION_SUBMISSION_SHUTDOWN) {
      ++session->callback_count;
    }
    ++session->transport_submission_count;
  }
  iree_slim_mutex_unlock(&session->callback_mutex);
  return status;
}

// Releases the synchronous submission gate. This may start a deferred
// connection drain whose callback destroys |session|.
static void iree_net_session_end_transport_submission(
    iree_net_session_t* session) {
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(session->transport_submission_count > 0);
  --session->transport_submission_count;
  bool submit_deactivation =
      iree_net_session_prepare_connection_deactivation_locked(session);
  iree_slim_mutex_unlock(&session->callback_mutex);
  if (submit_deactivation) {
    iree_net_session_submit_connection_deactivation(session);
  }
}

// Begins tearing down the session. Called from the last session_release().
//
// Synchronous cleanup (remote axes) happens immediately. Then, if the session
// has a connection, deactivation is initiated — the connection drains all its
// carriers and fires a callback when done. The callback completes the teardown
// by releasing the control channel, connection, and remaining resources.
//
// The control channel is NOT released here. Its frame_sender may have in-flight
// send completions pending on the proactor thread. Those completions reference
// the frame_sender (via context->sender) and must drain before the control
// channel is destroyed. The control channel is released in complete_teardown
// after all carrier completions have fired.
//
// If there is no connection (bootstrap failed early), teardown completes
// synchronously.
static void iree_net_session_begin_teardown(iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clean up remote axes and proxy semaphores.
  iree_net_session_cleanup_remote_axes(session);

  bool complete_synchronously = false;
  bool begin_connection_deactivation = false;
  bool wait_for_transport_submission = false;
  iree_slim_mutex_lock(&session->callback_mutex);
  if (!session->connection || session->connection_deactivated) {
    complete_synchronously = true;
  } else {
    IREE_ASSERT(!session->deactivated_callback.fn,
                "final session reference released before explicit "
                "deactivation completed");
    IREE_ASSERT(!session->connection_deactivation_requested,
                "session teardown started more than once");
    session->connection_deactivation_requested = true;
    session->destroy_after_deactivation = true;
    begin_connection_deactivation =
        iree_net_session_prepare_connection_deactivation_locked(session);
    wait_for_transport_submission = !begin_connection_deactivation;
  }
  iree_slim_mutex_unlock(&session->callback_mutex);

  // Deactivate the connection's carriers before releasing. This ensures all
  // pending proactor operations (NOPs, sends) complete before we free the
  // carrier memory they reference.
  if (begin_connection_deactivation) {
    iree_net_session_submit_connection_deactivation(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  if (wait_for_transport_submission) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  IREE_ASSERT(complete_synchronously);
  IREE_TRACE_ZONE_END(z0);
  iree_net_session_complete_teardown(session);
}

//===----------------------------------------------------------------------===//
// Public API: creation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_session_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t server_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(factory);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(out_session);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_session = NULL;

  iree_net_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_session_create_common(IREE_NET_SESSION_ROLE_CLIENT, options,
                                         callbacks, host_allocator, &session));

  // Retain factory.
  session->transport_factory = factory;
  iree_net_transport_factory_retain(factory);

  // Retain async infrastructure referenced by the session callbacks/timer.
  session->frontier_tracker = frontier_tracker;
  iree_async_frontier_tracker_retain(frontier_tracker);
  session->proactor = proactor;
  iree_async_proactor_retain(proactor);
  session->recv_pool = recv_pool;
  iree_async_buffer_pool_retain(recv_pool);

  // Copy server address (the string_view may not outlive this call).
  if (server_address.size > 0) {
    iree_status_t status =
        iree_allocator_malloc(host_allocator, server_address.size,
                              (void**)&session->server_address_storage);
    if (!iree_status_is_ok(status)) {
      iree_net_session_begin_teardown(session);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    memcpy(session->server_address_storage, server_address.data,
           server_address.size);
    session->server_address = iree_make_string_view(
        session->server_address_storage, server_address.size);
  }

  // Publish the session before any async submission. A fast transport may
  // execute callbacks on another proactor thread before this function returns.
  // Hold a function reference so those callbacks may release the caller's
  // published reference without invalidating setup still in progress here.
  *out_session = session;
  iree_net_session_t* function_session = session;
  iree_net_session_retain(function_session);

  // Start the bootstrap timeout timer before any other async operations.
  iree_status_t status = iree_net_session_start_bootstrap_timer(
      session, session->bootstrap_timeout_ns);
  if (!iree_status_is_ok(status)) {
    *out_session = NULL;
    iree_net_session_release(function_session);
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Transfer a reference to the async connect callback. The callback may
  // arrive after the timeout reference and caller reference have been released.
  iree_net_session_t* connect_session = session;
  iree_net_session_retain(connect_session);
  status = iree_net_transport_factory_connect(
      factory, session->server_address, proactor, recv_pool,
      iree_net_session_on_connect, connect_session);
  if (!iree_status_is_ok(status)) {
    // Synchronous submission failure means the connect callback will not
    // consume its transferred reference.
    // Timer is in flight (holds a ref), so we cannot destroy the session.
    // Fail it instead — the timer callback will see ERROR/CANCELLED and
    // release its ref. The caller releases theirs after on_error fires.
    iree_net_session_fail(
        session,
        iree_status_annotate(status, IREE_SV("failed to connect to server")));
    iree_net_session_release(connect_session);
    iree_net_session_release(function_session);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_net_session_release(function_session);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_session_accept(
    iree_net_connection_t* connection, iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(connection);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  if (!options->session_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "server sessions require a nonzero session_id in options");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_session_create_common(IREE_NET_SESSION_ROLE_SERVER, options,
                                         callbacks, host_allocator, &session));

  // Copy server-assigned session ID from options.
  session->session_id = options->session_id;

  // Retain connection.
  session->connection = connection;
  iree_net_connection_retain(connection);

  // Retain async infrastructure referenced by the session callbacks/timer.
  session->frontier_tracker = frontier_tracker;
  iree_async_frontier_tracker_retain(frontier_tracker);
  session->proactor = proactor;
  iree_async_proactor_retain(proactor);

  iree_status_t capacity_status =
      iree_net_session_validate_endpoint_capacity(session, connection);
  if (!iree_status_is_ok(capacity_status)) {
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return iree_status_annotate(
        capacity_status,
        IREE_SV("connection endpoint capacity is insufficient"));
  }

  // Publish the session before any async submission and retain it until this
  // function returns. Endpoint callbacks may run concurrently on the proactor.
  *out_session = session;
  iree_net_session_t* function_session = session;
  iree_net_session_retain(function_session);

  // Start the bootstrap timeout timer before any other async operations.
  iree_status_t status = iree_net_session_start_bootstrap_timer(
      session, session->bootstrap_timeout_ns);
  if (!iree_status_is_ok(status)) {
    *out_session = NULL;
    iree_net_session_release(function_session);
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Open control endpoint to begin bootstrap.
  iree_net_session_t* endpoint_session = session;
  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_net_session_on_control_endpoint_ready,
      .user_data = endpoint_session,
  };
  iree_net_session_retain(endpoint_session);
  status = iree_net_connection_open_endpoint(connection, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    // Synchronous submission failure means the endpoint callback will not
    // consume its transferred reference.
    // Endpoint open failed. The timer is in flight (holds a ref), so we
    // can't destroy the session. Fail it — the timer callback will release
    // its ref, and the caller releases theirs after on_error fires.
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to open control endpoint")));
    iree_net_session_release(endpoint_session);
    iree_net_session_release(function_session);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_net_session_release(function_session);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API: lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_net_session_retain(iree_net_session_t* session) {
  if (!session) return;
  iree_atomic_ref_count_inc(&session->ref_count);
}

IREE_API_EXPORT void iree_net_session_release(iree_net_session_t* session) {
  if (!session) return;
  if (iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_net_session_begin_teardown(session);
  }
}

IREE_API_EXPORT void iree_net_session_detach_callbacks(
    iree_net_session_t* session,
    iree_net_session_callbacks_detached_callback_t callback) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(callback.fn);

  bool complete_synchronously = false;
  iree_slim_mutex_lock(&session->callback_mutex);
  IREE_ASSERT(!session->callbacks_detached,
              "session callbacks detached more than once");
  session->callbacks_detached = true;
  if (session->callback_count == 0) {
    memset(&session->callbacks, 0, sizeof(session->callbacks));
    complete_synchronously = true;
  } else {
    session->detach_callback = callback;
  }
  iree_slim_mutex_unlock(&session->callback_mutex);

  if (complete_synchronously) {
    callback.fn(callback.user_data);
  }
}

IREE_API_EXPORT iree_status_t
iree_net_session_deactivate(iree_net_session_t* session,
                            iree_net_session_deactivated_callback_t callback) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(callback.fn);

  bool request_accepted = false;
  bool submit_connection_deactivation = false;
  iree_net_session_deactivated_callback_t synchronous_callback = {0};
  iree_net_session_state_t state = IREE_NET_SESSION_STATE_BOOTSTRAPPING;
  iree_slim_mutex_lock(&session->callback_mutex);
  state = iree_net_session_load_state(session);
  if (state != IREE_NET_SESSION_STATE_BOOTSTRAPPING && !session->connection &&
      !session->connection_deactivation_requested) {
    session->connection_deactivation_requested = true;
    session->deactivated_callback = callback;
    request_accepted = true;
  } else if (state != IREE_NET_SESSION_STATE_BOOTSTRAPPING &&
             session->connection &&
             !session->connection_deactivation_requested) {
    session->connection_deactivation_requested = true;
    session->deactivated_callback = callback;
    if (state == IREE_NET_SESSION_STATE_OPERATIONAL) {
      iree_net_session_set_state(session, IREE_NET_SESSION_STATE_DRAINING);
    }
    request_accepted = true;
    submit_connection_deactivation =
        iree_net_session_prepare_connection_deactivation_locked(session);
  }
  if (request_accepted) {
    synchronous_callback =
        iree_net_session_take_deactivated_callback_locked(session);
  }
  iree_slim_mutex_unlock(&session->callback_mutex);
  if (!request_accepted) {
    if (state == IREE_NET_SESSION_STATE_BOOTSTRAPPING) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "session connection bootstrap must complete before deactivation");
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session deactivation already started");
  }

  if (synchronous_callback.fn) {
    synchronous_callback.fn(synchronous_callback.user_data);
  } else if (submit_connection_deactivation) {
    iree_net_session_submit_connection_deactivation(session);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_net_session_state_t
iree_net_session_state(const iree_net_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_net_session_load_state(session);
}

IREE_API_EXPORT uint64_t
iree_net_session_id(const iree_net_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->session_id;
}

IREE_API_EXPORT iree_net_carrier_t* iree_net_session_carrier(
    iree_net_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->connection ? iree_net_connection_carrier(session->connection)
                             : NULL;
}

//===----------------------------------------------------------------------===//
// Public API: operations
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_session_open_endpoint(
    iree_net_session_t* session, iree_net_endpoint_ready_callback_t callback) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(callback.fn);

  iree_net_session_endpoint_callback_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      session->host_allocator, sizeof(*context), (void**)&context));

  iree_status_t status = iree_net_session_begin_transport_submission(
      session, IREE_NET_SESSION_SUBMISSION_ENDPOINT);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(session->host_allocator, context);
    return status;
  }

  context->session = session;
  context->callback = callback;
  iree_net_session_retain(session);
  iree_net_endpoint_ready_callback_t retained_callback = {
      .fn = iree_net_session_on_application_endpoint_ready,
      .user_data = context,
  };
  status =
      iree_net_connection_open_endpoint(session->connection, retained_callback);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(session->host_allocator, context);
    iree_net_session_end_callback(session);
    iree_net_session_release(session);
  }
  iree_net_session_end_transport_submission(session);
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_session_send_control_data(
    iree_net_session_t* session, iree_net_control_frame_flags_t flags,
    iree_async_span_list_t payload, uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_RETURN_IF_ERROR(iree_net_session_begin_transport_submission(
      session, IREE_NET_SESSION_SUBMISSION_DATA_SEND));
  iree_status_t status = iree_net_control_channel_send_data(
      session->control_channel, flags, payload, operation_user_data);
  if (!iree_status_is_ok(status)) {
    iree_net_session_end_callback(session);
  }
  iree_net_session_end_transport_submission(session);
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_session_send_control_data_copy(
    iree_net_session_t* session, iree_net_control_frame_flags_t flags,
    iree_async_span_list_t payload, uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_RETURN_IF_ERROR(iree_net_session_begin_transport_submission(
      session, IREE_NET_SESSION_SUBMISSION_DATA_SEND));
  iree_status_t status = iree_net_control_channel_send_data_copy(
      session->control_channel, flags, payload, operation_user_data);
  if (!iree_status_is_ok(status)) {
    iree_net_session_end_callback(session);
  }
  iree_net_session_end_transport_submission(session);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_net_session_shutdown(iree_net_session_t* session, uint32_t reason_code,
                          iree_string_view_t message) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_RETURN_IF_ERROR(iree_net_session_begin_transport_submission(
      session, IREE_NET_SESSION_SUBMISSION_SHUTDOWN));

  iree_status_t status = iree_net_control_channel_send_goaway(
      session->control_channel, reason_code, message);

  // Clean up remote axes proactively once shutdown has claimed the session.
  iree_net_session_cleanup_remote_axes(session);
  iree_net_session_end_transport_submission(session);
  return status;
}
