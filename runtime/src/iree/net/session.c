// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/session.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/session_bootstrap.h"

typedef enum iree_net_session_lifecycle_flag_bits_e {
  IREE_NET_SESSION_LIFECYCLE_FLAG_NONE = 0u,
  IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING = 1u << 0,
  IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED = 1u << 1,
  IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATION_SUBMITTED = 1u << 2,
  IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATED = 1u << 3,
  IREE_NET_SESSION_LIFECYCLE_FLAG_ERROR_REPORTED = 1u << 4,
  IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING = 1u << 5,
} iree_net_session_lifecycle_flag_bits_t;
typedef uint32_t iree_net_session_lifecycle_flags_t;

struct iree_net_session_t {
  // Reference count including one internal lifecycle reference.
  iree_atomic_ref_count_t ref_count;
  // Host allocator owning the session and its components.
  iree_allocator_t host_allocator;
  // Serializes lifecycle and session-level operation admission.
  iree_slim_mutex_t mutex;
  // Public lifecycle state read with atomic acquire ordering.
  iree_atomic_int32_t state;
  // Internal lifecycle facts protected by |mutex|.
  iree_net_session_lifecycle_flags_t lifecycle_flags;
  // Immutable application callback bundle.
  iree_net_session_callbacks_t callbacks;
  // Role-specific bootstrap state and encoded local peer message.
  iree_net_session_bootstrap_t bootstrap;
  // Client factory attempt, detached before on_connect; idle for servers.
  iree_net_transport_connect_operation_t connect_operation;
  // Owned connection after accept or successful client connect.
  iree_net_connection_t* connection;
  // Borrowed endpoint ordinal zero owned by |connection|.
  iree_net_message_endpoint_t control_endpoint;
  // Owned protocol adapter borrowing |control_endpoint|.
  iree_net_control_channel_t* control_channel;
  // Number of accepted application endpoint opens.
  uint32_t opened_application_endpoint_count;
};

static void iree_net_session_destroy(iree_net_session_t* session) {
  IREE_ASSERT(iree_net_session_state(session) ==
              IREE_NET_SESSION_STATE_DEACTIVATED);
  iree_net_control_channel_free(session->control_channel);
  iree_net_connection_release(session->connection);
  iree_net_session_bootstrap_deinitialize(&session->bootstrap);
  iree_net_transport_connect_operation_deinitialize(
      &session->connect_operation);
  iree_slim_mutex_deinitialize(&session->mutex);
  iree_allocator_free(session->host_allocator, session);
}

void iree_net_session_retain(iree_net_session_t* session) {
  if (!session) {
    return;
  }
  iree_atomic_ref_count_inc(&session->ref_count);
}

void iree_net_session_release(iree_net_session_t* session) {
  if (!session) {
    return;
  }
  if (iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_net_session_destroy(session);
  }
}

iree_net_session_state_t iree_net_session_state(
    const iree_net_session_t* session) {
  return (iree_net_session_state_t)iree_atomic_load(
      &((iree_net_session_t*)session)->state, iree_memory_order_acquire);
}

static void iree_net_session_set_state_locked(iree_net_session_t* session,
                                              iree_net_session_state_t state) {
  iree_atomic_store(&session->state, (int32_t)state, iree_memory_order_release);
}

static bool iree_net_session_try_finalize_locked(iree_net_session_t* session) {
  if (!iree_any_bit_set(
          session->lifecycle_flags,
          IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED) ||
      iree_any_bit_set(session->lifecycle_flags,
                       IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING |
                           IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING)) {
    return false;
  }
  if (session->connection &&
      !iree_any_bit_set(
          session->lifecycle_flags,
          IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATED)) {
    return false;
  }
  if (iree_net_session_state(session) == IREE_NET_SESSION_STATE_DEACTIVATED) {
    return false;
  }
  iree_net_session_set_state_locked(session,
                                    IREE_NET_SESSION_STATE_DEACTIVATED);
  return true;
}

static void iree_net_session_finish_deactivated(iree_net_session_t* session) {
  session->callbacks.on_deactivated(session->callbacks.user_data, session);
  // Drop the lifecycle reference after the callback can release its reference.
  iree_net_session_release(session);
}

static void iree_net_session_on_connection_deactivated(void* user_data) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_slim_mutex_lock(&session->mutex);
  session->lifecycle_flags |=
      IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATED;
  const bool finalize = iree_net_session_try_finalize_locked(session);
  iree_slim_mutex_unlock(&session->mutex);
  if (finalize) {
    iree_net_session_finish_deactivated(session);
  }
}

static void iree_net_session_drive_deactivation(iree_net_session_t* session) {
  iree_net_connection_t* connection = NULL;
  bool finalize = false;
  iree_slim_mutex_lock(&session->mutex);
  if (iree_all_bits_set(session->lifecycle_flags,
                        IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED |
                            IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING) &&
      !session->connection) {
    // The factory detaches before invoking on_connect, so it never acquires
    // the session mutex while holding the operation mutex.
    iree_net_transport_connect_operation_cancel(&session->connect_operation);
  }
  if (iree_any_bit_set(
          session->lifecycle_flags,
          IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED) &&
      !iree_any_bit_set(session->lifecycle_flags,
                        IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING) &&
      session->connection &&
      !iree_any_bit_set(
          session->lifecycle_flags,
          IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATION_SUBMITTED)) {
    session->lifecycle_flags |=
        IREE_NET_SESSION_LIFECYCLE_FLAG_CONNECTION_DEACTIVATION_SUBMITTED;
    connection = session->connection;
  } else {
    finalize = iree_net_session_try_finalize_locked(session);
  }
  iree_slim_mutex_unlock(&session->mutex);

  if (connection) {
    iree_net_connection_deactivate(
        connection, (iree_net_connection_deactivate_callback_t){
                        .fn = iree_net_session_on_connection_deactivated,
                        .user_data = session,
                    });
  } else if (finalize) {
    iree_net_session_finish_deactivated(session);
  }
}

static void iree_net_session_finish_setup(iree_net_session_t* session) {
  iree_slim_mutex_lock(&session->mutex);
  IREE_ASSERT(iree_any_bit_set(session->lifecycle_flags,
                               IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING));
  session->lifecycle_flags &= ~IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING;
  iree_slim_mutex_unlock(&session->mutex);
  iree_net_session_drive_deactivation(session);
}

static void iree_net_session_fail(iree_net_session_t* session,
                                  iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));
  bool report_error = false;
  iree_slim_mutex_lock(&session->mutex);
  // Explicit deactivation owns later operation failures as teardown outcomes.
  if (!iree_any_bit_set(session->lifecycle_flags,
                        IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED |
                            IREE_NET_SESSION_LIFECYCLE_FLAG_ERROR_REPORTED) &&
      iree_net_session_state(session) != IREE_NET_SESSION_STATE_DEACTIVATED) {
    session->lifecycle_flags |=
        IREE_NET_SESSION_LIFECYCLE_FLAG_ERROR_REPORTED |
        IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED;
    iree_net_session_set_state_locked(session, IREE_NET_SESSION_STATE_DRAINING);
    report_error = true;
  }
  iree_slim_mutex_unlock(&session->mutex);

  if (report_error) {
    session->callbacks.on_error(session->callbacks.user_data, session, status);
  } else {
    iree_status_free(status);
  }
  iree_net_session_drive_deactivation(session);
}

static void iree_net_session_on_bootstrap_send_complete(
    void* user_data, iree_status_t status, iree_host_size_t bytes_transferred) {
  (void)bytes_transferred;
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session,
        iree_status_annotate(status, IREE_SV("bootstrap send failed")));
  } else {
    iree_status_free(status);
  }
}

static void iree_net_session_on_reject_send_complete(
    void* user_data, iree_status_t status, iree_host_size_t bytes_transferred) {
  (void)bytes_transferred;
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_status_free(status);
  iree_slim_mutex_lock(&session->mutex);
  IREE_ASSERT(iree_any_bit_set(session->lifecycle_flags,
                               IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING));
  session->lifecycle_flags &= ~IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING;
  iree_slim_mutex_unlock(&session->mutex);
  iree_net_session_drive_deactivation(session);
}

// Sends the role-specific HELLO or HELLO_ACK. The session mutex must be held.
static iree_status_t iree_net_session_send_bootstrap_locked(
    iree_net_session_t* session) {
  const iree_const_byte_span_t message =
      iree_net_session_bootstrap_outbound_message(&session->bootstrap);
  const iree_net_message_endpoint_send_params_t send_params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(message),
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = iree_net_session_on_bootstrap_send_complete,
              .user_data = session,
          },
  };
  iree_status_t status =
      iree_net_message_endpoint_send(session->control_endpoint, &send_params);
  if (iree_status_is_ok(status)) {
    iree_net_session_bootstrap_consume_outbound_message(&session->bootstrap);
  }
  return status;
}

// Sends a stable rejection diagnostic. The session mutex must be held.
static iree_status_t iree_net_session_send_reject_locked(
    iree_net_session_t* session, iree_status_code_t status_code) {
  const iree_string_view_t reason =
      iree_make_cstring_view(iree_status_code_string(status_code));
  iree_net_bootstrap_message_t message;
  memset(&message, 0, sizeof(message));
  message.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  message.value.reject.status_code = status_code;
  message.value.reject.reason = reason;

  iree_host_size_t message_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_net_bootstrap_message_calculate_size(&message, &message_size));
  uint8_t* message_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      session->host_allocator, message_size, (void**)&message_data));
  iree_status_t status = iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(message_data, message_size));
  if (iree_status_is_ok(status)) {
    const iree_net_message_endpoint_send_params_t send_params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(message_data, message_size)),
        .data = iree_async_span_list_empty(),
        .completion_callback =
            {
                .fn = iree_net_session_on_reject_send_complete,
                .user_data = session,
            },
    };
    session->lifecycle_flags |= IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING;
    status =
        iree_net_message_endpoint_send(session->control_endpoint, &send_params);
    if (!iree_status_is_ok(status)) {
      session->lifecycle_flags &=
          ~IREE_NET_SESSION_LIFECYCLE_FLAG_REJECT_PENDING;
    }
  }
  iree_allocator_free(session->host_allocator, message_data);
  return status;
}

static iree_status_t iree_net_session_on_control_data(
    void* user_data, iree_net_control_data_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  const iree_net_session_state_t state = iree_net_session_state(session);
  if (state != IREE_NET_SESSION_STATE_OPERATIONAL &&
      state != IREE_NET_SESSION_STATE_DRAINING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "control DATA received in session state %d",
                            (int)state);
  }
  return session->callbacks.on_control_data(session->callbacks.user_data,
                                            session, flags, payload, lease);
}

static void iree_net_session_on_control_goaway(void* user_data,
                                               uint32_t reason_code) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_slim_mutex_lock(&session->mutex);
  if (iree_net_session_state(session) == IREE_NET_SESSION_STATE_OPERATIONAL) {
    iree_net_session_set_state_locked(session, IREE_NET_SESSION_STATE_DRAINING);
  }
  iree_slim_mutex_unlock(&session->mutex);
  session->callbacks.on_goaway(session->callbacks.user_data, session,
                               reason_code);
}

static void iree_net_session_on_control_error(void* user_data,
                                              iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_fail(
      session,
      iree_status_annotate(status, IREE_SV("control endpoint failed")));
}

static iree_status_t iree_net_session_on_bootstrap_message(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  (void)lease;
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_bootstrap_peer_info_view_t remote_peer;
  memset(&remote_peer, 0, sizeof(remote_peer));
  iree_net_bootstrap_capabilities_t negotiated_capabilities =
      IREE_NET_BOOTSTRAP_CAPABILITY_NONE;

  iree_slim_mutex_lock(&session->mutex);
  if (iree_any_bit_set(
          session->lifecycle_flags,
          IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED)) {
    iree_slim_mutex_unlock(&session->mutex);
    return iree_ok_status();
  }
  const bool is_server = session->bootstrap.phase ==
                         IREE_NET_SESSION_BOOTSTRAP_PHASE_SERVER_WAIT_HELLO;
  iree_status_t status = iree_net_session_bootstrap_process_message(
      &session->bootstrap, message, &remote_peer, &negotiated_capabilities);
  if (iree_status_is_ok(status) && is_server) {
    status = iree_net_session_send_bootstrap_locked(session);
  }
  if (iree_status_is_ok(status)) {
    iree_net_control_channel_attach(session->control_channel);
    iree_net_session_set_state_locked(session,
                                      IREE_NET_SESSION_STATE_OPERATIONAL);
  } else if (is_server) {
    iree_status_t reject_status =
        iree_net_session_send_reject_locked(session, iree_status_code(status));
    if (!iree_status_is_ok(reject_status)) {
      status = iree_status_join(
          status,
          iree_status_annotate(reject_status,
                               IREE_SV("failed to send bootstrap REJECT")));
    }
  }
  iree_slim_mutex_unlock(&session->mutex);

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(session, status);
    return iree_ok_status();
  }
  session->callbacks.on_ready(session->callbacks.user_data, session,
                              &remote_peer, negotiated_capabilities);
  return iree_ok_status();
}

static void iree_net_session_on_bootstrap_error(void* user_data,
                                                iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_fail(
      session,
      iree_status_annotate(status, IREE_SV("bootstrap endpoint failed")));
}

static void iree_net_session_on_control_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_slim_mutex_lock(&session->mutex);
  const bool deactivation_requested =
      iree_any_bit_set(session->lifecycle_flags,
                       IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED);
  if (deactivation_requested) {
    iree_slim_mutex_unlock(&session->mutex);
    iree_status_free(status);
    iree_net_session_finish_setup(session);
    return;
  }

  if (iree_status_is_ok(status)) {
    session->control_endpoint = endpoint;
    const iree_net_control_channel_callbacks_t control_callbacks = {
        .on_data = iree_net_session_on_control_data,
        .on_goaway = iree_net_session_on_control_goaway,
        .on_error = iree_net_session_on_control_error,
        .user_data = session,
    };
    status = iree_net_control_channel_allocate(endpoint, control_callbacks,
                                               session->host_allocator,
                                               &session->control_channel);
  }
  if (iree_status_is_ok(status)) {
    const iree_net_message_endpoint_callbacks_t bootstrap_callbacks = {
        .on_message = iree_net_session_on_bootstrap_message,
        .on_error = iree_net_session_on_bootstrap_error,
        .user_data = session,
    };
    iree_net_message_endpoint_set_callbacks(endpoint, bootstrap_callbacks);
    status = iree_net_message_endpoint_activate(endpoint);
  }
  if (iree_status_is_ok(status) &&
      session->bootstrap.phase ==
          IREE_NET_SESSION_BOOTSTRAP_PHASE_CLIENT_SEND_HELLO) {
    status = iree_net_session_send_bootstrap_locked(session);
  }
  iree_slim_mutex_unlock(&session->mutex);

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to initialize control endpoint")));
  } else {
    iree_status_free(status);
  }
  iree_net_session_finish_setup(session);
}

static iree_status_t iree_net_session_validate_endpoint_capacity(
    const iree_net_session_t* session, iree_net_connection_t* connection) {
  const uint32_t application_endpoint_count =
      session->bootstrap.application_endpoint_count;
  const uint32_t required_endpoint_count = application_endpoint_count + 1u;
  const uint32_t max_endpoint_count =
      iree_net_connection_max_endpoint_count(connection);
  if (max_endpoint_count < required_endpoint_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "connection has %" PRIu32
                            " endpoint slots but the session requires %" PRIu32
                            " (1 control + %" PRIu32 " application)",
                            max_endpoint_count, required_endpoint_count,
                            application_endpoint_count);
  }
  return iree_ok_status();
}

static void iree_net_session_on_connect(void* user_data, iree_status_t status,
                                        iree_net_connection_t* connection) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_slim_mutex_lock(&session->mutex);
  const bool deactivation_requested =
      iree_any_bit_set(session->lifecycle_flags,
                       IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED);
  if (iree_status_is_ok(status)) {
    session->connection = connection;
  }
  if (deactivation_requested) {
    iree_slim_mutex_unlock(&session->mutex);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      iree_net_connection_release(connection);
    } else {
      iree_status_free(status);
    }
    iree_net_session_finish_setup(session);
    return;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_session_validate_endpoint_capacity(session, connection);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_connection_open_endpoint(
        connection, (iree_net_endpoint_ready_callback_t){
                        .fn = iree_net_session_on_control_endpoint_ready,
                        .user_data = session,
                    });
  }
  iree_slim_mutex_unlock(&session->mutex);

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to establish connection")));
    iree_net_session_finish_setup(session);
  } else {
    iree_status_free(status);
  }
}

static iree_status_t iree_net_session_create(
    iree_net_session_bootstrap_role_t role,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  *out_session = NULL;
  if (!callbacks.on_ready || !callbacks.on_control_data ||
      !callbacks.on_goaway || !callbacks.on_error ||
      !callbacks.on_deactivated) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all session callbacks are required");
  }

  iree_net_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*session),
                                             (void**)&session));
  memset(session, 0, sizeof(*session));
  iree_atomic_ref_count_init(&session->ref_count);
  session->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&session->mutex);
  iree_net_transport_connect_operation_initialize(&session->connect_operation);
  iree_atomic_store(&session->state, IREE_NET_SESSION_STATE_BOOTSTRAPPING,
                    iree_memory_order_relaxed);
  session->lifecycle_flags = IREE_NET_SESSION_LIFECYCLE_FLAG_SETUP_PENDING;
  session->callbacks = callbacks;

  iree_status_t status = iree_net_session_bootstrap_initialize(
      role, &options->local_peer, options->required_capabilities,
      host_allocator, &session->bootstrap);
  if (!iree_status_is_ok(status)) {
    iree_net_transport_connect_operation_deinitialize(
        &session->connect_operation);
    iree_slim_mutex_deinitialize(&session->mutex);
    iree_allocator_free(host_allocator, session);
    return status;
  }

  // Retained until after the fixed on_deactivated callback returns.
  iree_net_session_retain(session);
  *out_session = session;
  return iree_ok_status();
}

static void iree_net_session_abort_creation(iree_net_session_t* session) {
  iree_atomic_store(&session->state, IREE_NET_SESSION_STATE_DEACTIVATED,
                    iree_memory_order_relaxed);
  iree_net_session_destroy(session);
}

iree_status_t iree_net_session_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  if (!out_session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "session output is required");
  }
  *out_session = NULL;
  if (!factory || !proactor || !receive_pool || !options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "factory, proactor, receive pool, and options are required");
  }

  iree_net_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_net_session_create(IREE_NET_SESSION_BOOTSTRAP_ROLE_CLIENT, options,
                              callbacks, host_allocator, &session));
  iree_status_t status = iree_net_transport_factory_connect(
      factory, address, proactor, receive_pool,
      (iree_net_transport_connect_callback_t){
          .fn = iree_net_session_on_connect,
          .user_data = session,
      },
      &session->connect_operation);
  if (!iree_status_is_ok(status)) {
    iree_net_session_abort_creation(session);
    return status;
  }

  *out_session = session;
  return iree_ok_status();
}

iree_status_t iree_net_session_accept(iree_net_connection_t* connection,
                                      const iree_net_session_options_t* options,
                                      iree_net_session_callbacks_t callbacks,
                                      iree_allocator_t host_allocator,
                                      iree_net_session_t** out_session) {
  if (!out_session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "session output is required");
  }
  *out_session = NULL;
  if (!connection || !options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "connection and options are required");
  }

  iree_net_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_net_session_create(IREE_NET_SESSION_BOOTSTRAP_ROLE_SERVER, options,
                              callbacks, host_allocator, &session));
  iree_net_connection_retain(connection);
  session->connection = connection;
  iree_status_t status =
      iree_net_session_validate_endpoint_capacity(session, connection);
  if (iree_status_is_ok(status)) {
    status = iree_net_connection_open_endpoint(
        connection, (iree_net_endpoint_ready_callback_t){
                        .fn = iree_net_session_on_control_endpoint_ready,
                        .user_data = session,
                    });
  }
  if (!iree_status_is_ok(status)) {
    iree_net_session_abort_creation(session);
    return status;
  }

  *out_session = session;
  return iree_ok_status();
}

static iree_status_t iree_net_session_validate_endpoint_admission_locked(
    iree_net_session_t* session) {
  if (iree_net_session_state(session) != IREE_NET_SESSION_STATE_OPERATIONAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "application endpoints require an operational "
                            "session");
  }
  if (session->opened_application_endpoint_count >=
      session->bootstrap.application_endpoint_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "all %" PRIu32
                            " application endpoint ordinals are claimed",
                            session->bootstrap.application_endpoint_count);
  }
  return iree_ok_status();
}

iree_status_t iree_net_session_open_endpoint(
    iree_net_session_t* session, iree_net_endpoint_ready_callback_t callback) {
  iree_slim_mutex_lock(&session->mutex);
  iree_status_t status =
      iree_net_session_validate_endpoint_admission_locked(session);
  if (iree_status_is_ok(status)) {
    status = iree_net_connection_open_endpoint(session->connection, callback);
    if (iree_status_is_ok(status)) {
      ++session->opened_application_endpoint_count;
    }
  }
  iree_slim_mutex_unlock(&session->mutex);
  return status;
}

iree_status_t iree_net_session_open_direct_endpoint(
    iree_net_session_t* session,
    iree_net_direct_endpoint_ready_callback_t callback) {
  iree_slim_mutex_lock(&session->mutex);
  iree_status_t status =
      iree_net_session_validate_endpoint_admission_locked(session);
  if (iree_status_is_ok(status)) {
    status =
        iree_net_connection_open_direct_endpoint(session->connection, callback);
    if (iree_status_is_ok(status)) {
      ++session->opened_application_endpoint_count;
    }
  }
  iree_slim_mutex_unlock(&session->mutex);
  return status;
}

iree_status_t iree_net_session_send_control_data(
    iree_net_session_t* session, iree_net_control_data_flags_t flags,
    iree_async_span_list_t payload,
    iree_net_send_completion_callback_t completion_callback) {
  iree_slim_mutex_lock(&session->mutex);
  const bool operational =
      iree_net_session_state(session) == IREE_NET_SESSION_STATE_OPERATIONAL;
  iree_net_control_channel_t* control_channel = session->control_channel;
  iree_slim_mutex_unlock(&session->mutex);
  if (!operational) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "control DATA requires an operational session");
  }
  return iree_net_control_channel_send_data(control_channel, flags, payload,
                                            completion_callback);
}

iree_status_t iree_net_session_send_control_data_copy(
    iree_net_session_t* session, iree_net_control_data_flags_t flags,
    iree_async_span_list_t payload,
    iree_net_send_completion_callback_t completion_callback) {
  iree_slim_mutex_lock(&session->mutex);
  const bool operational =
      iree_net_session_state(session) == IREE_NET_SESSION_STATE_OPERATIONAL;
  iree_net_control_channel_t* control_channel = session->control_channel;
  iree_slim_mutex_unlock(&session->mutex);
  if (!operational) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "copied control DATA requires an operational session");
  }
  return iree_net_control_channel_send_data_copy(control_channel, flags,
                                                 payload, completion_callback);
}

iree_status_t iree_net_session_send_goaway(
    iree_net_session_t* session, uint32_t reason_code,
    iree_net_send_completion_callback_t completion_callback) {
  iree_slim_mutex_lock(&session->mutex);
  iree_status_t status = iree_ok_status();
  if (iree_net_session_state(session) != IREE_NET_SESSION_STATE_OPERATIONAL) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "GOAWAY requires an operational session");
  } else {
    status = iree_net_control_channel_send_goaway(
        session->control_channel, reason_code, completion_callback);
    if (iree_status_is_ok(status)) {
      iree_net_session_set_state_locked(session,
                                        IREE_NET_SESSION_STATE_DRAINING);
    }
  }
  iree_slim_mutex_unlock(&session->mutex);
  return status;
}

void iree_net_session_deactivate(iree_net_session_t* session) {
  if (!session) {
    return;
  }
  iree_slim_mutex_lock(&session->mutex);
  if (iree_net_session_state(session) != IREE_NET_SESSION_STATE_DEACTIVATED) {
    session->lifecycle_flags |=
        IREE_NET_SESSION_LIFECYCLE_FLAG_DEACTIVATION_REQUESTED;
    iree_net_session_set_state_locked(session, IREE_NET_SESSION_STATE_DRAINING);
  }
  iree_slim_mutex_unlock(&session->mutex);
  iree_net_session_drive_deactivation(session);
}
