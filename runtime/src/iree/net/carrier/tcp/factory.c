// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/tcp/factory.h"

#include <string.h>

#include "iree/async/address.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/slab.h"
#include "iree/base/alignment.h"
#include "iree/base/threading/mutex.h"

typedef struct iree_net_tcp_factory_t iree_net_tcp_factory_t;
typedef struct iree_net_tcp_listener_t iree_net_tcp_listener_t;

typedef enum iree_net_tcp_listener_state_e {
  IREE_NET_TCP_LISTENER_STATE_LISTENING = 0,
  IREE_NET_TCP_LISTENER_STATE_STOPPING = 1,
  IREE_NET_TCP_LISTENER_STATE_STOPPED = 2,
} iree_net_tcp_listener_state_t;

typedef enum iree_net_tcp_listener_flag_bits_e {
  IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING = 1u << 0,
  IREE_NET_TCP_LISTENER_FLAG_CONTROL_PENDING = 1u << 1,
  IREE_NET_TCP_LISTENER_FLAG_CANCELLATION_PENDING = 1u << 2,
} iree_net_tcp_listener_flag_bits_t;
typedef uint32_t iree_net_tcp_listener_flags_t;

struct iree_net_tcp_factory_t {
  // Public transport factory base; must be first.
  iree_net_transport_factory_t base;

  // Immutable options copied at factory creation.
  iree_net_tcp_factory_options_t options;

  // Allocator used for factory and outbound connection state.
  iree_allocator_t host_allocator;
};

struct iree_net_tcp_listener_t {
  // Public listener base; must be first.
  iree_net_listener_t base;

  // Factory retained through listener and accepted connection setup.
  iree_net_tcp_factory_t* factory;

  // Proactor dispatching accept and stopped callbacks. Retained.
  iree_async_proactor_t* proactor;

  // Bound socket retained until all native accept ownership has retired.
  iree_async_socket_t* listen_socket;

  // Immutable contact retained for diagnostics after the socket closes.
  iree_async_address_t bound_address;

  // Serializes stop requests with accept completion.
  iree_slim_mutex_t mutex;

  // Current listener lifecycle phase.
  iree_net_tcp_listener_state_t state;

  // Independent accept/control/cancellation ownership, protected by mutex.
  iree_net_tcp_listener_flags_t flags;

  // Flags selecting multishot or emulated single-shot acceptance.
  iree_async_operation_flags_t accept_operation_flags;

  // Persistent accept operation, rearmed only on single-shot backends or
  // after an unexpected terminal accept result.
  iree_async_socket_accept_operation_t accept_operation;

  // One monotonic stop handoff followed by native cancellation if needed.
  union {
    // Software-only dispatch onto the accept's poll owner.
    iree_async_nop_operation_t control;
    // Independent accept-identity retirement, after the handoff retires.
    iree_async_cancel_request_t cancellation;
  } stop;

  // Callback receiving accepted connections and listener errors.
  iree_net_listener_accept_callback_t accept_callback;

  // Callback awaiting the final listener drain.
  iree_net_listener_stopped_callback_t stopped_callback;

  // Allocator used for listener and accepted connection storage.
  iree_allocator_t host_allocator;
};

typedef enum iree_net_tcp_connect_flag_bits_e {
  IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING = 1u << 0,
  IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING = 1u << 1,
  IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_REQUESTED = 1u << 2,
  IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_PENDING = 1u << 3,
} iree_net_tcp_connect_flag_bits_t;
typedef uint32_t iree_net_tcp_connect_flags_t;

typedef struct iree_net_tcp_connect_state_t {
  // Operation establishing the outbound socket connection.
  iree_async_socket_connect_operation_t operation;

  // Caller storage protecting cancellation intent and terminal publication.
  iree_net_transport_connect_operation_t* connect_operation;

  // Independent owner/native obligations protected by connect_operation->mutex.
  iree_net_tcp_connect_flags_t flags;

  // Native result retained until all admitted ownership has retired.
  iree_status_t result_status;

  // Owner handoff storage is reusable once its callback begins.
  union {
    // Kickoff, then at most one cancellation handoff from an application
    // thread.
    iree_async_nop_operation_t control;
    // Native identity borrow issued after the cancellation handoff retires.
    iree_async_cancel_request_t cancellation;
  } dispatch;

  // Factory retained through connection publication.
  iree_net_tcp_factory_t* factory;

  // Proactor dispatching completion. Retained.
  iree_async_proactor_t* proactor;

  // Connected socket transferred into the connection on success. Retained.
  iree_async_socket_t* socket;

  // Dedicated receive pool transferred into the connection on success.
  iree_async_buffer_pool_t* receive_pool;

  // Callback receiving the published connection or terminal failure.
  iree_net_transport_connect_callback_t callback;
} iree_net_tcp_connect_state_t;

//===----------------------------------------------------------------------===//
// Connection resources
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_tcp_parse_address(
    iree_string_view_t value, iree_async_socket_type_t* out_socket_type,
    iree_async_address_t* out_address) {
  if (!value.data || iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP address is required");
  }
  if (iree_string_view_starts_with(value, IREE_SV("unix:"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP does not accept Unix socket addresses");
  }
  IREE_RETURN_IF_ERROR(iree_async_address_from_string(value, out_address));
  *out_socket_type = iree_string_view_starts_with_char(value, '[')
                         ? IREE_ASYNC_SOCKET_TYPE_TCP6
                         : IREE_ASYNC_SOCKET_TYPE_TCP;
  return iree_ok_status();
}

static iree_async_socket_options_t iree_net_tcp_client_socket_options(void) {
  return IREE_ASYNC_SOCKET_OPTION_NO_DELAY |
         IREE_ASYNC_SOCKET_OPTION_LOW_LATENCY_ACK |
         IREE_ASYNC_SOCKET_OPTION_KEEP_ALIVE |
         IREE_ASYNC_SOCKET_OPTION_ZERO_COPY;
}

static iree_async_socket_options_t iree_net_tcp_listener_socket_options(void) {
  iree_async_socket_options_t options = iree_net_tcp_client_socket_options();
#if !defined(IREE_PLATFORM_WINDOWS)
  options |= IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR;
#endif  // !IREE_PLATFORM_WINDOWS
  return options;
}

static iree_status_t iree_net_tcp_receive_pool_create(
    const iree_net_tcp_factory_options_t* options,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_async_buffer_pool_t** out_receive_pool) {
  *out_receive_pool = NULL;
  if (options->connection_options.receive_mode ==
      IREE_NET_TCP_RECEIVE_MODE_PLACEMENT) {
    return iree_ok_status();
  }
  iree_async_slab_options_t slab_options = iree_async_slab_options_default();
  slab_options.buffer_size = options->receive_buffer_size;
  slab_options.buffer_count = options->receive_buffer_count;

  iree_async_slab_t* slab = NULL;
  iree_async_region_t* region = NULL;
  iree_async_buffer_pool_t* receive_pool = NULL;
  iree_status_t status =
      iree_async_slab_create(slab_options, host_allocator, &slab);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_slab(
        proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_async_buffer_pool_create(region, host_allocator, &receive_pool);
  }
  iree_async_region_release(region);
  iree_async_slab_release(slab);
  if (iree_status_is_ok(status)) {
    *out_receive_pool = receive_pool;
  } else {
    iree_async_buffer_pool_release(receive_pool);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

static void iree_net_tcp_listener_finish_stop(
    iree_net_tcp_listener_t* listener) {
  iree_net_listener_stopped_callback_t callback = {0};
  iree_async_socket_t* listen_socket = NULL;
  iree_slim_mutex_lock(&listener->mutex);
  if (listener->state == IREE_NET_TCP_LISTENER_STATE_STOPPING &&
      !listener->flags) {
    listener->state = IREE_NET_TCP_LISTENER_STATE_STOPPED;
    callback = listener->stopped_callback;
    listen_socket = listener->listen_socket;
    listener->listen_socket = NULL;
  }
  iree_slim_mutex_unlock(&listener->mutex);
  // A stopped listener cannot service the kernel backlog. Close its native
  // endpoint before publishing retirement so later peers fail to connect.
  iree_async_socket_release(listen_socket);
  if (callback.fn) {
    callback.fn(callback.user_data);
  }
}

static iree_status_t iree_net_tcp_listener_submit_accept_locked(
    iree_net_tcp_listener_t* listener,
    iree_async_completion_fn_t completion_fn) {
  IREE_ASSERT(!iree_any_bit_set(listener->flags,
                                IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING),
              "TCP listener already has an accept pending");
  iree_async_operation_zero(&listener->accept_operation.base,
                            sizeof(listener->accept_operation));
  iree_async_operation_initialize(
      &listener->accept_operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
      listener->accept_operation_flags, completion_fn, listener);
  listener->accept_operation.listen_socket = listener->listen_socket;
  iree_status_t status = iree_async_proactor_submit_one(
      listener->proactor, &listener->accept_operation.base);
  if (iree_status_is_ok(status)) {
    listener->flags |= IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING;
  }
  return status;
}

static void iree_net_tcp_listener_free(iree_net_listener_t* base_listener) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;
  IREE_ASSERT(listener->state == IREE_NET_TCP_LISTENER_STATE_STOPPED,
              "TCP listener freed before it stopped");
  IREE_ASSERT(!listener->flags, "TCP listener freed with pending ownership");

  iree_net_tcp_factory_t* factory = listener->factory;
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_async_socket_release(listener->listen_socket);
  iree_async_proactor_release(listener->proactor);
  iree_slim_mutex_deinitialize(&listener->mutex);
  iree_allocator_free(host_allocator, listener);
  iree_net_transport_factory_release(&factory->base);
}

static void iree_net_tcp_listener_cancellation_complete(void* user_data) {
  iree_net_tcp_listener_t* listener = user_data;
  iree_slim_mutex_lock(&listener->mutex);
  listener->flags &= ~IREE_NET_TCP_LISTENER_FLAG_CANCELLATION_PENDING;
  iree_slim_mutex_unlock(&listener->mutex);
  iree_net_tcp_listener_finish_stop(listener);
}

static void iree_net_tcp_listener_stop_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE),
              "TCP listener stop NOP produced a nonterminal completion");
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)user_data;

  iree_slim_mutex_lock(&listener->mutex);
  listener->flags &= ~IREE_NET_TCP_LISTENER_FLAG_CONTROL_PENDING;
  if (iree_any_bit_set(listener->flags,
                       IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING)) {
    listener->flags |= IREE_NET_TCP_LISTENER_FLAG_CANCELLATION_PENDING;
    iree_async_cancel_request_initialize(
        (iree_async_cancel_callback_t){
            .fn = iree_net_tcp_listener_cancellation_complete,
            .user_data = listener,
        },
        &listener->stop.cancellation);
    IREE_CHECK_OK(iree_async_proactor_request_cancel(
        listener->proactor, &listener->accept_operation.base,
        &listener->stop.cancellation));
  }
  iree_slim_mutex_unlock(&listener->mutex);

  if (!iree_status_is_ok(status)) {
    listener->accept_callback.fn(listener->accept_callback.user_data, status,
                                 NULL);
  } else {
    iree_status_free(status);
  }
  iree_net_tcp_listener_finish_stop(listener);
}

static void iree_net_tcp_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)user_data;
  iree_async_socket_accept_operation_t* accept_operation =
      (iree_async_socket_accept_operation_t*)operation;
  const bool is_final =
      !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE);
  const bool is_cancelled =
      iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);

  iree_async_socket_t* accepted_socket = accept_operation->accepted_socket;
  accept_operation->accepted_socket = NULL;

  iree_status_t rearm_status = iree_ok_status();
  iree_slim_mutex_lock(&listener->mutex);
  const bool listening =
      listener->state == IREE_NET_TCP_LISTENER_STATE_LISTENING;
  if (is_final) {
    IREE_ASSERT(iree_any_bit_set(listener->flags,
                                 IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING),
                "TCP accept completed without listener ownership");
    listener->flags &= ~IREE_NET_TCP_LISTENER_FLAG_ACCEPT_PENDING;
    if (listener->state == IREE_NET_TCP_LISTENER_STATE_LISTENING &&
        !is_cancelled) {
      rearm_status = iree_net_tcp_listener_submit_accept_locked(
          listener, iree_net_tcp_listener_accept_complete);
    }
    IREE_ASSERT(!is_cancelled ||
                    listener->state == IREE_NET_TCP_LISTENER_STATE_STOPPING,
                "TCP accept cancelled while listener was not stopping");
  }
  iree_slim_mutex_unlock(&listener->mutex);

  if (is_cancelled) {
    IREE_ASSERT(!accepted_socket,
                "cancelled TCP accept produced an accepted socket");
    iree_async_socket_release(accepted_socket);
    iree_status_free(status);
  } else if (!iree_status_is_ok(status)) {
    iree_async_socket_release(accepted_socket);
    listener->accept_callback.fn(listener->accept_callback.user_data, status,
                                 NULL);
  } else if (!accepted_socket) {
    iree_status_free(status);
    listener->accept_callback.fn(
        listener->accept_callback.user_data,
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "TCP accept completed without a socket"),
        NULL);
  } else if (!listening) {
    // Stop owns unpublished native results. Releasing the socket here avoids
    // constructing a connection that the application would have to drain.
    iree_async_socket_release(accepted_socket);
    iree_status_free(status);
  } else {
    iree_async_buffer_pool_t* receive_pool = NULL;
    iree_status_t connection_status = iree_net_tcp_receive_pool_create(
        &listener->factory->options, listener->proactor,
        listener->host_allocator, &receive_pool);
    iree_net_connection_t* connection = NULL;
    if (iree_status_is_ok(connection_status)) {
      connection_status = iree_net_tcp_connection_create(
          listener->proactor, accepted_socket, receive_pool,
          &listener->factory->options.connection_options,
          listener->host_allocator, &connection);
    }
    iree_async_buffer_pool_release(receive_pool);
    iree_async_socket_release(accepted_socket);
    iree_status_free(status);
    listener->accept_callback.fn(listener->accept_callback.user_data,
                                 connection_status, connection);
  }

  if (!iree_status_is_ok(rearm_status)) {
    listener->accept_callback.fn(listener->accept_callback.user_data,
                                 rearm_status, NULL);
  }
  if (is_final) {
    // Retire the target only after accepted sockets and claimed callbacks are
    // done: withdrawal may publish stopped and destroy the entire listener.
    iree_slim_mutex_lock(&listener->mutex);
    const bool cancellation_pending = iree_any_bit_set(
        listener->flags, IREE_NET_TCP_LISTENER_FLAG_CANCELLATION_PENDING);
    iree_slim_mutex_unlock(&listener->mutex);
    if (cancellation_pending) {
      iree_async_proactor_cancel_request_target_retired(
          listener->proactor, &listener->stop.cancellation);
      return;
    }
    iree_net_tcp_listener_finish_stop(listener);
  }
}

static iree_status_t iree_net_tcp_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;

  iree_slim_mutex_lock(&listener->mutex);
  iree_status_t status = iree_ok_status();
  if (listener->state != IREE_NET_TCP_LISTENER_STATE_LISTENING) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "TCP listener is not listening");
  } else {
    listener->state = IREE_NET_TCP_LISTENER_STATE_STOPPING;
    listener->stopped_callback = callback;
    listener->flags |= IREE_NET_TCP_LISTENER_FLAG_CONTROL_PENDING;
    iree_async_operation_initialize(
        &listener->stop.control.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_listener_stop_complete,
        listener);
    // A retained proactor admits this private software-only NOP without
    // allocation or native queue capacity. Native cancellation is owner work.
    IREE_CHECK_OK(iree_async_proactor_submit_one(listener->proactor,
                                                 &listener->stop.control.base));
  }
  iree_slim_mutex_unlock(&listener->mutex);
  return status;
}

static iree_status_t iree_net_tcp_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_tcp_listener_t* listener = (iree_net_tcp_listener_t*)base_listener;
  if (!out_address) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bound address output is required");
  }
  *out_address = iree_string_view_empty();
  if (!buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bound address storage is required");
  }

  return iree_async_address_format(&listener->bound_address, buffer_capacity,
                                   buffer, out_address);
}

static const iree_net_listener_vtable_t iree_net_tcp_listener_vtable = {
    .free = iree_net_tcp_listener_free,
    .stop = iree_net_tcp_listener_stop,
    .query_bound_address = iree_net_tcp_listener_query_bound_address,
};

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

static void iree_net_tcp_connect_state_destroy(
    iree_net_tcp_connect_state_t* state) {
  iree_net_tcp_factory_t* factory = state->factory;
  iree_allocator_t host_allocator = factory->host_allocator;
  iree_async_socket_release(state->socket);
  iree_async_buffer_pool_release(state->receive_pool);
  iree_async_proactor_release(state->proactor);
  iree_allocator_free(host_allocator, state);
  iree_net_transport_factory_release(&factory->base);
}

static void iree_net_tcp_connect_finish(iree_net_tcp_connect_state_t* state) {
  iree_net_transport_connect_operation_t* operation = state->connect_operation;
  iree_slim_mutex_lock(&operation->mutex);
  if (iree_any_bit_set(state->flags,
                       IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING |
                           IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING |
                           IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_PENDING)) {
    iree_slim_mutex_unlock(&operation->mutex);
    return;
  }
  operation->binding.cancel_fn = NULL;
  operation->binding.user_data = NULL;
  iree_status_t status = state->result_status;
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(state->flags,
                       IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_REQUESTED)) {
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
  iree_slim_mutex_unlock(&operation->mutex);

  iree_net_transport_connect_callback_t callback = state->callback;
  iree_net_connection_t* connection = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_tcp_connection_create(
        state->proactor, state->socket, state->receive_pool,
        &state->factory->options.connection_options,
        state->factory->host_allocator, &connection);
  }
  iree_net_tcp_connect_state_destroy(state);
  callback.fn(callback.user_data, status, connection);
}

static void iree_net_tcp_connect_cancellation_complete(void* user_data) {
  iree_net_tcp_connect_state_t* state = user_data;
  iree_slim_mutex_lock(&state->connect_operation->mutex);
  state->flags &= ~IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_PENDING;
  iree_slim_mutex_unlock(&state->connect_operation->mutex);
  iree_net_tcp_connect_finish(state);
}

static void iree_net_tcp_connect_complete(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags) {
  (void)operation;
  IREE_ASSERT(!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE),
              "TCP connect produced a nonterminal completion");
  iree_net_tcp_connect_state_t* state =
      (iree_net_tcp_connect_state_t*)user_data;
  iree_slim_mutex_lock(&state->connect_operation->mutex);
  state->result_status = iree_status_join(state->result_status, status);
  state->flags &= ~IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING;
  const bool cancellation_pending = iree_any_bit_set(
      state->flags, IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_PENDING);
  iree_slim_mutex_unlock(&state->connect_operation->mutex);
  if (cancellation_pending) {
    // Withdrawal can finish the join inline and destroy state.
    iree_async_proactor_cancel_request_target_retired(
        state->proactor, &state->dispatch.cancellation);
    return;
  }
  iree_net_tcp_connect_finish(state);
}

static void iree_net_tcp_connect_cancel_dispatch(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_tcp_connect_state_t* state = user_data;
  iree_slim_mutex_lock(&state->connect_operation->mutex);
  state->result_status = iree_status_join(state->result_status, status);
  state->flags &= ~IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING;
  if (iree_any_bit_set(state->flags,
                       IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING)) {
    state->flags |= IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_PENDING;
    iree_async_cancel_request_initialize(
        (iree_async_cancel_callback_t){
            .fn = iree_net_tcp_connect_cancellation_complete,
            .user_data = state,
        },
        &state->dispatch.cancellation);
    // This private unlinked target and idle request satisfy admission's only
    // fallible preconditions. Native errors are owned by the poll result.
    IREE_CHECK_OK(iree_async_proactor_request_cancel(
        state->proactor, &state->operation.base,
        &state->dispatch.cancellation));
  }
  iree_slim_mutex_unlock(&state->connect_operation->mutex);
  iree_net_tcp_connect_finish(state);
}

// Called with the public operation mutex held. The software-only NOP requires
// no native submission slot or allocation on a retained proactor.
static void iree_net_tcp_connect_cancel(void* user_data) {
  iree_net_tcp_connect_state_t* state = user_data;
  if (iree_any_bit_set(state->flags,
                       IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_REQUESTED)) {
    return;
  }
  state->flags |= IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_REQUESTED;
  if (iree_any_bit_set(state->flags,
                       IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING) ||
      !iree_any_bit_set(state->flags,
                        IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING)) {
    return;
  }
  state->flags |= IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING;
  iree_async_operation_initialize(&state->dispatch.control.base,
                                  IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_tcp_connect_cancel_dispatch, state);
  IREE_CHECK_OK(iree_async_proactor_submit_one(state->proactor,
                                               &state->dispatch.control.base));
}

static void iree_net_tcp_connect_start(void* user_data,
                                       iree_async_operation_t* operation,
                                       iree_status_t status,
                                       iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_tcp_connect_state_t* state = user_data;
  iree_slim_mutex_lock(&state->connect_operation->mutex);
  state->flags &= ~IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING;
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(state->flags,
                        IREE_NET_TCP_CONNECT_FLAG_CANCELLATION_REQUESTED)) {
    status =
        iree_async_proactor_submit_one(state->proactor, &state->operation.base);
    if (iree_status_is_ok(status)) {
      state->flags |= IREE_NET_TCP_CONNECT_FLAG_NATIVE_PENDING;
    }
  }
  state->result_status = status;
  iree_slim_mutex_unlock(&state->connect_operation->mutex);
  iree_net_tcp_connect_finish(state);
}

static void iree_net_tcp_factory_destroy(
    iree_net_transport_factory_t* base_factory) {
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  iree_allocator_t host_allocator = factory->host_allocator;
  iree_allocator_free(host_allocator, factory);
}

static iree_net_transport_capabilities_t
iree_net_tcp_factory_query_capabilities(
    iree_net_transport_factory_t* base_factory) {
  (void)base_factory;
  return IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
         IREE_NET_TRANSPORT_CAPABILITY_ORDERED;
}

static iree_status_t iree_net_tcp_factory_connect(
    iree_net_transport_factory_t* base_factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    iree_net_transport_connect_callback_t callback,
    iree_net_transport_connect_operation_t* operation) {
  (void)receive_pool;
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  if (!proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP client proactor is required");
  }

  iree_async_socket_type_t socket_type = IREE_ASYNC_SOCKET_TYPE_TCP;
  iree_async_address_t remote_address;
  IREE_RETURN_IF_ERROR(
      iree_net_tcp_parse_address(address, &socket_type, &remote_address));

  iree_net_tcp_connect_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(factory->host_allocator,
                                             sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->factory = factory;
  iree_net_transport_factory_retain(base_factory);
  state->proactor = proactor;
  iree_async_proactor_retain(proactor);
  state->callback = callback;
  state->connect_operation = operation;

  iree_status_t status = iree_net_tcp_receive_pool_create(
      &factory->options, proactor, factory->host_allocator,
      &state->receive_pool);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_create(proactor, socket_type,
                                      iree_net_tcp_client_socket_options(),
                                      &state->socket);
  }
  if (iree_status_is_ok(status)) {
    iree_async_operation_initialize(
        &state->operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_connect_complete, state);
    state->operation.socket = state->socket;
    state->operation.address = remote_address;
    operation->binding.cancel_fn = iree_net_tcp_connect_cancel;
    operation->binding.user_data = state;
    state->flags = IREE_NET_TCP_CONNECT_FLAG_CONTROL_PENDING;
    iree_async_operation_initialize(
        &state->dispatch.control.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_tcp_connect_start, state);
    status =
        iree_async_proactor_submit_one(proactor, &state->dispatch.control.base);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_tcp_connect_state_destroy(state);
  }
  return status;
}

static iree_status_t iree_net_tcp_factory_create_listener(
    iree_net_transport_factory_t* base_factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    iree_net_listener_accept_callback_t accept_callback,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  (void)receive_pool;
  iree_net_tcp_factory_t* factory = (iree_net_tcp_factory_t*)base_factory;
  *out_listener = NULL;
  if (!proactor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP listener proactor is required");
  }

  iree_async_socket_type_t socket_type = IREE_ASYNC_SOCKET_TYPE_TCP;
  iree_async_address_t address;
  IREE_RETURN_IF_ERROR(
      iree_net_tcp_parse_address(bind_address, &socket_type, &address));

  iree_async_socket_t* listen_socket = NULL;
  IREE_RETURN_IF_ERROR(iree_async_socket_create(
      proactor, socket_type, iree_net_tcp_listener_socket_options(),
      &listen_socket));
  iree_status_t status = iree_async_socket_bind(listen_socket, &address);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listen_socket,
                                      factory->options.listen_backlog);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_query_local_address(listen_socket, &address);
  }

  iree_net_tcp_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*listener),
                                   (void**)&listener);
  }
  if (iree_status_is_ok(status)) {
    memset(listener, 0, sizeof(*listener));
    listener->base.vtable = &iree_net_tcp_listener_vtable;
    listener->factory = factory;
    iree_net_transport_factory_retain(base_factory);
    listener->proactor = proactor;
    iree_async_proactor_retain(proactor);
    listener->listen_socket = listen_socket;
    listener->bound_address = address;
    listen_socket = NULL;
    iree_slim_mutex_initialize(&listener->mutex);
    listener->state = IREE_NET_TCP_LISTENER_STATE_LISTENING;
    listener->accept_operation_flags =
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS;
    if (iree_any_bit_set(iree_async_proactor_query_capabilities(proactor),
                         IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
      listener->accept_operation_flags |= IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
    }
    listener->accept_callback = accept_callback;
    listener->host_allocator = host_allocator;

    iree_slim_mutex_lock(&listener->mutex);
    status = iree_net_tcp_listener_submit_accept_locked(
        listener, iree_net_tcp_listener_accept_complete);
    iree_slim_mutex_unlock(&listener->mutex);
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    if (listener) {
      iree_async_socket_release(listener->listen_socket);
      iree_async_proactor_release(listener->proactor);
      iree_net_transport_factory_release(&listener->factory->base);
      iree_slim_mutex_deinitialize(&listener->mutex);
      iree_allocator_free(host_allocator, listener);
    }
    iree_async_socket_release(listen_socket);
  }
  return status;
}

static const iree_net_transport_factory_vtable_t iree_net_tcp_factory_vtable = {
    .destroy = iree_net_tcp_factory_destroy,
    .query_capabilities = iree_net_tcp_factory_query_capabilities,
    .connect = iree_net_tcp_factory_connect,
    .create_listener = iree_net_tcp_factory_create_listener,
};

iree_status_t iree_net_tcp_factory_create(
    const iree_net_tcp_factory_options_t* options,
    iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_ASSERT_ARGUMENT(out_factory);
  *out_factory = NULL;
  iree_net_tcp_factory_options_t default_options =
      iree_net_tcp_factory_options_default();
  if (!options) {
    options = &default_options;
  }
  IREE_RETURN_IF_ERROR(
      iree_net_tcp_connection_options_validate(&options->connection_options));
  if (options->receive_buffer_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TCP receive buffer size must be nonzero");
  }
  if (!iree_host_size_is_power_of_two(options->receive_buffer_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "TCP receive buffer count must be a nonzero power of two");
  }
  if (options->receive_buffer_count > IREE_NET_TCP_MAX_RECEIVE_BUFFER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TCP receive buffer count %" PRIhsz
                            " exceeds the portable maximum %u",
                            options->receive_buffer_count,
                            (unsigned)IREE_NET_TCP_MAX_RECEIVE_BUFFER_COUNT);
  }
  iree_host_size_t receive_storage_size = 0;
  if (!iree_host_size_checked_mul(options->receive_buffer_size,
                                  options->receive_buffer_count,
                                  &receive_storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TCP receive storage size overflows host size");
  }

  iree_net_tcp_factory_t* factory = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*factory),
                                             (void**)&factory));
  memset(factory, 0, sizeof(*factory));
  iree_net_transport_factory_initialize(&iree_net_tcp_factory_vtable,
                                        &factory->base);
  factory->options = *options;
  factory->host_allocator = host_allocator;
  *out_factory = &factory->base;
  return iree_ok_status();
}
