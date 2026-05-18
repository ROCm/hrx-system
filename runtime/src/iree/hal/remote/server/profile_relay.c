// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/profile_relay.h"

#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"

#define IREE_HAL_REMOTE_PROFILE_ACK_WINDOW_INITIAL_CAPACITY 64

typedef struct iree_hal_remote_server_profile_response_node_t {
  // Intrusive node held by the relay ACK window until callbacks are observed.
  iree_net_sequence_node_t sequence_node;

  // Host allocator used to release this node.
  iree_allocator_t host_allocator;

  // Session slot used to send the deferred lifecycle response.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the response is ready.
  uint64_t session_id;

  // Request envelope copied from the profiling lifecycle message.
  iree_hal_remote_control_envelope_t request_envelope;

  // Lifecycle operation status. Consumed when the response is sent or dropped.
  iree_status_t status;
} iree_hal_remote_server_profile_response_node_t;

static void iree_hal_remote_server_profile_relay_free_response_nodes(
    iree_net_sequence_node_t* pending_list) {
  while (pending_list) {
    iree_net_sequence_node_t* next = pending_list->next;
    iree_hal_remote_server_profile_response_node_t* response_node =
        iree_containerof(pending_list,
                         iree_hal_remote_server_profile_response_node_t,
                         sequence_node);
    iree_status_free(response_node->status);
    iree_allocator_free(response_node->host_allocator, response_node);
    pending_list = next;
  }
}

static iree_status_t iree_hal_remote_server_profile_relay_send_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    uint64_t target_sequence, iree_status_t status,
    iree_status_code_t transfer_failure_code,
    uint64_t transfer_failure_sequence) {
  if (transfer_failure_sequence != 0 &&
      transfer_failure_sequence <= target_sequence) {
    status = iree_status_join(
        status,
        iree_make_status(transfer_failure_code,
                         "remote profile transfer failed before lifecycle "
                         "response was ready"));
  }
  if (iree_status_is_ok(status)) {
    return iree_hal_remote_server_session_send_response(
        session_slot, request_envelope, IREE_STATUS_OK, NULL, 0);
  }
  return iree_hal_remote_server_session_send_error_response(
      session_slot, request_envelope, status);
}

static iree_status_t
iree_hal_remote_server_profile_relay_process_ready_responses(
    iree_net_sequence_node_t* ready_list,
    iree_status_code_t transfer_failure_code,
    uint64_t transfer_failure_sequence) {
  iree_status_t status = iree_ok_status();
  while (ready_list) {
    iree_net_sequence_node_t* next = ready_list->next;
    iree_hal_remote_server_profile_response_node_t* response_node =
        iree_containerof(ready_list,
                         iree_hal_remote_server_profile_response_node_t,
                         sequence_node);
    bool session_active = false;
    iree_slim_mutex_lock(&response_node->session_slot->server->session_mutex);
    session_active =
        response_node->session_slot->session_id == response_node->session_id &&
        response_node->session_slot->session != NULL;
    iree_slim_mutex_unlock(&response_node->session_slot->server->session_mutex);
    iree_status_t send_status = iree_ok_status();
    if (session_active) {
      send_status = iree_hal_remote_server_profile_relay_send_response(
          response_node->session_slot, &response_node->request_envelope,
          response_node->sequence_node.sequence, response_node->status,
          transfer_failure_code, transfer_failure_sequence);
    } else {
      iree_status_free(response_node->status);
    }
    response_node->status = iree_ok_status();
    status = iree_status_join(status, send_status);
    iree_allocator_t host_allocator = response_node->host_allocator;
    iree_allocator_free(host_allocator, response_node);
    ready_list = next;
  }
  return status;
}

iree_status_t iree_hal_remote_server_profile_relay_initialize(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  return iree_net_sequence_window_initialize(
      /*initial_observed_sequence=*/0,
      IREE_HAL_REMOTE_PROFILE_ACK_WINDOW_INITIAL_CAPACITY, host_allocator,
      &iree_hal_remote_server_bulk_session_profile_relay(session_slot)
           ->ack_window);
}

bool iree_hal_remote_server_profile_relay_is_empty(
    const iree_hal_remote_server_profile_relay_t* relay) {
  return !relay->pending_transfer_head && !relay->pending_transfer_tail &&
         !relay->active_sink && !relay->ack_window.storage &&
         relay->transfer_failure_code == IREE_STATUS_OK &&
         relay->transfer_failure_sequence == 0;
}

void iree_hal_remote_server_profile_relay_deinitialize(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_profile_relay_t* relay =
      iree_hal_remote_server_bulk_session_profile_relay(session_slot);
  iree_net_sequence_node_t* pending_responses = NULL;
  iree_net_sequence_window_take_pending(&relay->ack_window, &pending_responses);
  iree_net_sequence_window_deinitialize(&relay->ack_window);
  iree_hal_remote_server_profile_relay_free_response_nodes(pending_responses);

  iree_hal_profile_sink_release(relay->active_sink);
  relay->active_sink = NULL;
  relay->transfer_failure_code = IREE_STATUS_OK;
  relay->transfer_failure_sequence = 0;
}

iree_hal_profile_sink_t*
iree_hal_remote_server_profile_relay_retain_active_sink(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_profile_sink_t* profile_sink = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  profile_sink = iree_hal_remote_server_bulk_session_profile_relay(session_slot)
                     ->active_sink;
  iree_hal_profile_sink_retain(profile_sink);
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  return profile_sink;
}

iree_hal_profile_sink_t*
iree_hal_remote_server_profile_relay_detach_active_sink(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_profile_sink_t* expected_sink) {
  iree_hal_profile_sink_t* profile_sink = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  if (!expected_sink ||
      iree_hal_remote_server_bulk_session_profile_relay(session_slot)
              ->active_sink == expected_sink) {
    profile_sink =
        iree_hal_remote_server_bulk_session_profile_relay(session_slot)
            ->active_sink;
    iree_hal_remote_server_bulk_session_profile_relay(session_slot)
        ->active_sink = NULL;
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  return profile_sink;
}

void iree_hal_remote_server_profile_relay_cancel(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->server) return;
  iree_hal_profile_sink_t* profile_sink =
      iree_hal_remote_server_profile_relay_detach_active_sink(session_slot,
                                                              NULL);
  if (!profile_sink) return;
  iree_status_t status =
      iree_hal_device_profiling_end(session_slot->server->devices[0]);
  iree_status_free(status);
  iree_hal_profile_sink_release(profile_sink);
}

iree_status_t iree_hal_remote_server_profile_relay_prepare_begin(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_profile_sink_t* profile_sink) {
  iree_hal_remote_server_profile_relay_t* relay =
      iree_hal_remote_server_bulk_session_profile_relay(session_slot);
  iree_status_t status = iree_ok_status();
  iree_net_sequence_node_t* pending_responses = NULL;

  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  if (!session_slot->session) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (relay->active_sink) {
    status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "remote profiling session already active");
  }
  if (iree_status_is_ok(status)) {
    iree_net_sequence_window_take_pending(&relay->ack_window,
                                          &pending_responses);
    iree_net_sequence_window_deinitialize(&relay->ack_window);
    status = iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0,
        IREE_HAL_REMOTE_PROFILE_ACK_WINDOW_INITIAL_CAPACITY,
        session_slot->server->host_allocator, &relay->ack_window);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_sink_retain(profile_sink);
    relay->active_sink = profile_sink;
    relay->transfer_failure_code = IREE_STATUS_OK;
    relay->transfer_failure_sequence = 0;
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);

  iree_hal_remote_server_profile_relay_free_response_nodes(pending_responses);
  return status;
}

iree_status_t iree_hal_remote_server_profile_relay_defer_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    uint64_t target_sequence, iree_status_t operation_status) {
  iree_hal_remote_server_profile_relay_t* relay =
      iree_hal_remote_server_bulk_session_profile_relay(session_slot);
  iree_status_code_t transfer_failure_code = IREE_STATUS_OK;
  uint64_t transfer_failure_sequence = 0;
  if (target_sequence == 0) {
    iree_slim_mutex_lock(&session_slot->server->session_mutex);
    transfer_failure_code = relay->transfer_failure_code;
    transfer_failure_sequence = relay->transfer_failure_sequence;
    iree_slim_mutex_unlock(&session_slot->server->session_mutex);
    return iree_hal_remote_server_profile_relay_send_response(
        session_slot, request_envelope, target_sequence, operation_status,
        transfer_failure_code, transfer_failure_sequence);
  }

  iree_allocator_t host_allocator = session_slot->server->host_allocator;
  iree_hal_remote_server_profile_response_node_t* response_node = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*response_node), (void**)&response_node);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(operation_status, status);
    iree_slim_mutex_lock(&session_slot->server->session_mutex);
    transfer_failure_code = relay->transfer_failure_code;
    transfer_failure_sequence = relay->transfer_failure_sequence;
    iree_slim_mutex_unlock(&session_slot->server->session_mutex);
    return iree_hal_remote_server_profile_relay_send_response(
        session_slot, request_envelope, target_sequence, status,
        transfer_failure_code, transfer_failure_sequence);
  }

  memset(response_node, 0, sizeof(*response_node));
  response_node->host_allocator = host_allocator;
  response_node->session_slot = session_slot;
  response_node->session_id = session_slot->session_id;
  response_node->request_envelope = *request_envelope;
  response_node->status = operation_status;
  operation_status = iree_ok_status();

  iree_net_sequence_node_t* ready_list = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  transfer_failure_code = relay->transfer_failure_code;
  transfer_failure_sequence = relay->transfer_failure_sequence;
  status = iree_net_sequence_window_defer_until(
      &relay->ack_window, target_sequence, &response_node->sequence_node,
      &ready_list);
  if (iree_status_is_ok(status)) response_node = NULL;
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);

  if (!iree_status_is_ok(status)) {
    status = iree_status_join(response_node->status, status);
    response_node->status = iree_ok_status();
    iree_allocator_free(host_allocator, response_node);
    return iree_hal_remote_server_profile_relay_send_response(
        session_slot, request_envelope, target_sequence, status,
        transfer_failure_code, transfer_failure_sequence);
  }
  return iree_hal_remote_server_profile_relay_process_ready_responses(
      ready_list, transfer_failure_code, transfer_failure_sequence);
}

iree_status_t iree_hal_remote_server_profile_relay_observe_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t sequence,
    iree_status_t status) {
  iree_hal_remote_server_profile_relay_t* relay =
      iree_hal_remote_server_bulk_session_profile_relay(session_slot);
  iree_status_code_t transfer_failure_code = IREE_STATUS_OK;
  uint64_t transfer_failure_sequence = 0;
  iree_net_sequence_node_t* ready_list = NULL;
  iree_status_t observe_status = iree_ok_status();

  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  if (sequence == 0) {
    observe_status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "profile transfer ACK sequence must be nonzero");
  } else {
    if (!iree_status_is_ok(status) &&
        (relay->transfer_failure_sequence == 0 ||
         sequence < relay->transfer_failure_sequence)) {
      relay->transfer_failure_code = iree_status_code(status);
      relay->transfer_failure_sequence = sequence;
    }
    transfer_failure_code = relay->transfer_failure_code;
    transfer_failure_sequence = relay->transfer_failure_sequence;
    observe_status = iree_net_sequence_window_observe(&relay->ack_window,
                                                      sequence, &ready_list);
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);

  iree_status_free(status);
  if (iree_status_is_ok(observe_status)) {
    observe_status =
        iree_hal_remote_server_profile_relay_process_ready_responses(
            ready_list, transfer_failure_code, transfer_failure_sequence);
  } else {
    iree_hal_remote_server_profile_relay_free_response_nodes(ready_list);
  }
  return observe_status;
}
