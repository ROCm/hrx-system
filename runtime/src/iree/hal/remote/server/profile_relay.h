// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server profile relay state.
//
// Owns per-session state that connects HAL-native profiling callbacks to remote
// lifecycle responses: active sink ownership, callback acknowledgement
// ordering, first-failed callback sequencing, and server-originated callback
// backlog. The bulk sender owns the active transfer state machine; this relay
// owns the session-level invariants those transfers report into.

#ifndef IREE_HAL_REMOTE_SERVER_PROFILE_RELAY_H_
#define IREE_HAL_REMOTE_SERVER_PROFILE_RELAY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/net/channel/util/sequence_window.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_control_envelope_t
    iree_hal_remote_control_envelope_t;
typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;
typedef struct iree_hal_remote_server_profile_pending_transfer_t
    iree_hal_remote_server_profile_pending_transfer_t;

typedef struct iree_hal_remote_server_profile_relay_t {
  // FIFO head of profile callbacks awaiting active bulk transfer capacity.
  // Protected by the owning bulk session mutex.
  iree_hal_remote_server_profile_pending_transfer_t* pending_transfer_head;

  // FIFO tail of profile callbacks awaiting active bulk transfer capacity.
  // Protected by the owning bulk session mutex.
  iree_hal_remote_server_profile_pending_transfer_t* pending_transfer_tail;

  // Active server-created sink, or NULL when no profiling session is active.
  // Protected by iree_hal_remote_server_t::session_mutex.
  iree_hal_profile_sink_t* active_sink;

  // Callback ACK sequence window gating lifecycle responses.
  // Protected by iree_hal_remote_server_t::session_mutex.
  iree_net_sequence_window_t ack_window;

  // First failed callback transfer status code, or IREE_STATUS_OK.
  // Protected by iree_hal_remote_server_t::session_mutex.
  iree_status_code_t transfer_failure_code;

  // First failed callback sequence, or 0 when no transfer has failed.
  // Protected by iree_hal_remote_server_t::session_mutex.
  uint64_t transfer_failure_sequence;
} iree_hal_remote_server_profile_relay_t;

// Initializes session profile relay state.
iree_status_t iree_hal_remote_server_profile_relay_initialize(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator);

// Returns true when the relay contains no reusable-session state.
bool iree_hal_remote_server_profile_relay_is_empty(
    const iree_hal_remote_server_profile_relay_t* relay);

// Deinitializes session profile relay state and releases pending responses.
void iree_hal_remote_server_profile_relay_deinitialize(
    iree_hal_remote_server_session_t* session_slot);

// Cancels an active profiling session during connection teardown.
void iree_hal_remote_server_profile_relay_cancel(
    iree_hal_remote_server_session_t* session_slot);

// Prepares the relay for a new active profiling session.
iree_status_t iree_hal_remote_server_profile_relay_prepare_begin(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_profile_sink_t* profile_sink);

// Retains the active profile sink, or returns NULL if profiling is inactive.
iree_hal_profile_sink_t*
iree_hal_remote_server_profile_relay_retain_active_sink(
    iree_hal_remote_server_session_t* session_slot);

// Detaches the active sink if it matches |expected_sink|, or unconditionally
// when |expected_sink| is NULL. Returns a retained sink reference to release.
iree_hal_profile_sink_t*
iree_hal_remote_server_profile_relay_detach_active_sink(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_profile_sink_t* expected_sink);

// Defers a lifecycle response until profile callbacks up to |target_sequence|
// have been observed.
iree_status_t iree_hal_remote_server_profile_relay_defer_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    uint64_t target_sequence, iree_status_t operation_status);

// Observes client completion of one server-originated profile transfer.
// Consumes |status|, using non-OK values to fail lifecycle responses whose
// callback range includes |sequence|.
iree_status_t iree_hal_remote_server_profile_relay_observe_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t sequence,
    iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_PROFILE_RELAY_H_
