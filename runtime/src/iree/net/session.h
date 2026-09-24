// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transport-independent peer session lifecycle.
//
// A session connects or adopts one connection, exchanges bootstrap peer
// information on endpoint ordinal 0, and hands that endpoint to the control
// DATA/GOAWAY/ERROR protocol. Remaining endpoint ordinals are opened directly
// by the application in matching order on both peers.
//
// The session owns no HAL state, frontier proxies, transport carrier, timeout,
// or callback ledger. Connection deactivation is the callback retirement
// boundary. The callback user data and all application endpoint callback
// targets must remain valid until on_deactivated returns.

#ifndef IREE_NET_SESSION_H_
#define IREE_NET_SESSION_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/net/bootstrap.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/connection.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_session_t iree_net_session_t;

// Monotonic session lifecycle state.
typedef enum iree_net_session_state_e {
  // Transport connection or peer bootstrap is in progress.
  IREE_NET_SESSION_STATE_BOOTSTRAPPING = 0,
  // Peer bootstrap completed and application operations are admitted.
  IREE_NET_SESSION_STATE_OPERATIONAL = 1,
  // New session operations are closed while existing work may drain.
  IREE_NET_SESSION_STATE_DRAINING = 2,
  // Connection deactivation and every accepted callback have completed.
  IREE_NET_SESSION_STATE_DEACTIVATED = 3,
} iree_net_session_state_t;

// Called after peer bootstrap completes.
//
// |remote_peer| borrows the received bootstrap message and is valid only for
// this callback. |negotiated_capabilities| is the intersection of both peers'
// advertised capability sets. The session is OPERATIONAL before this call.
typedef void(IREE_API_PTR* iree_net_session_ready_fn_t)(
    void* user_data, iree_net_session_t* session,
    const iree_net_bootstrap_peer_info_view_t* remote_peer,
    iree_net_bootstrap_capabilities_t negotiated_capabilities);

// Handles one application DATA message received on the control endpoint.
//
// The payload excludes the control header and borrows |lease|. Returning a
// non-OK status terminates the control endpoint and session.
typedef iree_status_t(IREE_API_PTR* iree_net_session_control_data_fn_t)(
    void* user_data, iree_net_session_t* session,
    iree_net_control_data_flags_t flags, iree_const_byte_span_t payload,
    iree_async_buffer_lease_t* lease);

// Called after a peer GOAWAY closes new session operation admission.
//
// GOAWAY does not deactivate the connection. The owner drains its application
// endpoints and calls iree_net_session_deactivate when ready.
typedef void(IREE_API_PTR* iree_net_session_goaway_fn_t)(
    void* user_data, iree_net_session_t* session, uint32_t reason_code);

// Called for the first terminal bootstrap or network error.
//
// Status ownership transfers to the callback. The session is DRAINING and has
// already requested automatic connection deactivation before this call.
typedef void(IREE_API_PTR* iree_net_session_error_fn_t)(
    void* user_data, iree_net_session_t* session, iree_status_t status);

// Called after setup and connection deactivation have fully drained.
//
// This fires exactly once. After it returns the owner may free every callback
// target. The callback may release the returned session reference only if the
// successful connect or accept call that published it has already returned.
typedef void(IREE_API_PTR* iree_net_session_deactivated_fn_t)(
    void* user_data, iree_net_session_t* session);

// Immutable application callbacks for one session.
//
// All functions are required. Message and terminal-error callbacks may overlap
// when the connection proactor dispatches concurrently. on_deactivated is the
// join point after which none can execute again.
typedef struct iree_net_session_callbacks_t {
  // Reports successful peer bootstrap.
  iree_net_session_ready_fn_t on_ready;
  // Handles application DATA on the control endpoint.
  iree_net_session_control_data_fn_t on_control_data;
  // Reports peer-initiated graceful drain.
  iree_net_session_goaway_fn_t on_goaway;
  // Reports the first terminal bootstrap or network failure.
  iree_net_session_error_fn_t on_error;
  // Reports final setup and connection drain completion.
  iree_net_session_deactivated_fn_t on_deactivated;
  // Opaque pointer passed to every callback.
  void* user_data;
} iree_net_session_callbacks_t;

// Session creation options.
typedef struct iree_net_session_options_t {
  // Local peer information captured into the bootstrap message at creation.
  iree_net_bootstrap_peer_info_t local_peer;
  // Capabilities required in the negotiated peer intersection.
  iree_net_bootstrap_capabilities_t required_capabilities;
} iree_net_session_options_t;

// Returns zero-initialized session options.
static inline iree_net_session_options_t iree_net_session_options_default(
    void) {
  iree_net_session_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

// Begins a client session connection and bootstrap.
//
// A successful call returns a BOOTSTRAPPING session immediately. The factory
// owns all resources needed by its pending connect, so |factory|, |address|,
// |proactor|, and |receive_pool| are borrowed only through this call.
// |receive_pool| may be NULL when the transport owns its receive storage;
// its requirement is determined by the factory, not the session protocol.
// Asynchronous callbacks may race with the return. The returned reference is
// owned only after this call returns and must not be released by a callback
// that wins that race.
//
// On synchronous failure no session is published and no callback fires.
IREE_API_EXPORT iree_status_t iree_net_session_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* receive_pool,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session);

// Begins a server session on an accepted connection.
//
// The session retains |connection| on success; the caller keeps its existing
// reference. Asynchronous callbacks may race with the return. The returned
// reference is owned only after this call returns and must not be released by
// a callback that wins that race.
//
// On synchronous failure no session is published and no callback fires.
IREE_API_EXPORT iree_status_t iree_net_session_accept(
    iree_net_connection_t* connection,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session);

// Retains a session reference. NULL-safe and thread-safe.
IREE_API_EXPORT void iree_net_session_retain(iree_net_session_t* session);

// Releases a session reference. NULL-safe and thread-safe.
//
// A successfully published session holds an internal lifecycle reference
// through on_deactivated. Owners may release their reference after requesting
// deactivation or from a later on_deactivated callback.
IREE_API_EXPORT void iree_net_session_release(iree_net_session_t* session);

// Returns the current lifecycle state with acquire ordering.
IREE_API_EXPORT iree_net_session_state_t
iree_net_session_state(const iree_net_session_t* session);

// Opens the next application endpoint ordinal.
//
// Requires OPERATIONAL state and admits at most the application endpoint count
// exchanged during bootstrap. Message and direct endpoints share this ordinal
// budget and must be opened in matching order on both peers. A rejected open
// consumes no ordinal and invokes no callback. The callback is forwarded
// directly to the connection and its target must remain valid through session
// deactivation.
IREE_API_EXPORT iree_status_t iree_net_session_open_endpoint(
    iree_net_session_t* session, iree_net_endpoint_ready_callback_t callback);

// Opens the next application ordinal as a direct-placement endpoint.
//
// Shares admission and lifetime semantics with iree_net_session_open_endpoint.
// Returns UNIMPLEMENTED when the connection does not support direct placement;
// it never substitutes message delivery or implicit staging. Deactivation joins
// the ready callback and every accepted direct operation before on_deactivated.
IREE_API_EXPORT iree_status_t iree_net_session_open_direct_endpoint(
    iree_net_session_t* session,
    iree_net_direct_endpoint_ready_callback_t callback);

// Sends control DATA while borrowing payload storage through completion.
//
// Requires OPERATIONAL state. An OK return guarantees exactly one completion;
// a non-OK return guarantees none.
IREE_API_EXPORT iree_status_t iree_net_session_send_control_data(
    iree_net_session_t* session, iree_net_control_data_flags_t flags,
    iree_async_span_list_t payload,
    iree_net_send_completion_callback_t completion_callback);

// Sends control DATA after copying all payload bytes before returning.
//
// Requires OPERATIONAL state. Payload spans must be CPU-accessible.
IREE_API_EXPORT iree_status_t iree_net_session_send_control_data_copy(
    iree_net_session_t* session, iree_net_control_data_flags_t flags,
    iree_async_span_list_t payload,
    iree_net_send_completion_callback_t completion_callback);

// Sends GOAWAY and closes new session operation admission.
//
// Requires OPERATIONAL state. A rejected send leaves the session operational
// and can be retried. An accepted send transitions to DRAINING. This does not
// start connection deactivation.
IREE_API_EXPORT iree_status_t iree_net_session_send_goaway(
    iree_net_session_t* session, uint32_t reason_code,
    iree_net_send_completion_callback_t completion_callback);

// Requests infallible asynchronous session deactivation.
//
// Idempotent and valid in every state. This immediately closes session-level
// admission, waits for pending connection setup when necessary, and then uses
// connection deactivation to drain every endpoint operation and callback.
// Completion is reported only through the fixed on_deactivated callback.
IREE_API_EXPORT void iree_net_session_deactivate(iree_net_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_SESSION_H_
