// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Message endpoint: unified interface for sending and receiving discrete
// messages.
//
// This abstraction sits between carriers and message protocols. A
// connection-facing endpoint exposes payload messages and hides any transport
// framing needed to preserve their boundaries. Transport-internal endpoints
// may instead use complete wire frames as their endpoint-defined messages.
//
// ## Borrowed View Semantics
//
// IMPORTANT: Message endpoints are BORROWED VIEWS, not owned objects. Callers
// never free an endpoint directly. The endpoint is valid only while the
// underlying endpoint implementation is alive.
//
// Conversion functions like framing_adapter_as_endpoint() return stack-copyable
// structs that point into the underlying object. When that object is freed,
// all endpoint references become invalid.
//
// ## Protocol Handoff
//
// During connection bootstrap, ownership of an endpoint transfers from the
// bootstrap handler to the operational protocol. Use set_callbacks() to
// atomically swap both message and error handlers. Message callbacks for one
// endpoint are serialized in delivery order. A swap performed from inside a
// message callback therefore takes effect before any later message callback,
// ensuring no later message is delivered to the stale handler. A terminal
// error racing the swap may observe either complete callback bundle.

#ifndef IREE_NET_MESSAGE_ENDPOINT_H_
#define IREE_NET_MESSAGE_ENDPOINT_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/message_receive.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Callbacks and parameters
//===----------------------------------------------------------------------===//

// Message handler invoked when a complete message is received.
//
// Called on the proactor thread for each complete message. Calls for one
// endpoint are serialized in delivery order and never overlap; one call returns
// before the next begins. This serialization applies only to message callbacks;
// terminal-error and send-completion callbacks remain independently
// asynchronous.
//
// The handler receives a view of the message data and a lease to the backing
// storage. The lease is always valid (non-NULL) whether the message came from a
// recv buffer or was reassembled from fragments.
//
// To keep the message data valid beyond the callback, move the lease by copying
// it and clearing the callback's lease value. Release the moved lease when
// done.
//
// Return iree_ok_status() to continue receiving. Returning an error triggers
// the endpoint's error handler and may cause deactivation.
typedef iree_status_t(IREE_API_PTR* iree_net_message_endpoint_message_fn_t)(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease);

// Error handler invoked for the first terminal endpoint error.
//
// Called on the proactor thread after receive progress has stopped. Subsequent
// operations fail with the recorded terminal status. Status ownership is
// transferred to the handler, which must propagate or release it. Outstanding
// sends retain their own completion ownership and may finish afterward.
typedef void(IREE_API_PTR* iree_net_message_endpoint_error_fn_t)(
    void* user_data, iree_status_t status);

// Deactivation complete callback invoked when graceful shutdown finishes.
//
// After this callback fires, the endpoint is fully drained and operations will
// fail. The underlying object can now be safely freed.
typedef void(IREE_API_PTR* iree_net_message_endpoint_deactivate_fn_t)(
    void* user_data);

// Bundled message and error handlers for atomic handoff.
//
// During protocol transitions (e.g., bootstrap to operational), callbacks must
// change atomically to prevent messages from being delivered to a stale
// handler. The shared user_data ensures consistency across the bundle. Callback
// functions and user data remain caller-owned and must stay valid until the
// endpoint deactivation callback fires, including after this bundle is
// superseded by a later call to set_callbacks().
typedef struct iree_net_message_endpoint_callbacks_t {
  // Function invoked for each complete received message.
  iree_net_message_endpoint_message_fn_t on_message;

  // Function invoked when the endpoint reports a transport error.
  iree_net_message_endpoint_error_fn_t on_error;

  // Opaque user data passed to every callback in this bundle.
  void* user_data;

  // Optional notification that shared send capacity may have become available
  // without a send completion on this endpoint (for example, another endpoint
  // returned a connection slot). Invoked on the proactor with no transport
  // locks held. This is advisory; the callback queries budget before sending.
  // Endpoint deactivation joins these callbacks along with message delivery.
  void (*on_capacity)(void* user_data);
} iree_net_message_endpoint_callbacks_t;

// Parameters for send operations.
typedef struct iree_net_message_endpoint_send_params_t {
  // Transient leading bytes generated before the send call returns.
  iree_net_send_prefix_t generated_prefix;

  // Scatter-gather message data borrowed through terminal completion.
  iree_async_span_list_t data;

  // Required callback invoked when the send completes.
  iree_net_send_completion_callback_t completion_callback;
} iree_net_message_endpoint_send_params_t;

//===----------------------------------------------------------------------===//
// iree_net_message_endpoint_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_message_endpoint_vtable_t
    iree_net_message_endpoint_vtable_t;

// A borrowed view into a message-oriented transport endpoint.
//
// This is a lightweight handle (two pointers) that can be copied by value.
// The endpoint is valid only while the underlying object is alive. There is
// no retain/release - when the underlying object is freed, the endpoint
// becomes invalid.
typedef struct iree_net_message_endpoint_t {
  // Opaque endpoint implementation pointer passed to vtable methods.
  void* self;
  // Vtable implementing endpoint operations.
  const iree_net_message_endpoint_vtable_t* vtable;
} iree_net_message_endpoint_t;

struct iree_net_message_endpoint_vtable_t {
  // Atomically replaces the message and terminal-error callback bundle.
  void (*set_callbacks)(void* self,
                        iree_net_message_endpoint_callbacks_t callbacks);
  // Activates receive progress for the endpoint.
  iree_status_t (*activate)(void* self);
  // Begins asynchronous endpoint deactivation.
  iree_status_t (*deactivate)(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data);
  // Submits one callback-completed message send.
  iree_status_t (*send)(void* self,
                        const iree_net_message_endpoint_send_params_t* params);
  // Queries current message send admission capacity.
  iree_net_carrier_send_budget_t (*query_send_budget)(void* self);
  // Optional destination capture for a connection-facing payload message.
  iree_status_t (*receive)(
      void* self, iree_net_message_receive_operation_t* operation,
      const iree_net_message_endpoint_receive_params_t* params);
};

// Posts a current writable body for a message with an exact opaque prefix.
//
// All posted receives on an endpoint use the same nonzero prefix width. The
// transport validates the complete body extent before writing it. Matched
// messages complete this operation instead of invoking on_message; unmatched
// messages keep ordinary leased delivery. No wire envelope is added.
//
// An OK return captures the prefix, destination and callback and guarantees one
// terminal completion. A non-OK return captures nothing and invokes no
// callback. The operation and destination survive errors until that completion;
// endpoint error publication alone does not return a native destination borrow.
// A matched body can keep logical endpoint deactivation pending until it
// completes or the owning connection deactivates its native receive. No
// registration occurs here.
//
// Receives may be posted from any thread. An operation is zero-initialized
// before first use and cannot be reused until its terminal callback begins.
// Internal wire-frame endpoint views do not expose payload placement and return
// UNIMPLEMENTED; use the connection-facing endpoint.
static inline iree_status_t iree_net_message_endpoint_receive(
    iree_net_message_endpoint_t endpoint,
    iree_net_message_receive_operation_t* operation,
    const iree_net_message_endpoint_receive_params_t* params) {
  if (!operation || !params || !params->completion_callback.fn ||
      !params->prefix.data || !params->prefix.data_length ||
      params->prefix.data_length > IREE_NET_MESSAGE_RECEIVE_PREFIX_CAPACITY ||
      (params->data.data_length && !params->data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "receive operation, callback, bounded prefix and "
                            "writable destination are required");
  }
  if (params->data.data_length >
      IREE_HOST_SIZE_MAX - params->prefix.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "message receive extent overflow");
  }
  if (!endpoint.vtable->receive) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "endpoint does not support payload receive placement");
  }
  return endpoint.vtable->receive(endpoint.self, operation, params);
}

// Sets message and error handlers atomically.
//
// Used for protocol handoff (e.g., bootstrap completes, operational channel
// takes over). Both handlers and user_data change in a single operation. When
// called from an on_message handler, all later messages use the new bundle. A
// terminal-error callback already racing the handoff may use either the old or
// new bundle, but never a mixture. All installed callback targets must remain
// valid until endpoint deactivation completes.
//
// Must be called on the proactor thread after activation, or from any thread
// before activation.
static inline void iree_net_message_endpoint_set_callbacks(
    iree_net_message_endpoint_t endpoint,
    iree_net_message_endpoint_callbacks_t callbacks) {
  endpoint.vtable->set_callbacks(endpoint.self, callbacks);
}

// Activates the endpoint, enabling message receipt.
//
// After activation, the endpoint auto-receives and delivers complete messages
// to the on_message handler. Both on_message and on_error must be set first.
//
// Returns IREE_STATUS_FAILED_PRECONDITION if callbacks are not set.
static inline iree_status_t iree_net_message_endpoint_activate(
    iree_net_message_endpoint_t endpoint) {
  return endpoint.vtable->activate(endpoint.self);
}

// Begins graceful deactivation of the endpoint.
//
// This drains outstanding operations and stops receiving new messages. The
// optional callback fires when deactivation completes and the endpoint is safe
// to abandon. After the callback, operations on this endpoint will fail.
//
// Deactivation does not block waiting for pending work. If no work remains,
// the callback may fire synchronously from this call and invalidate the
// endpoint owner before the call returns.
//
// An OK return accepts the drain, which then completes even if transport
// cleanup fails; cleanup failures use the endpoint error handler. A non-OK
// return rejects the request and does not invoke the callback.
static inline iree_status_t iree_net_message_endpoint_deactivate(
    iree_net_message_endpoint_t endpoint,
    iree_net_message_endpoint_deactivate_fn_t callback, void* user_data) {
  return endpoint.vtable->deactivate(endpoint.self, callback, user_data);
}

// Sends a message via the endpoint.
//
// |params->generated_prefix| followed by |params->data| comprises one
// endpoint-defined message. The prefix writer runs synchronously at most once
// before this call returns; an endpoint may reject before invoking it when
// transport storage is unavailable. Once any endpoint layer acquires bounded
// send capacity the operation is accepted; prefix-generation and lower-layer
// failures are then reported through its terminal completion. Data buffers
// remain caller-owned until the completion callback fires. Either part may be
// empty, but the complete message must contain at least one byte.
//
// Connection-facing endpoints preserve the message boundary while hiding any
// transport framing they add. Lower-level transport endpoints may define a
// message as a complete wire frame and send the bytes unchanged.
//
// An OK return guarantees exactly one terminal completion through
// |params->completion_callback|. The callback may race with the return on
// another proactor thread. Its byte count covers the complete logical message,
// including the generated prefix. A non-OK return means the callback will not
// fire, the prefix writer was not invoked, and no bounded capacity remains
// owned by the operation.
//
// The prefix writer, its user data, and the span descriptor array are needed
// only for the duration of this call. The data buffers must remain valid until
// the completion callback fires.
static inline iree_status_t iree_net_message_endpoint_send(
    iree_net_message_endpoint_t endpoint,
    const iree_net_message_endpoint_send_params_t* params) {
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send parameters are required");
  }
  if (!params->completion_callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send completion callback is required");
  }
  if ((params->generated_prefix.length == 0) !=
      (params->generated_prefix.write == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send prefix length and writer disagree");
  }
  if (params->generated_prefix.length == 0 &&
      params->generated_prefix.user_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty send prefix has user data");
  }
  if (params->data.count > 0 && !params->data.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send span list has null storage");
  }
  iree_host_size_t total_length = params->generated_prefix.length;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    if (!iree_host_size_checked_add(total_length, params->data.values[i].length,
                                    &total_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "send payload length overflow");
    }
  }
  if (total_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send payload must be non-empty");
  }
  return endpoint.vtable->send(endpoint.self, params);
}

// Queries send budget for backpressure management.
//
// Returns both byte budget and operation slot budget. When either reaches
// zero, the endpoint is backpressured. A later send completion or on_capacity
// callback indicates that callers may query the budget and attempt admission
// again. Shared-capacity endpoints notify on_capacity when a different owner
// returns capacity; dedicated endpoints use their own send completions.
// This is an advisory snapshot; send remains the authoritative admission check.
static inline iree_net_carrier_send_budget_t
iree_net_message_endpoint_query_send_budget(
    iree_net_message_endpoint_t endpoint) {
  return endpoint.vtable->query_send_budget(endpoint.self);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_MESSAGE_ENDPOINT_H_
