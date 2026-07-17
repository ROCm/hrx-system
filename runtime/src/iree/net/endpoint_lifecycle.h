// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Coordinates endpoint- and connection-level transport deactivation.
//
// Message endpoints are borrowed from connections. An endpoint consumer may
// begin deactivation immediately before its owning connection begins draining
// all endpoints. Both requests must join the same carrier drain: exactly one
// request starts deactivation, the endpoint callback observes completion, and
// the connection callback fires only after every joined endpoint has drained.

#ifndef IREE_NET_ENDPOINT_LIFECYCLE_H_
#define IREE_NET_ENDPOINT_LIFECYCLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Actions a transport owner must perform after a lifecycle request.
typedef enum iree_net_endpoint_lifecycle_action_bits_e {
  IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE = 0u,

  // The caller won the transition to DRAINING and must start carrier
  // deactivation with iree_net_endpoint_lifecycle_complete_deactivation as the
  // completion path.
  IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION = 1u << 0,
} iree_net_endpoint_lifecycle_action_bits_t;
typedef uint32_t iree_net_endpoint_lifecycle_actions_t;

typedef enum iree_net_endpoint_lifecycle_state_e {
  // The endpoint has not been activated.
  IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED = 0,
  // The endpoint is active and may initiate transport operations.
  IREE_NET_ENDPOINT_LIFECYCLE_STATE_ACTIVE = 1,
  // Carrier deactivation has started but has not completed.
  IREE_NET_ENDPOINT_LIFECYCLE_STATE_DRAINING = 2,
  // All carrier operations have drained and the endpoint may be destroyed.
  IREE_NET_ENDPOINT_LIFECYCLE_STATE_DEACTIVATED = 3,
} iree_net_endpoint_lifecycle_state_t;

// Barrier joining all endpoint drains owned by one connection.
//
// Initialize immediately before visiting endpoint lifecycles, join each
// lifecycle, and then commit. The callback may fire synchronously from commit
// when no endpoint requires a drain. The barrier must remain live until its
// callback fires.
typedef struct iree_net_endpoint_deactivation_barrier_t {
  // Pending endpoint drains plus one setup hold released by commit.
  iree_atomic_int32_t pending_count;
  // Callback invoked when the setup hold and all endpoint drains are released.
  iree_net_connection_deactivate_callback_t callback;
} iree_net_endpoint_deactivation_barrier_t;

// Lifecycle state embedded in each message endpoint implementation.
typedef struct iree_net_endpoint_lifecycle_t {
  // Serializes deactivation requests against carrier completion.
  iree_slim_mutex_t mutex;
  // Current endpoint lifecycle state.
  iree_net_endpoint_lifecycle_state_t state;
  // Endpoint-consumer callback registered by endpoint deactivation.
  struct {
    // Function invoked when the endpoint carrier has drained.
    iree_net_message_endpoint_deactivate_fn_t fn;
    // Opaque value passed to |fn|.
    void* user_data;
  } endpoint_callback;
  // Connection barrier joined while this endpoint is draining.
  iree_net_endpoint_deactivation_barrier_t* connection_barrier;
} iree_net_endpoint_lifecycle_t;

// Initializes an endpoint lifecycle in the CREATED state.
IREE_API_EXPORT void iree_net_endpoint_lifecycle_initialize(
    iree_net_endpoint_lifecycle_t* out_lifecycle);

// Deinitializes an endpoint lifecycle.
//
// The lifecycle must be CREATED or DEACTIVATED with no pending callbacks.
IREE_API_EXPORT void iree_net_endpoint_lifecycle_deinitialize(
    iree_net_endpoint_lifecycle_t* lifecycle);

// Claims the endpoint for carrier activation.
//
// Carrier activation and this call are externally serialized with endpoint
// and connection deactivation. Returns FAILED_PRECONDITION unless the
// lifecycle is CREATED. On success the lifecycle is ACTIVE before the carrier
// begins delivering callbacks.
IREE_API_EXPORT iree_status_t
iree_net_endpoint_lifecycle_activate(iree_net_endpoint_lifecycle_t* lifecycle);

// Rolls back a successful activation claim after carrier activation fails.
//
// The lifecycle must be ACTIVE and carrier activation must not have succeeded.
IREE_API_EXPORT void iree_net_endpoint_lifecycle_rollback_activation(
    iree_net_endpoint_lifecycle_t* lifecycle);

// Begins endpoint-consumer deactivation.
//
// Returns BEGIN_DEACTIVATION when the caller must start the carrier drain.
// Completion must call iree_net_endpoint_lifecycle_complete_deactivation.
IREE_API_EXPORT iree_status_t iree_net_endpoint_lifecycle_request_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle,
    iree_net_message_endpoint_deactivate_fn_t callback, void* user_data,
    iree_net_endpoint_lifecycle_actions_t* out_actions);

// Initializes a connection-level endpoint deactivation barrier.
IREE_API_EXPORT void iree_net_endpoint_deactivation_barrier_initialize(
    iree_net_connection_deactivate_callback_t callback,
    iree_net_endpoint_deactivation_barrier_t* out_barrier);

// Joins a connection-level barrier to an endpoint drain.
//
// CREATED and DEACTIVATED endpoints require no work. ACTIVE endpoints return
// BEGIN_DEACTIVATION. DRAINING endpoints join the in-flight carrier drain
// without starting a second one.
IREE_API_EXPORT iree_net_endpoint_lifecycle_actions_t
iree_net_endpoint_lifecycle_join_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle,
    iree_net_endpoint_deactivation_barrier_t* barrier);

// Releases the barrier setup hold after all endpoint lifecycles are joined.
//
// The connection callback may fire synchronously and release the connection.
IREE_API_EXPORT void iree_net_endpoint_deactivation_barrier_commit(
    iree_net_endpoint_deactivation_barrier_t* barrier);

// Completes the single carrier drain associated with an endpoint.
//
// The endpoint callback runs first. The connection barrier is released last
// because its callback may destroy the endpoint lifecycle and its owner.
IREE_API_EXPORT void iree_net_endpoint_lifecycle_complete_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_ENDPOINT_LIFECYCLE_H_
