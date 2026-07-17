// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/endpoint_lifecycle.h"

#include <string.h>

void iree_net_endpoint_lifecycle_initialize(
    iree_net_endpoint_lifecycle_t* out_lifecycle) {
  IREE_ASSERT_ARGUMENT(out_lifecycle);
  memset(out_lifecycle, 0, sizeof(*out_lifecycle));
  iree_slim_mutex_initialize(&out_lifecycle->mutex);
  out_lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED;
}

void iree_net_endpoint_lifecycle_deinitialize(
    iree_net_endpoint_lifecycle_t* lifecycle) {
  IREE_ASSERT_ARGUMENT(lifecycle);
  IREE_ASSERT(
      lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED ||
          lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_DEACTIVATED,
      "endpoint lifecycle destroyed in state %d", (int)lifecycle->state);
  IREE_ASSERT(!lifecycle->endpoint_callback.fn,
              "endpoint lifecycle destroyed with a pending callback");
  IREE_ASSERT(!lifecycle->connection_barrier,
              "endpoint lifecycle destroyed with a pending connection drain");
  iree_slim_mutex_deinitialize(&lifecycle->mutex);
}

iree_status_t iree_net_endpoint_lifecycle_activate(
    iree_net_endpoint_lifecycle_t* lifecycle) {
  IREE_ASSERT_ARGUMENT(lifecycle);
  iree_slim_mutex_lock(&lifecycle->mutex);
  iree_status_t status = iree_ok_status();
  if (lifecycle->state != IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "endpoint cannot activate from state %d",
                              (int)lifecycle->state);
  } else {
    lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_ACTIVE;
  }
  iree_slim_mutex_unlock(&lifecycle->mutex);
  return status;
}

void iree_net_endpoint_lifecycle_rollback_activation(
    iree_net_endpoint_lifecycle_t* lifecycle) {
  IREE_ASSERT_ARGUMENT(lifecycle);
  iree_slim_mutex_lock(&lifecycle->mutex);
  IREE_ASSERT(lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_ACTIVE,
              "endpoint lifecycle activation rolled back from state %d",
              (int)lifecycle->state);
  lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED;
  iree_slim_mutex_unlock(&lifecycle->mutex);
}

iree_status_t iree_net_endpoint_lifecycle_request_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle,
    iree_net_message_endpoint_deactivate_fn_t callback, void* user_data,
    iree_net_endpoint_lifecycle_actions_t* out_actions) {
  IREE_ASSERT_ARGUMENT(lifecycle);
  IREE_ASSERT_ARGUMENT(out_actions);
  *out_actions = IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;

  iree_slim_mutex_lock(&lifecycle->mutex);
  iree_status_t status = iree_ok_status();
  if (lifecycle->state != IREE_NET_ENDPOINT_LIFECYCLE_STATE_ACTIVE) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "endpoint is not active (state=%d)",
                              (int)lifecycle->state);
  } else {
    lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_DRAINING;
    lifecycle->endpoint_callback.fn = callback;
    lifecycle->endpoint_callback.user_data = user_data;
    *out_actions = IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION;
  }
  iree_slim_mutex_unlock(&lifecycle->mutex);
  return status;
}

void iree_net_endpoint_deactivation_barrier_initialize(
    iree_net_connection_deactivate_callback_t callback,
    iree_net_endpoint_deactivation_barrier_t* out_barrier) {
  IREE_ASSERT_ARGUMENT(callback.fn);
  IREE_ASSERT_ARGUMENT(out_barrier);
  iree_atomic_store(&out_barrier->pending_count, 1, iree_memory_order_relaxed);
  out_barrier->callback = callback;
}

static void iree_net_endpoint_deactivation_barrier_arrive(
    iree_net_endpoint_deactivation_barrier_t* barrier) {
  if (iree_atomic_fetch_sub(&barrier->pending_count, 1,
                            iree_memory_order_acq_rel) == 1) {
    barrier->callback.fn(barrier->callback.user_data);
  }
}

iree_net_endpoint_lifecycle_actions_t
iree_net_endpoint_lifecycle_join_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle,
    iree_net_endpoint_deactivation_barrier_t* barrier) {
  IREE_ASSERT_ARGUMENT(lifecycle);
  IREE_ASSERT_ARGUMENT(barrier);

  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  iree_slim_mutex_lock(&lifecycle->mutex);
  if (lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_ACTIVE) {
    lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_DRAINING;
    actions |= IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION;
  }
  if (lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_DRAINING) {
    IREE_ASSERT(!lifecycle->connection_barrier,
                "endpoint lifecycle joined by multiple connections");
    iree_atomic_fetch_add(&barrier->pending_count, 1,
                          iree_memory_order_relaxed);
    lifecycle->connection_barrier = barrier;
  } else {
    IREE_ASSERT(
        lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_CREATED ||
            lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_DEACTIVATED,
        "invalid endpoint lifecycle state %d", (int)lifecycle->state);
  }
  iree_slim_mutex_unlock(&lifecycle->mutex);
  return actions;
}

void iree_net_endpoint_deactivation_barrier_commit(
    iree_net_endpoint_deactivation_barrier_t* barrier) {
  IREE_ASSERT_ARGUMENT(barrier);
  iree_net_endpoint_deactivation_barrier_arrive(barrier);
}

void iree_net_endpoint_lifecycle_complete_deactivation(
    iree_net_endpoint_lifecycle_t* lifecycle) {
  IREE_ASSERT_ARGUMENT(lifecycle);

  iree_net_message_endpoint_deactivate_fn_t endpoint_callback = NULL;
  void* endpoint_user_data = NULL;
  iree_net_endpoint_deactivation_barrier_t* connection_barrier = NULL;
  iree_slim_mutex_lock(&lifecycle->mutex);
  IREE_ASSERT(lifecycle->state == IREE_NET_ENDPOINT_LIFECYCLE_STATE_DRAINING,
              "endpoint lifecycle completed from state %d",
              (int)lifecycle->state);
  lifecycle->state = IREE_NET_ENDPOINT_LIFECYCLE_STATE_DEACTIVATED;
  endpoint_callback = lifecycle->endpoint_callback.fn;
  endpoint_user_data = lifecycle->endpoint_callback.user_data;
  lifecycle->endpoint_callback.fn = NULL;
  lifecycle->endpoint_callback.user_data = NULL;
  connection_barrier = lifecycle->connection_barrier;
  lifecycle->connection_barrier = NULL;
  iree_slim_mutex_unlock(&lifecycle->mutex);

  if (endpoint_callback) endpoint_callback(endpoint_user_data);
  if (connection_barrier) {
    iree_net_endpoint_deactivation_barrier_arrive(connection_barrier);
  }
}
