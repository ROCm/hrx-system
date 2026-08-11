// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/factory_bootstrap.h"

#include <string.h>

#include "iree/async/event.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"
#include "iree/net/carrier/shm/factory_state.h"
#include "iree/net/carrier/shm/handshake.h"

// A bootstrap is connection setup on a cold path. The protocol uses platform
// handle transfer operations that cannot be represented by the generic socket
// operations, so a dedicated worker performs the blocking exchange while an
// async notification gives the proactor an exact completion edge.
struct iree_net_shm_bootstrap_t {
  // Factory retained for options, shared wake ownership, and allocation state.
  iree_net_shm_factory_t* factory;
  // Proactor retained through terminal callback delivery.
  iree_async_proactor_t* proactor;
  // Channel owned after prepare succeeds until attached or closed.
  iree_async_primitive_t channel;
  // Shared wake owned by |factory| and stable while the factory is retained.
  iree_net_shm_shared_wake_t* shared_wake;
  // Server or client side of the handshake protocol.
  iree_net_shm_bootstrap_role_t role;
  // Number of endpoint exchanges required by the connection.
  uint16_t endpoint_count;
  // Per-endpoint results populated by the worker.
  iree_net_shm_handshake_result_t* handshake_results;
  // Terminal result produced by the worker and consumed by completion.
  iree_status_t worker_status;
  // Notification signaled once after |worker_status| and results are final.
  iree_async_notification_t* completion_notification;
  // Event waking a worker blocked in platform handshake I/O on cancellation.
  iree_async_event_t* cancellation_event;
  // Proactor operation awaiting |completion_notification|.
  iree_async_notification_wait_operation_t completion_wait_operation;
  // Dedicated blocking handshake worker.
  iree_thread_t* worker_thread;
  // Protects the terminal result of a cancellation request.
  iree_slim_mutex_t cancellation_mutex;
  // Failure encountered while interrupting peer I/O, if any.
  iree_status_t cancellation_status;
  // Set before the suspended worker is resumed for a successful prepare.
  iree_atomic_int32_t launch_requested;
  // Set once when cancellation is requested.
  iree_atomic_int32_t cancel_requested;
  // Callback receiving the terminal result on the proactor thread.
  iree_net_shm_bootstrap_callback_t callback;
  // Allocator used for the state, result array, and connection.
  iree_allocator_t host_allocator;
};

static void iree_net_shm_bootstrap_deinitialize_results(
    iree_net_shm_bootstrap_t* bootstrap) {
  if (!bootstrap->handshake_results) return;
  for (uint16_t i = 0; i < bootstrap->endpoint_count; ++i) {
    iree_net_shm_handshake_result_deinitialize(
        &bootstrap->handshake_results[i]);
  }
}

static void iree_net_shm_bootstrap_destroy(
    iree_net_shm_bootstrap_t* bootstrap) {
  if (!bootstrap) return;
  IREE_ASSERT(!bootstrap->worker_thread);
  IREE_ASSERT(iree_status_is_ok(bootstrap->worker_status));
  IREE_ASSERT(iree_status_is_ok(bootstrap->cancellation_status));
  iree_net_shm_bootstrap_deinitialize_results(bootstrap);
  iree_allocator_free(bootstrap->host_allocator, bootstrap->handshake_results);
  iree_async_primitive_close(&bootstrap->channel);
  iree_async_event_release(bootstrap->cancellation_event);
  iree_async_notification_release(bootstrap->completion_notification);
  iree_async_proactor_release(bootstrap->proactor);
  iree_net_transport_factory_release(&bootstrap->factory->base);
  iree_slim_mutex_deinitialize(&bootstrap->cancellation_mutex);
  iree_allocator_t host_allocator = bootstrap->host_allocator;
  iree_allocator_free(host_allocator, bootstrap);
}

static bool iree_net_shm_bootstrap_is_cancelled(
    const iree_net_shm_bootstrap_t* bootstrap) {
  return iree_atomic_load(&bootstrap->cancel_requested,
                          iree_memory_order_acquire) != 0;
}

static int iree_net_shm_bootstrap_worker_main(void* user_data) {
  iree_net_shm_bootstrap_t* bootstrap = (iree_net_shm_bootstrap_t*)user_data;

  // A failed prepare resumes the suspended thread only so it can exit and be
  // joined. It must not touch the caller-owned channel or signal a wait that
  // was never submitted.
  if (iree_atomic_load(&bootstrap->launch_requested,
                       iree_memory_order_acquire) == 0) {
    return 0;
  }

  iree_net_shm_handshake_cancellation_t cancellation = {
      .requested = &bootstrap->cancel_requested,
      .interrupt_primitive = bootstrap->cancellation_event->primitive,
  };
  iree_status_t status = iree_ok_status();
  if (iree_net_shm_bootstrap_is_cancelled(bootstrap)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "SHM connection bootstrap cancelled");
  }
  for (uint16_t i = 0;
       i < bootstrap->endpoint_count && iree_status_is_ok(status); ++i) {
    if (bootstrap->role == IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER) {
      status = iree_net_shm_handshake_server_endpoint(
          bootstrap->channel, &cancellation, bootstrap->shared_wake,
          bootstrap->factory->options, bootstrap->proactor,
          bootstrap->host_allocator, &bootstrap->handshake_results[i]);
    } else {
      status = iree_net_shm_handshake_client_endpoint(
          bootstrap->channel, &cancellation, bootstrap->shared_wake,
          bootstrap->proactor, bootstrap->host_allocator,
          &bootstrap->handshake_results[i]);
    }
  }

  bootstrap->worker_status = status;
  iree_async_notification_signal(bootstrap->completion_notification, 1);
  return 0;
}

static void iree_net_shm_bootstrap_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_bootstrap_t* bootstrap = (iree_net_shm_bootstrap_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_bootstrap_completion_flags_t completion_flags =
      IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_NONE;

  // Cancellation of the proactor wait must first interrupt the worker. The
  // terminal join below is the ownership boundary proving that its stack and
  // all platform OVERLAPPED/iovec state are no longer live.
  bool wait_cancelled =
      iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);
  if (wait_cancelled || !iree_status_is_ok(status)) {
    iree_net_shm_bootstrap_cancel(bootstrap);
  }
  iree_thread_release(bootstrap->worker_thread);
  bootstrap->worker_thread = NULL;

  iree_slim_mutex_lock(&bootstrap->cancellation_mutex);
  status = iree_status_join(status, bootstrap->cancellation_status);
  bootstrap->cancellation_status = iree_ok_status();
  iree_slim_mutex_unlock(&bootstrap->cancellation_mutex);

  if (iree_net_shm_bootstrap_is_cancelled(bootstrap) &&
      iree_status_is_cancelled(bootstrap->worker_status)) {
    iree_status_free(bootstrap->worker_status);
    bootstrap->worker_status = iree_ok_status();
    completion_flags |= IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED;
  }
  status = iree_status_join(status, bootstrap->worker_status);
  bootstrap->worker_status = iree_ok_status();

  bool was_cancelled = iree_any_bit_set(
      completion_flags, IREE_NET_SHM_BOOTSTRAP_COMPLETION_FLAG_CANCELLED);
  if (iree_status_is_ok(status) && !was_cancelled) {
    status = iree_net_shm_handshake_result_attach_file_transfer(
        &bootstrap->channel, bootstrap->host_allocator,
        &bootstrap->handshake_results[0]);
  }

  iree_net_connection_t* connection = NULL;
  if (iree_status_is_ok(status) && !was_cancelled) {
    status = iree_net_shm_connection_create_from_handshake_results(
        bootstrap->proactor, bootstrap->endpoint_count,
        bootstrap->handshake_results, bootstrap->host_allocator, &connection);
  }

  bootstrap->callback.fn(bootstrap->callback.user_data, status,
                         completion_flags, connection);
  iree_net_shm_bootstrap_destroy(bootstrap);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_net_shm_bootstrap_prepare(
    iree_net_shm_factory_t* factory, iree_net_shm_bootstrap_role_t role,
    iree_async_primitive_t* channel, iree_async_proactor_t* proactor,
    iree_net_shm_bootstrap_callback_t callback, iree_allocator_t host_allocator,
    iree_net_shm_bootstrap_t** out_bootstrap) {
  IREE_ASSERT_ARGUMENT(factory);
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(callback.fn);
  IREE_ASSERT_ARGUMENT(out_bootstrap);
  *out_bootstrap = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (role != IREE_NET_SHM_BOOTSTRAP_ROLE_SERVER &&
      role != IREE_NET_SHM_BOOTSTRAP_ROLE_CLIENT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid SHM bootstrap role %d", (int)role);
  }
  if (factory->options.max_endpoint_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM connection requires at least one endpoint");
  }

  iree_net_shm_bootstrap_t* bootstrap = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*bootstrap),
                                (void**)&bootstrap));
  memset(bootstrap, 0, sizeof(*bootstrap));
  bootstrap->factory = factory;
  iree_net_transport_factory_retain(&factory->base);
  bootstrap->proactor = proactor;
  iree_async_proactor_retain(proactor);
  bootstrap->channel = iree_async_primitive_none();
  bootstrap->role = role;
  bootstrap->endpoint_count = factory->options.max_endpoint_count;
  bootstrap->worker_status = iree_ok_status();
  iree_slim_mutex_initialize(&bootstrap->cancellation_mutex);
  bootstrap->cancellation_status = iree_ok_status();
  iree_atomic_store(&bootstrap->launch_requested, 0, iree_memory_order_relaxed);
  iree_atomic_store(&bootstrap->cancel_requested, 0, iree_memory_order_relaxed);
  bootstrap->callback = callback;
  bootstrap->host_allocator = host_allocator;

  iree_slim_mutex_lock(&factory->mutex);
  iree_status_t status = iree_net_shm_factory_get_or_create_shared_wake(
      factory, proactor, &bootstrap->shared_wake);
  iree_slim_mutex_unlock(&factory->mutex);

  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, bootstrap->endpoint_count,
                                    sizeof(*bootstrap->handshake_results),
                                    (void**)&bootstrap->handshake_results);
  }
  if (iree_status_is_ok(status)) {
    memset(bootstrap->handshake_results, 0,
           bootstrap->endpoint_count * sizeof(*bootstrap->handshake_results));
    status = iree_async_notification_create(
        proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE,
        &bootstrap->completion_notification);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_event_create(proactor, &bootstrap->cancellation_event);
  }
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    thread_params.name = IREE_SV("iree-shm-bootstrap");
    thread_params.create_suspended = true;
    status = iree_thread_create(iree_net_shm_bootstrap_worker_main, bootstrap,
                                thread_params, host_allocator,
                                &bootstrap->worker_thread);
  }
  if (iree_status_is_ok(status)) {
    uint32_t wait_token = iree_async_notification_begin_observe(
        bootstrap->completion_notification);
    iree_async_operation_initialize(
        &bootstrap->completion_wait_operation.base,
        IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        iree_net_shm_bootstrap_complete, bootstrap);
    bootstrap->completion_wait_operation.notification =
        bootstrap->completion_notification;
    bootstrap->completion_wait_operation.wait_flags =
        IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
    bootstrap->completion_wait_operation.wait_token = wait_token;
    status = iree_async_proactor_submit_one(
        proactor, &bootstrap->completion_wait_operation.base);
    iree_async_notification_end_observe(bootstrap->completion_notification);
  }

  if (iree_status_is_ok(status)) {
    bootstrap->channel = *channel;
    *channel = iree_async_primitive_none();
    *out_bootstrap = bootstrap;
  } else {
    // The thread was created suspended. Resume it without launch permission so
    // it exits without touching the caller-owned channel, then join exactly.
    if (bootstrap->worker_thread) {
      iree_thread_resume(bootstrap->worker_thread);
      iree_thread_release(bootstrap->worker_thread);
      bootstrap->worker_thread = NULL;
    }
    iree_net_shm_bootstrap_destroy(bootstrap);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_net_shm_bootstrap_launch(iree_net_shm_bootstrap_t* bootstrap) {
  IREE_ASSERT_ARGUMENT(bootstrap);
  int32_t expected = 0;
  bool did_launch = iree_atomic_compare_exchange_strong(
      &bootstrap->launch_requested, &expected, 1, iree_memory_order_release,
      iree_memory_order_relaxed);
  IREE_ASSERT(did_launch, "SHM bootstrap launched more than once");
  iree_thread_resume(bootstrap->worker_thread);
}

void iree_net_shm_bootstrap_cancel(iree_net_shm_bootstrap_t* bootstrap) {
  IREE_ASSERT_ARGUMENT(bootstrap);
  iree_slim_mutex_lock(&bootstrap->cancellation_mutex);
  int32_t was_requested = iree_atomic_exchange(&bootstrap->cancel_requested, 1,
                                               iree_memory_order_acq_rel);
  if (was_requested == 0) {
    bootstrap->cancellation_status =
        iree_async_event_set(bootstrap->cancellation_event);
  }
  iree_slim_mutex_unlock(&bootstrap->cancellation_mutex);
}
