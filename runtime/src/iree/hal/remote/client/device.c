// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/remote/client/allocator.h"
#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/command_buffer.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/client/file.h"
#include "iree/hal/remote/client/queue.h"
#include "iree/hal/remote/client/semaphore.h"
#include "iree/hal/remote/client/slab_provider.h"
#include "iree/hal/remote/protocol/bootstrap.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/net/bootstrap.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/status_wire.h"
#include "iree/net/transport_factory.h"

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable;
static void iree_hal_remote_client_device_destroy(
    iree_hal_device_t* base_device);

// Block size used for command-buffer retained resource sets.
#define IREE_HAL_REMOTE_CLIENT_RESOURCE_SET_BLOCK_SIZE 4096

iree_hal_remote_client_device_t* iree_hal_remote_client_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_device_vtable);
  return (iree_hal_remote_client_device_t*)base_value;
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_client_device_options_initialize(
    iree_hal_remote_client_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_options_parse(
    iree_hal_remote_client_device_options_t* options,
    iree_string_pair_list_t params) {
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    iree_string_view_t key = params.pairs[i].key;
    iree_string_view_t value = params.pairs[i].value;

    if (iree_string_view_equal(key, IREE_SV("server"))) {
      options->server_address = value;
    } else if (iree_string_view_equal(key, IREE_SV("connect_timeout"))) {
      uint32_t timeout_ms = 0;
      if (!iree_string_view_atoi_uint32(value, &timeout_ms)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid connect_timeout value");
      }
      options->connect_timeout_ns =
          (iree_duration_t)timeout_ms * 1000000;  // ms to ns.
    } else if (iree_string_view_equal(key, IREE_SV("rdma"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rdma must be 'true' or 'false'");
      }
    } else if (iree_string_view_equal(key, IREE_SV("trace"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trace must be 'true' or 'false'");
      }
    }
    // Unknown parameters are ignored for forward compatibility.
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_options_verify(
    const iree_hal_remote_client_device_options_t* options) {
  if (!options->transport_factory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport_factory is required");
  }
  if (iree_string_view_is_empty(options->server_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "server_address is required");
  }
  if (iree_all_bits_set(options->flags,
                        IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA)) {
    iree_net_transport_capabilities_t capabilities =
        iree_net_transport_factory_query_capabilities(
            options->transport_factory);
    if (!iree_all_bits_set(capabilities, IREE_NET_TRANSPORT_CAPABILITY_RDMA)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "rdma=true requires a transport factory with RDMA capability");
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_t
//===----------------------------------------------------------------------===//

// Pending control RPC state. Stack-allocated by the blocking caller, linked
// into the device's pending_rpcs list for the duration of the RPC.
typedef struct iree_hal_remote_pending_rpc_t {
  uint32_t request_id;
  iree_notification_t notification;
  // Written by the proactor thread with release ordering, read by the app
  // thread with acquire ordering. This is the C11 atomic bridge between the
  // proactor's writes to response fields and the app thread's reads. The
  // notification post/commit_wait also provides ordering, but the sentinel
  // check between prepare_wait and commit_wait requires explicit atomics.
  iree_atomic_int32_t response_status_code;
  // Terminal device failure cloned while failure_mutex protects this entry.
  iree_status_t failure_status;
  iree_const_byte_span_t response_payload;  // points into retained lease
  iree_async_buffer_lease_t response_lease;
  struct iree_hal_remote_pending_rpc_t* next;
} iree_hal_remote_pending_rpc_t;

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_hal_remote_recv_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;
  (void)create_params;

  iree_status_t status = iree_hal_remote_client_device_options_verify(options);

  // Calculate trailing storage layout.
  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t server_address_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_client_device_t), &total_size,
        IREE_STRUCT_FIELD(identifier.size, char, &identifier_offset),
        IREE_STRUCT_FIELD(options->server_address.size, char,
                          &server_address_offset));
  }

  iree_hal_remote_client_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&device);
  }

  if (iree_status_is_ok(status)) {
    memset(device, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_remote_client_device_vtable,
                                 &device->resource);

    // Copy strings to trailing storage.
    iree_string_view_append_to_buffer(identifier, &device->identifier,
                                      (char*)device + identifier_offset);
    device->options = *options;
    iree_string_view_append_to_buffer(options->server_address,
                                      &device->options.server_address,
                                      (char*)device + server_address_offset);

    iree_net_transport_factory_retain(device->options.transport_factory);

    device->host_allocator = host_allocator;
    iree_arena_block_pool_initialize(
        IREE_HAL_REMOTE_CLIENT_RESOURCE_SET_BLOCK_SIZE, host_allocator,
        &device->resource_set_block_pool);

    status = iree_hal_device_spec_create_minimal(
        identifier, identifier, IREE_SV("remote"), IREE_SV("remote"),
        host_allocator, &device->device_spec);
  }

  if (iree_status_is_ok(status)) {
    device->recv_pool = recv_pool;
    iree_hal_remote_recv_pool_retain(recv_pool);
    device->proactor = iree_hal_remote_recv_pool_proactor(recv_pool);
    iree_async_proactor_retain(device->proactor);

    iree_atomic_store(&device->queue_channel, 0, iree_memory_order_relaxed);
    iree_atomic_store(&device->next_submission_epoch, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&device->next_provisional_generation, 1,
                      iree_memory_order_relaxed);
    iree_slim_mutex_initialize(&device->provisional_mutex);
    iree_atomic_store(&device->next_request_id, 1, iree_memory_order_relaxed);
    iree_slim_mutex_initialize(&device->failure_mutex);
    iree_atomic_store(&device->state,
                      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED,
                      iree_memory_order_relaxed);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_bulk_session_initialize(
        host_allocator, &device->bulk_session);
  }

  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = 256;
    status = iree_async_frontier_tracker_create(tracker_options, host_allocator,
                                                &device->frontier_tracker);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_allocator_create(
        device, identifier, host_allocator, &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_slab_provider_create(
        device, host_allocator, &device->queue_slab_provider);
  }

  if (iree_status_is_ok(status)) {
    status = iree_async_notification_create(device->proactor,
                                            IREE_ASYNC_NOTIFICATION_FLAG_NONE,
                                            &device->queue_pool_notification);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_initialize_bulk_transfers(device);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else if (device) {
    iree_hal_remote_client_device_destroy((iree_hal_device_t*)device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_net_queue_channel_t*
iree_hal_remote_client_device_exchange_queue_channel(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* new_queue_channel) {
  return (iree_net_queue_channel_t*)iree_atomic_exchange(
      &device->queue_channel, (intptr_t)new_queue_channel,
      iree_memory_order_acq_rel);
}

// Closes queue channel admission without dropping the device's owning
// reference. The object remains alive for callers that observed CONNECTED
// before the terminal state transition.
static void iree_hal_remote_client_device_detach_queue_channel(
    iree_hal_remote_client_device_t* device) {
  iree_net_queue_channel_t* queue_channel =
      (iree_net_queue_channel_t*)iree_atomic_load(&device->queue_channel,
                                                  iree_memory_order_acquire);
  iree_net_queue_channel_detach(queue_channel);
}

static void iree_hal_remote_client_device_release_queue_channel(
    iree_hal_remote_client_device_t* device) {
  iree_net_queue_channel_t* queue_channel =
      iree_hal_remote_client_device_exchange_queue_channel(device, NULL);
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);
}

static iree_net_bulk_channel_t*
iree_hal_remote_client_device_exchange_bulk_channel(
    iree_hal_remote_client_device_t* device,
    iree_net_bulk_channel_t* new_bulk_channel) {
  return iree_hal_remote_client_bulk_session_exchange_channel(
      &device->bulk_session, new_bulk_channel);
}

// Closes bulk channel admission without dropping the device's owning
// reference. The object remains alive for callers that observed CONNECTED
// before the terminal state transition.
static void iree_hal_remote_client_device_detach_bulk_channel(
    iree_hal_remote_client_device_t* device) {
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  iree_net_bulk_channel_detach(bulk_channel);
}

static void iree_hal_remote_client_device_release_bulk_channel(
    iree_hal_remote_client_device_t* device) {
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_device_exchange_bulk_channel(device, NULL);
  iree_net_bulk_channel_detach(bulk_channel);
  iree_net_bulk_channel_release(bulk_channel);
}

static void iree_hal_remote_client_device_dispatch_error(
    iree_hal_remote_client_device_t* device, iree_status_t status) {
  if (device->options.error_callback.fn) {
    device->options.error_callback.fn(device->options.error_callback.user_data,
                                      status);
  } else {
    iree_status_free(status);
  }
}

bool iree_hal_remote_client_device_try_commit_connected(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* queue_channel,
    iree_net_bulk_channel_t* bulk_channel) {
  IREE_ASSERT_ARGUMENT(queue_channel);
  IREE_ASSERT_ARGUMENT(bulk_channel);

  iree_net_queue_channel_t* old_queue_channel = NULL;
  iree_net_bulk_channel_t* old_bulk_channel = NULL;
  iree_hal_remote_client_device_connected_callback_t connect_callback = {0};
  bool committed = false;
  iree_slim_mutex_lock(&device->failure_mutex);
  if (iree_status_is_ok(device->terminal_status) &&
      iree_hal_remote_client_device_load_state(device) ==
          IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    old_queue_channel = iree_hal_remote_client_device_exchange_queue_channel(
        device, queue_channel);
    old_bulk_channel = iree_hal_remote_client_device_exchange_bulk_channel(
        device, bulk_channel);
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
    connect_callback = device->connect_callback;
    memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    if (connect_callback.fn) {
      device->lifecycle_flags |=
          IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_CONNECT_DISPATCHING;
    }
    committed = true;
  }
  iree_slim_mutex_unlock(&device->failure_mutex);
  if (!committed) return false;

  iree_net_queue_channel_detach(old_queue_channel);
  iree_net_queue_channel_release(old_queue_channel);
  iree_net_bulk_channel_detach(old_bulk_channel);
  iree_net_bulk_channel_release(old_bulk_channel);

  if (connect_callback.fn) {
    connect_callback.fn(connect_callback.user_data, iree_ok_status());
  }

  iree_status_t deferred_error = iree_ok_status();
  iree_slim_mutex_lock(&device->failure_mutex);
  device->lifecycle_flags &=
      ~IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_CONNECT_DISPATCHING;
  if (iree_all_bits_set(
          device->lifecycle_flags,
          IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_ERROR_DEFERRED)) {
    device->lifecycle_flags &=
        ~IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_ERROR_DEFERRED;
    deferred_error = iree_status_clone(device->terminal_status);
  }
  iree_slim_mutex_unlock(&device->failure_mutex);
  if (!iree_status_is_ok(deferred_error)) {
    iree_hal_remote_client_device_dispatch_error(device, deferred_error);
  }
  return true;
}

// Releases the device-owned network graph during final destruction. The
// session and all endpoints must already be deactivated or absent. Keeping the
// graph retained until this point lets operations that raced a terminal device
// state transition safely reach the closed transport submission gates.
static void iree_hal_remote_client_device_release_network_graph(
    iree_hal_remote_client_device_t* device) {
  // Detach and release channels before the session. Deactivation has already
  // drained their endpoints, so no callback can race this release.
  iree_hal_remote_client_device_release_bulk_channel(device);
  iree_hal_remote_client_device_release_queue_channel(device);

  // Clear the session before releasing to prevent re-entrancy.
  iree_net_session_t* session = device->session;
  device->session = NULL;
  device->session_carrier = NULL;
  device->file_registration_capabilities =
      IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  iree_net_session_release(session);
}

static void iree_hal_remote_client_device_complete_destroy(
    iree_hal_remote_client_device_t* device) {
  iree_hal_device_t* base_device = (iree_hal_device_t*)device;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_device_release_network_graph(device);

  iree_async_notification_release(device->queue_pool_notification);
  iree_hal_slab_provider_release(device->queue_slab_provider);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_net_transport_factory_release(device->options.transport_factory);

  // Release any unresolved provisional buffers (shouldn't happen in normal
  // operation — all provisionals are resolved by ADVANCE before teardown).
  for (iree_host_size_t i = 0; i < device->provisional_buffers.count; ++i) {
    iree_hal_buffer_release(device->provisional_buffers.buffers[i]);
  }
  iree_allocator_free(host_allocator,
                      device->provisional_buffers.provisional_ids);
  iree_allocator_free(host_allocator, device->provisional_buffers.buffers);
  iree_slim_mutex_deinitialize(&device->provisional_mutex);

  iree_hal_remote_client_device_deinitialize_bulk_transfers(device);
  iree_hal_remote_client_bulk_session_deinitialize(&device->bulk_session);

  iree_hal_remote_recv_pool_release(device->recv_pool);
  iree_async_frontier_tracker_release(device->frontier_tracker);
  iree_async_proactor_release(device->proactor);
  iree_hal_device_spec_release(device->device_spec);
  iree_arena_block_pool_deinitialize(&device->resource_set_block_pool);

  iree_status_free(device->terminal_status);
  iree_slim_mutex_deinitialize(&device->failure_mutex);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_deactivated(
    void* user_data) {
  iree_hal_remote_client_device_complete_destroy(
      (iree_hal_remote_client_device_t*)user_data);
}

static void iree_hal_remote_client_device_complete_deactivate(
    iree_hal_remote_client_device_t* device) {
  // Session deactivation has drained the endpoints. Close channel admission
  // and detach callbacks while retaining the objects until device destruction.
  iree_hal_remote_client_device_detach_bulk_channel(device);
  iree_hal_remote_client_device_detach_queue_channel(device);
  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);

  iree_hal_remote_client_device_deactivated_callback_t callback =
      device->deactivate_callback;
  memset(&device->deactivate_callback, 0, sizeof(device->deactivate_callback));
  callback.fn(callback.user_data);

  // Balances the retain taken when deactivation began. The application may
  // have released its final reference from the callback above.
  iree_hal_device_release((iree_hal_device_t*)device);
}

static void iree_hal_remote_client_device_on_deactivate_session_deactivated(
    void* user_data) {
  iree_hal_remote_client_device_complete_deactivate(
      (iree_hal_remote_client_device_t*)user_data);
}

static void iree_hal_remote_client_device_on_deactivate_callbacks_detached(
    void* user_data) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_session_deactivated_callback_t callback = {
      .fn = iree_hal_remote_client_device_on_deactivate_session_deactivated,
      .user_data = device,
  };
  iree_status_t status = iree_net_session_deactivate(device->session, callback);
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_hal_remote_client_device_on_session_callbacks_detached(
    void* user_data) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;

  iree_hal_remote_client_device_connected_callback_t connect_callback = {0};
  iree_slim_mutex_lock(&device->failure_mutex);
  connect_callback = device->connect_callback;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  iree_slim_mutex_unlock(&device->failure_mutex);
  if (connect_callback.fn) {
    connect_callback.fn(
        connect_callback.user_data,
        iree_make_status(IREE_STATUS_CANCELLED, "remote device destroyed"));
  }

  iree_net_queue_channel_t* queue_channel =
      (iree_net_queue_channel_t*)iree_atomic_load(&device->queue_channel,
                                                  iree_memory_order_acquire);
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  if (!queue_channel && !bulk_channel) {
    iree_hal_remote_client_device_complete_destroy(device);
    return;
  }

  iree_net_session_deactivated_callback_t callback = {
      .fn = iree_hal_remote_client_device_on_session_deactivated,
      .user_data = device,
  };
  iree_status_t status = iree_net_session_deactivate(device->session, callback);
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

static void iree_hal_remote_client_device_destroy(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_remote_client_device_state_t previous_state =
      iree_hal_remote_client_device_load_state(device);
  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    iree_async_frontier_tracker_fail_axis(
        device->frontier_tracker, device->remote_queue_axis,
        iree_make_status(IREE_STATUS_CANCELLED,
                         "device destroyed while connected"));
  }
  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);

  if (!device->session ||
      previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED) {
    // Explicit deactivation already detached application callbacks and drained
    // the connection. The retained network graph can be released directly.
    iree_hal_remote_client_device_complete_destroy(device);
    return;
  }

  // The session may have already admitted a callback carrying |device| while
  // the final HAL reference was being released. Keep the device allocation
  // alive until that callback returns and prevent any later callback from
  // claiming the user data.
  iree_net_session_callbacks_detached_callback_t callback = {
      .fn = iree_hal_remote_client_device_on_session_callbacks_detached,
      .user_data = device,
  };
  iree_net_session_detach_callbacks(device->session, callback);
}

static iree_string_view_t iree_hal_remote_client_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_remote_client_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_remote_client_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->device_allocator;
}

static iree_status_t iree_hal_remote_client_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
  return iree_ok_status();
}

static void iree_hal_remote_client_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_remote_client_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (device->device_allocator) {
    IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  }
  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_remote_client_device_spec(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return device->device_spec;
}

static iree_status_t iree_hal_remote_client_device_sample_observation(
    iree_hal_device_t* base_device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (iree_any_bit_set(requested_flags,
                       IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_device_observation_populate_memory_total_from_spec(
            device->device_spec, out_observation));
  }
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_remote_client_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_remote_client_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (!topology_info) {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
  } else {
    device->topology_info = *topology_info;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collective channels not supported on remote device");
}

static iree_status_t iree_hal_remote_client_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_command_buffer_create(
      device, mode, command_categories, queue_affinity, binding_capacity,
      device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_remote_client_device_load_executable(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_executable_load(
      device, queue_affinity, target, load_params, device->host_allocator,
      out_executable);
}

static iree_status_t iree_hal_remote_client_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  return iree_hal_remote_client_file_import(device, queue_affinity, access,
                                            handle, flags, device->proactor,
                                            device->host_allocator, out_file);
}

static iree_status_t iree_hal_remote_client_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_semaphore_create(
      device->proactor, initial_value, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_remote_client_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
         IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_SIGNAL;
}

static bool iree_hal_remote_client_device_query_pool_epoch(
    void* user_data, iree_async_axis_t axis, uint64_t epoch) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  return iree_async_frontier_tracker_query_epoch(device->frontier_tracker, axis,
                                                 epoch);
}

static iree_status_t iree_hal_remote_client_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  iree_hal_slab_provider_properties_t properties;
  iree_hal_slab_provider_query_properties(device->queue_slab_provider,
                                          &properties);
  if (!iree_all_bits_set(properties.supported_usage,
                         IREE_HAL_BUFFER_USAGE_DEFAULT)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "remote device has no memory type suitable for queue pool slabs");
  }
  out_backend->slab_provider = device->queue_slab_provider;
  out_backend->notification = device->queue_pool_notification;
  out_backend->epoch_query = (iree_hal_pool_epoch_query_t){
      .fn = iree_hal_remote_client_device_query_pool_epoch,
      .user_data = device,
  };
  return iree_ok_status();
}

typedef struct iree_hal_remote_client_profiling_begin_message_t {
  iree_hal_remote_control_envelope_t envelope;
  iree_hal_remote_profiling_begin_request_t body;
} iree_hal_remote_client_profiling_begin_message_t;
static_assert(sizeof(iree_hal_remote_client_profiling_begin_message_t) == 72,
              "");

typedef struct iree_hal_remote_client_profiling_flush_message_t {
  iree_hal_remote_control_envelope_t envelope;
  iree_hal_remote_profiling_flush_request_t body;
} iree_hal_remote_client_profiling_flush_message_t;
static_assert(sizeof(iree_hal_remote_client_profiling_flush_message_t) == 24,
              "");

typedef struct iree_hal_remote_client_profiling_end_message_t {
  iree_hal_remote_control_envelope_t envelope;
  iree_hal_remote_profiling_end_request_t body;
} iree_hal_remote_client_profiling_end_message_t;
static_assert(sizeof(iree_hal_remote_client_profiling_end_message_t) == 24, "");

static iree_status_t iree_hal_remote_client_profiling_get_string_length(
    iree_string_view_t value, const char* field_name, uint32_t* out_length) {
  *out_length = 0;
  if (value.size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile %s length %" PRIhsz
                            " exceeds wire limit %u",
                            field_name, value.size, UINT32_MAX);
  }
  *out_length = (uint32_t)value.size;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_profiling_count_counter_names(
    const iree_hal_device_profiling_options_t* options,
    uint32_t* out_counter_set_count, uint32_t* out_counter_name_count) {
  *out_counter_set_count = 0;
  *out_counter_name_count = 0;
  if (options->counter_set_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile counter set count %" PRIhsz
                            " exceeds wire limit %u",
                            options->counter_set_count, UINT32_MAX);
  }
  *out_counter_set_count = (uint32_t)options->counter_set_count;

  iree_host_size_t counter_name_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < options->counter_set_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_counter_set_selection_t* counter_set =
        &options->counter_sets[i];
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            counter_name_count, counter_set->counter_name_count,
            &counter_name_count))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "profile counter name count overflow");
    }
  }
  if (iree_status_is_ok(status) && counter_name_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile counter name count %" PRIhsz
                              " exceeds wire limit %u",
                              counter_name_count, UINT32_MAX);
  }
  if (iree_status_is_ok(status)) {
    *out_counter_name_count = (uint32_t)counter_name_count;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_profiling_layout_counter_sets(
    const iree_hal_device_profiling_options_t* options,
    iree_host_size_t* inout_total_size) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < options->counter_set_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_counter_set_selection_t* counter_set =
        &options->counter_sets[i];
    uint32_t counter_set_name_length = 0;
    status = iree_hal_remote_client_profiling_get_string_length(
        counter_set->name, "counter set name", &counter_set_name_length);
    if (iree_status_is_ok(status) &&
        counter_set->counter_name_count > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "profile counter set %" PRIhsz
                                " counter name count %" PRIhsz
                                " exceeds wire limit %u",
                                i, counter_set->counter_name_count, UINT32_MAX);
    }
    if (iree_status_is_ok(status)) {
      status = IREE_STRUCT_LAYOUT(
          *inout_total_size, inout_total_size,
          IREE_STRUCT_FIELD_ALIGNED(
              1, iree_hal_remote_profile_counter_set_selection_t, 8, NULL),
          IREE_STRUCT_FIELD_ALIGNED(counter_set_name_length, char, 8, NULL));
    }
    for (iree_host_size_t j = 0;
         j < counter_set->counter_name_count && iree_status_is_ok(status);
         ++j) {
      uint32_t counter_name_length = 0;
      status = iree_hal_remote_client_profiling_get_string_length(
          counter_set->counter_names[j], "counter name", &counter_name_length);
      if (iree_status_is_ok(status)) {
        status = IREE_STRUCT_LAYOUT(
            *inout_total_size, inout_total_size,
            IREE_STRUCT_FIELD_ALIGNED(1, iree_hal_remote_profile_counter_name_t,
                                      8, NULL),
            IREE_STRUCT_FIELD_ALIGNED(counter_name_length, char, 8, NULL));
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_profiling_write_counter_sets(
    const iree_hal_device_profiling_options_t* options, uint8_t* base,
    iree_host_size_t* inout_offset) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < options->counter_set_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_counter_set_selection_t* counter_set =
        &options->counter_sets[i];
    iree_host_size_t counter_set_offset = 0;
    iree_host_size_t counter_set_name_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        *inout_offset, inout_offset,
        IREE_STRUCT_FIELD_ALIGNED(
            1, iree_hal_remote_profile_counter_set_selection_t, 8,
            &counter_set_offset),
        IREE_STRUCT_FIELD_ALIGNED(counter_set->name.size, char, 8,
                                  &counter_set_name_offset));
    if (iree_status_is_ok(status)) {
      iree_hal_remote_profile_counter_set_selection_t* remote_counter_set =
          (iree_hal_remote_profile_counter_set_selection_t*)(base +
                                                             counter_set_offset);
      remote_counter_set->flags = counter_set->flags;
      remote_counter_set->counter_name_count =
          (uint32_t)counter_set->counter_name_count;
      remote_counter_set->name_length = (uint32_t)counter_set->name.size;
      if (counter_set->name.size > 0) {
        memcpy(base + counter_set_name_offset, counter_set->name.data,
               counter_set->name.size);
      }
    }
    for (iree_host_size_t j = 0;
         j < counter_set->counter_name_count && iree_status_is_ok(status);
         ++j) {
      iree_string_view_t counter_name = counter_set->counter_names[j];
      iree_host_size_t counter_name_offset = 0;
      iree_host_size_t counter_name_data_offset = 0;
      status = IREE_STRUCT_LAYOUT(
          *inout_offset, inout_offset,
          IREE_STRUCT_FIELD_ALIGNED(1, iree_hal_remote_profile_counter_name_t,
                                    8, &counter_name_offset),
          IREE_STRUCT_FIELD_ALIGNED(counter_name.size, char, 8,
                                    &counter_name_data_offset));
      if (iree_status_is_ok(status)) {
        iree_hal_remote_profile_counter_name_t* remote_counter_name =
            (iree_hal_remote_profile_counter_name_t*)(base +
                                                      counter_name_offset);
        remote_counter_name->name_length = (uint32_t)counter_name.size;
        if (counter_name.size > 0) {
          memcpy(base + counter_name_data_offset, counter_name.data,
                 counter_name.size);
        }
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_profiling_begin_request_allocate(
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator, iree_byte_span_t* out_message) {
  *out_message = iree_byte_span_empty();

  iree_string_view_t executable_function_pattern = iree_string_view_empty();
  if (iree_any_bit_set(
          options->capture_filter.flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN)) {
    executable_function_pattern =
        options->capture_filter.executable_function_pattern;
  }

  uint32_t executable_function_pattern_length = 0;
  uint32_t counter_set_count = 0;
  uint32_t counter_name_count = 0;
  iree_status_t status = iree_hal_remote_client_profiling_get_string_length(
      executable_function_pattern, "executable function pattern",
      &executable_function_pattern_length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_profiling_count_counter_names(
        options, &counter_set_count, &counter_name_count);
  }

  iree_host_size_t message_size =
      sizeof(iree_hal_remote_client_profiling_begin_message_t);
  iree_host_size_t executable_function_pattern_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        message_size, &message_size,
        IREE_STRUCT_FIELD_ALIGNED(executable_function_pattern_length, char, 8,
                                  &executable_function_pattern_offset));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_profiling_layout_counter_sets(
        options, &message_size);
  }

  uint8_t* message_data = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, message_size,
                                   (void**)&message_data);
  }
  if (iree_status_is_ok(status)) {
    memset(message_data, 0, message_size);
    iree_hal_remote_client_profiling_begin_message_t* message =
        (iree_hal_remote_client_profiling_begin_message_t*)message_data;
    message->envelope.message_type = IREE_HAL_REMOTE_CONTROL_PROFILING_BEGIN;
    message->body.data_families = options->data_families;
    message->body.command_buffer_id = options->capture_filter.command_buffer_id;
    message->body.flags = options->flags;
    message->body.capture_filter_flags = options->capture_filter.flags;
    message->body.command_index = options->capture_filter.command_index;
    message->body.physical_device_ordinal =
        options->capture_filter.physical_device_ordinal;
    message->body.queue_ordinal = options->capture_filter.queue_ordinal;
    message->body.executable_function_pattern_length =
        executable_function_pattern_length;
    message->body.counter_set_count = counter_set_count;
    message->body.counter_name_count = counter_name_count;
    if (executable_function_pattern_length > 0) {
      memcpy(message_data + executable_function_pattern_offset,
             executable_function_pattern.data,
             executable_function_pattern.size);
    }
    iree_host_size_t write_offset =
        executable_function_pattern_offset +
        iree_host_align(executable_function_pattern_length, 8);
    status = iree_hal_remote_client_profiling_write_counter_sets(
        options, message_data, &write_offset);
  }
  if (iree_status_is_ok(status)) {
    *out_message = iree_make_byte_span(message_data, message_size);
  } else {
    iree_allocator_free(host_allocator, message_data);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));

  iree_byte_span_t message = iree_byte_span_empty();
  iree_status_t status =
      iree_hal_remote_client_profiling_begin_request_allocate(
          options, device->host_allocator, &message);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_bulk_begin_profile_session(device,
                                                               options->sink);
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        device, iree_make_const_byte_span(message.data, message.data_length),
        &response_payload, &response_lease);
  }
  iree_async_buffer_lease_release(&response_lease);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_end_profile_session(device);
  }
  iree_allocator_free(device->host_allocator, message.data);
  return status;
}

static iree_status_t iree_hal_remote_client_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    return iree_ok_status();
  }
  if (!iree_hal_remote_client_bulk_has_profile_session(device)) {
    return iree_ok_status();
  }
  iree_hal_remote_client_profiling_flush_message_t message = {
      .envelope =
          {
              .message_type = IREE_HAL_REMOTE_CONTROL_PROFILING_FLUSH,
          },
  };
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease = {0};
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      device, iree_make_const_byte_span(&message, sizeof(message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    iree_hal_remote_client_bulk_end_profile_session(device);
    return iree_ok_status();
  }
  if (!iree_hal_remote_client_bulk_has_profile_session(device)) {
    return iree_ok_status();
  }
  iree_hal_remote_client_profiling_end_message_t message = {
      .envelope =
          {
              .message_type = IREE_HAL_REMOTE_CONTROL_PROFILING_END,
          },
  };
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease = {0};
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      device, iree_make_const_byte_span(&message, sizeof(message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  iree_hal_remote_client_bulk_end_profile_session(device);
  return status;
}

//===----------------------------------------------------------------------===//
// Session callbacks
//===----------------------------------------------------------------------===//

static iree_hal_remote_file_registration_capabilities_t
iree_hal_remote_client_device_file_registration_capabilities(
    iree_net_carrier_t* carrier) {
  iree_hal_remote_file_registration_capabilities_t capabilities =
      IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  if (carrier) {
    iree_net_carrier_capabilities_t carrier_capabilities =
        iree_net_carrier_capabilities(carrier);
    if (iree_any_bit_set(carrier_capabilities,
                         IREE_NET_CARRIER_CAPABILITY_POSIX_FD_TRANSFER)) {
      capabilities |= IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_POSIX_FD;
    }
    if (iree_any_bit_set(carrier_capabilities,
                         IREE_NET_CARRIER_CAPABILITY_WIN32_HANDLE_TRANSFER)) {
      capabilities |= IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_WIN32_HANDLE;
    }
  }
  return capabilities;
}

static void iree_hal_remote_client_device_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_device_spec_t* remote_device_spec = NULL;
  iree_status_t status = iree_hal_remote_bootstrap_device_catalog_parse_spec(
      remote_topology->application_data, /*device_ordinal=*/0,
      device->host_allocator, &remote_device_spec);
  if (!iree_status_is_ok(status)) {
    iree_hal_device_spec_release(remote_device_spec);
    status = iree_status_join(
        status,
        iree_net_session_shutdown(session, IREE_STATUS_INVALID_ARGUMENT,
                                  IREE_SV("invalid remote device catalog")));
    iree_hal_remote_client_device_fail(device, status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_hal_remote_client_slab_provider_configure(device->queue_slab_provider,
                                                 remote_device_spec);

  iree_hal_device_spec_t* old_device_spec = device->device_spec;
  device->device_spec = remote_device_spec;
  remote_device_spec = NULL;
  iree_hal_device_spec_release(old_device_spec);

  // Save the remote queue axis for building signal frontiers.
  // The server's topology must have at least one axis (the queue axis).
  if (remote_topology->axis_count > 0) {
    device->remote_queue_axis = remote_topology->axes[0];
  }
  iree_atomic_store(&device->next_submission_epoch, 1,
                    iree_memory_order_relaxed);
  iree_atomic_store(&device->next_provisional_generation, 1,
                    iree_memory_order_relaxed);
  device->session_carrier = iree_net_session_carrier(session);
  device->file_registration_capabilities =
      iree_hal_remote_client_device_file_registration_capabilities(
          device->session_carrier);

  // The connect callback is NOT fired here — it is deferred until queue and
  // bulk endpoint provisioning complete so CONNECTED means all production
  // channels are usable.
  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_hal_remote_client_device_on_queue_endpoint_ready,
      .user_data = device,
  };
  status = iree_net_session_open_endpoint(session, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_fail(device, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_goaway(
    void* user_data, iree_net_session_t* session, uint32_t reason_code,
    iree_string_view_t message) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;
  iree_hal_remote_client_device_fail(
      device,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "server sent GOAWAY (%u): %.*s",
                       reason_code, (int)message.size,
                       message.data ? message.data : ""));

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_error(
    void* user_data, iree_net_session_t* session, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)session;

  iree_hal_remote_client_device_fail(device, status);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_client_device_on_session_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  (void)operation_user_data;
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_device_state_t state =
      iree_hal_remote_client_device_load_state(device);
  if (state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATING ||
      state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED) {
    iree_status_free(status);
    return;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_fail(device, status);
  } else {
    iree_status_free(status);
  }
}

iree_status_t iree_hal_remote_client_device_check_connected(
    iree_hal_remote_client_device_t* device) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&device->failure_mutex);
  if (!iree_status_is_ok(device->terminal_status)) {
    status = iree_status_clone(device->terminal_status);
  } else if (iree_hal_remote_client_device_load_state(device) !=
             IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "device is not connected");
  }
  iree_slim_mutex_unlock(&device->failure_mutex);
  return status;
}

typedef struct iree_hal_remote_client_terminal_effects_t {
  // Status transferred to the bulk transfer subsystem.
  iree_status_t bulk_status;

  // Status transferred to the remote queue frontier, when registered.
  iree_status_t frontier_status;
} iree_hal_remote_client_terminal_effects_t;

// Publishes the preserved terminal status to pending operation state. The
// caller must hold |failure_mutex| across any associated state transition.
static void iree_hal_remote_client_device_prepare_terminal_effects_locked(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_device_state_t previous_state, iree_status_t status,
    iree_hal_remote_client_terminal_effects_t* out_effects) {
  IREE_ASSERT(!iree_status_is_ok(status));
  memset(out_effects, 0, sizeof(*out_effects));

  if (iree_status_is_ok(device->terminal_status)) {
    device->terminal_status = status;
  } else {
    iree_status_free(status);
  }
  for (iree_hal_remote_pending_rpc_t* pending = device->pending_rpcs; pending;
       pending = pending->next) {
    if (iree_status_is_ok(pending->failure_status)) {
      pending->failure_status = iree_status_clone(device->terminal_status);
      iree_atomic_store(&pending->response_status_code,
                        (int32_t)iree_status_code(device->terminal_status),
                        iree_memory_order_release);
      iree_notification_post(&pending->notification, IREE_ALL_WAITERS);
    }
  }
  out_effects->bulk_status = iree_status_clone(device->terminal_status);
  if (previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    out_effects->frontier_status = iree_status_clone(device->terminal_status);
  }
}

// Applies terminal effects that may acquire subsystem locks or close channel
// admission. The caller must not hold |failure_mutex|.
static void iree_hal_remote_client_device_apply_terminal_effects(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_terminal_effects_t effects) {
  iree_hal_remote_client_bulk_fail_transfers(device, effects.bulk_status);
  if (!iree_status_is_ok(effects.frontier_status)) {
    iree_async_frontier_tracker_fail_axis(device->frontier_tracker,
                                          device->remote_queue_axis,
                                          effects.frontier_status);
  }
  iree_hal_remote_client_device_detach_bulk_channel(device);
  iree_hal_remote_client_device_detach_queue_channel(device);
}

void iree_hal_remote_client_device_fail(iree_hal_remote_client_device_t* device,
                                        iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));

  iree_hal_remote_client_device_state_t previous_state;
  iree_hal_remote_client_terminal_effects_t effects;
  iree_hal_remote_client_device_connected_callback_t connect_callback = {0};
  iree_status_t callback_status = iree_ok_status();
  bool dispatch_error = false;
  iree_slim_mutex_lock(&device->failure_mutex);
  previous_state = iree_hal_remote_client_device_load_state(device);
  if (!iree_status_is_ok(device->terminal_status) ||
      previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATING ||
      previous_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED) {
    iree_slim_mutex_unlock(&device->failure_mutex);
    iree_status_free(status);
    return;
  }

  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  iree_hal_remote_client_device_prepare_terminal_effects_locked(
      device, previous_state, status, &effects);
  callback_status = iree_status_clone(device->terminal_status);
  connect_callback = device->connect_callback;
  memset(&device->connect_callback, 0, sizeof(device->connect_callback));
  bool connect_callback_dispatching = iree_all_bits_set(
      device->lifecycle_flags,
      IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_CONNECT_DISPATCHING);
  if (!connect_callback.fn && connect_callback_dispatching) {
    device->lifecycle_flags |=
        IREE_HAL_REMOTE_CLIENT_DEVICE_LIFECYCLE_FLAG_ERROR_DEFERRED;
  }
  dispatch_error = !connect_callback.fn && !connect_callback_dispatching;
  iree_slim_mutex_unlock(&device->failure_mutex);

  iree_hal_remote_client_device_apply_terminal_effects(device, effects);

  if (connect_callback.fn) {
    connect_callback.fn(connect_callback.user_data, callback_status);
  } else if (dispatch_error) {
    iree_hal_remote_client_device_dispatch_error(device, callback_status);
  } else {
    iree_status_free(callback_status);
  }
}

iree_status_t iree_hal_remote_client_device_control_rpc_with_after_send(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t request,
    iree_hal_remote_client_device_control_rpc_after_send_t after_send,
    iree_const_byte_span_t* out_response_payload,
    iree_async_buffer_lease_t* out_response_lease) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_response_payload = iree_const_byte_span_empty();
  memset(out_response_lease, 0, sizeof(*out_response_lease));

  iree_status_t status = iree_ok_status();
  if (request.data_length < sizeof(iree_hal_remote_control_envelope_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "control RPC request too small for envelope: %" PRIhsz " bytes",
        request.data_length);
  }

  // Assign a unique request_id and patch it into the envelope.
  uint32_t request_id = 0;
  if (iree_status_is_ok(status)) {
    request_id = (uint32_t)iree_atomic_fetch_add(&device->next_request_id, 1,
                                                 iree_memory_order_relaxed);
    // The envelope is at the start of the request buffer. We cast away const
    // to patch the request_id — the caller allocated this on their stack.
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)request.data;
    envelope->request_id = request_id;
  }

  // Initialize the pending RPC entry on the stack.
  iree_hal_remote_pending_rpc_t pending;
  memset(&pending, 0, sizeof(pending));
  bool pending_initialized = false;
  if (iree_status_is_ok(status)) {
    pending.request_id = request_id;
    iree_notification_initialize(&pending.notification);
    pending_initialized = true;
    // IREE_STATUS_INTERNAL is the sentinel: "no response yet."
    iree_atomic_store(&pending.response_status_code, IREE_STATUS_INTERNAL,
                      iree_memory_order_relaxed);
    pending.failure_status = iree_ok_status();
  }

  // Link into the pending list while serialized with terminal failure. The
  // state check must happen under this mutex so an RPC cannot register after a
  // terminal transition has already walked the list.
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&device->failure_mutex);
    if (!iree_status_is_ok(device->terminal_status)) {
      status = iree_status_clone(device->terminal_status);
    } else if (iree_hal_remote_client_device_load_state(device) !=
               IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "device is not connected");
    } else if (!device->session) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "session is not available");
    } else {
      pending.next = device->pending_rpcs;
      device->pending_rpcs = &pending;
    }
    iree_slim_mutex_unlock(&device->failure_mutex);
  }

  // Send the request.
  bool request_submitted = false;
  if (iree_status_is_ok(status)) {
    iree_async_span_t span =
        iree_async_span_from_ptr((void*)request.data, request.data_length);
    iree_async_span_list_t payload = {&span, 1};
    status = iree_net_session_send_control_data_copy(
        device->session, /*flags=*/0, payload, /*operation_user_data=*/0);
    request_submitted = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status) && after_send.fn) {
    status = after_send.fn(after_send.user_data);
  }

  if (request_submitted) {
    // Block until the proactor thread delivers the response.
    iree_wait_token_t token =
        iree_notification_prepare_wait(&pending.notification);
    // Check if the response already arrived (between send and prepare_wait).
    // Acquire pairs with the proactor thread's release store.
    if (iree_atomic_load(&pending.response_status_code,
                         iree_memory_order_acquire) == IREE_STATUS_INTERNAL) {
      // Still waiting — commit the wait.
      iree_notification_commit_wait(&pending.notification, token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    } else {
      iree_notification_cancel_wait(&pending.notification);
    }
  }

  // Unlink from the pending list.
  if (pending_initialized) {
    iree_slim_mutex_lock(&device->failure_mutex);
    iree_hal_remote_pending_rpc_t** prev = &device->pending_rpcs;
    while (*prev && *prev != &pending) prev = &(*prev)->next;
    if (*prev == &pending) *prev = pending.next;
    iree_slim_mutex_unlock(&device->failure_mutex);

    iree_notification_deinitialize(&pending.notification);
  }

  if (!iree_status_is_ok(status)) {
    // Send failed — clean up any lease that might have been set.
    iree_async_buffer_lease_release(&pending.response_lease);
    iree_status_free(pending.failure_status);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  if (!iree_status_is_ok(pending.failure_status)) {
    iree_async_buffer_lease_release(&pending.response_lease);
    iree_status_t failure_status = pending.failure_status;
    pending.failure_status = iree_ok_status();
    IREE_TRACE_ZONE_END(z0);
    return failure_status;
  }

  // Check the response status. The acquire load is redundant after
  // commit_wait (which already provides acquire ordering), but explicit
  // for the cancel_wait early-return path and for TSAN visibility.
  iree_status_code_t response_code = (iree_status_code_t)iree_atomic_load(
      &pending.response_status_code, iree_memory_order_acquire);
  if (response_code != IREE_STATUS_OK) {
    // Try to deserialize the full status from the response body. The server
    // serializes the status using status_wire format as the body payload.
    iree_status_t error_status = iree_ok_status();
    if (pending.response_payload.data_length > 0) {
      iree_status_t deserialize_status = iree_net_status_wire_deserialize(
          pending.response_payload, &error_status);
      if (!iree_status_is_ok(deserialize_status)) {
        // Deserialization failed (truncated, version mismatch, etc.).
        // Propagate the deserialization failure annotated with the server's
        // response code so the caller sees both what went wrong on the
        // server AND that the detailed status couldn't be recovered.
        error_status = iree_status_annotate_f(
            deserialize_status, "server returned %s; status details lost",
            iree_status_code_string(response_code));
      }
    } else {
      // No body — the server sent only a status code (old server, or the
      // error path didn't serialize a body). Annotate so the caller knows
      // this came from a remote RPC.
      error_status =
          iree_make_status(response_code, "remote control RPC failed");
    }
    iree_async_buffer_lease_release(&pending.response_lease);
    IREE_TRACE_ZONE_END(z0);
    return error_status;
  }

  *out_response_payload = pending.response_payload;
  *out_response_lease = pending.response_lease;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Sends a control channel message and blocks until the response arrives.
// The request span must contain the full message ([envelope + body]).
// On success, |out_response_payload| points into the retained lease and
// |out_response_lease| holds the backing buffer. The caller must release
// the lease after processing the response.
iree_status_t iree_hal_remote_client_device_control_rpc(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t request,
    iree_const_byte_span_t* out_response_payload,
    iree_async_buffer_lease_t* out_response_lease) {
  iree_hal_remote_client_device_control_rpc_after_send_t after_send = {0};
  return iree_hal_remote_client_device_control_rpc_with_after_send(
      device, request, after_send, out_response_payload, out_response_lease);
}

// Sends a fire-and-forget control message (no response expected).
iree_status_t iree_hal_remote_client_device_send_fire_and_forget(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t message) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_check_connected(device));
  if (!device->session) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session is not available");
  }
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message.data, message.data_length);
  iree_async_span_list_t payload = {&span, 1};
  return iree_net_session_send_control_data_copy(device->session, /*flags=*/0,
                                                 payload,
                                                 /*operation_user_data=*/0);
}

iree_status_t iree_hal_remote_client_device_release_resource(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id) {
  if (resource_id == 0) return iree_ok_status();

  iree_net_queue_channel_t* queue_channel =
      (iree_net_queue_channel_t*)iree_atomic_load(&device->queue_channel,
                                                  iree_memory_order_acquire);
  if (!queue_channel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "queue channel not available");
  }

  const iree_host_size_t payload_length =
      sizeof(iree_hal_remote_resource_release_op_t) +
      sizeof(iree_hal_remote_resource_id_t);
  uint64_t next_submission_epoch = (uint64_t)iree_atomic_load(
      &device->next_submission_epoch, iree_memory_order_relaxed);
  uint64_t required_observed_epoch =
      next_submission_epoch > 1 ? next_submission_epoch - 1 : 0;

  iree_net_queue_channel_command_reservation_t reservation;
  iree_status_t status = iree_net_queue_channel_begin_command(
      queue_channel, /*stream_id=*/0, /*wait_frontier_entry_count=*/0,
      /*signal_frontier_entry_count=*/0, payload_length, &reservation);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_resource_release_op_t* op =
        (iree_hal_remote_resource_release_op_t*)
            reservation.command_payload.data;
    memset(op, 0, sizeof(*op));
    op->header.type = IREE_HAL_REMOTE_QUEUE_OP_RESOURCE_RELEASE_BATCH;
    op->required_observed_epoch = required_observed_epoch;
    op->resource_count = 1;
    iree_hal_remote_resource_id_t* resource_ids =
        (iree_hal_remote_resource_id_t*)(reservation.command_payload.data +
                                         sizeof(*op));
    resource_ids[0] = resource_id;
    status = iree_net_queue_channel_commit_command(queue_channel, &reservation);
  }
  return status;
}

// Returns the device's session (for use by the allocator module).
iree_net_session_t* iree_hal_remote_client_device_session(
    iree_hal_remote_client_device_t* device) {
  return device->session;
}

//===----------------------------------------------------------------------===//
// Provisional buffer tracking
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_device_register_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id, iree_hal_buffer_t* buffer) {
  iree_slim_mutex_lock(&device->provisional_mutex);

  iree_host_size_t minimum_capacity = device->provisional_buffers.count + 1;
  if (minimum_capacity > device->provisional_buffers.capacity) {
    iree_host_size_t ids_capacity = device->provisional_buffers.capacity;
    iree_status_t status = iree_allocator_grow_array(
        device->host_allocator, minimum_capacity,
        sizeof(iree_hal_remote_resource_id_t), &ids_capacity,
        (void**)&device->provisional_buffers.provisional_ids);
    if (iree_status_is_ok(status)) {
      iree_host_size_t bufs_capacity = device->provisional_buffers.capacity;
      status = iree_allocator_grow_array(
          device->host_allocator, minimum_capacity, sizeof(iree_hal_buffer_t*),
          &bufs_capacity, (void**)&device->provisional_buffers.buffers);
      device->provisional_buffers.capacity =
          iree_min(ids_capacity, bufs_capacity);
    } else {
      device->provisional_buffers.capacity = ids_capacity;
    }
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&device->provisional_mutex);
      return status;
    }
  }

  iree_host_size_t index = device->provisional_buffers.count++;
  device->provisional_buffers.provisional_ids[index] = provisional_id;
  device->provisional_buffers.buffers[index] = buffer;
  iree_hal_buffer_retain(buffer);

  iree_slim_mutex_unlock(&device->provisional_mutex);
  return iree_ok_status();
}

iree_hal_buffer_t* iree_hal_remote_client_device_resolve_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id) {
  iree_slim_mutex_lock(&device->provisional_mutex);

  iree_hal_buffer_t* buffer = NULL;
  for (iree_host_size_t i = 0; i < device->provisional_buffers.count; ++i) {
    if (device->provisional_buffers.provisional_ids[i] == provisional_id) {
      buffer = device->provisional_buffers.buffers[i];
      // Remove by swapping with the last entry.
      iree_host_size_t last = device->provisional_buffers.count - 1;
      if (i != last) {
        device->provisional_buffers.provisional_ids[i] =
            device->provisional_buffers.provisional_ids[last];
        device->provisional_buffers.buffers[i] =
            device->provisional_buffers.buffers[last];
      }
      --device->provisional_buffers.count;
      // Release the tracking reference. The buffer stays alive via the
      // caller's reference from queue_alloca.
      iree_hal_buffer_release(buffer);
      break;
    }
  }

  iree_slim_mutex_unlock(&device->provisional_mutex);
  return buffer;
}

static iree_status_t iree_hal_remote_client_device_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;

  // Parse control envelope.
  if (payload.data_length < sizeof(iree_hal_remote_control_envelope_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control data too small for envelope: %" PRIhsz
                            " bytes",
                            payload.data_length);
  }
  const iree_hal_remote_control_envelope_t* envelope =
      (const iree_hal_remote_control_envelope_t*)payload.data;

  if (envelope->message_flags & IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE) {
    // Response to a pending RPC. Match by request_id.
    const uint8_t* after_envelope =
        payload.data + sizeof(iree_hal_remote_control_envelope_t);
    iree_host_size_t remaining =
        payload.data_length - sizeof(iree_hal_remote_control_envelope_t);

    // Parse response prefix.
    if (remaining < sizeof(iree_hal_remote_control_response_prefix_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "control response too small for response prefix: %" PRIhsz " bytes",
          remaining);
    }
    const iree_hal_remote_control_response_prefix_t* prefix =
        (const iree_hal_remote_control_response_prefix_t*)after_envelope;
    const uint8_t* response_body =
        after_envelope + sizeof(iree_hal_remote_control_response_prefix_t);
    iree_host_size_t response_body_length =
        remaining - sizeof(iree_hal_remote_control_response_prefix_t);

    // Find the matching pending RPC and publish the response while holding the
    // mutex. The waiter unlinks stack-owned pending entries after waking; the
    // lock keeps fail/disconnect paths from racing a response writer that has
    // found an entry but not yet posted its notification.
    iree_slim_mutex_lock(&device->failure_mutex);
    iree_hal_remote_pending_rpc_t* pending = device->pending_rpcs;
    while (pending && pending->request_id != envelope->request_id) {
      pending = pending->next;
    }

    if (!pending) {
      iree_slim_mutex_unlock(&device->failure_mutex);
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no pending RPC with request_id=%u",
                              envelope->request_id);
    }

    // Populate the pending RPC with response data. The payload and lease
    // must be written before the status code (which is the release store
    // that makes them visible to the app thread's acquire load).
    pending->response_payload =
        iree_make_const_byte_span(response_body, response_body_length);
    // Steal ownership of the lease so the response payload data stays valid
    // after this callback returns. Zeroing the original makes the session's
    // post-callback release a no-op (release.fn == NULL).
    pending->response_lease = *lease;
    memset(lease, 0, sizeof(*lease));

    // Release store: makes all prior writes (payload, lease) visible to the
    // app thread when it does an acquire load of response_status_code.
    iree_atomic_store(&pending->response_status_code,
                      (int32_t)prefix->status_code, iree_memory_order_release);

    // Wake the blocked caller.
    iree_notification_post(&pending->notification, IREE_ALL_WAITERS);
    iree_slim_mutex_unlock(&device->failure_mutex);
    return iree_ok_status();
  }

  // Server-initiated notification (request_id == 0).
  // TODO: dispatch NOTIFY_RESOURCE_ERROR, NOTIFY_DEVICE_LOST, etc.
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_connect(
    iree_hal_device_t* base_device,
    iree_hal_remote_client_device_connected_callback_t callback) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(callback.fn);
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&device->failure_mutex);
  iree_hal_remote_client_device_state_t state =
      iree_hal_remote_client_device_load_state(device);
  if (!iree_status_is_ok(device->terminal_status)) {
    status = iree_status_clone(device->terminal_status);
  } else if (state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "device is already connected");
  } else if (state != IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "device must be in DISCONNECTED state to connect "
                              "(current state: %d)",
                              (int)state);
  } else {
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING);
    device->connect_callback = callback;
  }
  iree_slim_mutex_unlock(&device->failure_mutex);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Configure session options. The client provides no local topology (no
  // device queues to advertise) — the server will advertise its topology in
  // the HELLO_ACK.
  iree_net_session_options_t session_options =
      iree_net_session_options_default();
  session_options.capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;
  if (iree_all_bits_set(device->options.flags,
                        IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA)) {
    session_options.capabilities |= IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
    session_options.required_capabilities |= IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
  }
  session_options.application_endpoint_count =
      IREE_HAL_REMOTE_APPLICATION_ENDPOINT_COUNT;
  if (device->options.connect_timeout_ns) {
    session_options.bootstrap_timeout_ns = device->options.connect_timeout_ns;
  }

  // Wire session callbacks.
  iree_net_session_callbacks_t session_callbacks = {
      .on_ready = iree_hal_remote_client_device_on_session_ready,
      .on_goaway = iree_hal_remote_client_device_on_session_goaway,
      .on_error = iree_hal_remote_client_device_on_session_error,
      .on_control_data = iree_hal_remote_client_device_on_control_data,
      .on_send_complete =
          iree_hal_remote_client_device_on_session_send_complete,
      .user_data = device,
  };

  // Initiate async connection + session bootstrap.
  status = iree_net_session_connect(
      device->options.transport_factory, device->options.server_address,
      device->proactor,
      iree_hal_remote_recv_pool_buffer_pool(device->recv_pool),
      device->frontier_tracker, &session_options, session_callbacks,
      device->host_allocator, &device->session);

  if (!iree_status_is_ok(status)) {
    iree_status_t bulk_status = iree_ok_status();
    iree_slim_mutex_lock(&device->failure_mutex);
    if (iree_status_is_ok(device->terminal_status)) {
      device->terminal_status = iree_status_clone(status);
      bulk_status = iree_status_clone(status);
      iree_hal_remote_client_device_store_state(
          device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
      memset(&device->connect_callback, 0, sizeof(device->connect_callback));
    }
    iree_slim_mutex_unlock(&device->failure_mutex);
    if (!iree_status_is_ok(bulk_status)) {
      iree_hal_remote_client_bulk_fail_transfers(device, bulk_status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_deactivate(
    iree_hal_device_t* base_device,
    iree_hal_remote_client_device_deactivated_callback_t callback) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(callback.fn);
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_hal_device_retain(base_device);

  iree_hal_remote_client_terminal_effects_t effects;
  iree_slim_mutex_lock(&device->failure_mutex);
  int32_t expected = (int32_t)iree_hal_remote_client_device_load_state(device);
  for (;;) {
    if (expected == (int32_t)IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
      iree_status_t status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "device connection must complete before deactivation");
      iree_slim_mutex_unlock(&device->failure_mutex);
      iree_hal_device_release(base_device);
      return status;
    }
    if (expected == (int32_t)IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATING ||
        expected == (int32_t)IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED) {
      iree_status_t status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "device deactivation already started");
      iree_slim_mutex_unlock(&device->failure_mutex);
      iree_hal_device_release(base_device);
      return status;
    }
    if (iree_atomic_compare_exchange_weak(
            &device->state, &expected,
            (int32_t)IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      break;
    }
  }

  iree_hal_remote_client_device_prepare_terminal_effects_locked(
      device, (iree_hal_remote_client_device_state_t)expected,
      iree_make_status(IREE_STATUS_CANCELLED, "remote device deactivated"),
      &effects);
  iree_slim_mutex_unlock(&device->failure_mutex);
  iree_hal_remote_client_device_apply_terminal_effects(device, effects);

  if (!device->session) {
    iree_hal_remote_client_device_store_state(
        device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);
    callback.fn(callback.user_data);
    iree_hal_device_release(base_device);
    return iree_ok_status();
  }

  device->deactivate_callback = callback;
  iree_net_session_callbacks_detached_callback_t detached_callback = {
      .fn = iree_hal_remote_client_device_on_deactivate_callbacks_detached,
      .user_data = device,
  };
  iree_net_session_detach_callbacks(device->session, detached_callback);
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_state(iree_hal_device_t* base_device) {
  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  return iree_hal_remote_client_device_load_state(device);
}

static iree_status_t iree_hal_remote_client_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  (void)base_device;
  (void)options;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote devices do not support process-local external capture");
}

static iree_status_t iree_hal_remote_client_device_external_capture_end(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote devices do not support process-local external capture");
}

static const iree_hal_device_vtable_t iree_hal_remote_client_device_vtable = {
    .destroy = iree_hal_remote_client_device_destroy,
    .id = iree_hal_remote_client_device_id,
    .host_allocator = iree_hal_remote_client_device_host_allocator,
    .device_allocator = iree_hal_remote_client_device_allocator,
    .replace_device_allocator = iree_hal_remote_client_replace_device_allocator,
    .replace_channel_provider = iree_hal_remote_client_replace_channel_provider,
    .trim = iree_hal_remote_client_device_trim,
    .device_spec = iree_hal_remote_client_device_spec,
    .sample_observation = iree_hal_remote_client_device_sample_observation,
    .topology_info = iree_hal_remote_client_device_topology_info,
    .refine_topology_edge = iree_hal_remote_client_device_refine_topology_edge,
    .assign_topology_info = iree_hal_remote_client_device_assign_topology_info,
    .create_channel = iree_hal_remote_client_device_create_channel,
    .create_command_buffer =
        iree_hal_remote_client_device_create_command_buffer,
    .load_executable = iree_hal_remote_client_device_load_executable,
    .import_file = iree_hal_remote_client_device_import_file,
    .create_semaphore = iree_hal_remote_client_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_remote_client_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_remote_client_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_remote_client_device_queue_alloca,
    .queue_dealloca = iree_hal_remote_client_device_queue_dealloca,
    .queue_fill = iree_hal_remote_client_device_queue_fill,
    .queue_update = iree_hal_remote_client_device_queue_update,
    .queue_copy = iree_hal_remote_client_device_queue_copy,
    .queue_read = iree_hal_remote_client_device_queue_read,
    .queue_write = iree_hal_remote_client_device_queue_write,
    .queue_host_call = iree_hal_remote_client_device_queue_host_call,
    .queue_dispatch = iree_hal_remote_client_device_queue_dispatch,
    .queue_execute = iree_hal_remote_client_device_queue_execute,
    .queue_atomic_wait = iree_hal_remote_client_device_queue_atomic_wait,
    .queue_atomic_store = iree_hal_remote_client_device_queue_atomic_store,
    .queue_atomic_rmw = iree_hal_remote_client_device_queue_atomic_rmw,
    .queue_timestamp = iree_hal_remote_client_device_queue_timestamp,
    .queue_flush = iree_hal_remote_client_device_queue_flush,
    .profiling_begin = iree_hal_remote_client_device_profiling_begin,
    .profiling_flush = iree_hal_remote_client_device_profiling_flush,
    .profiling_end = iree_hal_remote_client_device_profiling_end,
    .external_capture_begin =
        iree_hal_remote_client_device_external_capture_begin,
    .external_capture_end = iree_hal_remote_client_device_external_capture_end,
};
