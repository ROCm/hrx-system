// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/server.h"

#include "iree/async/operations/scheduling.h"
#include "iree/hal/device_spec.h"
#include "iree/hal/remote/protocol/bootstrap.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/file_index.h"
#include "iree/hal/remote/server/profile.h"
#include "iree/net/bootstrap.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/transport_factory.h"

// Initial capacity for the per-session resource table.
// Each session can hold up to this many concurrently allocated resources
// (buffers, semaphores, executables, etc.) across all types.
#define IREE_HAL_REMOTE_SERVER_RESOURCE_TABLE_CAPACITY 256
// Initial capacity for per-session sequence reconstruction windows.
#define IREE_HAL_REMOTE_SERVER_SEQUENCE_WINDOW_INITIAL_CAPACITY 256

static void iree_hal_remote_server_destroy(iree_hal_remote_server_t* server);

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_remote_server_options_initialize(
    iree_hal_remote_server_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_options_parse(
    iree_hal_remote_server_options_t* options, iree_string_pair_list_t params) {
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    iree_string_view_t key = params.pairs[i].key;
    iree_string_view_t value = params.pairs[i].value;

    if (iree_string_view_equal(key, IREE_SV("bind"))) {
      options->bind_address = value;
    } else if (iree_string_view_equal(key, IREE_SV("max_connections"))) {
      uint32_t max_connections = 0;
      if (!iree_string_view_atoi_uint32(value, &max_connections)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid max_connections value");
      }
      options->max_connections = max_connections;
    } else if (iree_string_view_equal(key, IREE_SV("rdma"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rdma must be 'true' or 'false'");
      }
    } else if (iree_string_view_equal(key, IREE_SV("trace"))) {
      if (iree_string_view_equal(value, IREE_SV("true"))) {
        options->flags |= IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
      } else if (iree_string_view_equal(value, IREE_SV("false"))) {
        options->flags &= ~IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "trace must be 'true' or 'false'");
      }
    }
    // Unknown parameters are ignored for forward compatibility.
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_options_verify(
    const iree_hal_remote_server_options_t* options) {
  if (!options->transport_factory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transport_factory is required");
  }
  if (iree_string_view_is_empty(options->bind_address)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bind_address is required");
  }
  if (!options->local_topology) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local_topology is required");
  }
  if (options->local_topology->axis_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local_topology must expose a queue axis");
  }
  if (iree_all_bits_set(options->flags,
                        IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA)) {
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
// iree_hal_remote_server_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_server_create_device_catalog(
    iree_hal_device_t* const* devices, iree_host_size_t device_count,
    iree_allocator_t host_allocator, iree_byte_span_t* out_catalog) {
  *out_catalog = iree_byte_span_empty();

  if (device_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote device catalog supports at most %u "
                            "devices but got %" PRIhsz,
                            UINT32_MAX, device_count);
  }

  iree_byte_span_t* serialized_specs = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, device_count,
                                                   sizeof(*serialized_specs),
                                                   (void**)&serialized_specs));
  memset(serialized_specs, 0, device_count * sizeof(*serialized_specs));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_device_spec_t* device_spec =
        iree_hal_device_spec(devices[i]);
    if (!device_spec) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "remote device %" PRIhsz " does not expose a HAL device spec", i);
    } else {
      status = iree_hal_device_spec_serialize(device_spec, host_allocator,
                                              &serialized_specs[i]);
    }
  }

  iree_host_size_t entry_table_offset = 0;
  iree_host_size_t spec_data_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_bootstrap_device_catalog_header_t),
        &spec_data_offset,
        IREE_STRUCT_FIELD(device_count,
                          iree_hal_remote_bootstrap_device_spec_entry_t,
                          &entry_table_offset));
  }

  iree_host_size_t total_size = spec_data_offset;
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t padded_spec_length = 0;
    if (!iree_host_size_checked_align(serialized_specs[i].data_length, 8,
                                      &padded_spec_length) ||
        !iree_host_size_checked_add(total_size, padded_spec_length,
                                    &total_size)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "remote device catalog size overflow");
    }
  }

  iree_byte_span_t catalog = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&catalog.data);
    catalog.data_length = total_size;
  }

  if (iree_status_is_ok(status)) {
    memset(catalog.data, 0, catalog.data_length);

    iree_hal_remote_bootstrap_device_catalog_header_t* header =
        (iree_hal_remote_bootstrap_device_catalog_header_t*)catalog.data;
    header->magic = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_MAGIC;
    header->version = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_VERSION;
    header->flags = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_CATALOG_FLAG_NONE;
    header->device_count = (uint32_t)device_count;

    iree_hal_remote_bootstrap_device_spec_entry_t* entries =
        (iree_hal_remote_bootstrap_device_spec_entry_t*)(catalog.data +
                                                         entry_table_offset);
    iree_host_size_t write_offset = spec_data_offset;
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      entries[i].device_ordinal = (uint32_t)i;
      entries[i].flags = IREE_HAL_REMOTE_BOOTSTRAP_DEVICE_SPEC_ENTRY_FLAG_NONE;
      entries[i].spec_offset = (uint64_t)write_offset;
      entries[i].spec_length = (uint64_t)serialized_specs[i].data_length;
      memcpy(catalog.data + write_offset, serialized_specs[i].data,
             serialized_specs[i].data_length);

      iree_host_size_t padded_spec_length = 0;
      if (!iree_host_size_checked_align(serialized_specs[i].data_length, 8,
                                        &padded_spec_length)) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "remote device catalog size overflow");
        break;
      }
      write_offset += padded_spec_length;
    }

    if (iree_status_is_ok(status)) {
      *out_catalog = catalog;
      catalog = iree_byte_span_empty();
    }
  }

  iree_allocator_free(host_allocator, catalog.data);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    iree_allocator_free(host_allocator, serialized_specs[i].data);
  }
  iree_allocator_free(host_allocator, serialized_specs);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_create(
    const iree_hal_remote_server_options_t* options,
    iree_hal_device_t* const* devices, iree_host_size_t device_count,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_remote_server_t** out_server) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(devices);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(out_server);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_server = NULL;

  iree_status_t status = iree_ok_status();
  if (device_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "at least one device is required");
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_options_verify(options);
  }

  uint32_t max_connections = 0;
  uint32_t axis_count = 0;
  if (iree_status_is_ok(status)) {
    max_connections = options->max_connections
                          ? options->max_connections
                          : IREE_HAL_REMOTE_DEFAULT_MAX_CONNECTIONS;
    axis_count = options->local_topology->axis_count;
  }

  // Calculate trailing storage layout.
  iree_host_size_t total_size = 0;
  iree_host_size_t bind_address_offset = 0;
  iree_host_size_t axes_offset = 0;
  iree_host_size_t epochs_offset = 0;
  iree_host_size_t devices_offset = 0;
  iree_host_size_t sessions_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_server_t), &total_size,
        IREE_STRUCT_FIELD(options->bind_address.size, char,
                          &bind_address_offset),
        IREE_STRUCT_FIELD_ALIGNED(axis_count, iree_async_axis_t,
                                  iree_alignof(iree_async_axis_t),
                                  &axes_offset),
        IREE_STRUCT_FIELD(axis_count, uint64_t, &epochs_offset),
        IREE_STRUCT_FIELD(device_count, iree_hal_device_t*, &devices_offset),
        IREE_STRUCT_FIELD(max_connections, iree_hal_remote_server_session_t,
                          &sessions_offset));
  }

  iree_hal_remote_server_t* server = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&server);
  }

  if (iree_status_is_ok(status)) {
    memset(server, 0, total_size);
    iree_atomic_ref_count_init(&server->ref_count);
    server->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&server->session_mutex);

    // Bind all trailing storage before any fallible child setup. The zeroed
    // server is then always a complete representation accepted by destroy.
    server->devices = (iree_hal_device_t**)((uint8_t*)server + devices_offset);
    server->sessions =
        (iree_hal_remote_server_session_t*)((uint8_t*)server + sessions_offset);

    // Copy options and bind_address to trailing storage.
    server->options = *options;
    server->options.max_connections = max_connections;
    iree_string_view_append_to_buffer(options->bind_address,
                                      &server->options.bind_address,
                                      (char*)server + bind_address_offset);

    // Copy topology arrays to trailing storage.
    iree_async_axis_t* local_axes =
        (iree_async_axis_t*)((uint8_t*)server + axes_offset);
    uint64_t* local_epochs = (uint64_t*)((uint8_t*)server + epochs_offset);
    memcpy(local_axes, options->local_topology->axes,
           axis_count * sizeof(iree_async_axis_t));
    memcpy(local_epochs, options->local_topology->current_epochs,
           axis_count * sizeof(uint64_t));
    server->local_topology.axes = local_axes;
    server->local_topology.current_epochs = local_epochs;
    server->local_topology.axis_count = axis_count;
    server->local_topology.machine_index =
        options->local_topology->machine_index;
    server->local_topology.session_epoch =
        options->local_topology->session_epoch;

    // Clear topology pointer in options (we own the copy now, not the
    // original).
    server->options.local_topology = NULL;

    iree_net_transport_factory_retain(server->options.transport_factory);
    iree_hal_remote_file_index_retain(server->options.file_index);

    server->device_count = device_count;
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      server->devices[i] = devices[i];
      iree_hal_device_retain(devices[i]);
    }

    // Borrow infrastructure.
    server->proactor = proactor;
    server->frontier_tracker = frontier_tracker;
    server->recv_pool = recv_pool;

    server->next_session_id = 1;
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_create_device_catalog(
        server->devices, server->device_count, host_allocator,
        &server->bootstrap_device_catalog);
  }

  if (iree_status_is_ok(status)) {
    server->local_topology.application_data =
        iree_const_cast_byte_span(server->bootstrap_device_catalog);
  }

  if (iree_status_is_ok(status)) {
    *out_server = server;
  } else if (server) {
    iree_hal_remote_server_destroy(server);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_remote_server_retain(
    iree_hal_remote_server_t* server) {
  if (IREE_LIKELY(server)) {
    iree_atomic_ref_count_inc(&server->ref_count);
  }
}

static void iree_hal_remote_server_destroy(iree_hal_remote_server_t* server) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = server->host_allocator;

  // Release all active sessions (should be empty if stop() was called).
  // Release application channels before sessions (channels reference endpoints
  // owned by sessions). Channels own their header pools (freed on channel
  // destroy). Clear each slot before releasing to prevent re-entrancy: if
  // release drops the last ref and the session's destructor synchronously
  // fires an error callback, on_session_error → remove_session must not
  // find the session still in the slot.
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    // Release resource table before releasing the session.
    iree_hal_remote_server_session_deinitialize_resource_table(
        &server->sessions[i], host_allocator);

    iree_hal_remote_server_epoch_semaphore_map_deinitialize(
        &server->sessions[i].epoch_semaphore_map, host_allocator);

    iree_hal_remote_server_session_deinitialize_provisionals(
        &server->sessions[i], host_allocator);

    iree_hal_remote_server_session_deinitialize_windows(&server->sessions[i]);
    iree_hal_remote_server_bulk_session_free(server->sessions[i].bulk_session);
    server->sessions[i].bulk_session = NULL;

    iree_net_queue_channel_detach(server->sessions[i].queue_channel);
    iree_net_queue_channel_release(server->sessions[i].queue_channel);
    server->sessions[i].queue_channel = NULL;

    iree_net_session_t* session = server->sessions[i].session;
    server->sessions[i].session = NULL;
    iree_net_session_release(session);
  }

  // Free listener if still alive (shouldn't be if stop() was called).
  if (server->listener) {
    iree_net_listener_free(server->listener);
    server->listener = NULL;
  }

  // Release retained objects.
  iree_net_transport_factory_release(server->options.transport_factory);
  iree_hal_remote_file_index_release(server->options.file_index);
  for (iree_host_size_t i = 0; i < server->device_count; ++i) {
    iree_hal_device_release(server->devices[i]);
  }
  iree_allocator_free(host_allocator, server->bootstrap_device_catalog.data);

  iree_slim_mutex_deinitialize(&server->session_mutex);
  iree_allocator_free(host_allocator, server);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_remote_server_release(
    iree_hal_remote_server_t* server) {
  if (IREE_LIKELY(server) &&
      iree_atomic_ref_count_dec(&server->ref_count) == 1) {
    iree_hal_remote_server_destroy(server);
  }
}

//===----------------------------------------------------------------------===//
// Session management
//===----------------------------------------------------------------------===//

// Finds a free session slot. Returns the slot index or -1 if full.
static int32_t iree_hal_remote_server_find_free_slot(
    iree_hal_remote_server_t* server) {
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (!server->sessions[i].session && !server->sessions[i].queue_channel &&
        iree_hal_remote_server_bulk_session_is_empty(
            server->sessions[i].bulk_session) &&
        server->sessions[i].flags == 0 &&
        server->sessions[i].queue_flags == 0) {
      return (int32_t)i;
    }
  }
  return -1;
}

static void iree_hal_remote_server_on_accept(
    void* user_data, iree_status_t status, iree_net_connection_t* connection) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check state and find a free slot under the lock.
  int32_t slot = -1;
  uint64_t session_id = 0;
  iree_slim_mutex_lock(&server->session_mutex);
  if (iree_status_is_ok(status) &&
      server->state != IREE_HAL_REMOTE_SERVER_STATE_STARTING &&
      server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  if (iree_status_is_ok(status)) {
    slot = iree_hal_remote_server_find_free_slot(server);
    if (slot < 0) {
      status = iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    } else {
      session_id = server->next_session_id++;
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  // Set the server back-pointer on the session entry before session_accept
  // so that session callbacks can access it via user_data. The slot is
  // reserved (found free under lock above) and on_accept runs on the single
  // proactor thread, so no other accept can claim this slot concurrently.
  if (iree_status_is_ok(status)) {
    server->sessions[slot].server = server;
  }

  // Initialize the resource table for this session.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_initialize(
        IREE_HAL_REMOTE_SERVER_RESOURCE_TABLE_CAPACITY, server->host_allocator,
        &server->sessions[slot].resource_table);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0,
        IREE_HAL_REMOTE_SERVER_SEQUENCE_WINDOW_INITIAL_CAPACITY,
        server->host_allocator,
        &server->sessions[slot].observed_submission_window);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0,
        IREE_HAL_REMOTE_SERVER_SEQUENCE_WINDOW_INITIAL_CAPACITY,
        server->host_allocator,
        &server->sessions[slot].completed_signal_window);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_session_options_t bulk_session_options =
        iree_hal_remote_server_bulk_session_options_default();
    status = iree_hal_remote_server_bulk_session_create(
        &server->sessions[slot], &bulk_session_options, server->host_allocator,
        &server->sessions[slot].bulk_session);
  }

  // Create the server-side session outside the lock (allocation + network
  // setup). Session callbacks use &server->sessions[slot] as user_data so
  // they have direct access to the per-session resource table.
  iree_net_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    iree_net_session_options_t session_options =
        iree_net_session_options_default();
    session_options.local_topology = server->local_topology;
    session_options.capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;
    if (iree_all_bits_set(server->options.flags,
                          IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA)) {
      session_options.capabilities |= IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
      session_options.required_capabilities |=
          IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
    }
    session_options.application_endpoint_count =
        IREE_HAL_REMOTE_APPLICATION_ENDPOINT_COUNT;
    session_options.session_id = session_id;

    iree_net_session_callbacks_t callbacks = {
        .on_ready = iree_hal_remote_server_on_session_ready,
        .on_goaway = iree_hal_remote_server_on_session_goaway,
        .on_error = iree_hal_remote_server_on_session_error,
        .on_control_data = iree_hal_remote_server_on_control_data,
        .on_send_complete = iree_hal_remote_server_on_session_send_complete,
        .user_data = &server->sessions[slot],
    };

    status = iree_net_session_accept(
        connection, server->proactor, server->frontier_tracker,
        &session_options, callbacks, server->host_allocator, &session);
  }

  // Release the accept callback's connection reference. The session retains it
  // internally if accept succeeded; connection is NULL when the transport
  // delivers an error status.
  iree_net_connection_release(connection);

  // Store the session under the lock.
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    server->sessions[slot].session = session;
    server->sessions[slot].session_id = session_id;
    ++server->active_session_count;
    iree_slim_mutex_unlock(&server->session_mutex);
  } else {
    // Clean up partially initialized session entry. Guard on slot >= 0
    // because the error may be from before a slot was acquired (server not
    // running, or all slots full).
    if (slot >= 0) {
      iree_hal_remote_server_session_deinitialize_resource_table(
          &server->sessions[slot], server->host_allocator);
      iree_hal_remote_server_session_deinitialize_windows(
          &server->sessions[slot]);
      iree_hal_remote_server_bulk_session_free(
          server->sessions[slot].bulk_session);
      server->sessions[slot].bulk_session = NULL;
      server->sessions[slot].server = NULL;
    }
    iree_status_free(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_server_on_listener_stopped(void* user_data) {
  iree_hal_remote_server_t* server = (iree_hal_remote_server_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Listener has fully stopped — safe to free it.
  iree_net_listener_free(server->listener);

  iree_hal_remote_server_stopped_callback_t stopped_callback;
  memset(&stopped_callback, 0, sizeof(stopped_callback));

  iree_slim_mutex_lock(&server->session_mutex);
  server->listener = NULL;
  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
      server->active_session_count == 0) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    stopped_callback = server->stopped_callback;
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (stopped_callback.fn) {
    stopped_callback.fn(stopped_callback.user_data);
  }

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_start(iree_hal_remote_server_t* server) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&server->session_mutex);
  iree_hal_remote_server_state_t current_state = server->state;
  iree_slim_mutex_unlock(&server->session_mutex);

  if (current_state != IREE_HAL_REMOTE_SERVER_STATE_STOPPED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in STOPPED state to start "
                            "(current state: %d)",
                            (int)current_state);
  }

  iree_slim_mutex_lock(&server->session_mutex);
  server->state = IREE_HAL_REMOTE_SERVER_STATE_STARTING;
  iree_slim_mutex_unlock(&server->session_mutex);

  // Create the listener via the transport factory (outside lock — allocation
  // and network setup). Accept completions may arrive before this call returns,
  // so the state is STARTING while the listener becomes visible to the kernel.
  iree_status_t status = iree_net_transport_factory_create_listener(
      server->options.transport_factory, server->options.bind_address,
      server->proactor, server->recv_pool, iree_hal_remote_server_on_accept,
      server, server->host_allocator, &server->listener);

  iree_slim_mutex_lock(&server->session_mutex);
  if (iree_status_is_ok(status)) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
  } else {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_ERROR;
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Performs stop processing on the proactor thread. The public stop entry point
// publishes STOPPING and schedules this routine so that session teardown is
// serialized with endpoint callbacks.
static void iree_hal_remote_server_stop_on_proactor(
    iree_hal_remote_server_t* server) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // STOPPING is published by the requesting thread before this operation is
  // submitted so that new accepts are rejected immediately.
  iree_slim_mutex_lock(&server->session_mutex);
  IREE_ASSERT_EQ(server->state, IREE_HAL_REMOTE_SERVER_STATE_STOPPING);

  // Send GOAWAY to all active sessions.
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (server->sessions[i].session) {
      iree_net_session_state_t session_state =
          iree_net_session_state(server->sessions[i].session);
      if (session_state == IREE_NET_SESSION_STATE_OPERATIONAL) {
        iree_status_t goaway_status = iree_net_session_shutdown(
            server->sessions[i].session, /*reason_code=*/0,
            iree_make_cstring_view("server stopping"));
        // Stop unconditionally removes and deactivates every session below, so
        // a GOAWAY send failure is represented by that terminal teardown.
        iree_status_free(goaway_status);
      }
    }
  }

  iree_slim_mutex_unlock(&server->session_mutex);

  // Relinquish ownership of each draining session. Sending GOAWAY transitions
  // the local session to DRAINING but does not produce a local completion
  // callback; waiting for the peer to send GOAWAY back would leave the server
  // stuck in STOPPING for transports that deactivate silently. Retain each
  // session while dropping the lock so a concurrent peer callback can remove
  // the same slot safely.
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    iree_net_session_t* session = NULL;
    iree_slim_mutex_lock(&server->session_mutex);
    session = server->sessions[i].session;
    iree_net_session_retain(session);
    iree_slim_mutex_unlock(&server->session_mutex);
    if (session) {
      iree_hal_remote_server_remove_session(server, session);
      iree_net_session_release(session);
    }
  }

  // Stop the listener (no more accepts). Done outside the lock because
  // listener_stop may have internal synchronization.
  if (server->listener) {
    iree_net_listener_stopped_callback_t listener_callback;
    listener_callback.fn = iree_hal_remote_server_on_listener_stopped;
    listener_callback.user_data = server;
    iree_status_t stop_status =
        iree_net_listener_stop(server->listener, listener_callback);
    if (!iree_status_is_ok(stop_status)) {
      // stop() has already returned to the application and its callback has no
      // status channel. Freeing a listener that still owns accept operations
      // would be a use-after-free, so fail loud instead of violating the
      // listener lifecycle contract.
      iree_status_abort(stop_status);
    }
  }

  // If there are no sessions and no listener, complete immediately.
  iree_hal_remote_server_stopped_callback_t stopped_callback;
  memset(&stopped_callback, 0, sizeof(stopped_callback));

  iree_slim_mutex_lock(&server->session_mutex);
  if (server->active_session_count == 0 && !server->listener) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    stopped_callback = server->stopped_callback;
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (stopped_callback.fn) {
    stopped_callback.fn(stopped_callback.user_data);
  }

  IREE_TRACE_ZONE_END(z0);
}

// Proactor operation carrying the retained server until stop processing has
// been dispatched. Allocated per request so the stopped callback may restart
// and stop the same server while this callback is still unwinding.
typedef struct iree_hal_remote_server_stop_operation_t {
  // NOP used to enter the server's proactor thread.
  iree_async_nop_operation_t nop;

  // Server retained until stop processing has been dispatched.
  iree_hal_remote_server_t* server;
} iree_hal_remote_server_stop_operation_t;

static void iree_hal_remote_server_stop_operation_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  iree_hal_remote_server_stop_operation_t* stop_operation =
      (iree_hal_remote_server_stop_operation_t*)user_data;
  iree_hal_remote_server_t* server = stop_operation->server;
  iree_hal_remote_server_stop_on_proactor(server);

  iree_allocator_t host_allocator = server->host_allocator;
  iree_allocator_free(host_allocator, stop_operation);
  iree_hal_remote_server_release(server);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_stop(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_stopped_callback_t callback) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_server_stop_operation_t* stop_operation = NULL;
  iree_status_t status = iree_allocator_malloc(
      server->host_allocator, sizeof(*stop_operation), (void**)&stop_operation);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(stop_operation, 0, sizeof(*stop_operation));

  // Publish STOPPING before scheduling so that accept callbacks reject new
  // connections while the stop operation waits for the proactor thread.
  iree_slim_mutex_lock(&server->session_mutex);
  if (server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    iree_hal_remote_server_state_t current_state = server->state;
    iree_slim_mutex_unlock(&server->session_mutex);
    iree_allocator_free(server->host_allocator, stop_operation);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in RUNNING state to stop "
                            "(current state: %d)",
                            (int)current_state);
  }
  server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPING;
  server->stopped_callback = callback;
  iree_slim_mutex_unlock(&server->session_mutex);

  stop_operation->server = server;
  iree_hal_remote_server_retain(server);
  iree_async_operation_initialize(
      &stop_operation->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_remote_server_stop_operation_complete, stop_operation);
  status = iree_async_proactor_submit_one(server->proactor,
                                          &stop_operation->nop.base);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    IREE_ASSERT_EQ(server->state, IREE_HAL_REMOTE_SERVER_STATE_STOPPING);
    server->state = IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
    memset(&server->stopped_callback, 0, sizeof(server->stopped_callback));
    iree_slim_mutex_unlock(&server->session_mutex);
    iree_allocator_free(server->host_allocator, stop_operation);
    iree_hal_remote_server_release(server);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_query_bound_address(
    iree_hal_remote_server_t* server, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  IREE_ASSERT_ARGUMENT(server);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_address);

  iree_slim_mutex_lock(&server->session_mutex);
  iree_hal_remote_server_state_t current_state = server->state;
  iree_net_listener_t* listener = server->listener;
  iree_slim_mutex_unlock(&server->session_mutex);

  if (current_state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "server must be in RUNNING state to query address "
                            "(current state: %d)",
                            (int)current_state);
  }

  if (!listener) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "server is RUNNING but has no listener");
  }

  return iree_net_listener_query_bound_address(listener, buffer_capacity,
                                               buffer, out_address);
}
