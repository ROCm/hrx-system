// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-serve-device: Exposes local HAL devices to remote clients.
//
// Usage:
//   iree-serve-device --device=amdgpu://0
//   iree-serve-device --device=vulkan://
//
// Verify native executable dispatch from a remote-only client:
//   iree-remote-check --device=remote-tcp://server-host:5000

#include "iree/async/address.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/server/file_index.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/session.h"
#include "iree/net/transport_factory.h"
#include "iree/tooling/device_util.h"
#include "iree/tools/iree-serve-device/transport.h"

#if defined(IREE_HAVE_NET_RDMA_TRANSPORT)
#define IREE_SERVE_DEVICE_RDMA_BIND_HELP \
  "  rdma://address       RDMA CM (Linux)\n"
#define IREE_SERVE_DEVICE_RDMA_USAGE_EXAMPLE \
  "\n"                                       \
  "  # Serve over RDMA\n"                    \
  "  iree-serve-device --device=amdgpu://0 " \
  "--bind=rdma://192.0.2.10:7471 --rdma\n"   \
  "\n"                                       \
  "  # Connect over RDMA\n"                  \
  "  iree-remote-check "                     \
  "--device='remote-rdma://192.0.2.10:7471?rdma=true'\n"
#else
#define IREE_SERVE_DEVICE_RDMA_BIND_HELP ""
#define IREE_SERVE_DEVICE_RDMA_USAGE_EXAMPLE ""
#endif  // IREE_HAVE_NET_RDMA_TRANSPORT

IREE_FLAG(string, bind, "tcp://0.0.0.0:5000",
          "Address to bind the server to.\n"
          "The default listens on all IPv4 interfaces for trusted-network "
          "use. Remote HAL delegates peer admission and transport protection "
          "to the surrounding network.\n"
          "Transport prefixes:\n"
          "  tcp://host:port       TCP sockets (default)\n"
          "  shm:///path           Shared memory (local "
          "IPC)\n" IREE_SERVE_DEVICE_RDMA_BIND_HELP);

IREE_FLAG(int32_t, max_connections, 16,
          "Maximum number of concurrent client connections.");

IREE_FLAG(bool, rdma, false,
          "Require an RDMA-capable transport and peer for bulk transfers.");

IREE_FLAG_LIST(
    string, remote_file_allow,
    "Allows a server-side file or directory to be opened read-only by remote "
    "clients. Repeat as --remote_file_allow=logical=host_path. The host path "
    "is stat-ed during startup to decide whether it is an exact file entry or "
    "a directory prefix entry.");
IREE_FLAG_LIST(
    string, remote_file_allow_write,
    "Allows a server-side file or directory to be opened read/write by remote "
    "clients. Repeat as --remote_file_allow_write=logical=host_path. The host "
    "path is stat-ed during startup to decide whether it is an exact file "
    "entry or a directory prefix entry.");

typedef struct iree_serve_device_state_t {
  // Host allocator used for all process-owned tool state.
  iree_allocator_t host_allocator;

  // Served local HAL device selected by --device.
  iree_hal_device_t* device;

  // Single-device group that assigns topology/frontier metadata to |device|.
  iree_hal_device_group_t* device_group;

  // Proactor pool passed to the served device for async HAL operations.
  iree_async_proactor_pool_t* device_proactor_pool;

  // Main-thread proactor that owns process signal delivery.
  iree_async_proactor_t* signal_proactor;

  // Server proactor pool whose entry is polled by |server_proactor_thread|.
  iree_async_proactor_pool_t* server_proactor_pool;

  // Thread that polls the server proactor after setup completes.
  iree_async_proactor_thread_t* server_proactor_thread;

  // Shared receive buffer pool for remote protocol traffic.
  iree_hal_remote_recv_pool_t* recv_pool;

  // Tracks server-side frontier axes for remote queue ordering.
  iree_async_frontier_tracker_t* tracker;

  // Optional explicit allow-list for server-side remote FILE_OPEN requests.
  iree_hal_remote_file_index_t* file_index;

  // Transport factory backing the listener and accepted connections.
  iree_net_transport_factory_t* factory;

  // Remote HAL server exposing |device|.
  iree_hal_remote_server_t* server;

  // True when |server| has started and must be stopped before release.
  bool server_running;

  // Subscription for SIGINT/Ctrl+C on |signal_proactor|.
  iree_async_signal_subscription_t* interrupt_subscription;

  // Subscription for SIGTERM/termination on |signal_proactor|.
  iree_async_signal_subscription_t* terminate_subscription;

  // Notification posted when signal or server-stop callbacks update state.
  iree_notification_t shutdown_notification;

  // Set to 1 after the first shutdown signal is delivered.
  iree_atomic_int32_t shutdown_requested;

  // Set to 1 by the server stopped callback once shutdown has drained.
  iree_atomic_int32_t server_stopped;
} iree_serve_device_state_t;

static iree_status_t iree_serve_device_add_file_allow_list(
    iree_hal_remote_file_index_t* file_index, iree_string_view_t flag_name,
    iree_flag_string_list_t flag_list, iree_hal_memory_access_t access) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < flag_list.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t logical_name = iree_string_view_empty();
    iree_string_view_t host_path = iree_string_view_empty();
    if (iree_string_view_split(flag_list.values[i], '=', &logical_name,
                               &host_path) < 0 ||
        iree_string_view_is_empty(logical_name) ||
        iree_string_view_is_empty(host_path)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--%.*s values must be logical=host_path",
                                (int)flag_name.size, flag_name.data);
    } else {
      status = iree_hal_remote_file_index_allow_path(file_index, logical_name,
                                                     host_path, access);
    }
  }
  return status;
}

static iree_status_t iree_serve_device_create_file_index_from_flags(
    iree_serve_device_state_t* state) {
  iree_flag_string_list_t read_only_list = FLAG_remote_file_allow_list();
  iree_flag_string_list_t read_write_list = FLAG_remote_file_allow_write_list();
  iree_status_t status = iree_ok_status();
  if (read_only_list.count == 0 && read_write_list.count == 0) {
    return status;
  }

  iree_hal_remote_file_index_t* file_index = NULL;
  status =
      iree_hal_remote_file_index_create(state->host_allocator, &file_index);
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_add_file_allow_list(
        file_index, IREE_SV("remote_file_allow"), read_only_list,
        IREE_HAL_MEMORY_ACCESS_READ);
  }
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_add_file_allow_list(
        file_index, IREE_SV("remote_file_allow_write"), read_write_list,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
  }
  if (iree_status_is_ok(status)) {
    state->file_index = file_index;
  } else {
    iree_hal_remote_file_index_release(file_index);
  }
  return status;
}

static void iree_serve_device_on_signal(void* user_data,
                                        iree_async_signal_t signal) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  int32_t was_requested = (int32_t)iree_atomic_exchange(
      &state->shutdown_requested, 1, iree_memory_order_acq_rel);
  if (!was_requested) {
    fprintf(stdout, "\nReceived %.*s, shutting down...\n",
            (int)iree_async_signal_name(signal).size,
            iree_async_signal_name(signal).data);
    fflush(stdout);
  }
  iree_notification_post(&state->shutdown_notification, IREE_ALL_WAITERS);
}

static bool iree_serve_device_is_shutdown_requested(void* user_data) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  return iree_atomic_load(&state->shutdown_requested,
                          iree_memory_order_acquire) != 0;
}

static bool iree_serve_device_is_server_stopped(void* user_data) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  return iree_atomic_load(&state->server_stopped, iree_memory_order_acquire) !=
         0;
}

static void iree_serve_device_on_server_stopped(void* user_data) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  iree_atomic_store(&state->server_stopped, 1, iree_memory_order_release);
  iree_notification_post(&state->shutdown_notification, IREE_ALL_WAITERS);
}

static iree_status_t iree_serve_device_initialize_signal_handling(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_async_signal_block_default();
  if (iree_status_is_ok(status)) {
    status = iree_async_signal_ignore_broken_pipe();
  }
  if (iree_status_is_ok(status)) {
    iree_async_proactor_options_t signal_proactor_options =
        iree_async_proactor_options_default();
    signal_proactor_options.debug_name = IREE_SV("serve-signal");
    status = iree_async_proactor_create_platform(signal_proactor_options,
                                                 state->host_allocator,
                                                 &state->signal_proactor);
  }
  if (iree_status_is_ok(status)) {
    iree_async_signal_callback_t signal_callback = {
        .fn = iree_serve_device_on_signal,
        .user_data = state,
    };
    status = iree_async_proactor_subscribe_signal(
        state->signal_proactor, IREE_ASYNC_SIGNAL_INTERRUPT, signal_callback,
        &state->interrupt_subscription);
  }
  if (iree_status_is_ok(status)) {
    iree_async_signal_callback_t signal_callback = {
        .fn = iree_serve_device_on_signal,
        .user_data = state,
    };
    status = iree_async_proactor_subscribe_signal(
        state->signal_proactor, IREE_ASYNC_SIGNAL_TERMINATE, signal_callback,
        &state->terminate_subscription);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_serve_device_wait_for_shutdown_signal(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         !iree_serve_device_is_shutdown_requested(state)) {
    status = iree_async_proactor_poll(state->signal_proactor,
                                      iree_infinite_timeout(),
                                      /*out_completed_count=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_serve_device_wait_for_server_stopped(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (state->server_proactor_thread) {
    iree_notification_await(&state->shutdown_notification,
                            iree_serve_device_is_server_stopped, state,
                            iree_infinite_timeout());
  } else {
    iree_async_proactor_t* proactor =
        iree_hal_remote_recv_pool_proactor(state->recv_pool);
    while (iree_status_is_ok(status) &&
           !iree_serve_device_is_server_stopped(state)) {
      status = iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                        /*out_completed_count=*/NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_serve_device_stop_server(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (state->server_running) {
    iree_atomic_store(&state->server_stopped, 0, iree_memory_order_release);

    iree_hal_remote_server_stopped_callback_t stopped_callback = {
        .fn = iree_serve_device_on_server_stopped,
        .user_data = state,
    };
    status = iree_hal_remote_server_stop(state->server, stopped_callback);
    if (iree_status_is_ok(status)) {
      status = iree_serve_device_wait_for_server_stopped(state);
    }
    if (iree_status_is_ok(status)) {
      state->server_running = false;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_serve_device_stop_server_thread(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (state->server_proactor_thread) {
    iree_async_proactor_thread_request_stop(state->server_proactor_thread);
    status = iree_async_proactor_thread_join(state->server_proactor_thread,
                                             IREE_DURATION_INFINITE);
    bool joined = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_thread_consume_status(
          state->server_proactor_thread);
    }
    if (joined) {
      iree_async_proactor_thread_release(state->server_proactor_thread);
      state->server_proactor_thread = NULL;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Creates and starts the server before its proactor begins polling.
static iree_status_t iree_serve_device_create_and_start_server(
    iree_serve_device_state_t* state, iree_string_view_t bind_address) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_t* proactor =
      iree_hal_remote_recv_pool_proactor(state->recv_pool);

  iree_async_axis_t server_axes[] = {0x0200};
  uint64_t server_epochs[] = {0};
  iree_net_session_topology_t server_topology = {
      .axes = server_axes,
      .current_epochs = server_epochs,
      .axis_count = 1,
      .machine_index = 1,
      .session_epoch = 1,
  };

  iree_hal_remote_server_options_t options;
  iree_hal_remote_server_options_initialize(&options);
  options.transport_factory = state->factory;
  options.bind_address = bind_address;
  options.local_topology = &server_topology;
  options.max_connections = (uint32_t)FLAG_max_connections;
  options.file_index = state->file_index;
  if (FLAG_rdma) {
    options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
  }

  iree_hal_device_t* devices[] = {state->device};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_create(
              &options, devices, 1, proactor, state->tracker,
              iree_hal_remote_recv_pool_buffer_pool(state->recv_pool),
              state->host_allocator, &state->server));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_start(state->server));
  state->server_running = true;

  iree_async_proactor_thread_options_t server_thread_options =
      iree_async_proactor_thread_options_default();
  server_thread_options.debug_name = IREE_SV("serve-net");
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_thread_create(proactor, server_thread_options,
                                            state->host_allocator,
                                            &state->server_proactor_thread));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_string_view_t iree_serve_device_extract_bound_port(
    iree_string_view_t bound_address) {
  iree_host_size_t separator = iree_string_view_find_last_of(
      bound_address, IREE_SV(":"), IREE_STRING_VIEW_NPOS);
  if (separator == IREE_STRING_VIEW_NPOS ||
      separator + 1 >= bound_address.size) {
    return iree_string_view_empty();
  }
  return iree_string_view_substr(bound_address, separator + 1,
                                 bound_address.size - separator - 1);
}

// Reports the actual listener and a client device flag or wildcard template.
static iree_status_t iree_serve_device_report_ready(
    iree_serve_device_state_t* state, iree_string_view_t transport_name) {
  char bound_address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t bound_address = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_query_bound_address(
      state->server, sizeof(bound_address_buffer), bound_address_buffer,
      &bound_address));

  iree_string_view_t device_id = iree_hal_device_id(state->device);
  iree_serve_device_bind_visibility_t visibility =
      iree_serve_device_classify_bind_visibility(transport_name, bound_address);
  const char* client_query = FLAG_rdma ? "?rdma=true" : "";

  fprintf(stdout,
          "\nRemote HAL server ready.\n"
          "  Device: %.*s\n"
          "  Listener: %.*s://%.*s\n",
          (int)device_id.size, device_id.data, (int)transport_name.size,
          transport_name.data, (int)bound_address.size, bound_address.data);
  if (visibility == IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD) {
    iree_string_view_t port =
        iree_serve_device_extract_bound_port(bound_address);
    if (!iree_string_view_is_empty(port)) {
      fprintf(stdout,
              "  Client: --device='remote-%.*s://<server-host>:%.*s%s'\n",
              (int)transport_name.size, transport_name.data, (int)port.size,
              port.data, client_query);
    } else {
      fprintf(stdout,
              "  Client: --device='remote-%.*s://<reachable-address>%s'\n",
              (int)transport_name.size, transport_name.data, client_query);
    }
  } else {
    fprintf(stdout, "  Client: --device='remote-%.*s://%.*s%s'\n",
            (int)transport_name.size, transport_name.data,
            (int)bound_address.size, bound_address.data, client_query);
  }

  if (visibility == IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY &&
      iree_string_view_equal(transport_name, IREE_SV("tcp"))) {
    fprintf(stdout, "  Access: same-host only.\n");
  } else if (visibility == IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY) {
    fprintf(stdout,
            "  Access: same-host IPC; operating-system endpoint permissions "
            "control access.\n");
  } else {
    fprintf(stdout,
            "  Trust: peers admitted by the surrounding network; no "
            "protocol-level authentication or encryption.\n");
  }
  fprintf(stdout, "  Press Ctrl+C to stop.\n");
  fflush(stdout);

  return iree_ok_status();
}

static iree_status_t iree_serve_device_teardown(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_serve_device_stop_server(state);
  status =
      iree_status_join(status, iree_serve_device_stop_server_thread(state));

  if (state->signal_proactor) {
    iree_async_proactor_unsubscribe_signal(state->signal_proactor,
                                           state->interrupt_subscription);
    state->interrupt_subscription = NULL;
    iree_async_proactor_unsubscribe_signal(state->signal_proactor,
                                           state->terminate_subscription);
    state->terminate_subscription = NULL;
  }

  iree_hal_remote_server_release(state->server);
  iree_async_proactor_release(state->signal_proactor);
  iree_net_transport_factory_release(state->factory);
  iree_hal_remote_file_index_release(state->file_index);
  iree_hal_remote_recv_pool_release(state->recv_pool);
  iree_async_proactor_pool_release(state->server_proactor_pool);
  iree_hal_device_group_release(state->device_group);
  iree_hal_device_release(state->device);
  iree_async_frontier_tracker_release(state->tracker);
  iree_async_proactor_pool_release(state->device_proactor_pool);
  iree_notification_deinitialize(&state->shutdown_notification);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_serve_device_run(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_serve_device_state_t state;
  memset(&state, 0, sizeof(state));
  state.host_allocator = iree_allocator_system();
  iree_notification_initialize(&state.shutdown_notification);
  iree_atomic_store(&state.shutdown_requested, 0, iree_memory_order_relaxed);
  iree_atomic_store(&state.server_stopped, 0, iree_memory_order_relaxed);

  // Process-wide signal state must be installed before creating proactor
  // threads or device workers so all child threads inherit the blocked mask.
  iree_status_t status = iree_serve_device_initialize_signal_handling(&state);

  // Create a proactor pool for the local device. Kept alive for the server's
  // lifetime — the device needs its proactor thread for async queue operations.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), state.host_allocator,
        &state.device_proactor_pool);
  }

  // Create the local device to serve (uses --device flag).
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = state.device_proactor_pool;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(),
        /*default_device=*/iree_string_view_empty(), &create_params,
        state.host_allocator, &state.device);
  }

  iree_serve_device_bind_t bind = {0};
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_parse_bind_uri(iree_make_cstring_view(FLAG_bind),
                                              &bind);
  }

  // Create a proactor pool for the server's networking.
  if (iree_status_is_ok(status)) {
    iree_async_proactor_pool_options_t server_proactor_pool_options =
        iree_async_proactor_pool_options_default();
    // Server start submits listener work and must be serialized before poll.
    memset(&server_proactor_pool_options.runner, 0,
           sizeof(server_proactor_pool_options.runner));
    status = iree_async_proactor_pool_create(
        1, /*node_ids=*/NULL, server_proactor_pool_options,
        state.host_allocator, &state.server_proactor_pool);
  }

  // Create the recv_pool from the server proactor pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_recv_pool_create(
        state.server_proactor_pool, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        state.host_allocator, &state.recv_pool);
  }

  // Create the frontier tracker.
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = 16;
    status = iree_async_frontier_tracker_create(
        tracker_options, state.host_allocator, &state.tracker);
  }

  // Assign topology/frontier metadata to the served device. Queue-ordered
  // allocation pools use the device frontier tracker to publish reservation
  // lifetimes, so a raw device without topology is not a valid served device.
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        state.device, state.tracker, state.host_allocator, &state.device_group);
  }

  if (iree_status_is_ok(status)) {
    status = iree_serve_device_create_transport_factory(
        bind.transport_name, state.host_allocator, &state.factory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_create_file_index_from_flags(&state);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_serve_device_create_and_start_server(&state, bind.bind_address);
  }
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_report_ready(&state, bind.transport_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_serve_device_wait_for_shutdown_signal(&state);
  }

  status = iree_status_join(status, iree_serve_device_teardown(&state));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-serve-device",
      "Exposes local HAL devices to remote HAL clients.\n"
      "Remote HAL is designed for trusted networks and delegates peer "
      "admission\n"
      "and transport protection to the surrounding network.\n"
      "\n"
      "Quick start (server and client machines):\n"
      "  # Serve an AMDGPU device on all IPv4 interfaces (default)\n"
      "  iree-serve-device --device=amdgpu://0\n"
      "\n"
      "  # Upload and dispatch a matching embedded native executable\n"
      "  iree-remote-check --device=remote-tcp://server-host:5000\n"
      "\n"
      "  # Restrict TCP to same-host development\n"
      "  iree-serve-device --device=amdgpu://0 "
      "--bind=tcp://127.0.0.1:5000\n"
      "\n"
      "  # Serve over shared memory (same-host IPC)\n"
      "  iree-serve-device --device=amdgpu://0 "
      "--bind=shm:///dev/shm/iree-gpu\n" IREE_SERVE_DEVICE_RDMA_USAGE_EXAMPLE);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_serve_device_run();

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
