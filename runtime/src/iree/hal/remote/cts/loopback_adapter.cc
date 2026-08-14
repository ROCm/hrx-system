// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Remote loopback adapter for HAL CTS backend registrations.
//
// Concrete HAL CTS packages own their driver registration, executable targets,
// hardware tags, and build guards. Linking this adapter into one of those CTS
// packages derives a "remote_<backend>" CTS backend that wraps the concrete
// backend behind a loopback remote server/client pair.

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/slab.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/session.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {
namespace {

static constexpr uint32_t kAxisTableCapacity = 16;

static bool StringVectorContains(const std::vector<std::string>& values,
                                 const char* candidate) {
  for (const auto& value : values) {
    if (value == candidate) return true;
  }
  return false;
}

// Supporting infrastructure that must outlive the remote client device. The CTS
// backend cache creates one device per backend, so there is at most one active
// context per adapted backend per process.
struct RemoteBackendContext {
  // CTS-owned proactor pool retained for the remote server/client lifetime.
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  // Receive pool shared by the loopback server and client.
  iree_hal_remote_recv_pool_t* recv_pool = nullptr;
  // Loopback transport factory shared by the server and client driver.
  iree_net_transport_factory_t* factory = nullptr;
  // Frontier tracker used by the server-side remote session.
  iree_async_frontier_tracker_t* server_tracker = nullptr;
  // Concrete HAL driver returned by the source backend factory.
  iree_hal_driver_t* server_driver = nullptr;
  // Concrete HAL device wrapped by the remote server.
  iree_hal_device_t* server_device = nullptr;
  // Queue frontier group for the wrapped server device.
  iree_hal_device_group_t* server_device_group = nullptr;
  // Remote HAL server exposing the wrapped server device.
  iree_hal_remote_server_t* server = nullptr;
  // Loopback address unique to this backend context.
  std::string server_address;

  ~RemoteBackendContext() { Teardown(); }

  void Teardown() {
    if (server) {
      struct StopState {
        // Posted when the server has drained its listener and sessions.
        iree_notification_t notification;
        // Set before the stop callback posts |notification|.
        std::atomic<bool> stopped;
      } state;
      iree_notification_initialize(&state.notification);
      state.stopped.store(false, std::memory_order_relaxed);

      iree_hal_remote_server_stopped_callback_t callback;
      callback.fn = [](void* user_data) {
        auto* state = static_cast<StopState*>(user_data);
        state->stopped.store(true, std::memory_order_release);
        iree_notification_post(&state->notification, IREE_ALL_WAITERS);
      };
      callback.user_data = &state;

      iree_status_t status = iree_hal_remote_server_stop(server, callback);
      if (iree_status_is_ok(status)) {
        auto server_stopped = +[](void* user_data) -> bool {
          auto* state = static_cast<StopState*>(user_data);
          return state->stopped.load(std::memory_order_acquire);
        };
        iree_notification_await(&state.notification, server_stopped, &state,
                                iree_infinite_timeout());
      }
      IREE_EXPECT_OK(status);
      iree_notification_deinitialize(&state.notification);
    }

    iree_hal_remote_server_release(server);
    server = nullptr;
    iree_hal_device_group_release(server_device_group);
    server_device_group = nullptr;
    iree_hal_device_release(server_device);
    server_device = nullptr;
    iree_hal_driver_release(server_driver);
    server_driver = nullptr;
    iree_net_transport_factory_release(factory);
    factory = nullptr;
    iree_async_frontier_tracker_release(server_tracker);
    server_tracker = nullptr;
    iree_hal_remote_recv_pool_release(recv_pool);
    recv_pool = nullptr;
    iree_async_proactor_pool_release(proactor_pool);
    proactor_pool = nullptr;
  }
};

// GTest environment that tears down remote backend contexts at program exit.
class RemoteBackendEnvironment : public ::testing::Environment {
 public:
  RemoteBackendContext* context(const char* backend_name) {
    auto& context = contexts_[backend_name];
    if (!context) context = std::make_unique<RemoteBackendContext>();
    return context.get();
  }

  void TearDown() override {
    for (auto& entry : contexts_) {
      entry.second->Teardown();
    }
    contexts_.clear();
  }

 private:
  // Remote backend contexts keyed by backend name.
  std::map<std::string, std::unique_ptr<RemoteBackendContext>> contexts_;
};

static RemoteBackendEnvironment* GetEnvironment() {
  static auto* environment = new RemoteBackendEnvironment();
  return environment;
}

// Register before test_main adds CtsBackendCacheEnvironment. GTest tears global
// environments down in reverse registration order, ensuring cached remote
// client devices are released before their server contexts.
static bool remote_backend_environment_registered_ = [] {
  ::testing::AddGlobalTestEnvironment(GetEnvironment());
  return true;
}();

static iree_status_t CreateRemoteDevice(
    const iree_hal_device_create_params_t* create_params,
    const std::string& backend_name, const DeviceFactory& source_factory,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  RemoteBackendContext* context =
      GetEnvironment()->context(backend_name.c_str());
  *out_driver = nullptr;
  *out_device = nullptr;

  iree_status_t status = iree_ok_status();
  if (!create_params || !create_params->proactor_pool) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote CTS backend requires a proactor pool");
  }

  if (iree_status_is_ok(status)) {
    context->proactor_pool = create_params->proactor_pool;
    iree_async_proactor_pool_retain(context->proactor_pool);
    status = iree_hal_remote_recv_pool_create(
        context->proactor_pool, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        iree_allocator_system(), &context->recv_pool);
  }

  iree_async_proactor_t* proactor =
      context->recv_pool
          ? iree_hal_remote_recv_pool_proactor(context->recv_pool)
          : nullptr;

  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = kAxisTableCapacity;
    status = iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &context->server_tracker);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_loopback_factory_create(
        iree_net_loopback_factory_options_default(), iree_allocator_system(),
        &context->factory);
  }

  if (iree_status_is_ok(status)) {
    status = source_factory(create_params, &context->server_driver,
                            &context->server_device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        context->server_device, context->server_tracker,
        iree_allocator_system(), &context->server_device_group);
  }

  if (iree_status_is_ok(status)) {
    context->server_address = backend_name;
    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topology = {};
    server_topology.axes = server_axes;
    server_topology.current_epochs = server_epochs;
    server_topology.axis_count = 1;
    server_topology.machine_index = 1;
    server_topology.session_epoch = 1;

    iree_hal_remote_server_options_t server_options;
    iree_hal_remote_server_options_initialize(&server_options);
    server_options.transport_factory = context->factory;
    server_options.bind_address =
        iree_make_cstring_view(context->server_address.c_str());
    server_options.local_topology = &server_topology;
    server_options.max_connections = 1;

    iree_hal_device_t* devices[] = {context->server_device};
    status = iree_hal_remote_server_create(
        &server_options, devices, 1, proactor, context->server_tracker,
        iree_hal_remote_recv_pool_buffer_pool(context->recv_pool),
        iree_allocator_system(), &context->server);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_start(context->server);
  }

  iree_hal_driver_t* client_driver = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_driver_options_t driver_options;
    iree_hal_remote_client_driver_options_initialize(&driver_options);
    driver_options.transport_factory = context->factory;
    driver_options.default_device_options.server_address =
        iree_make_cstring_view(context->server_address.c_str());

    status = iree_hal_remote_client_driver_create(
        IREE_SV("remote"), &driver_options, iree_allocator_system(),
        &client_driver);
  }

  iree_hal_device_t* client_device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_device_by_path(
        client_driver, IREE_SV("remote"), iree_string_view_empty(),
        /*param_count=*/0, /*params=*/nullptr, create_params,
        iree_allocator_system(), &client_device);
  }

  if (iree_status_is_ok(status)) {
    struct ConnectState {
      // Notification posted once the remote client connect callback fires.
      iree_notification_t notification;
      // True once the remote client connect callback fires.
      std::atomic<bool> fired;
      // Status transferred from the remote client connect callback.
      iree_status_t status;
    };
    ConnectState connect_state;
    iree_notification_initialize(&connect_state.notification);
    connect_state.fired.store(false, std::memory_order_relaxed);
    connect_state.status = iree_ok_status();

    iree_hal_remote_client_device_connected_callback_t callback;
    callback.fn = [](void* user_data, iree_status_t status) {
      auto* connect_state = static_cast<ConnectState*>(user_data);
      connect_state->status = status;
      connect_state->fired.store(true, std::memory_order_release);
      iree_notification_post(&connect_state->notification, IREE_ALL_WAITERS);
    };
    callback.user_data = &connect_state;

    status = iree_hal_remote_client_device_connect(client_device, callback);
    if (iree_status_is_ok(status)) {
      auto connection_fired = +[](void* user_data) -> bool {
        auto* connect_state = static_cast<ConnectState*>(user_data);
        return connect_state->fired.load(std::memory_order_acquire);
      };
      iree_notification_await(&connect_state.notification, connection_fired,
                              &connect_state, iree_infinite_timeout());
      status = connect_state.status;
      connect_state.status = iree_ok_status();
    }
    iree_status_free(connect_state.status);
    iree_notification_deinitialize(&connect_state.notification);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = client_driver;
    *out_device = client_device;
  } else {
    iree_hal_device_release(client_device);
    iree_hal_driver_release(client_driver);
    context->Teardown();
  }
  return status;
}

static std::vector<std::string> BuildRemoteTags(
    const BackendConfig& source_config) {
  std::vector<std::string> tags;
  tags.push_back("remote");
  if (StringVectorContains(source_config.tags, "async_queue")) {
    tags.push_back("async_queue");
  }
  if (StringVectorContains(source_config.tags, "file_io")) {
    tags.push_back("file_io");
  }
  if (StringVectorContains(source_config.tags, "host_calls")) {
    tags.push_back("host_calls");
  }
  if (StringVectorContains(source_config.tags, "indirect")) {
    tags.push_back("indirect");
  }
  if (StringVectorContains(source_config.tags, "mapping")) {
    tags.push_back("mapping");
  }
  return tags;
}

static void AdaptBackendThroughRemoteLoopback(
    const BackendConfig& source_config,
    std::vector<BackendConfig>* adapted_configs) {
  if (StringVectorContains(source_config.tags, "remote")) return;

  std::string remote_name = "remote_";
  remote_name.append(source_config.name);

  BackendInfo remote_info = source_config.info;
  remote_info.name = remote_name;
  DeviceFactory source_factory = source_config.info.factory;
  remote_info.factory =
      [remote_name, source_factory](
          const iree_hal_device_create_params_t* create_params,
          iree_hal_driver_t** out_driver,
          iree_hal_device_t** out_device) -> iree_status_t {
    return CreateRemoteDevice(create_params, remote_name, source_factory,
                              out_driver, out_device);
  };

  BackendConfig remote_config;
  remote_config.info = std::move(remote_info);
  remote_config.name = remote_config.info.name;
  remote_config.tags = BuildRemoteTags(source_config);
  remote_config.executable_targets = source_config.executable_targets;
  adapted_configs->push_back(std::move(remote_config));
}

static bool remote_loopback_adapter_registered_ =
    (CtsRegistry::RegisterBackendAdapter({
         "remote_loopback",
         AdaptBackendThroughRemoteLoopback,
     }),
     true);

}  // namespace
}  // namespace iree::hal::cts
