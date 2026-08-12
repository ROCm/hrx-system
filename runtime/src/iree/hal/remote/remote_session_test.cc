// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end integration tests for the HAL remote client ↔ server lifecycle.
//
// Tests the full path: loopback transport → session bootstrap → HAL remote
// server accepting connections → HAL remote client device connecting → both
// sides reaching operational/connected state → graceful shutdown.
//
// Uses the loopback carrier factory (in-memory, no network) and the mock HAL
// device (no GPU required). This validates the session-layer integration
// without hardware dependencies.

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/operation.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/command_buffer_test_util.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/server/file_index.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/net/session.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "runtime/src/iree/hal/remote/testdata_vmvx.h"

namespace {

static iree_status_t PollProactorOnce(
    iree_async_proactor_t* proactor,
    iree_host_size_t* out_completed_count = nullptr) {
  iree_status_t status = iree_async_proactor_poll(
      proactor, iree_infinite_timeout(), out_completed_count);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

iree_status_t WriteFileContents(iree_string_view_t path,
                                iree_const_byte_span_t contents) {
  return iree_io_file_contents_write(path, contents, iree_allocator_system());
}

static const iree_hal_executable_target_t* FindExecutableTarget(
    iree_hal_device_t* device, iree_string_view_t family,
    iree_string_view_t target_key) {
  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* target = &executable_spec->targets[i];
    if (iree_string_view_equal(target->family, family) &&
        iree_string_view_equal(target->target_key, target_key)) {
      return target;
    }
  }
  return nullptr;
}

static const iree_hal_executable_target_t* FirstExecutableTarget(
    iree_hal_device_t* device) {
  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(iree_hal_device_spec(device));
  return executable_spec->target_count > 0 ? &executable_spec->targets[0]
                                           : nullptr;
}

// Notification-backed completion used to join callbacks from proactor threads.
class CompletionNotification {
 public:
  CompletionNotification() { iree_notification_initialize(&notification_); }

  ~CompletionNotification() { iree_notification_deinitialize(&notification_); }

  void Signal() {
    completed_.store(true, std::memory_order_release);
    iree_notification_post(&notification_, IREE_ALL_WAITERS);
  }

  bool Await() {
    return iree_notification_await(&notification_, IsCompleted, this,
                                   iree_infinite_timeout());
  }

 private:
  static bool IsCompleted(void* user_data) {
    auto* self = static_cast<CompletionNotification*>(user_data);
    return self->completed_.load(std::memory_order_acquire);
  }

  // Notification posted after |completed_| is published.
  iree_notification_t notification_;
  // True once the joined callback has completed.
  std::atomic<bool> completed_{false};
};

// Completion state for callbacks transferring a status code.
struct StatusCompletion {
  // Notification posted after |status_code| is published.
  CompletionNotification notification;
  // Status code transferred from the callback.
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_OK};
};

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class RemoteSessionTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    // Create a runner-less proactor pool so this fixture can manually poll.
    iree_async_proactor_pool_options_t pool_options =
        iree_async_proactor_pool_options_default();
    memset(&pool_options.runner, 0, sizeof(pool_options.runner));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL, pool_options,
        iree_allocator_system(), &proactor_pool_));
    IREE_ASSERT_OK(iree_hal_remote_recv_pool_create(
        proactor_pool_, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        iree_allocator_system(), &recv_pool_));
    proactor_ = iree_hal_remote_recv_pool_proactor(recv_pool_);

    // Create the server-side frontier tracker.
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = kAxisTableCapacity;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &server_tracker_));

    // Create loopback transport factory.
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    if (client_device_) {
      DeactivateDeviceAndWait(client_device_);
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }

    if (server_ && !server_stopped_) {
      StopServerAndWait();
    }

    iree_hal_remote_server_release(server_);
    server_ = nullptr;

    iree_hal_device_release(mock_device_);
    mock_device_ = nullptr;
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;

    iree_async_frontier_tracker_release(server_tracker_);
    server_tracker_ = nullptr;

    iree_hal_remote_recv_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_proactor_pool_release(proactor_pool_);
    proactor_pool_ = nullptr;
    proactor_ = nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Polling helpers
  //===--------------------------------------------------------------------===//

  // Runs a synchronous HAL operation on a worker while this thread drives the
  // runner-less proactor needed to complete its control RPCs.
  template <typename Operation>
  iree_status_t RunWhilePolling(Operation operation) {
    iree::Status operation_status;
    std::atomic<bool> operation_complete = false;
    std::thread operation_thread([&]() {
      operation_status = iree::Status(operation());
      operation_complete.store(true, std::memory_order_release);
      iree_async_proactor_wake(proactor_);
    });

    iree::Status poll_status;
    while (!operation_complete.load(std::memory_order_acquire) &&
           poll_status.ok()) {
      poll_status = iree::Status(PollProactorOnce(proactor_));
    }
    operation_thread.join();
    return iree_status_join(operation_status.release(), poll_status.release());
  }

  // Polls the proactor until |condition| returns true.
  bool PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree_host_size_t completed = 0;
      iree::Status status(PollProactorOnce(proactor_, &completed));
      if (!status.ok()) {
        ADD_FAILURE() << status.ToString();
        return false;
      }
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Setup helpers
  //===--------------------------------------------------------------------===//

  // Creates a mock HAL device for the server to wrap.
  void CreateMockDevice() {
    iree_hal_mock_device_options_t mock_options;
    iree_hal_mock_device_options_initialize(&mock_options);
    mock_options.identifier = IREE_SV("mock");
    mock_options.executable_loading_enabled = true;
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &mock_options, iree_allocator_system(), &mock_device_));
  }

  // Creates and starts the server with a single-axis topology.
  void CreateAndStartServer(uint32_t max_connections = 4) {
    CreateMockDevice();

    // Build server topology: one axis representing the mock device's queue.
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
    server_options.transport_factory = factory_;
    server_options.bind_address = IREE_SV("test-server");
    server_options.local_topology = &server_topology;
    server_options.max_connections = max_connections;

    iree_hal_device_t* devices[] = {mock_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, 1, proactor_, server_tracker_,
        iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
        iree_allocator_system(), &server_));

    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  // Creates the client device configured to connect to the server.
  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address = IREE_SV("test-server");

    // Wire the error callback to track post-connect errors.
    client_options.error_callback.fn = OnClientError;
    client_options.error_callback.user_data = this;

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, &create_params, recv_pool_,
        iree_allocator_system(), &client_device_));
  }

  // Connects the client device and polls until the connect callback fires.
  // Returns the status delivered to the connect callback.
  iree_status_code_t ConnectAndWait() {
    client_connect_fired_ = false;
    client_connect_status_ = IREE_STATUS_OK;

    iree_hal_remote_client_device_connected_callback_t callback;
    callback.fn = OnClientConnected;
    callback.user_data = this;

    iree_status_t connect_status =
        iree_hal_remote_client_device_connect(client_device_, callback);
    if (!iree_status_is_ok(connect_status)) {
      iree_status_code_t code = iree_status_code(connect_status);
      iree_status_ignore(connect_status);
      return code;
    }

    EXPECT_TRUE(PollUntil([&]() { return client_connect_fired_; }))
        << "Client connect callback did not fire";
    return client_connect_status_;
  }

  // Stops the server and polls until the stopped callback fires.
  void StopServerAndWait() {
    server_stopped_ = false;
    iree_hal_remote_server_stopped_callback_t callback;
    callback.fn = OnServerStopped;
    callback.user_data = this;
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(PollUntil([&]() { return server_stopped_; }))
        << "Server stop callback did not fire";
  }

  void DeactivateDeviceAndWait(iree_hal_device_t* device) {
    bool deactivated = false;
    iree_hal_remote_client_device_deactivated_callback_t callback = {
        /*.fn=*/[](void* user_data) { *static_cast<bool*>(user_data) = true; },
        /*.user_data=*/&deactivated,
    };
    IREE_ASSERT_OK(iree_hal_remote_client_device_deactivate(device, callback));
    ASSERT_TRUE(PollUntil([&]() { return deactivated; }));
  }

  //===--------------------------------------------------------------------===//
  // Callback implementations
  //===--------------------------------------------------------------------===//

  static void OnClientConnected(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->client_connect_fired_ = true;
    self->client_connect_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->client_error_fired_ = true;
    self->client_error_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnServerStopped(void* user_data) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->server_stopped_ = true;
  }

  //===--------------------------------------------------------------------===//
  // Test state
  //===--------------------------------------------------------------------===//

  // Shared infrastructure.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  iree_async_proactor_t* proactor_ = nullptr;
  iree_hal_remote_recv_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;

  // Server-side frontier tracker.
  iree_async_frontier_tracker_t* server_tracker_ = nullptr;

  // Server side.
  iree_hal_device_t* mock_device_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;
  bool server_stopped_ = false;

  // Client side.
  iree_hal_device_t* client_device_ = nullptr;
  bool client_connect_fired_ = false;
  iree_status_code_t client_connect_status_ = IREE_STATUS_OK;
  bool client_error_fired_ = false;
  iree_status_code_t client_error_status_ = IREE_STATUS_OK;
};

//===----------------------------------------------------------------------===//
// Connection lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteSessionTest, DriverCreateDeviceByPathRequiresExplicitConnect) {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_remote_client_driver_options_t driver_options;
  iree_hal_remote_client_driver_options_initialize(&driver_options);
  driver_options.transport_factory = factory_;

  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_driver_create(
      IREE_SV("remote-loopback"), &driver_options, iree_allocator_system(),
      &driver));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_path(
      driver, IREE_SV("remote-loopback"), IREE_SV("missing-server"),
      /*param_count=*/0, /*params=*/nullptr, &create_params,
      iree_allocator_system(), &device));

  EXPECT_EQ(iree_hal_remote_client_device_state(device),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED);

  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_async_proactor_pool_release(proactor_pool);
}

TEST_F(RemoteSessionTest, ConnectFailsWhenServerMissing) {
  CreateClientDevice();

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED);
  EXPECT_NE(ConnectAndWait(), IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);

  DeactivateDeviceAndWait(client_device_);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);
  iree_hal_device_release(client_device_);
  client_device_ = nullptr;
}

TEST_F(RemoteSessionTest, DeactivateWithoutConnectionCompletesSynchronously) {
  CreateClientDevice();

  bool deactivated = false;
  iree_hal_remote_client_device_deactivated_callback_t callback = {
      /*.fn=*/[](void* user_data) { *static_cast<bool*>(user_data) = true; },
      /*.user_data=*/&deactivated,
  };
  IREE_ASSERT_OK(
      iree_hal_remote_client_device_deactivate(client_device_, callback));

  EXPECT_TRUE(deactivated);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);
  iree_hal_device_release(client_device_);
  client_device_ = nullptr;
}

TEST_F(RemoteSessionTest, ConnectFailsWithInsufficientEndpointCapacity) {
  iree_net_transport_factory_release(factory_);
  factory_ = nullptr;

  iree_net_loopback_factory_options_t factory_options =
      iree_net_loopback_factory_options_default();
  factory_options.max_endpoint_count =
      IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT - 1u;
  IREE_ASSERT_OK(iree_net_loopback_factory_create(
      factory_options, iree_allocator_system(), &factory_));

  CreateAndStartServer();
  CreateClientDevice();

  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
}

TEST_F(RemoteSessionTest, ConnectSucceeds) {
  CreateAndStartServer();
  CreateClientDevice();

  // Client should start disconnected.
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED);

  // Connect and verify success.
  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
}

TEST_F(RemoteSessionTest, DeactivateRetainsDeviceThroughCallback) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  struct CallbackState {
    // Device reference transferred from the fixture to the callback.
    iree_hal_device_t* device;
    // True after the callback releases the transferred reference.
    bool fired;
  } callback_state = {
      /*.device=*/client_device_,
      /*.fired=*/false,
  };
  client_device_ = nullptr;

  iree_hal_remote_client_device_deactivated_callback_t callback = {
      /*.fn=*/
      [](void* user_data) {
        auto* state = static_cast<CallbackState*>(user_data);
        EXPECT_EQ(iree_hal_remote_client_device_state(state->device),
                  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);
        iree_hal_device_release(state->device);
        state->device = nullptr;
        state->fired = true;
      },
      /*.user_data=*/&callback_state,
  };
  IREE_ASSERT_OK(iree_hal_remote_client_device_deactivate(callback_state.device,
                                                          callback));
  ASSERT_TRUE(PollUntil([&]() { return callback_state.fired; }));
  EXPECT_EQ(callback_state.device, nullptr);
}

TEST_F(RemoteSessionTest, ConnectAdvertisesServerDeviceSpec) {
  CreateAndStartServer();
  CreateClientDevice();

  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  const iree_hal_device_spec_t* client_spec =
      iree_hal_device_spec(client_device_);
  const iree_hal_device_spec_t* server_spec =
      iree_hal_device_spec(mock_device_);
  ASSERT_NE(client_spec, nullptr);
  ASSERT_NE(server_spec, nullptr);

  iree_byte_span_t client_bytes = iree_byte_span_empty();
  iree_byte_span_t server_bytes = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_device_spec_serialize(
      client_spec, iree_allocator_system(), &client_bytes));
  IREE_ASSERT_OK(iree_hal_device_spec_serialize(
      server_spec, iree_allocator_system(), &server_bytes));
  ASSERT_EQ(client_bytes.data_length, server_bytes.data_length);
  EXPECT_EQ(
      memcmp(client_bytes.data, server_bytes.data, client_bytes.data_length),
      0);
  iree_allocator_free(iree_allocator_system(), server_bytes.data);
  iree_allocator_free(iree_allocator_system(), client_bytes.data);
}

TEST_F(RemoteSessionTest, LoadsExecutableUsingAdvertisedTargetOrdinal) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  const iree_hal_executable_target_t* target = FindExecutableTarget(
      client_device_, IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY),
      IREE_SV(IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY));
  ASSERT_NE(target, nullptr);

  // One metadata-only function record followed by its name.
  static const uint8_t kExecutableData[] = {
      0x01, 0x00, 0x00, 0x00,  // Function count.
      0x02, 0x03, 0x01,        // Constants, bindings, and flags.
      0x04, 0x02, 0x01,        // Workgroup size.
      0x08,                    // Name length.
      0x20,                    // Native ABI parameter offset.
      0x10, 0x00,              // Parameter byte size.
      'd',  'i',  's',  'p',  'a', 't', 'c', 'h',
  };
  static const uint32_t kConstants[] = {0x11223344u, 0x55667788u};
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_make_const_byte_span(kExecutableData, sizeof(kExecutableData));
  load_params.constant_count = IREE_ARRAYSIZE(kConstants);
  load_params.constants = kConstants;

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(RunWhilePolling([&]() {
    return iree_hal_device_load_executable(client_device_,
                                           IREE_HAL_QUEUE_AFFINITY_ANY, target,
                                           &load_params, &executable);
  }));
  ASSERT_NE(executable, nullptr);
  EXPECT_EQ(iree_hal_executable_function_count(executable), 1u);

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("dispatch"), &function));
  iree_hal_executable_function_info_t info;
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &info));
  EXPECT_TRUE(iree_string_view_equal(info.name, IREE_SV("dispatch")));
  EXPECT_EQ(info.flags, IREE_HAL_EXECUTABLE_FUNCTION_FLAG_SEQUENTIAL);
  EXPECT_EQ(info.constant_byte_length, 2u * sizeof(uint32_t));
  EXPECT_EQ(info.binding_count, 3u);
  EXPECT_EQ(info.parameter_count, 1u);
  EXPECT_EQ(info.workgroup_size[0], 4u);
  EXPECT_EQ(info.workgroup_size[1], 2u);
  EXPECT_EQ(info.workgroup_size[2], 1u);

  iree_hal_executable_function_parameter_t parameter;
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable, function, /*capacity=*/1, &parameter));
  EXPECT_EQ(parameter.type,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
  EXPECT_EQ(parameter.flags,
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET);
  EXPECT_EQ(parameter.size, 0x10u);
  EXPECT_EQ(parameter.offset, 0u);
  EXPECT_EQ(parameter.native_abi_offset, 0x20u);
  EXPECT_TRUE(iree_string_view_is_empty(parameter.name));

  bool global_found = true;
  iree_hal_executable_global_t global =
      iree_hal_executable_global_from_value(0);
  IREE_ASSERT_OK(RunWhilePolling([&]() {
    return iree_hal_executable_try_lookup_global_by_name(
        executable, IREE_SV("missing"), &global_found, &global);
  }));
  EXPECT_FALSE(global_found);
  EXPECT_FALSE(iree_hal_executable_global_is_valid(global));

  IREE_ASSERT_OK(RunWhilePolling([&]() {
    return iree_hal_executable_try_lookup_global_by_name(
        executable, IREE_SV(IREE_HAL_MOCK_EXECUTABLE_GLOBAL_NAME),
        &global_found, &global);
  }));
  ASSERT_TRUE(global_found);
  ASSERT_TRUE(iree_hal_executable_global_is_valid(global));

  iree_hal_executable_global_info_t global_info;
  IREE_ASSERT_OK(
      iree_hal_executable_global_info(executable, global, &global_info));
  EXPECT_TRUE(iree_string_view_equal(
      global_info.name, IREE_SV(IREE_HAL_MOCK_EXECUTABLE_GLOBAL_NAME)));
  EXPECT_EQ(global_info.byte_length, sizeof(uint64_t));

  iree_hal_buffer_t* any_global_buffer = nullptr;
  IREE_ASSERT_OK(RunWhilePolling([&]() {
    return iree_hal_executable_global_buffer(
        executable, global, IREE_HAL_QUEUE_AFFINITY_ANY, &any_global_buffer);
  }));
  ASSERT_NE(any_global_buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(any_global_buffer), sizeof(uint64_t));

  iree_hal_buffer_t* cached_any_global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(executable, global,
                                                   IREE_HAL_QUEUE_AFFINITY_ANY,
                                                   &cached_any_global_buffer));
  EXPECT_EQ(cached_any_global_buffer, any_global_buffer);

  iree_hal_buffer_t* queue_global_buffer = nullptr;
  IREE_ASSERT_OK(RunWhilePolling([&]() {
    return iree_hal_executable_global_buffer(
        executable, global, /*queue_affinity=*/1, &queue_global_buffer);
  }));
  ASSERT_NE(queue_global_buffer, nullptr);
  EXPECT_NE(queue_global_buffer, any_global_buffer);
  EXPECT_EQ(iree_hal_buffer_byte_length(queue_global_buffer), sizeof(uint64_t));

  iree_hal_executable_release(executable);
}

TEST_F(RemoteSessionTest, ReusesSingleServerSlotAfterClientDisconnect) {
  CreateAndStartServer(/*max_connections=*/1);
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  DeactivateDeviceAndWait(client_device_);
  iree_hal_device_release(client_device_);
  client_device_ = nullptr;

  CreateClientDevice();
  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
}

TEST_F(RemoteSessionTest, ConnectThenGracefulShutdown) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  // Stop the server. This sends GOAWAY to the client session.
  StopServerAndWait();

  // GOAWAY is terminal for the device because its remote resource namespace
  // and retained network graph cannot be reused by a second session.
  EXPECT_TRUE(PollUntil([&]() {
    return iree_hal_remote_client_device_state(client_device_) ==
           IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR;
  })) << "Client did not enter the terminal error state after server GOAWAY";
}

TEST_F(RemoteSessionTest, DoubleConnectFails) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  // Second connect should fail with ALREADY_EXISTS.
  iree_hal_remote_client_device_connected_callback_t callback;
  callback.fn = OnClientConnected;
  callback.user_data = this;
  iree_status_t status =
      iree_hal_remote_client_device_connect(client_device_, callback);
  EXPECT_TRUE(iree_status_is_already_exists(status))
      << "Expected ALREADY_EXISTS, got " << iree_status_code(status);
  iree_status_ignore(status);
}

TEST_F(RemoteSessionTest, ConnectBeforeDisconnected) {
  CreateAndStartServer();
  CreateClientDevice();

  // Start a connect (transitions to CONNECTING).
  client_connect_fired_ = false;
  iree_hal_remote_client_device_connected_callback_t callback;
  callback.fn = OnClientConnected;
  callback.user_data = this;
  IREE_ASSERT_OK(
      iree_hal_remote_client_device_connect(client_device_, callback));

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING);

  // Attempting another connect while CONNECTING should fail.
  iree_status_t status =
      iree_hal_remote_client_device_connect(client_device_, callback);
  EXPECT_TRUE(iree_status_is_failed_precondition(status))
      << "Expected FAILED_PRECONDITION, got " << iree_status_code(status);
  iree_status_ignore(status);

  // Let the original connect complete.
  ASSERT_TRUE(PollUntil([&]() { return client_connect_fired_; }));
}

TEST_F(RemoteSessionTest, ClientCreateRejectsRdmaWithoutTransportCapability) {
  iree_hal_remote_client_device_options_t client_options;
  iree_hal_remote_client_device_options_initialize(&client_options);
  client_options.transport_factory = factory_;
  client_options.server_address = IREE_SV("test-server");
  client_options.flags |= IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA;

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  iree_hal_device_t* client_device = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_client_device_create(
          IREE_SV("remote"), &client_options, &create_params, recv_pool_,
          iree_allocator_system(), &client_device));
  iree_hal_device_release(client_device);
}

TEST_F(RemoteSessionTest, ServerCreateRejectsRdmaWithoutTransportCapability) {
  CreateMockDevice();

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
  server_options.transport_factory = factory_;
  server_options.bind_address = IREE_SV("test-server");
  server_options.local_topology = &server_topology;
  server_options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;

  iree_hal_device_t* devices[] = {mock_device_};
  iree_hal_remote_server_t* server = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_server_create(
          &server_options, devices, 1, proactor_, server_tracker_,
          iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
          iree_allocator_system(), &server));
  iree_hal_remote_server_release(server);
}

TEST_F(RemoteSessionTest, MultipleClientsConnect) {
  CreateAndStartServer();

  // Create and connect two separate client devices.
  iree_hal_device_t* client_a = nullptr;
  iree_hal_device_t* client_b = nullptr;

  iree_hal_remote_client_device_options_t options;
  iree_hal_remote_client_device_options_initialize(&options);
  options.transport_factory = factory_;
  options.server_address = IREE_SV("test-server");

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  IREE_ASSERT_OK(iree_hal_remote_client_device_create(
      IREE_SV("remote-a"), &options, &create_params, recv_pool_,
      iree_allocator_system(), &client_a));
  IREE_ASSERT_OK(iree_hal_remote_client_device_create(
      IREE_SV("remote-b"), &options, &create_params, recv_pool_,
      iree_allocator_system(), &client_b));

  // Connect both.
  struct ConnectCtx {
    bool fired = false;
    iree_status_code_t status = IREE_STATUS_OK;
  };
  auto connect_callback = [](void* user_data, iree_status_t status) {
    auto* ctx = static_cast<ConnectCtx*>(user_data);
    ctx->fired = true;
    ctx->status = iree_status_code(status);
    iree_status_ignore(status);
  };
  ConnectCtx ctx_a, ctx_b;

  iree_hal_remote_client_device_connected_callback_t cb_a;
  cb_a.fn = connect_callback;
  cb_a.user_data = &ctx_a;
  IREE_ASSERT_OK(iree_hal_remote_client_device_connect(client_a, cb_a));

  iree_hal_remote_client_device_connected_callback_t cb_b;
  cb_b.fn = connect_callback;
  cb_b.user_data = &ctx_b;
  IREE_ASSERT_OK(iree_hal_remote_client_device_connect(client_b, cb_b));

  ASSERT_TRUE(PollUntil([&]() { return ctx_a.fired && ctx_b.fired; }))
      << "Multi-client connect callbacks did not fire";

  EXPECT_EQ(ctx_a.status, IREE_STATUS_OK);
  EXPECT_EQ(ctx_b.status, IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_a),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_b),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);

  DeactivateDeviceAndWait(client_b);
  iree_hal_device_release(client_b);
  DeactivateDeviceAndWait(client_a);
  iree_hal_device_release(client_a);
  StopServerAndWait();
}

//===----------------------------------------------------------------------===//
// Device API tests (connected vs disconnected behavior)
//===----------------------------------------------------------------------===//

TEST_F(RemoteSessionTest, QueueOpsFailWhenDisconnected) {
  CreateAndStartServer();
  CreateClientDevice();

  // Queue operations should fail with FAILED_PRECONDITION before connecting.
  iree_hal_semaphore_list_t empty_list = iree_hal_semaphore_list_empty();
  iree_hal_buffer_t* buffer = nullptr;
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_status_t status = iree_hal_device_queue_alloca(
      client_device_, /*queue_affinity=*/0, empty_list, empty_list,
      /*pool=*/nullptr, buffer_params, /*allocation_size=*/1024,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer);
  EXPECT_TRUE(iree_status_is_failed_precondition(status))
      << "Expected FAILED_PRECONDITION for queue op while disconnected";
  iree_status_ignore(status);
}

TEST_F(RemoteSessionTest, DeviceSpecAvailableWithoutConnection) {
  CreateAndStartServer();
  CreateClientDevice();

  const iree_hal_device_spec_t* client_spec =
      iree_hal_device_spec(client_device_);
  ASSERT_NE(client_spec, nullptr);

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(client_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(identity->logical_device_id, IREE_SV("remote")));
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("remote")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("remote")));
}

//===----------------------------------------------------------------------===//
// Buffer operations fixture
//===----------------------------------------------------------------------===//

// Heavier fixture with a background poll thread and a real local-task device.
// The background thread drives the proactor so the test thread can make
// blocking control_rpc calls (buffer allocation, map, unmap).
class RemoteBufferTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    // Create a runner-less proactor pool. This fixture owns the explicit
    // proactor thread below so blocking control RPCs can make progress.
    iree_async_proactor_pool_options_t pool_options =
        iree_async_proactor_pool_options_default();
    memset(&pool_options.runner, 0, sizeof(pool_options.runner));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL, pool_options,
        iree_allocator_system(), &proactor_pool_));
    IREE_ASSERT_OK(iree_hal_remote_recv_pool_create(
        proactor_pool_, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        iree_allocator_system(), &recv_pool_));
    proactor_ = iree_hal_remote_recv_pool_proactor(recv_pool_);

    // Start a dedicated poll thread. This frees the test thread to make
    // blocking RPC calls.
    IREE_ASSERT_OK(iree_async_proactor_thread_create(
        proactor_, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &proactor_thread_));

    // Create the server-side frontier tracker.
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = kAxisTableCapacity;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &server_tracker_));

    // Create loopback transport factory.
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));

    // Create local-task device (real allocator, real async completion).
    CreateLocalTaskDevice();
    CreateServerFileIndex();

    // Create server + client and connect.
    CreateAndStartServer();
    CreateClientDevice();
    ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  }

  void TearDown() override {
    if (client_device_) {
      DeactivateClientDeviceAndWait();
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }
    if (server_) {
      StopServerAndWait();
      iree_hal_remote_server_release(server_);
      server_ = nullptr;
    }

    // Client deactivation and server stop join all transport callbacks before
    // the polling thread exits.
    if (proactor_thread_) {
      iree_async_proactor_thread_request_stop(proactor_thread_);
      IREE_ASSERT_OK(iree_async_proactor_thread_join(proactor_thread_,
                                                     IREE_DURATION_INFINITE));
      iree_status_ignore(
          iree_async_proactor_thread_consume_status(proactor_thread_));
      iree_async_proactor_thread_release(proactor_thread_);
      proactor_thread_ = nullptr;
    }

    // All async operations are complete. Release remaining infrastructure.
    iree_hal_device_group_release(local_task_device_group_);
    local_task_device_group_ = nullptr;
    iree_hal_device_release(local_task_device_);
    local_task_device_ = nullptr;
    iree_hal_driver_release(local_task_driver_);
    local_task_driver_ = nullptr;
    iree_hal_remote_file_index_release(file_index_);
    file_index_ = nullptr;
    server_read_file_.Remove();
    server_write_file_.Remove();
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;

    iree_async_frontier_tracker_release(server_tracker_);
    server_tracker_ = nullptr;

    iree_hal_remote_recv_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_proactor_pool_release(proactor_pool_);
    proactor_pool_ = nullptr;
    proactor_ = nullptr;
  }

  void DeactivateClientDeviceAndWait() {
    CompletionNotification completion;
    iree_hal_remote_client_device_deactivated_callback_t callback = {
        /*.fn=*/
        [](void* user_data) {
          static_cast<CompletionNotification*>(user_data)->Signal();
        },
        /*.user_data=*/&completion,
    };
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_deactivate(client_device_, callback));
    ASSERT_TRUE(completion.Await());
  }

  void StopServerAndWait() {
    CompletionNotification completion;
    iree_hal_remote_server_stopped_callback_t callback = {
        /*.fn=*/
        [](void* user_data) {
          static_cast<CompletionNotification*>(user_data)->Signal();
        },
        /*.user_data=*/&completion,
    };
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(completion.Await());
  }

  void CreateLocalTaskDevice() {
    iree_status_t status = iree_hal_local_task_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &local_task_driver_));

    iree_async_proactor_pool_t* proactor_pool = NULL;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool));

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        local_task_driver_, &create_params, iree_allocator_system(),
        &local_task_device_));
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        local_task_device_, server_tracker_, iree_allocator_system(),
        &local_task_device_group_));

    iree_async_proactor_pool_release(proactor_pool);
  }

  void CreateServerFileIndex() {
    static const char kReadContents[] = "remote server file read contents";
    server_read_contents_ =
        std::string(kReadContents, sizeof(kReadContents) - 1);
    server_read_file_ =
        iree::testing::TempFilePath("iree_hal_remote_server_read");
    IREE_ASSERT_OK(WriteFileContents(
        server_read_file_.path_view(),
        iree_make_const_byte_span(server_read_contents_.data(),
                                  server_read_contents_.size())));

    server_write_file_ =
        iree::testing::TempFilePath("iree_hal_remote_server_write");
    std::string initial_write_contents(64, '\0');
    IREE_ASSERT_OK(WriteFileContents(
        server_write_file_.path_view(),
        iree_make_const_byte_span(initial_write_contents.data(),
                                  initial_write_contents.size())));

    IREE_ASSERT_OK(iree_hal_remote_file_index_create(iree_allocator_system(),
                                                     &file_index_));
    IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
        file_index_, IREE_SV("server://read"), server_read_file_.path_view(),
        IREE_HAL_MEMORY_ACCESS_READ));
    IREE_ASSERT_OK(iree_hal_remote_file_index_allow_path(
        file_index_, IREE_SV("server://write"), server_write_file_.path_view(),
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE));
  }

  void CreateAndStartServer() {
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
    server_options.transport_factory = factory_;
    server_options.bind_address = IREE_SV("test-server");
    server_options.local_topology = &server_topology;
    server_options.max_connections = 4;
    server_options.file_index = file_index_;

    iree_hal_device_t* devices[] = {local_task_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, 1, proactor_, server_tracker_,
        iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
        iree_allocator_system(), &server_));

    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address = IREE_SV("test-server");
    client_options.error_callback.fn = OnClientError;
    client_options.error_callback.user_data = this;

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, &create_params, recv_pool_,
        iree_allocator_system(), &client_device_));
  }

  void AllocateMappableBuffer(iree_device_size_t allocation_size,
                              iree_hal_buffer_t** out_buffer) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

    iree_hal_buffer_params_t params = {0};
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator, params, allocation_size, out_buffer));
  }

  void QueueUpdateAndWait(const void* source_buffer,
                          iree_host_size_t source_length,
                          iree_hal_buffer_t* target_buffer,
                          iree_device_size_t target_offset) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

    iree_hal_semaphore_t* signal_semaphores[] = {semaphore};
    uint64_t signal_values[] = {1};
    iree_hal_semaphore_list_t signal_list = {
        /*.count=*/1,
        /*.semaphores=*/signal_semaphores,
        /*.payload_values=*/signal_values,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_update(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal_list, source_buffer,
        /*source_offset=*/0, target_buffer, target_offset, source_length,
        IREE_HAL_UPDATE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_semaphore_release(semaphore);
  }

  iree_status_code_t ConnectAndWait() {
    StatusCompletion completion;
    iree_hal_remote_client_device_connected_callback_t callback = {
        /*.fn=*/
        [](void* user_data, iree_status_t status) {
          auto* completion = static_cast<StatusCompletion*>(user_data);
          completion->status_code.store(iree_status_code(status),
                                        std::memory_order_relaxed);
          iree_status_free(status);
          completion->notification.Signal();
        },
        /*.user_data=*/&completion,
    };

    iree_status_t connect_status =
        iree_hal_remote_client_device_connect(client_device_, callback);
    if (!iree_status_is_ok(connect_status)) {
      iree_status_code_t code = iree_status_code(connect_status);
      iree_status_free(connect_status);
      return code;
    }

    EXPECT_TRUE(completion.notification.Await());
    return completion.status_code.load(std::memory_order_relaxed);
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    iree_status_ignore(status);
  }

  // Shared infrastructure.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_proactor_thread_t* proactor_thread_ = nullptr;
  iree_hal_remote_recv_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;

  // Server-side frontier tracker.
  iree_async_frontier_tracker_t* server_tracker_ = nullptr;

  // Server side.
  iree_hal_driver_t* local_task_driver_ = nullptr;
  iree_hal_device_t* local_task_device_ = nullptr;
  iree_hal_device_group_t* local_task_device_group_ = nullptr;
  iree_hal_remote_file_index_t* file_index_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;
  iree::testing::TempFilePath server_read_file_;
  iree::testing::TempFilePath server_write_file_;
  std::string server_read_contents_;

  // Client side.
  iree_hal_device_t* client_device_ = nullptr;
};

TEST_F(RemoteBufferTest, DeactivateKeepsLateResourceCleanupSafe) {
  iree_hal_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_event_create(client_device_,
                                       IREE_HAL_QUEUE_AFFINITY_ANY,
                                       IREE_HAL_EVENT_FLAG_NONE, &event));

  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_device_open_file(
      client_device_, IREE_SV("server://read"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &file));

  DeactivateClientDeviceAndWait();

  // Child resources may outlive terminal device deactivation. Their cleanup
  // paths reach the retired queue and control channels, which must remain alive
  // and reject the late sends through their closed admission gates.
  iree_hal_event_release(event);
  iree_hal_file_release(file);

  iree_hal_device_release(client_device_);
  client_device_ = nullptr;
}

#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)

// Fixture for exercising the real POSIX SHM cross-process transport path. The
// client and server use distinct proactors so the synchronous bootstrap
// handshakes are always paired by independent polling threads.
class RemoteShmFileRegistrationTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    CreateProactorAndRecvPool(&server_proactor_, &server_recv_pool_);
    CreateProactorAndRecvPool(&client_proactor_, &client_recv_pool_);

    IREE_ASSERT_OK(iree_async_proactor_thread_create(
        server_proactor_, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &server_proactor_thread_));
    IREE_ASSERT_OK(iree_async_proactor_thread_create(
        client_proactor_, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &client_proactor_thread_));

    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = kAxisTableCapacity;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &server_tracker_));

    iree_net_shm_carrier_options_t factory_options =
        iree_net_shm_carrier_options_default();
    factory_options.max_endpoint_count =
        IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT;
    IREE_ASSERT_OK(iree_net_shm_factory_create(
        factory_options, iree_allocator_system(), &factory_));

    socket_path_ =
        "/tmp/iree-rshm-" + std::to_string((uint64_t)iree_time_now()) + ".sock";
    RemoveSocketPathIfPresent();
    server_address_ = std::string("unix:") + socket_path_;

    CreateLocalTaskDevice();
    CreateAndStartServer();
    CreateClientDevice();
    ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  }

  void TearDown() override {
    if (client_device_) {
      DeactivateClientDeviceAndWait();
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }
    if (server_) {
      StopServerAndWait();
      iree_hal_remote_server_release(server_);
      server_ = nullptr;
    }

    StopProactorThread(&client_proactor_thread_);
    StopProactorThread(&server_proactor_thread_);

    iree_hal_device_group_release(local_task_device_group_);
    local_task_device_group_ = nullptr;
    iree_hal_device_release(local_task_device_);
    local_task_device_ = nullptr;
    iree_hal_driver_release(local_task_driver_);
    local_task_driver_ = nullptr;
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;

    RemoveSocketPathIfPresent();

    iree_async_frontier_tracker_release(server_tracker_);
    server_tracker_ = nullptr;

    iree_hal_remote_recv_pool_release(client_recv_pool_);
    client_recv_pool_ = nullptr;
    iree_hal_remote_recv_pool_release(server_recv_pool_);
    server_recv_pool_ = nullptr;
    iree_async_proactor_release(client_proactor_);
    client_proactor_ = nullptr;
    iree_async_proactor_release(server_proactor_);
    server_proactor_ = nullptr;
  }

  static void CreateProactorAndRecvPool(
      iree_async_proactor_t** out_proactor,
      iree_hal_remote_recv_pool_t** out_recv_pool) {
    *out_proactor = nullptr;
    *out_recv_pool = nullptr;

    iree_async_proactor_t* proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));

    iree_async_slab_t* slab = nullptr;
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 64 * 1024;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

    iree_async_region_t* region = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));

    iree_async_buffer_pool_t* buffer_pool = nullptr;
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        region, iree_allocator_system(), &buffer_pool));

    iree_hal_remote_recv_pool_t* recv_pool = nullptr;
    IREE_ASSERT_OK(
        iree_hal_remote_recv_pool_wrap(proactor, slab, region, buffer_pool,
                                       iree_allocator_system(), &recv_pool));

    iree_async_region_release(region);
    iree_async_slab_release(slab);

    *out_proactor = proactor;
    *out_recv_pool = recv_pool;
  }

  void CreateLocalTaskDevice() {
    iree_status_t status = iree_hal_local_task_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), IREE_SV("local-task"),
        iree_allocator_system(), &local_task_driver_));

    iree_async_proactor_pool_t* proactor_pool = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool));

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        local_task_driver_, &create_params, iree_allocator_system(),
        &local_task_device_));
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        local_task_device_, server_tracker_, iree_allocator_system(),
        &local_task_device_group_));

    iree_async_proactor_pool_release(proactor_pool);
  }

  void CreateAndStartServer() {
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
    server_options.transport_factory = factory_;
    server_options.bind_address =
        iree_make_string_view(server_address_.data(), server_address_.size());
    server_options.local_topology = &server_topology;
    server_options.max_connections = 1;

    iree_hal_device_t* devices[] = {local_task_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, 1, server_proactor_, server_tracker_,
        iree_hal_remote_recv_pool_buffer_pool(server_recv_pool_),
        iree_allocator_system(), &server_));

    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address =
        iree_make_string_view(server_address_.data(), server_address_.size());
    client_options.error_callback.fn = OnClientError;
    client_options.error_callback.user_data = this;

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote-shm"), &client_options, &create_params,
        client_recv_pool_, iree_allocator_system(), &client_device_));
  }

  void AllocateMappableBuffer(iree_device_size_t allocation_size,
                              iree_hal_buffer_t** out_buffer) {
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

    iree_hal_buffer_params_t params = {0};
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator, params, allocation_size, out_buffer));
  }

  void RemoveSocketPathIfPresent() {
    if (!socket_path_.empty()) {
      errno = 0;
      if (std::remove(socket_path_.c_str()) != 0 && errno != ENOENT) {
        ADD_FAILURE() << "failed to remove Unix socket path " << socket_path_
                      << ": " << std::strerror(errno);
      }
    }
  }

  iree_status_code_t ConnectAndWait() {
    StatusCompletion completion;
    iree_hal_remote_client_device_connected_callback_t callback = {
        /*.fn=*/
        [](void* user_data, iree_status_t status) {
          auto* completion = static_cast<StatusCompletion*>(user_data);
          completion->status_code.store(iree_status_code(status),
                                        std::memory_order_relaxed);
          iree_status_free(status);
          completion->notification.Signal();
        },
        /*.user_data=*/&completion,
    };

    iree_status_t connect_status =
        iree_hal_remote_client_device_connect(client_device_, callback);
    iree_status_code_t code = iree_status_code(connect_status);
    if (!iree_status_is_ok(connect_status)) {
      iree_status_free(connect_status);
    } else {
      EXPECT_TRUE(completion.notification.Await());
      code = completion.status_code.load(std::memory_order_relaxed);
    }
    return code;
  }

  void DeactivateClientDeviceAndWait() {
    CompletionNotification completion;
    iree_hal_remote_client_device_deactivated_callback_t callback = {
        /*.fn=*/
        [](void* user_data) {
          static_cast<CompletionNotification*>(user_data)->Signal();
        },
        /*.user_data=*/&completion,
    };
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_deactivate(client_device_, callback));
    ASSERT_TRUE(completion.Await());
  }

  void StopServerAndWait() {
    CompletionNotification completion;
    iree_hal_remote_server_stopped_callback_t callback = {
        /*.fn=*/
        [](void* user_data) {
          static_cast<CompletionNotification*>(user_data)->Signal();
        },
        /*.user_data=*/&completion,
    };
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(completion.Await());
  }

  static void StopProactorThread(iree_async_proactor_thread_t** thread) {
    if (*thread) {
      iree_async_proactor_thread_request_stop(*thread);
      IREE_EXPECT_OK(
          iree_async_proactor_thread_join(*thread, IREE_DURATION_INFINITE));
      IREE_EXPECT_OK(iree_async_proactor_thread_consume_status(*thread));
      iree_async_proactor_thread_release(*thread);
      *thread = nullptr;
    }
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    (void)user_data;
    iree_status_ignore(status);
  }

  // Server I/O proactor.
  iree_async_proactor_t* server_proactor_ = nullptr;
  // Client I/O proactor.
  iree_async_proactor_t* client_proactor_ = nullptr;
  // Polling thread for |server_proactor_|.
  iree_async_proactor_thread_t* server_proactor_thread_ = nullptr;
  // Polling thread for |client_proactor_|.
  iree_async_proactor_thread_t* client_proactor_thread_ = nullptr;
  // Server receive buffers.
  iree_hal_remote_recv_pool_t* server_recv_pool_ = nullptr;
  // Client receive buffers.
  iree_hal_remote_recv_pool_t* client_recv_pool_ = nullptr;
  // Server queue frontier tracker.
  iree_async_frontier_tracker_t* server_tracker_ = nullptr;

  // Shared SHM transport factory.
  iree_net_transport_factory_t* factory_ = nullptr;
  // Short Unix socket path.
  std::string socket_path_;
  // Transport address derived from |socket_path_|.
  std::string server_address_;

  // Server target driver.
  iree_hal_driver_t* local_task_driver_ = nullptr;
  // Server target device.
  iree_hal_device_t* local_task_device_ = nullptr;
  // Queue frontier group for |local_task_device_|.
  iree_hal_device_group_t* local_task_device_group_ = nullptr;
  // Remote HAL server.
  iree_hal_remote_server_t* server_ = nullptr;
  // Remote HAL client device.
  iree_hal_device_t* client_device_ = nullptr;
};

#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS

// Profile sink used to force client-side profile callback failure after the
// remote bulk relay has delivered and ordered a callback.
struct RejectingProfileSink {
  // HAL resource header for profile sink lifetime.
  iree_hal_resource_t resource;

  // Number of begin-session callbacks observed.
  std::atomic<uint32_t> begin_count{0};

  // Number of write callbacks observed.
  std::atomic<uint32_t> write_count{0};

  // Number of end-session callbacks observed.
  std::atomic<uint32_t> end_count{0};

  // True when write callbacks should reject delivered chunks.
  std::atomic<bool> reject_writes{false};
};

static RejectingProfileSink* RejectingProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<RejectingProfileSink*>(sink);
}

static void RejectingProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t RejectingProfileSinkBegin(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)metadata;
  RejectingProfileSinkCast(sink)->begin_count.fetch_add(
      1, std::memory_order_relaxed);
  return iree_ok_status();
}

static iree_status_t RejectingProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)metadata;
  (void)iovec_count;
  (void)iovecs;
  RejectingProfileSink* rejecting_sink = RejectingProfileSinkCast(sink);
  rejecting_sink->write_count.fetch_add(1, std::memory_order_relaxed);
  if (rejecting_sink->reject_writes.load(std::memory_order_relaxed)) {
    return iree_make_status(IREE_STATUS_CANCELLED,
                            "test profile sink rejected write");
  }
  return iree_ok_status();
}

static iree_status_t RejectingProfileSinkEnd(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)metadata;
  EXPECT_EQ(session_status_code, IREE_STATUS_OK);
  RejectingProfileSinkCast(sink)->end_count.fetch_add(
      1, std::memory_order_relaxed);
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kRejectingProfileSinkVTable = {
    /*.destroy=*/RejectingProfileSinkDestroy,
    /*.begin_session=*/RejectingProfileSinkBegin,
    /*.write=*/RejectingProfileSinkWrite,
    /*.end_session=*/RejectingProfileSinkEnd,
};

static void RejectingProfileSinkInitialize(RejectingProfileSink* sink) {
  iree_hal_resource_initialize(&kRejectingProfileSinkVTable, &sink->resource);
  sink->begin_count.store(0, std::memory_order_relaxed);
  sink->write_count.store(0, std::memory_order_relaxed);
  sink->end_count.store(0, std::memory_order_relaxed);
  sink->reject_writes.store(false, std::memory_order_relaxed);
}

static iree_hal_profile_sink_t* RejectingProfileSinkAsBase(
    RejectingProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

//===----------------------------------------------------------------------===//
// Buffer allocation and map/unmap tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteBufferTest, AllocateAndDeallocate) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/256, &buffer));
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 256);

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, VirtualMemoryUnsupportedMatchesServerAllocator) {
  iree_hal_allocator_t* remote_allocator =
      iree_hal_device_allocator(client_device_);
  iree_hal_allocator_t* local_allocator =
      iree_hal_device_allocator(local_task_device_);

  ASSERT_FALSE(iree_hal_allocator_supports_virtual_memory(local_allocator));
  EXPECT_FALSE(iree_hal_allocator_supports_virtual_memory(remote_allocator));

  iree_hal_buffer_params_t params = {0};
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_device_size_t minimum_page_size = 1;
  iree_device_size_t recommended_page_size = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_allocator_virtual_memory_query_granularity(
                            remote_allocator, params, &minimum_page_size,
                            &recommended_page_size));
  EXPECT_EQ(minimum_page_size, 0);
  EXPECT_EQ(recommended_page_size, 0);

  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_allocator_virtual_memory_reserve(
                            remote_allocator, IREE_HAL_QUEUE_AFFINITY_ANY, 4096,
                            &virtual_buffer));
  EXPECT_EQ(virtual_buffer, nullptr);

  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_allocator_physical_memory_allocate(
                            remote_allocator, params, 4096,
                            iree_allocator_system(), &physical_memory));
  EXPECT_EQ(physical_memory, nullptr);
}

TEST_F(RemoteBufferTest, WriteDiscardThenReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/64, &buffer));

  // Map WRITE|DISCARD: fill with a known pattern.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/64, &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);
  ASSERT_EQ(mapping.contents.data_length, 64);

  // Fill with 0xAB pattern.
  memset(mapping.contents.data, 0xAB, 64);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Map READ: verify the pattern persisted through the round-trip.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/64, &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);

  // Every byte should be 0xAB.
  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(data[i], 0xAB) << "Mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, PartialMapRange) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/256, &buffer));

  // Write 0xCC to the first 128 bytes.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/128, &mapping));
  memset(mapping.contents.data, 0xCC, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Write 0xDD to bytes 128-255.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/128,
                                           /*byte_length=*/128, &mapping));
  memset(mapping.contents.data, 0xDD, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Read the full buffer and verify both halves.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/256, &mapping));

  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(data[i], 0xCC) << "First half mismatch at byte " << i;
  }
  for (iree_host_size_t i = 128; i < 256; ++i) {
    EXPECT_EQ(data[i], 0xDD) << "Second half mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, ReadWriteModifiesInPlace) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/16, &buffer));

  // Write initial data.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/16, &mapping));
  for (iree_host_size_t i = 0; i < 16; ++i) {
    mapping.contents.data[i] = (uint8_t)i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Map READ|WRITE: read current data, modify, write back.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*byte_offset=*/0, /*byte_length=*/16, &mapping));

  // Verify initial data was pulled.
  for (iree_host_size_t i = 0; i < 16; ++i) {
    ASSERT_EQ(mapping.contents.data[i], (uint8_t)i)
        << "READ|WRITE initial data mismatch at byte " << i;
  }

  // Increment each byte.
  for (iree_host_size_t i = 0; i < 16; ++i) {
    mapping.contents.data[i] += 100;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Read back and verify modification.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/16, &mapping));
  for (iree_host_size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(mapping.contents.data[i], (uint8_t)(i + 100))
        << "Modified data mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, InvalidatesReadOnlyMapping) {
  iree_hal_buffer_t* buffer = nullptr;
  AllocateMappableBuffer(/*allocation_size=*/64, &buffer);

  uint8_t initial_data[64];
  memset(initial_data, 0x11, sizeof(initial_data));
  QueueUpdateAndWait(initial_data, sizeof(initial_data), buffer,
                     /*target_offset=*/0);

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/64, &mapping));

  uint8_t updated_data[16];
  memset(updated_data, 0x22, sizeof(updated_data));
  QueueUpdateAndWait(updated_data, sizeof(updated_data), buffer,
                     /*target_offset=*/16);

  IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(
      &mapping, /*byte_offset=*/16, /*byte_length=*/16));

  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(data[i], 0x11) << "prefix mismatch at byte " << i;
  }
  for (iree_host_size_t i = 16; i < 32; ++i) {
    EXPECT_EQ(data[i], 0x22) << "invalidated range mismatch at byte " << i;
  }
  for (iree_host_size_t i = 32; i < 64; ++i) {
    EXPECT_EQ(data[i], 0x11) << "suffix mismatch at byte " << i;
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, InvalidatesReadWriteMapping) {
  iree_hal_buffer_t* buffer = nullptr;
  AllocateMappableBuffer(/*allocation_size=*/64, &buffer);

  uint8_t initial_data[64];
  memset(initial_data, 0x33, sizeof(initial_data));
  QueueUpdateAndWait(initial_data, sizeof(initial_data), buffer,
                     /*target_offset=*/0);

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*byte_offset=*/0, /*byte_length=*/64, &mapping));
  memset(mapping.contents.data + 8, 0x44, 8);

  uint8_t updated_data[8];
  memset(updated_data, 0x55, sizeof(updated_data));
  QueueUpdateAndWait(updated_data, sizeof(updated_data), buffer,
                     /*target_offset=*/8);

  IREE_ASSERT_OK(iree_hal_buffer_mapping_invalidate_range(
      &mapping, /*byte_offset=*/8, /*byte_length=*/8));
  memset(mapping.contents.data + 32, 0x66, 8);
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(
      &mapping, /*byte_offset=*/32, /*byte_length=*/8));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/64, &mapping));
  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], 0x33) << "prefix mismatch at byte " << i;
  }
  for (iree_host_size_t i = 8; i < 16; ++i) {
    EXPECT_EQ(data[i], 0x55) << "invalidated range mismatch at byte " << i;
  }
  for (iree_host_size_t i = 16; i < 32; ++i) {
    EXPECT_EQ(data[i], 0x33) << "middle mismatch at byte " << i;
  }
  for (iree_host_size_t i = 32; i < 40; ++i) {
    EXPECT_EQ(data[i], 0x66) << "flushed range mismatch at byte " << i;
  }
  for (iree_host_size_t i = 40; i < 64; ++i) {
    EXPECT_EQ(data[i], 0x33) << "suffix mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, RejectsOverlappingClientMappings) {
  iree_hal_buffer_t* buffer = nullptr;
  AllocateMappableBuffer(/*allocation_size=*/64, &buffer);

  iree_hal_buffer_mapping_t first_mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/64, &first_mapping));

  iree_hal_buffer_mapping_t second_mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          /*byte_offset=*/0, /*byte_length=*/64, &second_mapping));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                /*byte_offset=*/0,
                                /*byte_length=*/64, &second_mapping));

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&first_mapping));
  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Queue fill, copy, and update tests
//===----------------------------------------------------------------------===//

// Helper to build a semaphore list with a single semaphore and value.
// Named storage avoids C++ compound literal lifetime issues.
struct SemaphoreListHelper {
  iree_hal_semaphore_t* semaphore;
  uint64_t value;
  iree_hal_semaphore_list_t list;
  SemaphoreListHelper(iree_hal_semaphore_t* input_semaphore, uint64_t val)
      : semaphore(input_semaphore), value(val) {
    list.count = 1;
    list.semaphores = &semaphore;
    list.payload_values = &value;
  }
};

TEST_F(RemoteBufferTest, RepeatedAllocateReleaseDoesNotExhaustRemoteTable) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  for (iree_host_size_t i = 0; i < 300; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator, params, /*allocation_size=*/256, &buffer));
    iree_hal_buffer_release(buffer);

    // The release is a fire-and-forget queue frame. A following barrier lets
    // the test observe that the server processed it without adding any wait to
    // the production release path.
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
    SemaphoreListHelper signal(semaphore, 1);
    IREE_ASSERT_OK(iree_hal_device_queue_barrier(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal.list,
        IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_semaphore_release(semaphore);
  }
}

TEST_F(RemoteBufferTest, QueueFillAndReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/1024, &buffer));

  // Create a semaphore to track fill completion.
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Queue fill with 4-byte pattern.
  uint32_t pattern = 0xDEADBEEF;
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, buffer,
      /*target_offset=*/0, /*length=*/1024, &pattern,
      /*pattern_length=*/sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Wait for fill to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back and verify.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/1024, &mapping));
  const uint32_t* data = (const uint32_t*)mapping.contents.data;
  for (iree_host_size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(data[i], 0xDEADBEEF) << "Fill pattern mismatch at uint32 " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, QueueCopyChained) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buf_a = nullptr;
  iree_hal_buffer_t* buf_b = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 1024, &buf_a));
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 1024, &buf_b));

  // Create two semaphores for chaining: fill → copy.
  iree_hal_semaphore_t* sem_a = nullptr;
  iree_hal_semaphore_t* sem_b = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_b));

  // Fill buf_a with 0xAA, signal sem_a.
  uint8_t fill_pattern = 0xAA;
  SemaphoreListHelper fill_signal(sem_a, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), fill_signal.list, buf_a, 0, 1024,
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));

  // Copy buf_a → buf_b, wait on sem_a, signal sem_b.
  // This is the critical frontier ordering test: the copy depends on the
  // fill via sem_a. If the wait frontier doesn't correctly chain them
  // through local semaphores on the server, the copy reads stale data.
  SemaphoreListHelper copy_wait(sem_a, 1);
  SemaphoreListHelper copy_signal(sem_b, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, copy_wait.list,
      copy_signal.list, buf_a, 0, buf_b, 0, 1024, IREE_HAL_COPY_FLAG_NONE));

  // Wait for copy to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem_b, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read buf_b and verify it has the fill pattern from buf_a.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buf_b, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0, 1024,
                                           &mapping));
  for (iree_host_size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(mapping.contents.data[i], 0xAA)
        << "Copy data mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(sem_b);
  iree_hal_semaphore_release(sem_a);
  iree_hal_buffer_release(buf_b);
  iree_hal_buffer_release(buf_a);
}

TEST_F(RemoteBufferTest, QueueUpdateAndReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 64, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  // Queue update with inline host data.
  const char host_data[] = "Hello, remote HAL!";
  iree_host_size_t data_length = sizeof(host_data) - 1;  // exclude NUL
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, host_data,
      /*source_offset=*/0, buffer, /*target_offset=*/0, data_length,
      IREE_HAL_UPDATE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back and verify.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           data_length, &mapping));
  ASSERT_EQ(memcmp(mapping.contents.data, host_data, data_length), 0)
      << "Update data mismatch";
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, ProfilingSinkAbortFailsLifecycleResponse) {
  RejectingProfileSink sink;
  RejectingProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
  profiling_options.sink = RejectingProfileSinkAsBase(&sink);
  IREE_ASSERT_OK(
      iree_hal_device_profiling_begin(client_device_, &profiling_options));

  uint32_t update_data = 0xABCD1234u;
  iree_hal_buffer_t* buffer = nullptr;
  AllocateMappableBuffer(sizeof(update_data), &buffer);
  QueueUpdateAndWait(&update_data, sizeof(update_data), buffer,
                     /*target_offset=*/0);

  sink.reject_writes.store(true, std::memory_order_relaxed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_device_profiling_flush(client_device_));
  EXPECT_EQ(1u, sink.begin_count.load(std::memory_order_relaxed));
  EXPECT_GT(sink.write_count.load(std::memory_order_relaxed), 0u);

  sink.reject_writes.store(false, std::memory_order_relaxed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_device_profiling_end(client_device_));
  EXPECT_EQ(1u, sink.end_count.load(std::memory_order_relaxed));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, ServerFileQueueRead) {
  iree_hal_file_t* source_file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_device_open_file(
      client_device_, IREE_SV("server://read"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &source_file));
  EXPECT_EQ(iree_hal_file_length(source_file), 0);

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, server_read_contents_.size(), &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, source_file,
      /*source_offset=*/0, buffer, /*target_offset=*/0,
      server_read_contents_.size(), IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      server_read_contents_.size(), &mapping));
  ASSERT_EQ(memcmp(mapping.contents.data, server_read_contents_.data(),
                   server_read_contents_.size()),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(source_file);
}

TEST_F(RemoteBufferTest, RegisteredClientFileQueueRead) {
  static const char kReadContents[] = "remote registered file read contents";
  iree::testing::TempFilePath source_path("iree_hal_remote_registered_read");
  IREE_ASSERT_OK(WriteFileContents(
      source_path.path_view(),
      iree_make_const_byte_span(kReadContents, sizeof(kReadContents) - 1)));

  iree_io_file_handle_t* source_handle = nullptr;
  iree_status_t status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_ASYNC,
      source_path.path_view(), iree_allocator_system(), &source_handle);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_file_t* source_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      source_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &source_file));
  iree_io_file_handle_release(source_handle);
  EXPECT_EQ(iree_hal_file_length(source_file), sizeof(kReadContents) - 1);

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(kReadContents) - 1, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, source_file,
      /*source_offset=*/0, buffer, /*target_offset=*/0,
      sizeof(kReadContents) - 1, IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      sizeof(kReadContents) - 1, &mapping));
  ASSERT_EQ(
      memcmp(mapping.contents.data, kReadContents, sizeof(kReadContents) - 1),
      0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(source_file);
  source_path.Remove();
}

TEST_F(RemoteBufferTest, ServerFileQueueReadOpenFailureSignalsSemaphore) {
  iree_hal_file_t* source_file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_device_open_file(
      client_device_, IREE_SV("server://missing"), IREE_HAL_MEMORY_ACCESS_READ,
      iree_allocator_system(), &source_file));

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, source_file,
      /*source_offset=*/0, buffer, /*target_offset=*/0, 16,
      IREE_HAL_READ_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(source_file);
}

TEST_F(RemoteBufferTest, ClientFileQueueReadLargeHostAllocation) {
  constexpr iree_device_size_t kLength = 3 * 64 * 1024 + 17;

  std::vector<uint8_t> source_contents(kLength);
  std::vector<uint8_t> target_contents(kLength, 0);
  for (iree_host_size_t i = 0; i < source_contents.size(); ++i) {
    source_contents[i] = (uint8_t)((i * 31 + 7) & 0xFF);
  }

  iree_io_file_handle_t* source_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(source_contents.data(), source_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &source_handle));
  iree_hal_file_t* source_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      source_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &source_file));
  iree_io_file_handle_release(source_handle);

  iree_io_file_handle_t* target_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_contents.data(), target_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &target_handle));
  iree_hal_file_t* target_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, target_handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file));
  iree_io_file_handle_release(target_handle);

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, kLength, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper read_signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), read_signal.list, source_file,
      /*source_offset=*/0, buffer, /*target_offset=*/0, kLength,
      IREE_HAL_READ_FLAG_NONE));

  SemaphoreListHelper write_wait(semaphore, 1);
  SemaphoreListHelper write_signal(semaphore, 2);
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, write_wait.list,
      write_signal.list, buffer, /*source_offset=*/0, target_file,
      /*target_offset=*/0, kLength, IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 2, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(target_contents, source_contents);

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
  iree_hal_file_release(source_file);
}

TEST_F(RemoteBufferTest, ServerFileQueueWrite) {
  iree_hal_file_t* target_file = nullptr;
  IREE_ASSERT_OK(iree_hal_remote_client_device_open_file(
      client_device_, IREE_SV("server://write"),
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      iree_allocator_system(), &target_file));

  static const char kWriteContents[] = "remote server file write contents";
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(kWriteContents) - 1, &buffer));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, sizeof(kWriteContents) - 1,
                                           &mapping));
  memcpy(mapping.contents.data, kWriteContents, sizeof(kWriteContents) - 1);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, buffer, /*source_offset=*/0,
      target_file, /*target_offset=*/0, sizeof(kWriteContents) - 1,
      IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      server_write_file_.path_view(), iree_allocator_system(), &contents));
  ASSERT_GE(contents->const_buffer.data_length, sizeof(kWriteContents) - 1);
  ASSERT_EQ(memcmp(contents->const_buffer.data, kWriteContents,
                   sizeof(kWriteContents) - 1),
            0);
  iree_io_file_contents_free(contents);

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
}

TEST_F(RemoteBufferTest, RegisteredClientFileQueueWrite) {
  static const char kWriteContents[] = "remote registered file write contents";
  iree::testing::TempFilePath target_path("iree_hal_remote_registered_write");
  std::string initial_contents(sizeof(kWriteContents) - 1, '\0');
  IREE_ASSERT_OK(
      WriteFileContents(target_path.path_view(),
                        iree_make_const_byte_span(initial_contents.data(),
                                                  initial_contents.size())));

  iree_io_file_handle_t* target_handle = nullptr;
  iree_status_t status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
      target_path.path_view(), iree_allocator_system(), &target_handle);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_file_t* target_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, target_handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file));
  iree_io_file_handle_release(target_handle);
  EXPECT_EQ(iree_hal_file_length(target_file), sizeof(kWriteContents) - 1);

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(kWriteContents) - 1, &buffer));
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, sizeof(kWriteContents) - 1,
                                           &mapping));
  memcpy(mapping.contents.data, kWriteContents, sizeof(kWriteContents) - 1);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, buffer, /*source_offset=*/0,
      target_file, /*target_offset=*/0, sizeof(kWriteContents) - 1,
      IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      target_path.path_view(), iree_allocator_system(), &contents));
  ASSERT_GE(contents->const_buffer.data_length, sizeof(kWriteContents) - 1);
  ASSERT_EQ(memcmp(contents->const_buffer.data, kWriteContents,
                   sizeof(kWriteContents) - 1),
            0);
  iree_io_file_contents_free(contents);

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
  target_path.Remove();
}

#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)

TEST_F(RemoteShmFileRegistrationTest,
       RegisteredClientFileQueueReadWriteOverUnixShm) {
  static const char kReadContents[] =
      "remote unix shm registered file read contents";
  iree::testing::TempFilePath source_path(
      "iree_hal_remote_shm_registered_read");
  IREE_ASSERT_OK(WriteFileContents(
      source_path.path_view(),
      iree_make_const_byte_span(kReadContents, sizeof(kReadContents) - 1)));

  iree_io_file_handle_t* source_handle = nullptr;
  iree_status_t status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS |
          IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_ASYNC,
      source_path.path_view(), iree_allocator_system(), &source_handle);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_file_t* source_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      source_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &source_file));
  iree_io_file_handle_release(source_handle);
  EXPECT_EQ(iree_hal_file_length(source_file), sizeof(kReadContents) - 1);

  iree_hal_buffer_t* read_buffer = nullptr;
  AllocateMappableBuffer(sizeof(kReadContents) - 1, &read_buffer);

  iree_hal_semaphore_t* read_semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &read_semaphore));
  SemaphoreListHelper read_signal(read_semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), read_signal.list, source_file,
      /*source_offset=*/0, read_buffer, /*target_offset=*/0,
      sizeof(kReadContents) - 1, IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      read_semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_mapping_t read_mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      read_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      sizeof(kReadContents) - 1, &read_mapping));
  ASSERT_EQ(memcmp(read_mapping.contents.data, kReadContents,
                   sizeof(kReadContents) - 1),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&read_mapping));

  iree_hal_semaphore_release(read_semaphore);
  iree_hal_buffer_release(read_buffer);
  iree_hal_file_release(source_file);
  source_path.Remove();

  static const char kWriteContents[] =
      "remote unix shm registered file write contents";
  iree::testing::TempFilePath target_path(
      "iree_hal_remote_shm_registered_write");
  std::string initial_contents(sizeof(kWriteContents) - 1, '\0');
  IREE_ASSERT_OK(
      WriteFileContents(target_path.path_view(),
                        iree_make_const_byte_span(initial_contents.data(),
                                                  initial_contents.size())));

  iree_io_file_handle_t* target_handle = nullptr;
  status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
      target_path.path_view(), iree_allocator_system(), &target_handle);
  if (iree_status_is_unavailable(status) ||
      iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Async platform file handles unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_hal_file_t* target_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, target_handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file));
  iree_io_file_handle_release(target_handle);
  EXPECT_EQ(iree_hal_file_length(target_file), sizeof(kWriteContents) - 1);

  iree_hal_buffer_t* write_buffer = nullptr;
  AllocateMappableBuffer(sizeof(kWriteContents) - 1, &write_buffer);
  iree_hal_buffer_mapping_t write_mapping;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_range(write_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
                                sizeof(kWriteContents) - 1, &write_mapping));
  memcpy(write_mapping.contents.data, kWriteContents,
         sizeof(kWriteContents) - 1);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&write_mapping));

  iree_hal_semaphore_t* write_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &write_semaphore));
  SemaphoreListHelper write_signal(write_semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), write_signal.list, write_buffer,
      /*source_offset=*/0, target_file, /*target_offset=*/0,
      sizeof(kWriteContents) - 1, IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      write_semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_io_file_contents_t* contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      target_path.path_view(), iree_allocator_system(), &contents));
  ASSERT_GE(contents->const_buffer.data_length, sizeof(kWriteContents) - 1);
  ASSERT_EQ(memcmp(contents->const_buffer.data, kWriteContents,
                   sizeof(kWriteContents) - 1),
            0);
  iree_io_file_contents_free(contents);

  iree_hal_semaphore_release(write_semaphore);
  iree_hal_buffer_release(write_buffer);
  iree_hal_file_release(target_file);
  target_path.Remove();
}

#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS

TEST_F(RemoteBufferTest, ClientFileQueueWriteLargeHostAllocation) {
  constexpr iree_device_size_t kLength = 3 * 64 * 1024 + 17;

  constexpr uint8_t kPattern = 0xA5;
  std::vector<uint8_t> expected(kLength, kPattern);
  std::vector<uint8_t> target_contents(kLength, 0);

  iree_io_file_handle_t* target_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_contents.data(), target_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &target_handle));

  iree_hal_file_t* target_file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, target_handle,
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &target_file));
  iree_io_file_handle_release(target_handle);

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, kLength, &buffer));

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper fill_signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), fill_signal.list, buffer,
      /*target_offset=*/0, kLength, &kPattern, sizeof(kPattern),
      IREE_HAL_FILL_FLAG_NONE));

  SemaphoreListHelper write_wait(semaphore, 1);
  SemaphoreListHelper write_signal(semaphore, 2);
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, write_wait.list,
      write_signal.list, buffer, /*source_offset=*/0, target_file,
      /*target_offset=*/0, kLength, IREE_HAL_WRITE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 2, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(target_contents, expected);

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
}

//===----------------------------------------------------------------------===//
// Queue alloca and dealloca tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteBufferTest, QueueAllocaAndVerify) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list,
      /*pool=*/NULL, params,
      /*allocation_size=*/256, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 256);

  // Wait for alloca to complete (ADVANCE resolves the provisional ID).
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the buffer is usable: map and access.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0, 256,
                                           &mapping));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, QueueAllocaFillChained) {
  iree_hal_semaphore_t* sem_a = nullptr;
  iree_hal_semaphore_t* sem_b = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_b));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  // Alloca → fill chain using frontier ordering.
  iree_hal_buffer_t* buffer = nullptr;
  SemaphoreListHelper alloca_signal(sem_a, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), alloca_signal.list,
      /*pool=*/NULL, params, /*allocation_size=*/1024,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));

  // Fill the alloca'd buffer, waiting on the alloca semaphore.
  uint32_t fill_pattern = 0xCAFEBABE;
  SemaphoreListHelper fill_wait(sem_a, 1);
  SemaphoreListHelper fill_signal(sem_b, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_wait.list,
      fill_signal.list, buffer, 0, 1024, &fill_pattern, sizeof(fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));

  // Wait for fill to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem_b, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back and verify fill pattern.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0, 1024,
                                           &mapping));
  const uint32_t* data = (const uint32_t*)mapping.contents.data;
  for (iree_host_size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(data[i], 0xCAFEBABE) << "Fill pattern mismatch at uint32 " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(sem_b);
  iree_hal_semaphore_release(sem_a);
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, QueueDeallocaOrdering) {
  iree_hal_semaphore_t* sem_a = nullptr;
  iree_hal_semaphore_t* sem_b = nullptr;
  iree_hal_semaphore_t* sem_c = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_b));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_c));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  // Alloca → fill → dealloca chain.
  iree_hal_buffer_t* buffer = nullptr;
  SemaphoreListHelper alloca_signal(sem_a, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), alloca_signal.list,
      /*pool=*/NULL, params, /*allocation_size=*/512, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer));

  uint8_t fill_pattern = 0x55;
  SemaphoreListHelper fill_wait(sem_a, 1);
  SemaphoreListHelper fill_signal(sem_b, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_wait.list,
      fill_signal.list, buffer, 0, 512, &fill_pattern, sizeof(fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));

  SemaphoreListHelper dealloca_wait(sem_b, 1);
  SemaphoreListHelper dealloca_signal(sem_c, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, dealloca_wait.list,
      dealloca_signal.list, buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  // Wait for the full chain to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem_c, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(sem_c);
  iree_hal_semaphore_release(sem_b);
  iree_hal_semaphore_release(sem_a);
  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Executable upload and dispatch tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteBufferTest, ExecutableUploadRejectsMalformedData) {
  const iree_hal_executable_target_t* target =
      FirstExecutableTarget(client_device_);
  ASSERT_NE(target, nullptr);

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  uint8_t fake_binary[] = {0xDE, 0xAD, 0xBE, 0xEF};
  load_params.executable_data =
      iree_make_const_byte_span(fake_binary, sizeof(fake_binary));

  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_device_load_executable(
                            client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, target,
                            &load_params, &executable));
}

// Looks up a compiled binary in the embedded VMVX testdata TOC by filename.
static iree_const_byte_span_t LookupTestdata(const char* filename) {
  const iree_file_toc_t* toc = iree_remote_testdata_vmvx_create();
  for (size_t i = 0; i < iree_remote_testdata_vmvx_size(); ++i) {
    if (strcmp(toc[i].name, filename) == 0) {
      return iree_make_const_byte_span(toc[i].data, toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

TEST_F(RemoteBufferTest, QueueDispatchAbsF32) {
  iree_const_byte_span_t binary =
      LookupTestdata("command_buffer_dispatch_test.bin");
  ASSERT_GT(binary.data_length, 0u);

  const iree_hal_executable_target_t* target = FindExecutableTarget(
      client_device_, IREE_SV("ireevm"), IREE_SV("bytecode"));
  if (!target) {
    GTEST_SKIP() << "VMVX executable loading is not enabled";
  }
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = binary;

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_device_load_executable(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, target, &load_params,
      &executable));

  // Allocate input and output buffers (2 floats each).
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                        IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, 2 * sizeof(float), &input_buffer));

  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, 2 * sizeof(float), &output_buffer));

  // Write input data: [-2.5, -2.5].
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      input_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0, 2 * sizeof(float), &mapping));
  float input_data[] = {-2.5f, -2.5f};
  memcpy(mapping.contents.data, input_data, sizeof(input_data));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Dispatch abs(input) → output.
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  iree_hal_buffer_ref_t binding_refs[2];
  memset(binding_refs, 0, sizeof(binding_refs));
  binding_refs[0].buffer = input_buffer;
  binding_refs[0].offset = 0;
  binding_refs[0].length = 2 * sizeof(float);
  binding_refs[1].buffer = output_buffer;
  binding_refs[1].offset = 0;
  binding_refs[1].length = 2 * sizeof(float);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/2,
      /*.values=*/binding_refs,
  };

  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, executable,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  // Wait for dispatch to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back output and verify abs values: [2.5, 2.5].
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      output_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      0, 2 * sizeof(float), &mapping));
  const float* output_data = (const float*)mapping.contents.data;
  EXPECT_EQ(output_data[0], 2.5f);
  EXPECT_EQ(output_data[1], 2.5f);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
  iree_hal_executable_release(executable);
}

//===----------------------------------------------------------------------===//
// Command buffer tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteBufferTest, CommandBufferUpdateOneShotAndReusable) {
  constexpr iree_host_size_t kUpdateLength = 1024;
  std::vector<uint8_t> source_data(kUpdateLength);
  for (iree_host_size_t i = 0; i < source_data.size(); ++i) {
    source_data[i] = static_cast<uint8_t>(i * 31u + 7u);
  }
  std::string maximum_label(UINT16_MAX, 'L');

  const iree_hal_command_buffer_mode_t modes[] = {
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
  for (iree_hal_command_buffer_mode_t mode : modes) {
    SCOPED_TRACE(mode);
    iree_hal_buffer_t* target_buffer = nullptr;
    AllocateMappableBuffer(kUpdateLength, &target_buffer);

    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        client_device_, mode, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin_debug_group(
        command_buffer,
        iree_make_string_view(maximum_label.data(), maximum_label.size()),
        iree_hal_label_color_unspecified(), /*location=*/nullptr));
    IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
        command_buffer, source_data.data(), /*source_offset=*/0,
        iree_hal_make_buffer_ref(target_buffer, /*offset=*/0, kUpdateLength),
        IREE_HAL_UPDATE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
    SemaphoreListHelper signal(semaphore, 1);
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal.list, command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_range(target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                  IREE_HAL_MEMORY_ACCESS_READ,
                                  /*byte_offset=*/0, kUpdateLength, &mapping));
    EXPECT_EQ(memcmp(mapping.contents.data, source_data.data(), kUpdateLength),
              0);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

    iree_hal_semaphore_release(semaphore);
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(target_buffer);
  }
}

TEST_F(RemoteBufferTest, CommandBufferBindingTableSupportsWideSlots) {
  constexpr iree_host_size_t kBindingCount = 33;
  constexpr uint32_t kTargetSlot = 32;
  constexpr iree_device_size_t kBufferLength = 256;
  constexpr uint32_t kFillPattern = 0xA5B6C7D8u;

  const iree_hal_command_buffer_mode_t modes[] = {
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
  for (iree_hal_command_buffer_mode_t mode : modes) {
    SCOPED_TRACE(mode);
    iree_hal_buffer_t* target_buffer = nullptr;
    AllocateMappableBuffer(kBufferLength, &target_buffer);

    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        client_device_, mode, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        IREE_HAL_QUEUE_AFFINITY_ANY, kBindingCount, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer,
        iree_hal_make_indirect_buffer_ref(kTargetSlot, /*offset=*/0,
                                          kBufferLength),
        &kFillPattern, sizeof(kFillPattern), IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    std::vector<iree_hal_buffer_binding_t> bindings(kBindingCount);
    bindings[kTargetSlot] = {
        /*.buffer=*/target_buffer,
        /*.offset=*/0,
        /*.length=*/kBufferLength,
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*.count=*/bindings.size(),
        /*.bindings=*/bindings.data(),
    };

    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
    SemaphoreListHelper signal(semaphore, 1);
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal.list, command_buffer,
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_range(target_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                  IREE_HAL_MEMORY_ACCESS_READ,
                                  /*byte_offset=*/0, kBufferLength, &mapping));
    const uint32_t* values = (const uint32_t*)mapping.contents.data;
    for (iree_host_size_t i = 0; i < kBufferLength / sizeof(*values); ++i) {
      EXPECT_EQ(values[i], kFillPattern) << "element " << i;
    }
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

    iree_hal_semaphore_release(semaphore);
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(target_buffer);
  }
}

TEST_F(RemoteBufferTest, CommandBufferDispatchConstantsUse32BitLength) {
  iree_const_byte_span_t binary =
      LookupTestdata("command_buffer_dispatch_test.bin");
  ASSERT_GT(binary.data_length, 0u);
  const iree_hal_executable_target_t* target = FindExecutableTarget(
      client_device_, IREE_SV("ireevm"), IREE_SV("bytecode"));
  if (!target) {
    GTEST_SKIP() << "VMVX executable loading is not enabled";
  }

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = binary;
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_device_load_executable(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, target, &load_params,
      &executable));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      client_device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  std::vector<uint32_t> constants(16383);
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(constants.data(),
                                constants.size() * sizeof(constants[0])),
      iree_hal_buffer_ref_list_empty(), IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_remote_command_view_t command;
  iree_byte_span_t stream =
      iree_hal_remote_client_command_buffer_test_stream(command_buffer);
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data, stream.data_length), &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_DISPATCH);
  EXPECT_EQ(command.header.length, 65632u);

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_executable_release(executable);
}

TEST_F(RemoteBufferTest, MalformedLaterCommandNeverSubmitsPartialRecording) {
  constexpr iree_host_size_t kBufferLength = 256;
  iree_hal_buffer_t* target_buffer = nullptr;
  AllocateMappableBuffer(kBufferLength, &target_buffer);
  std::vector<uint8_t> initial_data(kBufferLength, 0);
  QueueUpdateAndWait(initial_data.data(), initial_data.size(), target_buffer,
                     /*target_offset=*/0);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      client_device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint8_t pattern = 0xA5;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(target_buffer, /*offset=*/0, kBufferLength),
      &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_byte_span_t stream =
      iree_hal_remote_client_command_buffer_test_stream(command_buffer);
  iree_hal_remote_command_view_t first_command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data, stream.data_length),
      &first_command));
  ASSERT_LT(first_command.bytes.data_length, stream.data_length);
  auto* malformed_header = reinterpret_cast<iree_hal_remote_cmd_header_t*>(
      stream.data + first_command.bytes.data_length);
  malformed_header->reserved = 1;

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      target_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, kBufferLength, &mapping));
  EXPECT_EQ(memcmp(mapping.contents.data, initial_data.data(), kBufferLength),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(target_buffer);
}

TEST_F(RemoteBufferTest, OneShotCommandBufferDispatch) {
  iree_const_byte_span_t binary =
      LookupTestdata("command_buffer_dispatch_test.bin");
  ASSERT_GT(binary.data_length, 0u);

  const iree_hal_executable_target_t* target = FindExecutableTarget(
      client_device_, IREE_SV("ireevm"), IREE_SV("bytecode"));
  if (!target) {
    GTEST_SKIP() << "VMVX executable loading is not enabled";
  }
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = binary;

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(iree_hal_device_load_executable(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, target, &load_params,
      &executable));

  // Allocate input and output buffers.
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                        IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, 2 * sizeof(float), &input_buffer));
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, 2 * sizeof(float), &output_buffer));

  // Write input data: [-2.5, -2.5].
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      input_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0, 2 * sizeof(float), &mapping));
  float input_data[] = {-2.5f, -2.5f};
  memcpy(mapping.contents.data, input_data, sizeof(input_data));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Record a one-shot command buffer: dispatch abs(input) → output.
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      client_device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2];
  memset(binding_refs, 0, sizeof(binding_refs));
  binding_refs[0].buffer = input_buffer;
  binding_refs[0].offset = 0;
  binding_refs[0].length = 2 * sizeof(float);
  binding_refs[1].buffer = output_buffer;
  binding_refs[1].offset = 0;
  binding_refs[1].length = 2 * sizeof(float);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/2,
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // Submit via queue_execute.
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  // Wait for completion.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back output and verify abs values: [2.5, 2.5].
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      output_buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      0, 2 * sizeof(float), &mapping));
  const float* output_data = (const float*)mapping.contents.data;
  EXPECT_EQ(output_data[0], 2.5f);
  EXPECT_EQ(output_data[1], 2.5f);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
  iree_hal_executable_release(executable);
}

TEST_F(RemoteBufferTest, OneShotCommandBufferFillAndCopy) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                        IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* source_buffer = nullptr;
  iree_hal_buffer_t* dest_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                                    256, &source_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator, buffer_params,
                                                    256, &dest_buffer));

  // Record: fill source with 0xBB, barrier, copy source → dest.
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      client_device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  uint8_t pattern = 0xBB;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, iree_hal_make_buffer_ref(source_buffer, 0, 256), &pattern,
      sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, iree_hal_make_buffer_ref(source_buffer, 0, 256),
      iree_hal_make_buffer_ref(dest_buffer, 0, 256), IREE_HAL_COPY_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // The command buffer owns direct resources it recorded. The client can drop
  // its source reference before submitting without racing the server-side
  // replay of the inlined command stream.
  iree_hal_buffer_release(source_buffer);
  source_buffer = nullptr;

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
  SemaphoreListHelper signal(semaphore, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  iree_hal_command_buffer_release(command_buffer);
  command_buffer = nullptr;

  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify dest buffer has the fill pattern.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_range(dest_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_READ, 0, 256, &mapping));
  for (iree_host_size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(mapping.contents.data[i], 0xBB) << "Mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(dest_buffer);
}

}  // namespace
