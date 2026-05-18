// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ServerLifecycleTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    iree_async_proactor_options_t proactor_options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        proactor_options, iree_allocator_system(), &proactor_));

    iree_async_slab_t* slab = nullptr;
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
    iree_async_region_t* region = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));
    iree_async_buffer_pool_t* buffer_pool = nullptr;
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region, iree_allocator_system(), &buffer_pool));
    IREE_ASSERT_OK(
        iree_hal_remote_recv_pool_wrap(proactor_, slab, region, buffer_pool,
                                       iree_allocator_system(), &recv_pool_));
    iree_async_region_release(region);
    iree_async_slab_release(slab);

    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = kAxisTableCapacity;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &server_tracker_));

    iree_hal_mock_device_options_t mock_options;
    iree_hal_mock_device_options_initialize(&mock_options);
    mock_options.identifier = IREE_SV("mock");
    mock_options.executable_cache_enabled = true;
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &mock_options, iree_allocator_system(), &mock_device_));
  }

  void TearDown() override {
    DrainProactor();
    iree_hal_device_release(client_device_);
    client_device_ = nullptr;
    DrainProactor();
    if (server_ && server_->state == IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
      StopServerAndWait();
    }
    iree_hal_remote_server_release(server_);
    server_ = nullptr;
    DrainProactor();
    iree_hal_device_release(mock_device_);
    mock_device_ = nullptr;
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;
    iree_async_frontier_tracker_release(server_tracker_);
    server_tracker_ = nullptr;
    iree_hal_remote_recv_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  void CreateLoopbackFactory(uint32_t max_endpoint_count) {
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    factory_options.max_endpoint_count = max_endpoint_count;
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));
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
    server_options.max_connections = 1;

    iree_hal_device_t* devices[] = {mock_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, IREE_ARRAYSIZE(devices), proactor_,
        server_tracker_, iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
        iree_allocator_system(), &server_));
    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address = IREE_SV("test-server");

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, &create_params, recv_pool_,
        iree_allocator_system(), &client_device_));
  }

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
        << "Client connect callback timed out";
    return client_connect_status_;
  }

  void DrainProactor() {
    for (;;) {
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(0), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        break;
      } else if (!iree_status_is_ok(status)) {
        IREE_EXPECT_OK(status);
        break;
      }
      if (completed == 0) break;
    }
  }

  bool PollUntil(std::function<bool()> condition,
                 iree_duration_t budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline = iree_time_now() + budget;
    while (!condition()) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_deadline(deadline), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        return condition();
      } else if (!iree_status_is_ok(status)) {
        IREE_EXPECT_OK(status);
        return false;
      }
    }
    return true;
  }

  void StopServerAndWait() {
    server_stopped_ = false;
    iree_hal_remote_server_stopped_callback_t callback;
    callback.fn = OnServerStopped;
    callback.user_data = this;
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(PollUntil([&]() { return server_stopped_; }))
        << "Server stop timed out";
  }

  static void OnClientConnected(void* user_data, iree_status_t status) {
    auto* self = static_cast<ServerLifecycleTest*>(user_data);
    self->client_connect_fired_ = true;
    self->client_connect_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnServerStopped(void* user_data) {
    auto* self = static_cast<ServerLifecycleTest*>(user_data);
    self->server_stopped_ = true;
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_hal_remote_recv_pool_t* recv_pool_ = nullptr;
  iree_async_frontier_tracker_t* server_tracker_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_hal_device_t* mock_device_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;
  iree_hal_device_t* client_device_ = nullptr;
  bool client_connect_fired_ = false;
  iree_status_code_t client_connect_status_ = IREE_STATUS_OK;
  bool server_stopped_ = false;
};

TEST_F(ServerLifecycleTest, AcceptFailureClearsReservedSlot) {
  CreateLoopbackFactory(IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT - 1u);
  CreateAndStartServer();
  CreateClientDevice();

  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(server_->active_session_count, 0u);
  EXPECT_EQ(server_->next_session_id, 2u);
  ASSERT_EQ(server_->options.max_connections, 1u);
  const iree_hal_remote_server_session_t& session = server_->sessions[0];
  EXPECT_EQ(session.server, nullptr);
  EXPECT_EQ(session.session, nullptr);
  EXPECT_EQ(session.session_id, 0u);
  EXPECT_EQ(session.flags, 0u);
  EXPECT_EQ(session.queue_channel, nullptr);
  EXPECT_TRUE(
      iree_hal_remote_server_bulk_session_is_empty(session.bulk_session));
  EXPECT_EQ(session.resource_table.entries, nullptr);
  EXPECT_EQ(session.observed_submission_window.storage, nullptr);
  EXPECT_EQ(session.completed_signal_window.storage, nullptr);
}

}  // namespace
