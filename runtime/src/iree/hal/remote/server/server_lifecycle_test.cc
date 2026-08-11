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
#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::TestBufferPool;

struct FailNthAllocator {
  // One-based allocation ordinal to fail, or zero to allow all allocations.
  iree_host_size_t fail_allocation = 0;
  // Number of allocation commands observed.
  iree_host_size_t allocation_count = 0;
  // Number of successful allocations not yet freed.
  iree_host_size_t live_allocation_count = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailNthAllocator*>(self);
    iree_allocator_t system_allocator = iree_allocator_system();
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC: {
        ++allocator->allocation_count;
        if (allocator->allocation_count == allocator->fail_allocation) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        iree_status_t status = system_allocator.ctl(system_allocator.self,
                                                    command, params, inout_ptr);
        if (iree_status_is_ok(status) && *inout_ptr) {
          ++allocator->live_allocation_count;
        }
        return status;
      }
      case IREE_ALLOCATOR_COMMAND_REALLOC: {
        ++allocator->allocation_count;
        if (allocator->allocation_count == allocator->fail_allocation) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        void* old_ptr = *inout_ptr;
        iree_status_t status = system_allocator.ctl(system_allocator.self,
                                                    command, params, inout_ptr);
        if (iree_status_is_ok(status)) {
          if (!old_ptr && *inout_ptr) {
            ++allocator->live_allocation_count;
          } else if (old_ptr && !*inout_ptr) {
            --allocator->live_allocation_count;
          }
        }
        return status;
      }
      case IREE_ALLOCATOR_COMMAND_FREE:
        if (*inout_ptr) --allocator->live_allocation_count;
        return system_allocator.ctl(system_allocator.self, command, params,
                                    inout_ptr);
      default:
        return system_allocator.ctl(system_allocator.self, command, params,
                                    inout_ptr);
    }
  }
};

struct MissingDeviceSpecDevice {
  // HAL resource base; must remain the first field.
  iree_hal_resource_t resource;
  // Destruction count owned by the test.
  int* destruction_count;
};

static void DestroyMissingDeviceSpecDevice(iree_hal_device_t* base_device) {
  auto* device = reinterpret_cast<MissingDeviceSpecDevice*>(base_device);
  ++*device->destruction_count;
}

static const iree_hal_device_spec_t* QueryMissingDeviceSpec(
    iree_hal_device_t* base_device) {
  (void)base_device;
  return nullptr;
}

static const iree_hal_device_vtable_t kMissingDeviceSpecDeviceVTable = {
    /*.destroy=*/DestroyMissingDeviceSpecDevice,
    /*.id=*/nullptr,
    /*.host_allocator=*/nullptr,
    /*.device_allocator=*/nullptr,
    /*.replace_device_allocator=*/nullptr,
    /*.replace_channel_provider=*/nullptr,
    /*.trim=*/nullptr,
    /*.device_spec=*/QueryMissingDeviceSpec,
};

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
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
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
    mock_options.executable_loading_enabled = true;
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &mock_options, iree_allocator_system(), &mock_device_));
  }

  void TearDown() override {
    if (client_device_) {
      DeactivateDeviceAndWait(client_device_);
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }
    if (server_ && server_->state == IREE_HAL_REMOTE_SERVER_STATE_RUNNING) {
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

  iree_status_t CreateServer(iree_hal_device_t* device,
                             iree_allocator_t host_allocator,
                             iree_hal_remote_server_t** out_server) {
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

    iree_hal_device_t* devices[] = {device};
    return iree_hal_remote_server_create(
        &server_options, devices, IREE_ARRAYSIZE(devices), proactor_,
        server_tracker_, iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
        host_allocator, out_server);
  }

  void CreateAndStartServer() {
    IREE_ASSERT_OK(
        CreateServer(mock_device_, iree_allocator_system(), &server_));
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
        << "Client connect callback did not fire";
    return client_connect_status_;
  }

  iree_hal_remote_server_session_t* ActiveSessionSlot() {
    for (uint32_t i = 0; i < server_->options.max_connections; ++i) {
      if (server_->sessions[i].session) return &server_->sessions[i];
    }
    return nullptr;
  }

  iree_status_t CreateBulkChannel(
      iree_hal_remote_server_session_t* session_slot, MockEndpoint* endpoint,
      iree_net_bulk_channel_t** out_bulk_channel) {
    *out_bulk_channel = nullptr;
    TestBufferPool buffer_pool;
    iree_status_t status =
        buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024);
    if (iree_status_is_ok(status)) {
      iree_net_bulk_channel_callbacks_t callbacks =
          iree_hal_remote_server_bulk_session_channel_callbacks(session_slot);
      status = iree_net_bulk_channel_create(
          endpoint->as_endpoint(), /*options=*/nullptr, buffer_pool.release(),
          callbacks, iree_allocator_system(), out_bulk_channel);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(*out_bulk_channel);
    }
    return status;
  }

  void RequestServerStop() {
    server_stopped_ = false;
    iree_hal_remote_server_stopped_callback_t callback;
    callback.fn = OnServerStopped;
    callback.user_data = this;
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
  }

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

  void StopServerAndWait() {
    RequestServerStop();
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

TEST_F(ServerLifecycleTest, CreateUnwindsEveryAllocationFailure) {
  CreateLoopbackFactory(IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT);

  FailNthAllocator baseline_allocator;
  iree_hal_remote_server_t* server = nullptr;
  IREE_ASSERT_OK(
      CreateServer(mock_device_, baseline_allocator.allocator(), &server));
  ASSERT_NE(server, nullptr);
  const iree_host_size_t allocation_count = baseline_allocator.allocation_count;
  iree_hal_remote_server_release(server);
  EXPECT_EQ(baseline_allocator.live_allocation_count, 0u);
  ASSERT_GT(allocation_count, 1u);

  for (iree_host_size_t fail_allocation = 1;
       fail_allocation <= allocation_count; ++fail_allocation) {
    FailNthAllocator allocator;
    allocator.fail_allocation = fail_allocation;
    server = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        CreateServer(mock_device_, allocator.allocator(), &server));
    EXPECT_EQ(server, nullptr);
    EXPECT_EQ(allocator.allocation_count, fail_allocation);
    EXPECT_EQ(allocator.live_allocation_count, 0u);
  }
}

TEST_F(ServerLifecycleTest, CreateUnwindsMissingDeviceSpec) {
  CreateLoopbackFactory(IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT);

  int destruction_count = 0;
  MissingDeviceSpecDevice missing_spec_device = {
      /*.resource=*/{},
      /*.destruction_count=*/&destruction_count,
  };
  iree_hal_resource_initialize(&kMissingDeviceSpecDeviceVTable,
                               &missing_spec_device.resource);

  iree_hal_remote_server_t* server = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      CreateServer(reinterpret_cast<iree_hal_device_t*>(&missing_spec_device),
                   iree_allocator_system(), &server));
  EXPECT_EQ(server, nullptr);
  EXPECT_EQ(destruction_count, 0);

  iree_hal_device_release(
      reinterpret_cast<iree_hal_device_t*>(&missing_spec_device));
  EXPECT_EQ(destruction_count, 1);
}

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
  EXPECT_EQ(session.resource_table.slots, nullptr);
  EXPECT_EQ(session.observed_submission_window.storage, nullptr);
  EXPECT_EQ(session.completed_signal_window.storage, nullptr);
}

TEST_F(ServerLifecycleTest, StopWaitsForPendingBulkSendBeforeFreeingSlot) {
  CreateLoopbackFactory(IREE_HAL_REMOTE_REQUIRED_ENDPOINT_COUNT);
  CreateAndStartServer();
  CreateClientDevice();

  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  iree_hal_remote_server_session_t* session_slot = nullptr;
  ASSERT_TRUE(PollUntil([&]() {
    session_slot = ActiveSessionSlot();
    return session_slot &&
           iree_hal_remote_server_bulk_session_channel(session_slot);
  }));
  ASSERT_EQ(server_->active_session_count, 1u);

  iree_net_bulk_channel_t* live_bulk_channel =
      iree_hal_remote_server_bulk_session_take_channel(session_slot);
  ASSERT_NE(live_bulk_channel, nullptr);
  iree_net_bulk_channel_detach(live_bulk_channel);
  iree_net_bulk_channel_release(live_bulk_channel);

  std::unique_ptr<MockCarrier> carrier = MockCarrier::Create();
  MockEndpoint endpoint;
  endpoint.carrier = carrier.get();
  iree_net_bulk_channel_t* bulk_channel = nullptr;
  IREE_ASSERT_OK(CreateBulkChannel(session_slot, &endpoint, &bulk_channel));
  IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_attach_channel(
      session_slot, bulk_channel));

  IREE_ASSERT_OK(iree_net_bulk_channel_send_start(
      bulk_channel, /*transfer_id=*/1, /*total_size=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, /*operation_user_data=*/0));
  ASSERT_EQ(carrier->sends.size(), 1u);
  EXPECT_TRUE(iree_net_bulk_channel_has_pending_sends(bulk_channel));

  RequestServerStop();
  ASSERT_TRUE(PollUntil([&]() {
    return server_->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
           server_->listener == nullptr &&
           server_->active_session_count == 1u &&
           session_slot->session == nullptr &&
           iree_any_bit_set(
               session_slot->flags,
               IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING);
  }));
  EXPECT_FALSE(server_stopped_);
  EXPECT_TRUE(iree_net_bulk_channel_has_pending_sends(bulk_channel));
  EXPECT_NE(session_slot->bulk_session, nullptr);

  carrier->CompleteSend(/*send_index=*/0, iree_ok_status());
  ASSERT_TRUE(PollUntil([&]() { return server_stopped_; }));
  EXPECT_EQ(server_->state, IREE_HAL_REMOTE_SERVER_STATE_STOPPED);
  EXPECT_EQ(server_->active_session_count, 0u);
  EXPECT_EQ(session_slot->flags, 0u);
  EXPECT_TRUE(
      iree_hal_remote_server_bulk_session_is_empty(session_slot->bulk_session));

  iree_net_bulk_channel_release(bulk_channel);
}

}  // namespace
