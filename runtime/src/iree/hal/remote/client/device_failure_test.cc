// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t PollProactorOnce(iree_async_proactor_t* proactor) {
  return iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                  /*out_completed_count=*/nullptr);
}

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

  iree_notification_t notification_;
  std::atomic<bool> completed_{false};
};

class RemoteClientDeviceFailureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_pool_options_t pool_options =
        iree_async_proactor_pool_options_default();
    memset(&pool_options.runner, 0, sizeof(pool_options.runner));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr, pool_options,
        iree_allocator_system(), &proactor_pool_));
    IREE_ASSERT_OK(iree_hal_remote_recv_pool_create(
        proactor_pool_, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        iree_allocator_system(), &recv_pool_));
    proactor_ = iree_hal_remote_recv_pool_proactor(recv_pool_);

    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = 16;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, iree_allocator_system(), &server_tracker_));

    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));

    CreateAndStartServer();
  }

  void TearDown() override {
    if (client_device_) {
      DeactivateDeviceAndWait();
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }
    if (server_) {
      StopServerAndWait();
      iree_hal_remote_server_release(server_);
      server_ = nullptr;
    }

    iree_hal_device_release(mock_device_);
    iree_net_transport_factory_release(factory_);
    iree_async_frontier_tracker_release(server_tracker_);
    iree_hal_remote_recv_pool_release(recv_pool_);
    iree_async_proactor_pool_release(proactor_pool_);
  }

  bool PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree::Status status(PollProactorOnce(proactor_));
      if (!status.ok()) {
        ADD_FAILURE() << status.ToString();
        return false;
      }
    }
    return true;
  }

  void CreateAndStartServer() {
    iree_hal_mock_device_options_t mock_options;
    iree_hal_mock_device_options_initialize(&mock_options);
    mock_options.identifier = IREE_SV("mock");
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &mock_options, iree_allocator_system(), &mock_device_));

    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topology = {};
    server_topology.axes = server_axes;
    server_topology.current_epochs = server_epochs;
    server_topology.axis_count = IREE_ARRAYSIZE(server_axes);
    server_topology.machine_index = 1;
    server_topology.session_epoch = 1;

    iree_hal_remote_server_options_t server_options;
    iree_hal_remote_server_options_initialize(&server_options);
    server_options.transport_factory = factory_;
    server_options.bind_address = IREE_SV("test-server");
    server_options.local_topology = &server_topology;

    iree_hal_device_t* devices[] = {mock_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, IREE_ARRAYSIZE(devices), proactor_,
        server_tracker_, iree_hal_remote_recv_pool_buffer_pool(recv_pool_),
        iree_allocator_system(), &server_));
    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  void CreateAndConnectClient() {
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

    iree_hal_remote_client_device_connected_callback_t callback = {
        /*.fn=*/OnClientConnected,
        /*.user_data=*/this,
    };
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_connect(client_device_, callback));
    ASSERT_TRUE(PollUntil([&]() { return client_connect_fired_; }));
    ASSERT_EQ(client_connect_status_, IREE_STATUS_OK);
  }

  void DeactivateDeviceAndWait() {
    if (iree_hal_remote_client_device_state(client_device_) ==
        IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED) {
      return;
    }
    bool deactivated = false;
    iree_hal_remote_client_device_deactivated_callback_t callback = {
        /*.fn=*/[](void* user_data) { *static_cast<bool*>(user_data) = true; },
        /*.user_data=*/&deactivated,
    };
    IREE_ASSERT_OK(
        iree_hal_remote_client_device_deactivate(client_device_, callback));
    ASSERT_TRUE(PollUntil([&]() { return deactivated; }));
  }

  void StopServerAndWait() {
    bool stopped = false;
    iree_hal_remote_server_stopped_callback_t callback = {
        /*.fn=*/[](void* user_data) { *static_cast<bool*>(user_data) = true; },
        /*.user_data=*/&stopped,
    };
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(PollUntil([&]() { return stopped; }));
  }

  std::thread StartPendingTrimRpc(CompletionNotification* request_submitted,
                                  iree_status_code_t* out_status_code) {
    return std::thread([this, request_submitted, out_status_code]() {
      struct {
        iree_hal_remote_control_envelope_t envelope;
        iree_hal_remote_device_trim_request_t body;
      } request = {};
      request.envelope.message_type = IREE_HAL_REMOTE_CONTROL_DEVICE_TRIM;

      iree_hal_remote_client_device_control_rpc_after_send_t after_send = {
          /*.fn=*/
          [](void* user_data) -> iree_status_t {
            static_cast<CompletionNotification*>(user_data)->Signal();
            return iree_ok_status();
          },
          /*.user_data=*/request_submitted,
      };
      iree_const_byte_span_t response_payload;
      iree_async_buffer_lease_t response_lease;
      iree_status_t status =
          iree_hal_remote_client_device_control_rpc_with_after_send(
              iree_hal_remote_client_device_cast(client_device_),
              iree_make_const_byte_span(&request, sizeof(request)), after_send,
              &response_payload, &response_lease);
      *out_status_code = iree_status_code(status);
      iree_status_free(status);
      iree_async_buffer_lease_release(&response_lease);
    });
  }

  static void OnClientConnected(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteClientDeviceFailureTest*>(user_data);
    self->client_connect_status_ = iree_status_code(status);
    self->client_connect_fired_ = true;
    self->callback_order_.push_back("connect");
    iree_status_free(status);
    if (self->fail_from_connect_callback_) {
      iree_hal_remote_client_device_fail(
          iree_hal_remote_client_device_cast(self->client_device_),
          iree_make_status(IREE_STATUS_ABORTED,
                           "failure during connect callback"));
    } else if (self->deactivate_from_connect_callback_) {
      iree_hal_remote_client_device_deactivated_callback_t callback = {
          /*.fn=*/
          [](void* user_data) { *static_cast<bool*>(user_data) = true; },
          /*.user_data=*/&self->client_deactivated_,
      };
      IREE_EXPECT_OK(iree_hal_remote_client_device_deactivate(
          self->client_device_, callback));
    }
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteClientDeviceFailureTest*>(user_data);
    self->client_error_status_ = iree_status_code(status);
    ++self->client_error_count_;
    self->callback_order_.push_back("error");
    iree_status_free(status);
  }

  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  iree_async_proactor_t* proactor_ = nullptr;
  iree_hal_remote_recv_pool_t* recv_pool_ = nullptr;
  iree_async_frontier_tracker_t* server_tracker_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_hal_device_t* mock_device_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;
  iree_hal_device_t* client_device_ = nullptr;
  bool client_connect_fired_ = false;
  iree_status_code_t client_connect_status_ = IREE_STATUS_OK;
  bool fail_from_connect_callback_ = false;
  bool deactivate_from_connect_callback_ = false;
  bool client_deactivated_ = false;
  int client_error_count_ = 0;
  iree_status_code_t client_error_status_ = IREE_STATUS_OK;
  std::vector<std::string> callback_order_;
};

TEST_F(RemoteClientDeviceFailureTest,
       FailureDuringConnectCallbackPreservesCallbackOrder) {
  fail_from_connect_callback_ = true;
  CreateAndConnectClient();

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  EXPECT_EQ(client_error_count_, 1);
  EXPECT_EQ(client_error_status_, IREE_STATUS_ABORTED);
  EXPECT_EQ(callback_order_, (std::vector<std::string>{"connect", "error"}));
}

TEST_F(RemoteClientDeviceFailureTest,
       DeactivateDuringConnectCallbackDoesNotReportError) {
  deactivate_from_connect_callback_ = true;
  CreateAndConnectClient();
  ASSERT_TRUE(PollUntil([&]() { return client_deactivated_; }));

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DEACTIVATED);
  EXPECT_EQ(client_error_count_, 0);
  EXPECT_EQ(callback_order_, (std::vector<std::string>{"connect"}));
}

TEST_F(RemoteClientDeviceFailureTest, ServerShutdownPublishesTerminalFailure) {
  CreateAndConnectClient();

  StopServerAndWait();
  iree_hal_remote_server_release(server_);
  server_ = nullptr;

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  EXPECT_EQ(client_error_count_, 1);
  EXPECT_EQ(client_error_status_, IREE_STATUS_UNAVAILABLE);
  iree_status_t status = iree_hal_remote_client_device_check_connected(
      iree_hal_remote_client_device_cast(client_device_));
  EXPECT_TRUE(iree_status_is_unavailable(status));
  iree_status_free(status);
}

TEST_F(RemoteClientDeviceFailureTest,
       TerminalFailureWakesPendingRpcAndPreservesCause) {
  CreateAndConnectClient();

  CompletionNotification request_submitted;
  iree_status_code_t rpc_status_code = IREE_STATUS_OK;
  std::thread rpc_thread =
      StartPendingTrimRpc(&request_submitted, &rpc_status_code);
  ASSERT_TRUE(request_submitted.Await());

  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(client_device_);
  iree_hal_remote_client_device_fail(
      device,
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));
  rpc_thread.join();

  EXPECT_EQ(rpc_status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  EXPECT_EQ(client_error_count_, 1);
  EXPECT_EQ(client_error_status_, IREE_STATUS_ABORTED);

  iree_hal_remote_client_device_fail(
      device,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "later transport failure"));
  EXPECT_EQ(client_error_count_, 1);
  iree_status_t status = iree_hal_remote_client_device_check_connected(device);
  EXPECT_TRUE(iree_status_is_aborted(status));
  iree_status_free(status);
}

TEST_F(RemoteClientDeviceFailureTest, DeactivateWakesPendingRpc) {
  CreateAndConnectClient();

  CompletionNotification request_submitted;
  iree_status_code_t rpc_status_code = IREE_STATUS_OK;
  std::thread rpc_thread =
      StartPendingTrimRpc(&request_submitted, &rpc_status_code);
  ASSERT_TRUE(request_submitted.Await());

  DeactivateDeviceAndWait();
  rpc_thread.join();
  EXPECT_EQ(rpc_status_code, IREE_STATUS_CANCELLED);
}

TEST_F(RemoteClientDeviceFailureTest,
       TerminalFailureFailsSubmittedQueueFrontier) {
  CreateAndConnectClient();

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));

  iree_hal_semaphore_t* signal_semaphores[] = {semaphore};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/IREE_ARRAYSIZE(signal_semaphores),
      /*.semaphores=*/signal_semaphores,
      /*.payload_values=*/signal_values,
  };
  IREE_ASSERT_OK(
      iree_hal_device_queue_barrier(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                    iree_hal_semaphore_list_empty(),
                                    signal_list, IREE_HAL_EXECUTE_FLAG_NONE));

  iree_hal_remote_client_device_fail(
      iree_hal_remote_client_device_cast(client_device_),
      iree_make_status(IREE_STATUS_ABORTED, "injected terminal failure"));
  iree_status_t status = iree_hal_semaphore_wait(
      semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_TRUE(iree_status_is_aborted(status));
  iree_status_free(status);

  iree_hal_semaphore_release(semaphore);
}

}  // namespace
