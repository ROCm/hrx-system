// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/loopback/factory.h"

#include <functional>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/connection.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

enum PollSide {
  kNotPolling = 0,
  kClientPolling = 1,
  kServerPolling = 2,
};

struct ConnectState {
  // Stable caller-owned attempt, kept through callback and initiation return.
  iree_net_transport_connect_operation_t operation;

  ConnectState() {
    iree_net_transport_connect_operation_initialize(&operation);
  }
  ~ConnectState() {
    iree_net_transport_connect_operation_deinitialize(&operation);
  }
  ConnectState(const ConnectState&) = delete;
  ConnectState& operator=(const ConnectState&) = delete;

  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_connection_t* connection = nullptr;

  static void OnConnect(void* user_data, iree_status_t status,
                        iree_net_connection_t* connection) {
    auto* self = static_cast<ConnectState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->connection = connection;
    iree_status_free(status);
  }

  iree_net_transport_connect_callback_t callback() {
    return {.fn = OnConnect, .user_data = this};
  }
};

struct AcceptState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int* sequence = nullptr;
  int last_sequence = 0;
  std::vector<iree_status_code_t> status_codes;
  std::vector<iree_net_connection_t*> connections;

  static void OnAccept(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* self = static_cast<AcceptState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    if (self->sequence) {
      self->last_sequence = ++*self->sequence;
    }
    self->status_codes.push_back(iree_status_code(status));
    self->connections.push_back(connection);
    iree_status_free(status);
  }

  iree_net_listener_accept_callback_t callback() {
    return {.fn = OnAccept, .user_data = this};
  }
};

struct StopState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int* sequence = nullptr;
  int completed_sequence = 0;
  bool completed = false;

  static void OnStopped(void* user_data) {
    auto* self = static_cast<StopState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    if (self->sequence) {
      self->completed_sequence = ++*self->sequence;
    }
    self->completed = true;
  }

  iree_net_listener_stopped_callback_t callback() {
    return {.fn = OnStopped, .user_data = this};
  }
};

struct DeactivateState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  bool completed = false;

  static void OnDeactivated(void* user_data) {
    auto* self = static_cast<DeactivateState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    self->completed = true;
  }

  iree_net_connection_deactivate_callback_t callback() {
    return {.fn = OnDeactivated, .user_data = this};
  }
};

struct ControlledAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  int allocations_before_failure = -1;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ControlledAllocator*>(self);
    const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (is_allocation && allocator->allocations_before_failure == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "intentional allocation failure");
    }
    if (is_allocation && allocator->allocations_before_failure > 0) {
      --allocator->allocations_before_failure;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t value() { return {.self = this, .ctl = Control}; }
};

class LoopbackFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &client_proactor_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &server_proactor_));
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        /*options=*/nullptr, iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    StopAndFreeListener();
    for (iree_net_connection_t*& connection : client_connections_) {
      DeactivateAndRelease(connection, client_proactor_, kClientPolling);
    }
    for (iree_net_connection_t*& connection : server_connections_) {
      DeactivateAndRelease(connection, server_proactor_, kServerPolling);
    }
    iree_net_transport_factory_release(factory_);
    iree_async_proactor_release(client_proactor_);
    iree_async_proactor_release(server_proactor_);
  }

  void Poll(iree_async_proactor_t* proactor, PollSide side) {
    current_poll_side_ = side;
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                            /*out_completed_count=*/nullptr));
    current_poll_side_ = kNotPolling;
  }

  void PollUntil(iree_async_proactor_t* proactor, PollSide side,
                 const std::function<bool()>& condition) {
    while (!condition()) {
      Poll(proactor, side);
    }
  }

  void CreateListener(iree_string_view_t name, AcceptState* accept_state) {
    accept_state->current_poll_side = &current_poll_side_;
    accept_state->expected_poll_side = kServerPolling;
    IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
        factory_, name, server_proactor_, /*receive_pool=*/nullptr,
        accept_state->callback(), iree_allocator_system(), &listener_));
  }

  void StopAndFreeListener() {
    if (!listener_) {
      return;
    }
    StopState stop_state;
    stop_state.current_poll_side = &current_poll_side_;
    stop_state.expected_poll_side = kServerPolling;
    IREE_ASSERT_OK(iree_net_listener_stop(listener_, stop_state.callback()));
    PollUntil(server_proactor_, kServerPolling,
              [&] { return stop_state.completed; });
    iree_net_listener_free(listener_);
    listener_ = nullptr;
  }

  void DeactivateAndRelease(iree_net_connection_t*& connection,
                            iree_async_proactor_t* proactor, PollSide side) {
    if (!connection) {
      return;
    }
    DeactivateState deactivate_state;
    deactivate_state.current_poll_side = &current_poll_side_;
    deactivate_state.expected_poll_side = side;
    current_poll_side_ = side;
    iree_net_connection_deactivate(connection, deactivate_state.callback());
    current_poll_side_ = kNotPolling;
    if (!deactivate_state.completed) {
      PollUntil(proactor, side, [&] { return deactivate_state.completed; });
    }
    iree_net_connection_release(connection);
    connection = nullptr;
  }

  int current_poll_side_ = kNotPolling;
  iree_async_proactor_t* client_proactor_ = nullptr;
  iree_async_proactor_t* server_proactor_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_net_listener_t* listener_ = nullptr;
  std::vector<iree_net_connection_t*> client_connections_;
  std::vector<iree_net_connection_t*> server_connections_;
};

TEST_F(LoopbackFactoryTest, ValidatesOptions) {
  iree_net_loopback_factory_options_t options =
      iree_net_loopback_factory_options_default();
  options.max_endpoint_count = 0;
  iree_net_transport_factory_t* invalid_factory = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_loopback_factory_create(&options, iree_allocator_system(),
                                       &invalid_factory));
  EXPECT_EQ(invalid_factory, nullptr);
}

TEST_F(LoopbackFactoryTest,
       ConnectAllocationFailureRetiresClaimWithoutCallback) {
  iree_net_transport_factory_release(factory_);
  ControlledAllocator controlled_allocator;
  IREE_ASSERT_OK(iree_net_loopback_factory_create(
      /*options=*/nullptr, controlled_allocator.value(), &factory_));
  AcceptState accept_state;
  CreateListener(IREE_SV("failure"), &accept_state);

  // Allow the connect dispatch allocation and fail the accept dispatch
  // allocation after the listener has been claimed.
  controlled_allocator.allocations_before_failure = 1;
  ConnectState connect_state;
  connect_state.current_poll_side = &current_poll_side_;
  connect_state.expected_poll_side = kClientPolling;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("failure"), client_proactor_,
                            /*receive_pool=*/nullptr, connect_state.callback(),
                            &connect_state.operation));
  EXPECT_EQ(connect_state.callback_count, 0);

  StopAndFreeListener();
  iree_net_transport_factory_release(factory_);
  factory_ = nullptr;
}

TEST_F(LoopbackFactoryTest, RejectsDuplicateNameAndReusesStoppedName) {
  AcceptState accept_state;
  CreateListener(IREE_SV("unique"), &accept_state);

  iree_net_listener_t* duplicate_listener = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_net_transport_factory_create_listener(
                            factory_, IREE_SV("unique"), server_proactor_,
                            /*receive_pool=*/nullptr, accept_state.callback(),
                            iree_allocator_system(), &duplicate_listener));
  EXPECT_EQ(duplicate_listener, nullptr);

  StopAndFreeListener();
  CreateListener(IREE_SV("unique"), &accept_state);
}

TEST_F(LoopbackFactoryTest, StopWaitsForClaimedAcceptAndRejectsLaterConnects) {
  int sequence = 0;
  AcceptState accept_state;
  accept_state.sequence = &sequence;
  CreateListener(IREE_SV("drain"), &accept_state);

  ConnectState accepted_connect;
  accepted_connect.current_poll_side = &current_poll_side_;
  accepted_connect.expected_poll_side = kClientPolling;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, IREE_SV("drain"), client_proactor_,
      /*receive_pool=*/nullptr, accepted_connect.callback(),
      &accepted_connect.operation));

  StopState stop_state;
  stop_state.current_poll_side = &current_poll_side_;
  stop_state.expected_poll_side = kServerPolling;
  stop_state.sequence = &sequence;
  IREE_ASSERT_OK(iree_net_listener_stop(listener_, stop_state.callback()));
  EXPECT_FALSE(stop_state.completed);

  ConnectState rejected_connect;
  rejected_connect.current_poll_side = &current_poll_side_;
  rejected_connect.expected_poll_side = kClientPolling;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, IREE_SV("drain"), client_proactor_,
      /*receive_pool=*/nullptr, rejected_connect.callback(),
      &rejected_connect.operation));

  PollUntil(client_proactor_, kClientPolling, [&] {
    return accepted_connect.callback_count == 1 &&
           rejected_connect.callback_count == 1;
  });
  ASSERT_EQ(accepted_connect.status_code, IREE_STATUS_OK);
  ASSERT_NE(accepted_connect.connection, nullptr);
  EXPECT_EQ(rejected_connect.status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(rejected_connect.connection, nullptr);
  EXPECT_FALSE(stop_state.completed);

  PollUntil(server_proactor_, kServerPolling, [&] {
    return accept_state.connections.size() == 1 && stop_state.completed;
  });
  ASSERT_EQ(accept_state.status_codes[0], IREE_STATUS_OK);
  ASSERT_NE(accept_state.connections[0], nullptr);
  EXPECT_LT(accept_state.last_sequence, stop_state.completed_sequence);

  client_connections_.push_back(accepted_connect.connection);
  server_connections_.push_back(accept_state.connections[0]);
  iree_net_listener_free(listener_);
  listener_ = nullptr;
}

TEST_F(LoopbackFactoryTest, CancelledConnectRetiresStoppingListenerClaim) {
  AcceptState accepted;
  CreateListener(IREE_SV("cancel"), &accepted);
  ConnectState connected;
  connected.current_poll_side = &current_poll_side_;
  connected.expected_poll_side = kClientPolling;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, IREE_SV("cancel"), client_proactor_, nullptr,
      connected.callback(), &connected.operation));
  StopState stopped;
  stopped.current_poll_side = &current_poll_side_;
  stopped.expected_poll_side = kServerPolling;
  IREE_ASSERT_OK(iree_net_listener_stop(listener_, stopped.callback()));
  iree_net_transport_connect_operation_cancel(&connected.operation);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return connected.callback_count == 1; });
  EXPECT_EQ(connected.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(connected.connection, nullptr);
  EXPECT_FALSE(stopped.completed);
  PollUntil(server_proactor_, kServerPolling,
            [&] { return stopped.completed; });
  EXPECT_TRUE(accepted.connections.empty());
  iree_net_listener_free(listener_);
  listener_ = nullptr;
}

}  // namespace
}  // namespace iree
