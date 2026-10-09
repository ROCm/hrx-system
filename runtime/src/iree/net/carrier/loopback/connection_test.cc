// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/loopback/connection.h"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

enum PollSide {
  kNotPolling = 0,
  kClientPolling = 1,
  kServerPolling = 2,
};

struct EndpointReadyState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_net_message_endpoint_t endpoint = {};

  static void OnReady(void* user_data, iree_status_t status,
                      iree_net_message_endpoint_t endpoint) {
    auto* self = static_cast<EndpointReadyState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->endpoint = endpoint;
    iree_status_free(status);
  }

  iree_net_endpoint_ready_callback_t callback() {
    return {.fn = OnReady, .user_data = this};
  }
};

struct MessageState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  std::vector<std::string> messages;
  int error_count = 0;
  iree_status_code_t last_error = IREE_STATUS_OK;

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<MessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    EXPECT_NE(lease, nullptr);
    self->messages.emplace_back(reinterpret_cast<const char*>(message.data),
                                message.data_length);
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<MessageState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->error_count;
    self->last_error = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_message_endpoint_callbacks_t callbacks() {
    return {
        .on_message = OnMessage,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct SendState {
  int* current_poll_side = nullptr;
  PollSide expected_poll_side = kNotPolling;
  int callback_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_host_size_t bytes_transferred = 0;

  static void OnComplete(void* user_data, iree_status_t status,
                         iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SendState*>(user_data);
    EXPECT_EQ(*self->current_poll_side, self->expected_poll_side);
    ++self->callback_count;
    self->status_code = iree_status_code(status);
    self->bytes_transferred = bytes_transferred;
    iree_status_free(status);
  }

  iree_net_send_completion_callback_t callback() {
    return {.fn = OnComplete, .user_data = this};
  }
};

struct PrefixWriterGate {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered = false;
  bool released = false;
  iree_status_code_t result_code = IREE_STATUS_OK;

  static iree_status_t Write(void* user_data, iree_byte_span_t target) {
    auto* self = static_cast<PrefixWriterGate*>(user_data);
    std::unique_lock<std::mutex> lock(self->mutex);
    self->entered = true;
    self->condition.notify_all();
    self->condition.wait(lock, [&] { return self->released; });
    memset(target.data, 0xA5, target.data_length);
    return self->result_code == IREE_STATUS_OK
               ? iree_ok_status()
               : iree_make_status(self->result_code,
                                  "injected prefix writer failure");
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

class LoopbackConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &client_proactor_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &server_proactor_));
  }

  void TearDown() override {
    DeactivateAndRelease(client_connection_, client_proactor_, kClientPolling);
    DeactivateAndRelease(server_connection_, server_proactor_, kServerPolling);
    iree_async_proactor_release(client_proactor_);
    iree_async_proactor_release(server_proactor_);
  }

  void CreatePair(
      uint32_t max_endpoint_count,
      const iree_net_loopback_carrier_options_t* carrier_options = nullptr) {
    IREE_ASSERT_OK(iree_net_loopback_connection_create_pair(
        client_proactor_, server_proactor_, max_endpoint_count, carrier_options,
        iree_allocator_system(), &client_connection_, &server_connection_));
    iree_net_loopback_connection_publish(client_connection_);
    iree_net_loopback_connection_publish(server_connection_);
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

  iree_net_message_endpoint_t OpenEndpoint(iree_net_connection_t* connection,
                                           iree_async_proactor_t* proactor,
                                           PollSide side) {
    EndpointReadyState ready_state;
    ready_state.current_poll_side = &current_poll_side_;
    ready_state.expected_poll_side = side;
    IREE_EXPECT_OK(
        iree_net_connection_open_endpoint(connection, ready_state.callback()));
    EXPECT_EQ(ready_state.callback_count, 0);
    PollUntil(proactor, side, [&] { return ready_state.callback_count == 1; });
    EXPECT_EQ(ready_state.status_code, IREE_STATUS_OK);
    EXPECT_NE(ready_state.endpoint.self, nullptr);
    return ready_state.endpoint;
  }

  void ActivateEndpoint(iree_net_message_endpoint_t endpoint,
                        MessageState* message_state, PollSide side) {
    message_state->current_poll_side = &current_poll_side_;
    message_state->expected_poll_side = side;
    iree_net_message_endpoint_set_callbacks(endpoint,
                                            message_state->callbacks());
    IREE_ASSERT_OK(iree_net_message_endpoint_activate(endpoint));
  }

  void DeactivateAndRelease(iree_net_connection_t*& connection,
                            iree_async_proactor_t* proactor, PollSide side) {
    if (!connection) {
      return;
    }
    DeactivateState state;
    state.current_poll_side = &current_poll_side_;
    state.expected_poll_side = side;
    current_poll_side_ = side;
    iree_net_connection_deactivate(connection, state.callback());
    current_poll_side_ = kNotPolling;
    if (!state.completed) {
      PollUntil(proactor, side, [&] { return state.completed; });
    }
    iree_net_connection_release(connection);
    connection = nullptr;
  }

  int current_poll_side_ = kNotPolling;
  iree_async_proactor_t* client_proactor_ = nullptr;
  iree_async_proactor_t* server_proactor_ = nullptr;
  iree_net_connection_t* client_connection_ = nullptr;
  iree_net_connection_t* server_connection_ = nullptr;
};

TEST_F(LoopbackConnectionTest,
       PreservesEndpointOrdinalsAndFragmentedMessageBoundaries) {
  CreatePair(/*max_endpoint_count=*/2);

  iree_net_message_endpoint_t client_endpoints[2] = {
      OpenEndpoint(client_connection_, client_proactor_, kClientPolling),
      OpenEndpoint(client_connection_, client_proactor_, kClientPolling),
  };
  iree_net_message_endpoint_t server_endpoints[2] = {
      OpenEndpoint(server_connection_, server_proactor_, kServerPolling),
      OpenEndpoint(server_connection_, server_proactor_, kServerPolling),
  };

  MessageState client_messages[2];
  MessageState server_messages[2];
  for (int i = 0; i < 2; ++i) {
    ActivateEndpoint(client_endpoints[i], &client_messages[i], kClientPolling);
    ActivateEndpoint(server_endpoints[i], &server_messages[i], kServerPolling);
  }

  char first[] = "zero-";
  char second[] = "endpoint";
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first, sizeof(first) - 1),
      iree_async_span_from_ptr(second, sizeof(second) - 1),
  };
  SendState send_zero;
  send_zero.current_poll_side = &current_poll_side_;
  send_zero.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_send_params_t send_params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      .completion_callback = send_zero.callback(),
  };
  IREE_ASSERT_OK(
      iree_net_message_endpoint_send(client_endpoints[0], &send_params));
  first[0] = 'Z';

  char endpoint_one[] = "endpoint-one";
  iree_async_span_t endpoint_one_span =
      iree_async_span_from_ptr(endpoint_one, sizeof(endpoint_one) - 1);
  SendState send_one;
  send_one.current_poll_side = &current_poll_side_;
  send_one.expected_poll_side = kClientPolling;
  send_params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&endpoint_one_span, 1),
      .completion_callback = send_one.callback(),
  };
  IREE_ASSERT_OK(
      iree_net_message_endpoint_send(client_endpoints[1], &send_params));

  PollUntil(server_proactor_, kServerPolling, [&] {
    return server_messages[0].messages.size() == 1 &&
           server_messages[1].messages.size() == 1;
  });
  EXPECT_EQ(server_messages[0].messages[0], "Zero-endpoint");
  EXPECT_EQ(server_messages[1].messages[0], "endpoint-one");
  PollUntil(client_proactor_, kClientPolling, [&] {
    return send_zero.callback_count == 1 && send_one.callback_count == 1;
  });
  EXPECT_EQ(send_zero.status_code, IREE_STATUS_OK);
  EXPECT_EQ(send_zero.bytes_transferred,
            (sizeof(first) - 1) + (sizeof(second) - 1));
  EXPECT_EQ(send_one.status_code, IREE_STATUS_OK);

  EndpointReadyState exhausted;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_connection_open_endpoint(
                            client_connection_, exhausted.callback()));
  EXPECT_EQ(exhausted.callback_count, 0);
}

TEST_F(LoopbackConnectionTest,
       MaximumSpansPreserveGeneratedPrefixAndBorrowedPayload) {
  iree_net_loopback_carrier_options_t options =
      iree_net_loopback_carrier_options_default();
  options.max_send_operations = 2;
  options.max_send_spans = 2;
  CreatePair(/*max_endpoint_count=*/1, &options);

  iree_net_message_endpoint_t client_endpoint =
      OpenEndpoint(client_connection_, client_proactor_, kClientPolling);
  iree_net_message_endpoint_t server_endpoint =
      OpenEndpoint(server_connection_, server_proactor_, kServerPolling);
  MessageState client_messages;
  MessageState server_messages;
  ActivateEndpoint(client_endpoint, &client_messages, kClientPolling);
  ActivateEndpoint(server_endpoint, &server_messages, kServerPolling);

  char first[] = "copied-";
  char second[] = "overflow";
  char prefix[] = "prefix-";
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first, sizeof(first) - 1),
      iree_async_span_from_ptr(second, sizeof(second) - 1),
  };
  SendState send_result;
  send_result.current_poll_side = &current_poll_side_;
  send_result.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(prefix, sizeof(prefix) - 1)),
      .data = iree_async_span_list_make(spans, IREE_ARRAYSIZE(spans)),
      .completion_callback = send_result.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &params));
  prefix[0] = 'X';
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_messages.messages.size() == 1; });
  EXPECT_EQ(server_messages.messages[0], "prefix-copied-overflow");
  PollUntil(client_proactor_, kClientPolling,
            [&] { return send_result.callback_count == 1; });
  EXPECT_EQ(send_result.bytes_transferred,
            (sizeof(prefix) - 1) + (sizeof(first) - 1) + (sizeof(second) - 1));

  char generated_data[] = "direct";
  SendState direct_send;
  direct_send.current_poll_side = &current_poll_side_;
  direct_send.expected_poll_side = kServerPolling;
  params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(generated_data, 6)),
      .data = iree_async_span_list_empty(),
      .completion_callback = direct_send.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &params));
  PollUntil(client_proactor_, kClientPolling,
            [&] { return client_messages.messages.size() == 1; });
  EXPECT_EQ(client_messages.messages[0], "direct");
  PollUntil(server_proactor_, kServerPolling,
            [&] { return direct_send.callback_count == 1; });
  EXPECT_EQ(direct_send.status_code, IREE_STATUS_OK);
  EXPECT_EQ(direct_send.bytes_transferred, 6u);

  SendState prefix_failure;
  prefix_failure.current_poll_side = &current_poll_side_;
  prefix_failure.expected_poll_side = kServerPolling;
  params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(nullptr, 4)),
      .data = iree_async_span_list_empty(),
      .completion_callback = prefix_failure.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return prefix_failure.callback_count == 1; });
  EXPECT_EQ(prefix_failure.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(prefix_failure.bytes_transferred, 0u);
  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(server_endpoint).slots,
            options.max_send_operations);

  iree_async_region_t inaccessible_region = {};
  inaccessible_region.length = 1;
  iree_async_span_t inaccessible_span =
      iree_async_span_from_region(&inaccessible_region, 1);
  SendState lower_rejection;
  lower_rejection.current_poll_side = &current_poll_side_;
  lower_rejection.expected_poll_side = kServerPolling;
  params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&inaccessible_span, 1),
      .completion_callback = lower_rejection.callback(),
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &params));
  EXPECT_EQ(lower_rejection.callback_count, 0);
  PollUntil(server_proactor_, kServerPolling,
            [&] { return lower_rejection.callback_count == 1; });
  EXPECT_EQ(lower_rejection.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(lower_rejection.bytes_transferred, 0u);
  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(server_endpoint).slots,
            options.max_send_operations);
  EXPECT_EQ(client_messages.messages.size(), 1u);
}

TEST_F(LoopbackConnectionTest,
       FailingPrefixCompletionRefreshesNestedOneSlotBudget) {
  iree_net_loopback_carrier_options_t options =
      iree_net_loopback_carrier_options_default();
  options.max_send_operations = 1;
  CreatePair(/*max_endpoint_count=*/1, &options);

  iree_net_message_endpoint_t client_endpoint =
      OpenEndpoint(client_connection_, client_proactor_, kClientPolling);
  iree_net_message_endpoint_t server_endpoint =
      OpenEndpoint(server_connection_, server_proactor_, kServerPolling);
  MessageState client_messages;
  MessageState server_messages;
  ActivateEndpoint(client_endpoint, &client_messages, kClientPolling);
  ActivateEndpoint(server_endpoint, &server_messages, kServerPolling);

  PrefixWriterGate writer_gate;
  writer_gate.result_code = IREE_STATUS_CANCELLED;
  SendState first_send;
  first_send.current_poll_side = &current_poll_side_;
  first_send.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_send_params_t first_params = {
      .generated_prefix =
          {
              .length = 1,
              .write = PrefixWriterGate::Write,
              .user_data = &writer_gate,
          },
      .data = iree_async_span_list_empty(),
      .completion_callback = first_send.callback(),
  };
  iree_status_code_t first_submit_code = IREE_STATUS_UNKNOWN;
  std::thread send_thread([&] {
    iree_status_t status =
        iree_net_message_endpoint_send(client_endpoint, &first_params);
    first_submit_code = iree_status_code(status);
    iree_status_free(status);
  });
  {
    std::unique_lock<std::mutex> lock(writer_gate.mutex);
    writer_gate.condition.wait(lock, [&] { return writer_gate.entered; });
  }

  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(client_endpoint).slots,
            0u);
  char retry_payload = 'x';
  iree_async_span_t retry_span =
      iree_async_span_from_ptr(&retry_payload, sizeof(retry_payload));
  SendState retry_send;
  retry_send.current_poll_side = &current_poll_side_;
  retry_send.expected_poll_side = kClientPolling;
  iree_net_message_endpoint_send_params_t retry_params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&retry_span, 1),
      .completion_callback = retry_send.callback(),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_message_endpoint_send(client_endpoint, &retry_params));
  EXPECT_EQ(retry_send.callback_count, 0);

  {
    std::lock_guard<std::mutex> lock(writer_gate.mutex);
    writer_gate.released = true;
  }
  writer_gate.condition.notify_all();
  send_thread.join();

  EXPECT_EQ(first_submit_code, IREE_STATUS_OK);
  EXPECT_EQ(first_send.callback_count, 0);
  PollUntil(client_proactor_, kClientPolling,
            [&] { return first_send.callback_count == 1; });
  EXPECT_EQ(first_send.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(first_send.bytes_transferred, 0u);
  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(client_endpoint).slots,
            1u);
  EXPECT_TRUE(server_messages.messages.empty());

  IREE_ASSERT_OK(
      iree_net_message_endpoint_send(client_endpoint, &retry_params));
  PollUntil(server_proactor_, kServerPolling,
            [&] { return server_messages.messages.size() == 1; });
  EXPECT_EQ(server_messages.messages[0], "x");
  PollUntil(client_proactor_, kClientPolling,
            [&] { return retry_send.callback_count == 1; });
  EXPECT_EQ(retry_send.status_code, IREE_STATUS_OK);
}

TEST_F(LoopbackConnectionTest,
       DeactivationWaitsForAndCancelsPendingReadyCallback) {
  CreatePair(/*max_endpoint_count=*/1);

  EndpointReadyState ready_state;
  ready_state.current_poll_side = &current_poll_side_;
  ready_state.expected_poll_side = kClientPolling;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(client_connection_,
                                                   ready_state.callback()));

  DeactivateState deactivate_state;
  deactivate_state.current_poll_side = &current_poll_side_;
  deactivate_state.expected_poll_side = kClientPolling;
  iree_net_connection_deactivate(client_connection_,
                                 deactivate_state.callback());
  EXPECT_EQ(ready_state.callback_count, 0);
  EXPECT_FALSE(deactivate_state.completed);

  PollUntil(client_proactor_, kClientPolling,
            [&] { return deactivate_state.completed; });
  EXPECT_EQ(ready_state.callback_count, 1);
  EXPECT_EQ(ready_state.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(ready_state.endpoint.self, nullptr);

  iree_net_connection_release(client_connection_);
  client_connection_ = nullptr;
}

}  // namespace
}  // namespace iree
