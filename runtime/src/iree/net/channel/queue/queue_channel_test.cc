// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/queue/queue_channel.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::vector<uint8_t> MakeMessage(
    uint8_t type, uint32_t queue_id,
    const std::vector<iree_async_frontier_entry_t>& wait_frontier,
    const std::vector<iree_async_frontier_entry_t>& signal_frontier,
    std::vector<uint8_t> payload = {}) {
  const size_t wait_size =
      wait_frontier.size() * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  const size_t signal_size =
      signal_frontier.size() * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  std::vector<uint8_t> message(IREE_NET_QUEUE_MESSAGE_HEADER_SIZE + wait_size +
                               signal_size + payload.size());
  message[0] = IREE_NET_QUEUE_MESSAGE_VERSION;
  message[1] = type;
  message[2] = static_cast<uint8_t>(wait_frontier.size());
  message[3] = static_cast<uint8_t>(signal_frontier.size());
  iree_unaligned_store_le_u32(message.data() + 4, queue_id);
  uint8_t* entry_data = message.data() + IREE_NET_QUEUE_MESSAGE_HEADER_SIZE;
  for (const auto& entry : wait_frontier) {
    iree_unaligned_store_le_u64(entry_data, entry.axis);
    iree_unaligned_store_le_u64(entry_data + 8, entry.epoch);
    entry_data += IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  }
  for (const auto& entry : signal_frontier) {
    iree_unaligned_store_le_u64(entry_data, entry.axis);
    iree_unaligned_store_le_u64(entry_data + 8, entry.epoch);
    entry_data += IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  }
  if (!payload.empty()) {
    std::memcpy(entry_data, payload.data(), payload.size());
  }
  return message;
}

static std::vector<iree_async_frontier_entry_t> CaptureFrontier(
    const iree_net_queue_frontier_view_t* frontier) {
  std::vector<iree_async_frontier_entry_t> entries;
  entries.reserve(frontier->count);
  for (iree_host_size_t i = 0; i < frontier->count; ++i) {
    entries.push_back(iree_net_queue_frontier_view_get(frontier, i));
  }
  return entries;
}

static void ExpectFrontierEq(
    const std::vector<iree_async_frontier_entry_t>& actual,
    const std::vector<iree_async_frontier_entry_t>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].axis, expected[i].axis);
    EXPECT_EQ(actual[i].epoch, expected[i].epoch);
  }
}

struct SendCompletion {
  int count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_host_size_t bytes_transferred = 0;

  static void Callback(void* user_data, iree_status_t status,
                       iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SendCompletion*>(user_data);
    ++self->count;
    self->status_code = iree_status_code(status);
    self->bytes_transferred = bytes_transferred;
    iree_status_free(status);
  }

  iree_net_send_completion_callback_t callback() {
    return {
        .fn = Callback,
        .user_data = this,
    };
  }
};

class TestEndpoint {
 public:
  iree_net_message_endpoint_t endpoint() {
    return {
        .self = this,
        .vtable = &vtable_,
    };
  }

  iree_status_t InvokeMessage(iree_const_byte_span_t message,
                              iree_async_buffer_lease_t* lease = nullptr) {
    return callbacks_.on_message(callbacks_.user_data, message, lease);
  }

  iree_status_code_t DeliverMessage(
      iree_const_byte_span_t message,
      iree_async_buffer_lease_t* lease = nullptr) {
    iree_status_t status = InvokeMessage(message, lease);
    const iree_status_code_t status_code = iree_status_code(status);
    if (!iree_status_is_ok(status)) {
      callbacks_.on_error(callbacks_.user_data, status);
    }
    return status_code;
  }

  void DeliverError(iree_status_t status) {
    callbacks_.on_error(callbacks_.user_data, status);
  }

  iree_status_code_t reject_send_code = IREE_STATUS_OK;
  int set_callbacks_count = 0;
  int send_count = 0;
  int prefix_write_count = 0;
  iree_host_size_t last_borrowed_span_count = 0;
  std::vector<const uint8_t*> last_borrowed_span_pointers;
  std::vector<std::vector<uint8_t>> sent_messages;
  iree_net_carrier_send_budget_t budget = {
      .bytes = 12345,
      .slots = 7,
  };

 private:
  static TestEndpoint* Cast(void* self) {
    return static_cast<TestEndpoint*>(self);
  }

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    TestEndpoint* endpoint = Cast(self);
    endpoint->callbacks_ = callbacks;
    ++endpoint->set_callbacks_count;
  }

  static iree_status_t Activate(void* self) {
    (void)self;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    (void)self;
    if (callback) {
      callback(user_data);
    }
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    TestEndpoint* endpoint = Cast(self);
    ++endpoint->send_count;
    if (endpoint->reject_send_code != IREE_STATUS_OK) {
      return iree_status_from_code(endpoint->reject_send_code);
    }

    endpoint->last_borrowed_span_count = params->data.count;
    endpoint->last_borrowed_span_pointers.clear();
    std::vector<uint8_t> message(params->generated_prefix.length);
    if (params->generated_prefix.length > 0) {
      ++endpoint->prefix_write_count;
      iree_status_t status = params->generated_prefix.write(
          params->generated_prefix.user_data,
          iree_make_byte_span(message.data(), message.size()));
      if (!iree_status_is_ok(status)) {
        params->completion_callback.fn(params->completion_callback.user_data,
                                       status, 0);
        return iree_ok_status();
      }
    }
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      const iree_async_span_t span = params->data.values[i];
      const uint8_t* data = iree_async_span_ptr(span);
      endpoint->last_borrowed_span_pointers.push_back(data);
      message.insert(message.end(), data, data + span.length);
    }
    endpoint->sent_messages.push_back(std::move(message));
    params->completion_callback.fn(params->completion_callback.user_data,
                                   iree_ok_status(),
                                   endpoint->sent_messages.back().size());
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    return Cast(self)->budget;
  }

  static const iree_net_message_endpoint_vtable_t vtable_;

  iree_net_message_endpoint_callbacks_t callbacks_ = {};
};

const iree_net_message_endpoint_vtable_t TestEndpoint::vtable_ = {
    .set_callbacks = TestEndpoint::SetCallbacks,
    .activate = TestEndpoint::Activate,
    .deactivate = TestEndpoint::Deactivate,
    .send = TestEndpoint::Send,
    .query_send_budget = TestEndpoint::QuerySendBudget,
};

struct CallbackState {
  int command_count = 0;
  uint32_t command_queue_id = IREE_NET_QUEUE_ID_NONE;
  std::vector<iree_async_frontier_entry_t> command_wait_frontier;
  std::vector<iree_async_frontier_entry_t> command_signal_frontier;
  std::vector<uint8_t> command_payload;
  iree_async_buffer_lease_t* command_lease = nullptr;
  iree_status_code_t next_command_status = IREE_STATUS_OK;
  int advance_count = 0;
  std::vector<iree_async_frontier_entry_t> advance_signal_frontier;
  std::vector<uint8_t> advance_payload;
  iree_async_buffer_lease_t* advance_lease = nullptr;
  iree_status_code_t next_advance_status = IREE_STATUS_OK;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static iree_status_t OnCommand(
      void* user_data, uint32_t queue_id,
      const iree_net_queue_frontier_view_t* wait_frontier,
      const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->command_count;
    self->command_queue_id = queue_id;
    self->command_wait_frontier = CaptureFrontier(wait_frontier);
    self->command_signal_frontier = CaptureFrontier(signal_frontier);
    self->command_payload.assign(payload.data,
                                 payload.data + payload.data_length);
    self->command_lease = lease;
    return iree_status_from_code(self->next_command_status);
  }

  static iree_status_t OnAdvance(
      void* user_data, const iree_net_queue_frontier_view_t* signal_frontier,
      iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->advance_count;
    self->advance_signal_frontier = CaptureFrontier(signal_frontier);
    self->advance_payload.assign(payload.data,
                                 payload.data + payload.data_length);
    self->advance_lease = lease;
    return iree_status_from_code(self->next_advance_status);
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_queue_channel_callbacks_t callbacks() {
    return {
        .on_command = OnCommand,
        .on_advance = OnAdvance,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

struct BuildState {
  std::vector<iree_async_frontier_entry_t> wait_frontier;
  std::vector<iree_async_frontier_entry_t> signal_frontier;
  std::vector<uint8_t> generated_payload;
  iree_status_code_t status_code = IREE_STATUS_OK;
  int count = 0;

  static iree_status_t Build(void* user_data,
                             const iree_net_queue_message_builder_t* builder) {
    auto* self = static_cast<BuildState*>(user_data);
    ++self->count;
    if (self->status_code != IREE_STATUS_OK) {
      return iree_status_from_code(self->status_code);
    }
    if (builder->wait_frontier.count != self->wait_frontier.size() ||
        builder->signal_frontier.count != self->signal_frontier.size() ||
        builder->generated_payload.data_length !=
            self->generated_payload.size()) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "builder target layout mismatch");
    }
    for (iree_host_size_t i = 0; i < self->wait_frontier.size(); ++i) {
      iree_net_queue_frontier_builder_set(&builder->wait_frontier, i,
                                          self->wait_frontier[i]);
    }
    for (iree_host_size_t i = 0; i < self->signal_frontier.size(); ++i) {
      iree_net_queue_frontier_builder_set(&builder->signal_frontier, i,
                                          self->signal_frontier[i]);
    }
    if (!self->generated_payload.empty()) {
      std::memcpy(builder->generated_payload.data,
                  self->generated_payload.data(),
                  self->generated_payload.size());
    }
    return iree_ok_status();
  }

  iree_net_queue_channel_send_params_t params(
      iree_async_span_list_t payload,
      iree_net_send_completion_callback_t completion_callback) {
    const bool requires_builder = !wait_frontier.empty() ||
                                  !signal_frontier.empty() ||
                                  !generated_payload.empty();
    return {
        .wait_frontier_count = static_cast<uint8_t>(wait_frontier.size()),
        .signal_frontier_count = static_cast<uint8_t>(signal_frontier.size()),
        .generated_payload_length = generated_payload.size(),
        .build = requires_builder ? Build : nullptr,
        .build_user_data = requires_builder ? this : nullptr,
        .payload = payload,
        .completion_callback = completion_callback,
    };
  }
};

class QueueChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_net_queue_channel_allocate(
        endpoint_.endpoint(), callback_state_.callbacks(),
        iree_allocator_system(), &channel_));
  }

  void TearDown() override { iree_net_queue_channel_free(channel_); }

  void Attach() { iree_net_queue_channel_attach(channel_); }

  TestEndpoint endpoint_;
  CallbackState callback_state_;
  iree_net_queue_channel_t* channel_ = nullptr;
};

TEST(QueueChannelAllocationTest, RequiresEndpointAndCallbacks) {
  iree_net_queue_channel_t* channel = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_queue_channel_allocate(iree_net_message_endpoint_t{},
                                      iree_net_queue_channel_callbacks_t{},
                                      iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);

  TestEndpoint endpoint;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_queue_channel_allocate(endpoint.endpoint(),
                                      iree_net_queue_channel_callbacks_t{},
                                      iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);
}

TEST_F(QueueChannelTest, AttachDeliversUnalignedCommand) {
  Attach();
  const std::vector<iree_async_frontier_entry_t> wait_frontier = {
      {3, 5},
      {7, 11},
  };
  const std::vector<iree_async_frontier_entry_t> signal_frontier = {
      {9, 13},
  };
  const std::vector<uint8_t> message =
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 42, wait_frontier,
                  signal_frontier, {'c', 'm', 'd'});
  std::vector<uint8_t> unaligned_storage(message.size() + 1);
  std::memcpy(unaligned_storage.data() + 1, message.data(), message.size());
  iree_async_buffer_lease_t lease = {};

  EXPECT_EQ(endpoint_.DeliverMessage(
                iree_make_const_byte_span(unaligned_storage.data() + 1,
                                          message.size()),
                &lease),
            IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.command_count, 1);
  EXPECT_EQ(callback_state_.command_queue_id, 42u);
  ExpectFrontierEq(callback_state_.command_wait_frontier, wait_frontier);
  ExpectFrontierEq(callback_state_.command_signal_frontier, signal_frontier);
  EXPECT_EQ(callback_state_.command_payload,
            (std::vector<uint8_t>{'c', 'm', 'd'}));
  EXPECT_EQ(callback_state_.command_lease, &lease);
}

TEST_F(QueueChannelTest, DeliversAdvance) {
  Attach();
  const std::vector<iree_async_frontier_entry_t> signal_frontier = {
      {17, 19},
      {23, 29},
  };
  const std::vector<uint8_t> message =
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE, IREE_NET_QUEUE_ID_NONE,
                  {}, signal_frontier, {'o', 'k'});
  iree_async_buffer_lease_t lease = {};

  EXPECT_EQ(
      endpoint_.DeliverMessage(
          iree_make_const_byte_span(message.data(), message.size()), &lease),
      IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.advance_count, 1);
  ExpectFrontierEq(callback_state_.advance_signal_frontier, signal_frontier);
  EXPECT_EQ(callback_state_.advance_payload, (std::vector<uint8_t>{'o', 'k'}));
  EXPECT_EQ(callback_state_.advance_lease, &lease);
}

TEST_F(QueueChannelTest, CallbackFailureConvergesThroughEndpointError) {
  Attach();
  callback_state_.next_command_status = IREE_STATUS_ABORTED;
  const std::vector<uint8_t> message =
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0, {}, {}, {'x'});

  EXPECT_EQ(endpoint_.DeliverMessage(
                iree_make_const_byte_span(message.data(), message.size())),
            IREE_STATUS_ABORTED);
  EXPECT_EQ(callback_state_.command_count, 1);
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_ABORTED);
}

TEST_F(QueueChannelTest, MalformedMessagesFailBeforeDispatch) {
  Attach();
  std::vector<std::vector<uint8_t>> malformed;
  malformed.push_back({});
  malformed.push_back(std::vector<uint8_t>(7, 0));
  malformed.push_back(
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0, {}, {}));
  malformed.push_back(MakeMessage(0x7F, 0, {}, {}, {'x'}));
  malformed.push_back(
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE, 0, {}, {{1, 1}}));
  malformed.push_back(MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE,
                                  IREE_NET_QUEUE_ID_NONE, {{1, 1}}, {{2, 1}}));
  malformed.push_back(MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE,
                                  IREE_NET_QUEUE_ID_NONE, {}, {}));
  malformed.push_back(MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0,
                                  {{3, 1}, {3, 2}}, {}, {'x'}));
  malformed.push_back(MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0,
                                  {{5, 1}, {3, 2}}, {}, {'x'}));
  malformed.push_back(
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0, {{3, 0}}, {}, {'x'}));

  auto bad_version =
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0, {}, {}, {'x'});
  bad_version[0] = IREE_NET_QUEUE_MESSAGE_VERSION + 1;
  malformed.push_back(std::move(bad_version));
  auto truncated_frontier =
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 0, {}, {}, {'x'});
  truncated_frontier[2] = 1;
  malformed.push_back(std::move(truncated_frontier));

  for (const auto& message : malformed) {
    iree_status_t status = endpoint_.InvokeMessage(
        iree_make_const_byte_span(message.data(), message.size()));
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_DATA_LOSS);
    iree_status_free(status);
  }
  EXPECT_EQ(callback_state_.command_count, 0);
  EXPECT_EQ(callback_state_.advance_count, 0);
}

TEST_F(QueueChannelTest, GeneratedCommandPreservesBorrowedPayload) {
  Attach();
  BuildState build_state = {
      .wait_frontier = {{3, 5}, {7, 11}},
      .signal_frontier = {{9, 13}},
      .generated_payload = {'m', 'e', 't', 'a'},
  };
  std::vector<uint8_t> first = {1, 2, 3};
  std::vector<uint8_t> second = {4, 5};
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first.data(), first.size()),
      iree_async_span_from_ptr(second.data(), second.size()),
  };
  SendCompletion completion;
  const iree_net_queue_channel_send_params_t params = build_state.params(
      iree_async_span_list_make(spans, 2), completion.callback());

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(channel_, 42, &params));

  EXPECT_EQ(build_state.count, 1);
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0],
            MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, 42,
                        build_state.wait_frontier, build_state.signal_frontier,
                        {'m', 'e', 't', 'a', 1, 2, 3, 4, 5}));
  EXPECT_EQ(endpoint_.last_borrowed_span_count, 2u);
  EXPECT_EQ(endpoint_.last_borrowed_span_pointers,
            (std::vector<const uint8_t*>{first.data(), second.data()}));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(completion.bytes_transferred, endpoint_.sent_messages[0].size());
}

TEST_F(QueueChannelTest, StableCommandNeedsNoApplicationBuilder) {
  Attach();
  std::vector<uint8_t> payload = {'s', 't', 'a', 'b', 'l', 'e'};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendCompletion completion;
  BuildState build_state;
  const iree_net_queue_channel_send_params_t params = build_state.params(
      iree_async_span_list_make(&span, 1), completion.callback());

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      channel_, IREE_NET_QUEUE_ID_NONE, &params));
  EXPECT_EQ(build_state.count, 0);
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0],
            MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND,
                        IREE_NET_QUEUE_ID_NONE, {}, {}, payload));
}

TEST_F(QueueChannelTest, AdvanceUsesNoQueueId) {
  Attach();
  BuildState build_state = {
      .wait_frontier = {},
      .signal_frontier = {{9, 13}},
      .generated_payload = {},
  };
  SendCompletion completion;
  const iree_net_queue_channel_send_params_t params =
      build_state.params(iree_async_span_list_empty(), completion.callback());

  IREE_ASSERT_OK(iree_net_queue_channel_send_advance(channel_, &params));
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(
      endpoint_.sent_messages[0],
      MakeMessage(IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE, IREE_NET_QUEUE_ID_NONE,
                  {}, build_state.signal_frontier));
}

TEST_F(QueueChannelTest, SupportsMaximumFrontierCounts) {
  Attach();
  BuildState build_state;
  for (uint32_t i = 0; i < UINT8_MAX; ++i) {
    build_state.wait_frontier.push_back({2u * i + 1u, i + 1u});
    build_state.signal_frontier.push_back({2u * i + 2u, i + 1u});
  }
  build_state.generated_payload = {'x'};
  SendCompletion completion;
  const iree_net_queue_channel_send_params_t params =
      build_state.params(iree_async_span_list_empty(), completion.callback());

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(channel_, 0, &params));
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0].size(),
            IREE_NET_QUEUE_MESSAGE_HEADER_SIZE +
                2u * UINT8_MAX * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE + 1u);
  EXPECT_EQ(completion.count, 1);
}

TEST_F(QueueChannelTest, RejectedSendDoesNotRunBuilderOrCompletion) {
  Attach();
  endpoint_.reject_send_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  BuildState build_state = {
      .wait_frontier = {},
      .signal_frontier = {{9, 13}},
      .generated_payload = {'x'},
  };
  SendCompletion completion;
  const iree_net_queue_channel_send_params_t params =
      build_state.params(iree_async_span_list_empty(), completion.callback());

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_queue_channel_send_command(channel_, 0, &params));
  EXPECT_EQ(build_state.count, 0);
  EXPECT_EQ(completion.count, 0);
  EXPECT_TRUE(endpoint_.sent_messages.empty());
}

TEST_F(QueueChannelTest, BuilderFailureCompletesAcceptedSend) {
  Attach();
  BuildState build_state = {
      .wait_frontier = {},
      .signal_frontier = {{9, 13}},
      .generated_payload = {'x'},
      .status_code = IREE_STATUS_ABORTED,
  };
  SendCompletion completion;
  const iree_net_queue_channel_send_params_t params =
      build_state.params(iree_async_span_list_empty(), completion.callback());

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(channel_, 0, &params));
  EXPECT_EQ(build_state.count, 1);
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(completion.bytes_transferred, 0u);
  EXPECT_TRUE(endpoint_.sent_messages.empty());
}

TEST_F(QueueChannelTest, ValidatesSemanticsBeforeEndpointAdmission) {
  Attach();
  SendCompletion completion;
  iree_net_queue_channel_send_params_t params = {.completion_callback =
                                                     completion.callback()};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_queue_channel_send_command(channel_, 0, &params));
  EXPECT_EQ(endpoint_.send_count, 0);

  BuildState advance_with_wait = {
      .wait_frontier = {{1, 1}},
      .signal_frontier = {{2, 1}},
  };
  params = advance_with_wait.params(iree_async_span_list_empty(),
                                    completion.callback());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_queue_channel_send_advance(channel_, &params));
  EXPECT_EQ(endpoint_.send_count, 0);
}

TEST_F(QueueChannelTest, RejectsMessagesBeyondWireExtent) {
  Attach();
  iree_async_span_t span = iree_async_span_from_ptr(
      reinterpret_cast<void*>(uintptr_t{1}), UINT32_MAX);
  SendCompletion completion;
  BuildState build_state;
  const iree_net_queue_channel_send_params_t params = build_state.params(
      iree_async_span_list_make(&span, 1), completion.callback());

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_queue_channel_send_command(channel_, 0, &params));
  EXPECT_EQ(endpoint_.send_count, 0);
}

TEST_F(QueueChannelTest, AdmitsMaximumWireExtentToEndpoint) {
  Attach();
  endpoint_.reject_send_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  iree_async_span_t span =
      iree_async_span_from_ptr(reinterpret_cast<void*>(uintptr_t{1}),
                               UINT32_MAX - IREE_NET_QUEUE_MESSAGE_HEADER_SIZE);
  SendCompletion completion;
  BuildState build_state;
  const iree_net_queue_channel_send_params_t params = build_state.params(
      iree_async_span_list_make(&span, 1), completion.callback());

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_queue_channel_send_command(channel_, 0, &params));
  EXPECT_EQ(endpoint_.send_count, 1);
  EXPECT_EQ(completion.count, 0);
}

TEST_F(QueueChannelTest, QuerySendBudgetPassesThroughEndpoint) {
  const iree_net_carrier_send_budget_t budget =
      iree_net_queue_channel_query_send_budget(channel_);
  EXPECT_EQ(budget.bytes, endpoint_.budget.bytes);
  EXPECT_EQ(budget.slots, endpoint_.budget.slots);
}

TEST_F(QueueChannelTest, TransportErrorForwardsWithoutTranslation) {
  Attach();
  endpoint_.DeliverError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost"));
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_UNAVAILABLE);
}

}  // namespace
