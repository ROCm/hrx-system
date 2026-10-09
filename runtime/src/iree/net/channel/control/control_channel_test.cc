// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/control/control_channel.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::string FormatStatus(const iree_status_t status) {
  iree_host_size_t length = 0;
  if (!iree_status_format(status, 0, nullptr, &length)) {
    return {};
  }
  std::vector<char> buffer(length + 1);
  if (!iree_status_format(status, buffer.size(), buffer.data(), &length)) {
    return {};
  }
  return std::string(buffer.data(), length);
}

static std::vector<uint8_t> MakeMessage(uint8_t type, uint8_t flags,
                                        uint32_t value,
                                        std::vector<uint8_t> payload = {}) {
  std::vector<uint8_t> message(IREE_NET_CONTROL_MESSAGE_HEADER_SIZE +
                               payload.size());
  message[0] = IREE_NET_CONTROL_MESSAGE_VERSION;
  message[1] = type;
  message[2] = flags;
  message[3] = 0;
  iree_unaligned_store_le_u32(message.data() + 4, value);
  if (!payload.empty()) {
    std::memcpy(message.data() + IREE_NET_CONTROL_MESSAGE_HEADER_SIZE,
                payload.data(), payload.size());
  }
  return message;
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

  iree_status_t InvokeMessage(const std::vector<uint8_t>& message,
                              iree_async_buffer_lease_t* lease = nullptr) {
    return callbacks_.on_message(
        callbacks_.user_data,
        iree_make_const_byte_span(message.data(), message.size()), lease);
  }

  iree_status_code_t DeliverMessage(
      const std::vector<uint8_t>& message,
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
  iree_host_size_t last_borrowed_span_count = 0;
  std::vector<const uint8_t*> last_borrowed_span_pointers;
  std::vector<std::vector<uint8_t>> sent_messages;

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
    (void)self;
    return {
        /*.bytes=*/IREE_HOST_SIZE_MAX,
        /*.slots=*/UINT32_MAX,
    };
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
  int data_count = 0;
  iree_net_control_data_flags_t data_flags = 0;
  std::vector<uint8_t> data_payload;
  iree_async_buffer_lease_t* data_lease = nullptr;
  iree_status_code_t next_data_status = IREE_STATUS_OK;
  int goaway_count = 0;
  uint32_t goaway_reason = 0;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;
  std::string error_text;

  static iree_status_t OnData(void* user_data,
                              iree_net_control_data_flags_t flags,
                              iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->data_count;
    self->data_flags = flags;
    self->data_payload.assign(payload.data, payload.data + payload.data_length);
    self->data_lease = lease;
    return iree_status_from_code(self->next_data_status);
  }

  static void OnGoaway(void* user_data, uint32_t reason_code) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->goaway_count;
    self->goaway_reason = reason_code;
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    self->error_text = FormatStatus(status);
    iree_status_free(status);
  }

  iree_net_control_channel_callbacks_t callbacks() {
    return {
        .on_data = OnData,
        .on_goaway = OnGoaway,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

class ControlChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_net_control_channel_allocate(
        endpoint_.endpoint(), callback_state_.callbacks(),
        iree_allocator_system(), &channel_));
  }

  void TearDown() override { iree_net_control_channel_free(channel_); }

  void Attach() { iree_net_control_channel_attach(channel_); }

  TestEndpoint endpoint_;
  CallbackState callback_state_;
  iree_net_control_channel_t* channel_ = nullptr;
};

TEST(ControlChannelAllocationTest, RequiresEndpointAndCallbacks) {
  iree_net_control_channel_t* channel = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_control_channel_allocate(iree_net_message_endpoint_t{},
                                        iree_net_control_channel_callbacks_t{},
                                        iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);

  TestEndpoint endpoint;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_control_channel_allocate(endpoint.endpoint(),
                                        iree_net_control_channel_callbacks_t{},
                                        iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);
}

TEST_F(ControlChannelTest, AttachInstallsCallbacksAndDeliversData) {
  EXPECT_EQ(endpoint_.set_callbacks_count, 0);
  Attach();
  EXPECT_EQ(endpoint_.set_callbacks_count, 1);

  iree_async_buffer_lease_t lease = {};
  const std::vector<uint8_t> message = MakeMessage(
      IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0xA5, 0, {'d', 'a', 't', 'a'});
  EXPECT_EQ(endpoint_.DeliverMessage(message, &lease), IREE_STATUS_OK);

  EXPECT_EQ(callback_state_.data_count, 1);
  EXPECT_EQ(callback_state_.data_flags, 0xA5u);
  EXPECT_EQ(callback_state_.data_payload,
            (std::vector<uint8_t>{'d', 'a', 't', 'a'}));
  EXPECT_EQ(callback_state_.data_lease, &lease);
}

TEST_F(ControlChannelTest, DataFailureConvergesThroughEndpointError) {
  Attach();
  callback_state_.next_data_status = IREE_STATUS_ABORTED;

  EXPECT_EQ(endpoint_.DeliverMessage(
                MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0, 0)),
            IREE_STATUS_ABORTED);
  EXPECT_EQ(callback_state_.data_count, 1);
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_ABORTED);
}

TEST_F(ControlChannelTest, DeliversHeaderOnlyGoaway) {
  Attach();
  constexpr uint32_t kReason = 0x89ABCDEFu;
  EXPECT_EQ(endpoint_.DeliverMessage(
                MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_GOAWAY, 0, kReason)),
            IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.goaway_count, 1);
  EXPECT_EQ(callback_state_.goaway_reason, kReason);
  EXPECT_EQ(callback_state_.error_count, 0);
}

TEST_F(ControlChannelTest, MalformedMessagesFailWithoutDispatch) {
  Attach();
  std::vector<std::vector<uint8_t>> malformed;
  malformed.push_back({});
  malformed.push_back(std::vector<uint8_t>(7, 0));
  malformed.push_back(MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0, 1));
  malformed.push_back(MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_GOAWAY, 1, 0));
  malformed.push_back(
      MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_GOAWAY, 0, 0, {'n', 'o'}));
  malformed.push_back(MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_ERROR, 1, 0));
  malformed.push_back(MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_ERROR, 0, 1));
  malformed.push_back(MakeMessage(0x7F, 0, 0));
  malformed.push_back(MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_ERROR, 0, 0));

  auto bad_version = MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0, 0);
  bad_version[0] = IREE_NET_CONTROL_MESSAGE_VERSION + 1;
  malformed.push_back(std::move(bad_version));
  auto bad_reserved = MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0, 0);
  bad_reserved[3] = 1;
  malformed.push_back(std::move(bad_reserved));

  for (const auto& message : malformed) {
    iree_status_t status = endpoint_.InvokeMessage(message);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_DATA_LOSS);
    iree_status_free(status);
  }
  EXPECT_EQ(callback_state_.data_count, 0);
  EXPECT_EQ(callback_state_.goaway_count, 0);
}

TEST_F(ControlChannelTest, BorrowedDataUsesPrefixAndNativeCompletion) {
  Attach();
  std::vector<uint8_t> first = {1, 2, 3};
  std::vector<uint8_t> second = {4, 5};
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first.data(), first.size()),
      iree_async_span_from_ptr(second.data(), second.size()),
  };
  SendCompletion completion;

  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      channel_, 0x5A, iree_async_span_list_make(spans, 2),
      completion.callback()));

  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0],
            MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_DATA, 0x5A, 0,
                        {1, 2, 3, 4, 5}));
  EXPECT_EQ(endpoint_.last_borrowed_span_count, 2u);
  EXPECT_EQ(endpoint_.last_borrowed_span_pointers,
            (std::vector<const uint8_t*>{first.data(), second.data()}));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(completion.bytes_transferred, IREE_NET_CONTROL_MESSAGE_HEADER_SIZE +
                                              first.size() + second.size());
}

TEST_F(ControlChannelTest, CopiedDataUsesGeneratedPrefixWithoutSizeCliff) {
  Attach();
  std::vector<uint8_t> first(4097, 0xA5);
  std::vector<uint8_t> second(8192, 0x5A);
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first.data(), first.size()),
      iree_async_span_from_ptr(second.data(), second.size()),
  };
  SendCompletion completion;

  IREE_ASSERT_OK(iree_net_control_channel_send_data_copy(
      channel_, 7, iree_async_span_list_make(spans, 2), completion.callback()));
  std::fill(first.begin(), first.end(), 0);
  std::fill(second.begin(), second.end(), 0);

  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  const std::vector<uint8_t>& message = endpoint_.sent_messages[0];
  ASSERT_EQ(message.size(), IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + 12289u);
  EXPECT_EQ(message[0], IREE_NET_CONTROL_MESSAGE_VERSION);
  EXPECT_EQ(message[1], IREE_NET_CONTROL_MESSAGE_TYPE_DATA);
  EXPECT_EQ(message[2], 7);
  EXPECT_EQ(message[IREE_NET_CONTROL_MESSAGE_HEADER_SIZE], 0xA5);
  EXPECT_EQ(message[IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + 4096], 0xA5);
  EXPECT_EQ(message[IREE_NET_CONTROL_MESSAGE_HEADER_SIZE + 4097], 0x5A);
  EXPECT_EQ(message.back(), 0x5A);
  EXPECT_EQ(endpoint_.last_borrowed_span_count, 0u);
  EXPECT_EQ(endpoint_.send_count, 1);
  EXPECT_EQ(completion.count, 1);
}

TEST_F(ControlChannelTest, CopiedDataValidatesBeforeAdmission) {
  Attach();
  iree_async_span_t null_span = iree_async_span_from_ptr(nullptr, /*length=*/1);
  SendCompletion completion;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_control_channel_send_data_copy(
          channel_, 0, iree_async_span_list_make(&null_span, 1),
          completion.callback()));
  EXPECT_EQ(endpoint_.send_count, 0);
  EXPECT_EQ(completion.count, 0);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_control_channel_send_data_copy(channel_, UINT32_C(0x100),
                                              iree_async_span_list_empty(),
                                              completion.callback()));
  EXPECT_EQ(endpoint_.send_count, 0);
  EXPECT_EQ(completion.count, 0);
}

TEST_F(ControlChannelTest, RejectedGoawayCanBeRetried) {
  Attach();
  SendCompletion completion;
  endpoint_.reject_send_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_control_channel_send_goaway(
                            channel_, 42, completion.callback()));
  EXPECT_EQ(completion.count, 0);
  EXPECT_TRUE(endpoint_.sent_messages.empty());

  endpoint_.reject_send_code = IREE_STATUS_OK;
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(channel_, 42,
                                                      completion.callback()));
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0],
            MakeMessage(IREE_NET_CONTROL_MESSAGE_TYPE_GOAWAY, 0, 42));
  EXPECT_EQ(completion.count, 1);
}

TEST_F(ControlChannelTest, StructuredErrorConvergesThroughEndpointError) {
  Attach();
  SendCompletion completion;
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "remote device disappeared"),
      completion.callback()));
  ASSERT_EQ(endpoint_.sent_messages.size(), 1u);
  EXPECT_EQ(endpoint_.sent_messages[0][1], IREE_NET_CONTROL_MESSAGE_TYPE_ERROR);
  EXPECT_EQ(completion.count, 1);

  EXPECT_EQ(endpoint_.DeliverMessage(endpoint_.sent_messages[0]),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_NE(callback_state_.error_text.find("remote device disappeared"),
            std::string::npos);
}

TEST_F(ControlChannelTest, SendErrorConsumesStatusWhenAdmissionFails) {
  Attach();
  endpoint_.reject_send_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  SendCompletion completion;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_control_channel_send_error(
          channel_, iree_make_status(IREE_STATUS_INTERNAL, "lost"),
          completion.callback()));
  EXPECT_EQ(completion.count, 0);
  EXPECT_EQ(endpoint_.send_count, 1);
}

TEST_F(ControlChannelTest, TransportErrorForwardsWithoutTranslation) {
  Attach();
  endpoint_.DeliverError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost"));
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_NE(callback_state_.error_text.find("connection lost"),
            std::string::npos);
}

}  // namespace
