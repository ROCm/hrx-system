// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/bulk_channel.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::vector<uint8_t> MakeMessage(uint8_t type, uint64_t transfer_id,
                                        uint64_t value,
                                        std::vector<uint8_t> payload = {}) {
  std::vector<uint8_t> message(IREE_NET_BULK_MESSAGE_HEADER_SIZE +
                               payload.size());
  message[0] = IREE_NET_BULK_MESSAGE_VERSION;
  message[1] = type;
  iree_unaligned_store_le_u64(message.data() + 8, transfer_id);
  iree_unaligned_store_le_u64(message.data() + 16, value);
  if (!payload.empty()) {
    std::memcpy(message.data() + IREE_NET_BULK_MESSAGE_HEADER_SIZE,
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

  iree_status_code_t ParseMessage(iree_const_byte_span_t message,
                                  iree_async_buffer_lease_t* lease = nullptr) {
    iree_status_t status =
        callbacks_.on_message(callbacks_.user_data, message, lease);
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return status_code;
  }

  iree_status_code_t DeliverMessage(
      iree_const_byte_span_t message,
      iree_async_buffer_lease_t* lease = nullptr) {
    iree_status_t status =
        callbacks_.on_message(callbacks_.user_data, message, lease);
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
  int start_count = 0;
  uint64_t start_transfer_id = 0;
  uint64_t start_total_length = 0;
  iree_status_code_t next_start_status = IREE_STATUS_OK;
  int data_count = 0;
  uint64_t data_transfer_id = 0;
  uint64_t data_offset = 0;
  std::vector<uint8_t> data_payload;
  iree_async_buffer_lease_t* data_lease = nullptr;
  iree_status_code_t next_data_status = IREE_STATUS_OK;
  int complete_count = 0;
  uint64_t complete_transfer_id = 0;
  iree_status_code_t next_complete_status = IREE_STATUS_OK;
  int abort_count = 0;
  uint64_t abort_transfer_id = 0;
  std::vector<uint8_t> abort_detail;
  iree_async_buffer_lease_t* abort_lease = nullptr;
  iree_status_code_t next_abort_status = IREE_STATUS_OK;
  int credit_count = 0;
  uint64_t credit_delta = 0;
  uint64_t available_credit_count = 0;
  iree_status_code_t next_credit_status = IREE_STATUS_OK;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;

  static iree_status_t OnStart(void* user_data, uint64_t transfer_id,
                               uint64_t total_length) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->start_count;
    self->start_transfer_id = transfer_id;
    self->start_total_length = total_length;
    return iree_status_from_code(self->next_start_status);
  }

  static iree_status_t OnData(void* user_data, uint64_t transfer_id,
                              uint64_t offset, iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->data_count;
    self->data_transfer_id = transfer_id;
    self->data_offset = offset;
    self->data_payload.assign(payload.data, payload.data + payload.data_length);
    self->data_lease = lease;
    return iree_status_from_code(self->next_data_status);
  }

  static iree_status_t OnComplete(void* user_data, uint64_t transfer_id) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->complete_count;
    self->complete_transfer_id = transfer_id;
    return iree_status_from_code(self->next_complete_status);
  }

  static iree_status_t OnAbort(void* user_data, uint64_t transfer_id,
                               iree_const_byte_span_t detail,
                               iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->abort_count;
    self->abort_transfer_id = transfer_id;
    self->abort_detail.assign(detail.data, detail.data + detail.data_length);
    self->abort_lease = lease;
    return iree_status_from_code(self->next_abort_status);
  }

  static iree_status_t OnCredit(void* user_data, uint64_t credit_delta,
                                uint64_t available_credit_count) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->credit_count;
    self->credit_delta = credit_delta;
    self->available_credit_count = available_credit_count;
    return iree_status_from_code(self->next_credit_status);
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<CallbackState*>(user_data);
    ++self->error_count;
    self->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  iree_net_bulk_channel_callbacks_t callbacks() {
    return {
        .on_start = OnStart,
        .on_data = OnData,
        .on_complete = OnComplete,
        .on_abort = OnAbort,
        .on_credit = OnCredit,
        .on_error = OnError,
        .user_data = this,
    };
  }
};

class BulkChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_net_bulk_channel_allocate(
        endpoint_.endpoint(), callback_state_.callbacks(),
        iree_allocator_system(), &channel_));
  }

  void TearDown() override { iree_net_bulk_channel_free(channel_); }

  void Attach() { iree_net_bulk_channel_attach(channel_); }

  void GrantLocalCredit(uint64_t credit_delta) {
    SendCompletion completion;
    IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(channel_, credit_delta,
                                                     completion.callback()));
    EXPECT_EQ(completion.count, 1);
  }

  void DeliverRemoteCredit(uint64_t credit_limit) {
    const std::vector<uint8_t> message =
        MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, credit_limit);
    IREE_ASSERT_OK(endpoint_.ParseMessage(
        iree_make_const_byte_span(message.data(), message.size())));
  }

  TestEndpoint endpoint_;
  CallbackState callback_state_;
  iree_net_bulk_channel_t* channel_ = nullptr;
};

TEST(BulkChannelAllocationTest, RequiresEndpointAndCallbacks) {
  iree_net_bulk_channel_t* channel = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_allocate(iree_net_message_endpoint_t{},
                                     iree_net_bulk_channel_callbacks_t{},
                                     iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);

  TestEndpoint endpoint;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_allocate(endpoint.endpoint(),
                                     iree_net_bulk_channel_callbacks_t{},
                                     iree_allocator_system(), &channel));
  EXPECT_EQ(channel, nullptr);
}

TEST_F(BulkChannelTest, DeliversTransferLifecycleMessages) {
  Attach();
  GrantLocalCredit(1);
  iree_async_buffer_lease_t lease = {};

  std::vector<uint8_t> message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 3, 100);
  EXPECT_EQ(endpoint_.ParseMessage(
                iree_make_const_byte_span(message.data(), message.size())),
            IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.start_count, 1);
  EXPECT_EQ(callback_state_.start_transfer_id, 3u);
  EXPECT_EQ(callback_state_.start_total_length, 100u);

  message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 3, 16, {'d', 'a', 't', 'a'});
  EXPECT_EQ(
      endpoint_.ParseMessage(
          iree_make_const_byte_span(message.data(), message.size()), &lease),
      IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.data_count, 1);
  EXPECT_EQ(callback_state_.data_transfer_id, 3u);
  EXPECT_EQ(callback_state_.data_offset, 16u);
  EXPECT_EQ(callback_state_.data_payload,
            (std::vector<uint8_t>{'d', 'a', 't', 'a'}));
  EXPECT_EQ(callback_state_.data_lease, &lease);

  message = MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_COMPLETE, 3, 0);
  EXPECT_EQ(endpoint_.ParseMessage(
                iree_make_const_byte_span(message.data(), message.size())),
            IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.complete_count, 1);
  EXPECT_EQ(callback_state_.complete_transfer_id, 3u);

  message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_ABORT, 5, 0, {'f', 'a', 'i', 'l'});
  EXPECT_EQ(
      endpoint_.ParseMessage(
          iree_make_const_byte_span(message.data(), message.size()), &lease),
      IREE_STATUS_OK);
  EXPECT_EQ(callback_state_.abort_count, 1);
  EXPECT_EQ(callback_state_.abort_transfer_id, 5u);
  EXPECT_EQ(callback_state_.abort_detail,
            (std::vector<uint8_t>{'f', 'a', 'i', 'l'}));
  EXPECT_EQ(callback_state_.abort_lease, &lease);
}

TEST_F(BulkChannelTest, CumulativeCreditIsMonotonic) {
  Attach();
  DeliverRemoteCredit(2);
  EXPECT_EQ(callback_state_.credit_count, 1);
  EXPECT_EQ(callback_state_.credit_delta, 2u);
  EXPECT_EQ(callback_state_.available_credit_count, 2u);
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel_), 2u);

  DeliverRemoteCredit(1);
  DeliverRemoteCredit(2);
  EXPECT_EQ(callback_state_.credit_count, 1);

  std::vector<uint8_t> payload = {'x'};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendCompletion first_completion;
  SendCompletion second_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, 1, 0, iree_async_span_list_make(&span, 1),
      first_completion.callback()));
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, 1, 1, iree_async_span_list_make(&span, 1),
      second_completion.callback()));
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel_), 0u);

  SendCompletion blocked_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, 1, 2, iree_async_span_list_make(&span, 1),
      blocked_completion.callback()));
  EXPECT_EQ(blocked_completion.count, 1);
  EXPECT_EQ(blocked_completion.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(blocked_completion.bytes_transferred, 0u);

  DeliverRemoteCredit(5);
  EXPECT_EQ(callback_state_.credit_count, 2);
  EXPECT_EQ(callback_state_.credit_delta, 3u);
  EXPECT_EQ(callback_state_.available_credit_count, 3u);
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel_), 3u);
}

TEST_F(BulkChannelTest, ReceiveDataCannotExceedPublishedCredit) {
  Attach();
  const std::vector<uint8_t> message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 1, 0, {'x'});

  EXPECT_EQ(endpoint_.DeliverMessage(
                iree_make_const_byte_span(message.data(), message.size())),
            IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(callback_state_.data_count, 0);
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_DATA_LOSS);
}

TEST_F(BulkChannelTest, CreditGrantIsAdmissionCoupledAndCumulative) {
  Attach();
  endpoint_.reject_send_code = IREE_STATUS_UNAVAILABLE;
  SendCompletion rejected_completion;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_bulk_channel_send_credit(
                            channel_, 2, rejected_completion.callback()));
  EXPECT_EQ(endpoint_.prefix_write_count, 0);
  EXPECT_EQ(rejected_completion.count, 0);

  endpoint_.reject_send_code = IREE_STATUS_OK;
  SendCompletion first_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(
      channel_, 2, first_completion.callback()));
  SendCompletion second_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_credit(
      channel_, 3, second_completion.callback()));
  ASSERT_EQ(endpoint_.sent_messages.size(), 2u);
  EXPECT_EQ(endpoint_.sent_messages[0],
            MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, 2));
  EXPECT_EQ(endpoint_.sent_messages[1],
            MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, 5));

  const std::vector<uint8_t> data_message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 1, 0, {'x'});
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(endpoint_.ParseMessage(iree_make_const_byte_span(
                  data_message.data(), data_message.size())),
              IREE_STATUS_OK);
  }
  EXPECT_EQ(endpoint_.ParseMessage(iree_make_const_byte_span(
                data_message.data(), data_message.size())),
            IREE_STATUS_DATA_LOSS);
}

TEST_F(BulkChannelTest, RejectedDataSendDoesNotConsumeCredit) {
  Attach();
  DeliverRemoteCredit(1);
  std::vector<uint8_t> payload = {'x'};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());

  endpoint_.reject_send_code = IREE_STATUS_UNAVAILABLE;
  SendCompletion rejected_completion;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_bulk_channel_send_data(
                            channel_, 1, 0, iree_async_span_list_make(&span, 1),
                            rejected_completion.callback()));
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel_), 1u);
  EXPECT_EQ(rejected_completion.count, 0);

  endpoint_.reject_send_code = IREE_STATUS_OK;
  SendCompletion accepted_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, 1, 0, iree_async_span_list_make(&span, 1),
      accepted_completion.callback()));
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel_), 0u);
  EXPECT_EQ(accepted_completion.count, 1);
}

TEST_F(BulkChannelTest, SendsHeadersAndBorrowsPayloadSpans) {
  Attach();
  DeliverRemoteCredit(1);

  SendCompletion start_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_start(channel_, 7, 9,
                                                  start_completion.callback()));
  EXPECT_EQ(endpoint_.sent_messages.back(),
            MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 7, 9));

  std::vector<uint8_t> first = {1, 2, 3};
  std::vector<uint8_t> second = {4, 5};
  iree_async_span_t spans[] = {
      iree_async_span_from_ptr(first.data(), first.size()),
      iree_async_span_from_ptr(second.data(), second.size()),
  };
  SendCompletion data_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_data(
      channel_, 7, 4, iree_async_span_list_make(spans, 2),
      data_completion.callback()));
  EXPECT_EQ(
      endpoint_.sent_messages.back(),
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 7, 4, {1, 2, 3, 4, 5}));
  EXPECT_EQ(endpoint_.last_borrowed_span_count, 2u);
  EXPECT_EQ(endpoint_.last_borrowed_span_pointers,
            (std::vector<const uint8_t*>{first.data(), second.data()}));

  SendCompletion complete_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_complete(
      channel_, 7, complete_completion.callback()));
  EXPECT_EQ(endpoint_.sent_messages.back(),
            MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_COMPLETE, 7, 0));

  SendCompletion abort_completion;
  IREE_ASSERT_OK(iree_net_bulk_channel_send_abort(
      channel_, 9, iree_async_span_list_make(spans, 2),
      abort_completion.callback()));
  EXPECT_EQ(
      endpoint_.sent_messages.back(),
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_ABORT, 9, 0, {1, 2, 3, 4, 5}));
  EXPECT_EQ(abort_completion.bytes_transferred,
            IREE_NET_BULK_MESSAGE_HEADER_SIZE + 5u);
}

TEST_F(BulkChannelTest, CallbackFailureConvergesThroughEndpointError) {
  Attach();
  callback_state_.next_start_status = IREE_STATUS_ABORTED;
  const std::vector<uint8_t> message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 1, 10);

  EXPECT_EQ(endpoint_.DeliverMessage(
                iree_make_const_byte_span(message.data(), message.size())),
            IREE_STATUS_ABORTED);
  EXPECT_EQ(callback_state_.start_count, 1);
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_ABORTED);
}

TEST_F(BulkChannelTest, RejectsMalformedMessages) {
  Attach();
  std::vector<std::vector<uint8_t>> invalid_messages;
  invalid_messages.push_back(std::vector<uint8_t>(23));

  std::vector<uint8_t> message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 1, 0);
  message[0] = 2;
  invalid_messages.push_back(message);
  message = MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 1, 0);
  message[2] = 1;
  invalid_messages.push_back(message);
  invalid_messages.push_back(MakeMessage(0xFF, 1, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 0, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_START, 1, 0, {'x'}));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 0, 0, {'x'}));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 1, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_DATA, 1, UINT64_MAX, {'x'}));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_COMPLETE, 0, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_COMPLETE, 1, 1));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_COMPLETE, 1, 0, {'x'}));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_ABORT, 0, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_ABORT, 1, 1));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 1, 1));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, 0));
  invalid_messages.push_back(
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, 1, {'x'}));

  for (const auto& invalid_message : invalid_messages) {
    EXPECT_EQ(endpoint_.ParseMessage(iree_make_const_byte_span(
                  invalid_message.data(), invalid_message.size())),
              IREE_STATUS_DATA_LOSS);
  }
}

TEST_F(BulkChannelTest, ValidatesSendBeforeEndpointAdmission) {
  Attach();
  SendCompletion completion;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_send_start(channel_, 0, 0, completion.callback()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_send_data(
          channel_, 1, 0, iree_async_span_list_empty(), completion.callback()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_send_complete(channel_, 0, completion.callback()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_send_abort(
          channel_, 0, iree_async_span_list_empty(), completion.callback()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bulk_channel_send_credit(channel_, 0, completion.callback()));
  EXPECT_EQ(endpoint_.send_count, 0);
}

TEST_F(BulkChannelTest, RejectsMessagesBeyondWireExtent) {
  Attach();
  iree_async_span_t span = iree_async_span_from_ptr(
      reinterpret_cast<void*>(uintptr_t{1}), UINT32_MAX);
  SendCompletion completion;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_bulk_channel_send_data(
                            channel_, 1, 0, iree_async_span_list_make(&span, 1),
                            completion.callback()));
  EXPECT_EQ(endpoint_.send_count, 0);
}

TEST_F(BulkChannelTest, AdmitsMaximumWireExtentToEndpoint) {
  Attach();
  endpoint_.reject_send_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  iree_async_span_t span =
      iree_async_span_from_ptr(reinterpret_cast<void*>(uintptr_t{1}),
                               UINT32_MAX - IREE_NET_BULK_MESSAGE_HEADER_SIZE);
  SendCompletion completion;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_bulk_channel_send_data(
                            channel_, 1, 0, iree_async_span_list_make(&span, 1),
                            completion.callback()));
  EXPECT_EQ(endpoint_.send_count, 1);
  EXPECT_EQ(completion.count, 0);
}

TEST_F(BulkChannelTest, QuerySendBudgetPassesThroughEndpoint) {
  const iree_net_carrier_send_budget_t budget =
      iree_net_bulk_channel_query_send_budget(channel_);
  EXPECT_EQ(budget.bytes, endpoint_.budget.bytes);
  EXPECT_EQ(budget.slots, endpoint_.budget.slots);
}

TEST_F(BulkChannelTest, TerminalErrorForwardsOwnership) {
  Attach();
  endpoint_.DeliverError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport failed"));
  EXPECT_EQ(callback_state_.error_count, 1);
  EXPECT_EQ(callback_state_.error_code, IREE_STATUS_UNAVAILABLE);
}

class ConcurrentEndpoint {
 public:
  iree_net_message_endpoint_t endpoint() {
    return {
        .self = this,
        .vtable = &vtable_,
    };
  }

  iree_status_code_t DeliverMessage(iree_const_byte_span_t message) {
    iree_status_t status =
        callbacks_.on_message(callbacks_.user_data, message, /*lease=*/nullptr);
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return status_code;
  }

 private:
  static ConcurrentEndpoint* Cast(void* self) {
    return static_cast<ConcurrentEndpoint*>(self);
  }

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    Cast(self)->callbacks_ = callbacks;
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
    (void)self;
    uint8_t header[IREE_NET_BULK_MESSAGE_HEADER_SIZE];
    iree_status_t status = params->generated_prefix.write(
        params->generated_prefix.user_data,
        iree_make_byte_span(header, sizeof(header)));
    if (!iree_status_is_ok(status)) {
      params->completion_callback.fn(params->completion_callback.user_data,
                                     status, 0);
      return iree_ok_status();
    }
    iree_host_size_t total_length = params->generated_prefix.length;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      total_length += params->data.values[i].length;
    }
    params->completion_callback.fn(params->completion_callback.user_data,
                                   iree_ok_status(), total_length);
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    (void)self;
    return {};
  }

  static const iree_net_message_endpoint_vtable_t vtable_;

  iree_net_message_endpoint_callbacks_t callbacks_ = {};
};

const iree_net_message_endpoint_vtable_t ConcurrentEndpoint::vtable_ = {
    .set_callbacks = ConcurrentEndpoint::SetCallbacks,
    .activate = ConcurrentEndpoint::Activate,
    .deactivate = ConcurrentEndpoint::Deactivate,
    .send = ConcurrentEndpoint::Send,
    .query_send_budget = ConcurrentEndpoint::QuerySendBudget,
};

struct ConcurrentCompletion {
  std::atomic<int>* success_count = nullptr;
  std::atomic<int>* resource_exhausted_count = nullptr;
  std::atomic<int>* unexpected_count = nullptr;

  static void Callback(void* user_data, iree_status_t status,
                       iree_host_size_t bytes_transferred) {
    auto* self = static_cast<ConcurrentCompletion*>(user_data);
    const iree_status_code_t status_code = iree_status_code(status);
    if (status_code == IREE_STATUS_OK &&
        bytes_transferred == IREE_NET_BULK_MESSAGE_HEADER_SIZE + 1u) {
      self->success_count->fetch_add(1, std::memory_order_relaxed);
    } else if (status_code == IREE_STATUS_RESOURCE_EXHAUSTED &&
               bytes_transferred == 0) {
      self->resource_exhausted_count->fetch_add(1, std::memory_order_relaxed);
    } else {
      self->unexpected_count->fetch_add(1, std::memory_order_relaxed);
    }
    iree_status_free(status);
  }

  iree_net_send_completion_callback_t callback() {
    return {
        .fn = Callback,
        .user_data = this,
    };
  }
};

TEST(BulkChannelConcurrencyTest, PeerCreditCompletesExactConcurrentDataCount) {
  ConcurrentEndpoint endpoint;
  CallbackState callbacks;
  iree_net_bulk_channel_t* channel = nullptr;
  IREE_ASSERT_OK(
      iree_net_bulk_channel_allocate(endpoint.endpoint(), callbacks.callbacks(),
                                     iree_allocator_system(), &channel));
  iree_net_bulk_channel_attach(channel);
  const std::vector<uint8_t> credit_message =
      MakeMessage(IREE_NET_BULK_MESSAGE_TYPE_CREDIT, 0, 512);
  ASSERT_EQ(endpoint.DeliverMessage(iree_make_const_byte_span(
                credit_message.data(), credit_message.size())),
            IREE_STATUS_OK);

  std::atomic<int> accepted_count{0};
  std::atomic<int> unexpected_submit_count{0};
  std::atomic<int> completion_success_count{0};
  std::atomic<int> completion_resource_exhausted_count{0};
  std::atomic<int> completion_unexpected_count{0};
  ConcurrentCompletion completion = {
      .success_count = &completion_success_count,
      .resource_exhausted_count = &completion_resource_exhausted_count,
      .unexpected_count = &completion_unexpected_count,
  };
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < 8; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      uint8_t payload = static_cast<uint8_t>(thread_index);
      iree_async_span_t span = iree_async_span_from_ptr(&payload, 1);
      for (int i = 0; i < 128; ++i) {
        iree_status_t status = iree_net_bulk_channel_send_data(
            channel, 1, static_cast<uint64_t>(thread_index * 128 + i),
            iree_async_span_list_make(&span, 1), completion.callback());
        const iree_status_code_t status_code = iree_status_code(status);
        if (status_code == IREE_STATUS_OK) {
          accepted_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          unexpected_submit_count.fetch_add(1, std::memory_order_relaxed);
        }
        iree_status_free(status);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(accepted_count.load(std::memory_order_relaxed), 1024);
  EXPECT_EQ(unexpected_submit_count.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(completion_success_count.load(std::memory_order_relaxed), 512);
  EXPECT_EQ(completion_resource_exhausted_count.load(std::memory_order_relaxed),
            512);
  EXPECT_EQ(completion_unexpected_count.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(iree_net_bulk_channel_remote_credit_count(channel), 0u);
  iree_net_bulk_channel_free(channel);
}

}  // namespace
