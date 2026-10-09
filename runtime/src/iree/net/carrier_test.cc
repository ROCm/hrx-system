// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

struct MockCarrier {
  iree_net_carrier_t base;
  uint8_t prefix_storage[64];
  int query_budget_count = 0;
  int send_count = 0;
  int error_count = 0;
  iree_status_code_t error_code = IREE_STATUS_OK;
  iree_host_size_t prefix_length = 0;
  iree_net_send_completion_callback_t pending_completion = {0};

  static void Destroy(iree_net_carrier_t* carrier) {
    IREE_ASSERT(false && "stack carrier must not be released");
  }

  static iree_status_t Activate(iree_net_carrier_t* carrier) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_ACTIVE);
    return iree_ok_status();
  }

  static void Deactivate(iree_net_carrier_t* carrier,
                         iree_net_carrier_deactivate_callback_fn_t callback,
                         void* user_data) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_DEACTIVATED);
    callback(user_data);
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(
      iree_net_carrier_t* carrier) {
    MockCarrier* mock = reinterpret_cast<MockCarrier*>(carrier);
    ++mock->query_budget_count;
    return {.bytes = 4096, .slots = 4};
  }

  static iree_status_t Send(iree_net_carrier_t* carrier,
                            const iree_net_send_params_t* params) {
    MockCarrier* mock = reinterpret_cast<MockCarrier*>(carrier);
    ++mock->send_count;
    mock->prefix_length = params->generated_prefix.length;
    if (params->generated_prefix.length > sizeof(mock->prefix_storage)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock prefix storage exceeded");
    }
    if (params->generated_prefix.length > 0) {
      iree_status_t status = params->generated_prefix.write(
          params->generated_prefix.user_data,
          iree_make_byte_span(mock->prefix_storage,
                              params->generated_prefix.length));
      if (!iree_status_is_ok(status)) {
        params->completion_callback.fn(params->completion_callback.user_data,
                                       status, 0);
        return iree_ok_status();
      }
    }
    mock->pending_completion = params->completion_callback;
    return iree_ok_status();
  }

  static iree_status_t Shutdown(iree_net_carrier_t* carrier) {
    return iree_ok_status();
  }

  static iree_status_t HandleReceive(void* user_data, iree_async_span_t data,
                                     iree_async_buffer_lease_t* lease) {
    return iree_ok_status();
  }

  static void HandleError(void* user_data, iree_status_t status) {
    MockCarrier* mock = static_cast<MockCarrier*>(user_data);
    ++mock->error_count;
    mock->error_code = iree_status_code(status);
    iree_status_free(status);
  }

  void Initialize();
  void Deinitialize() { iree_net_carrier_deinitialize(&base); }

  static const iree_net_carrier_vtable_t kVtable;
};

const iree_net_carrier_vtable_t MockCarrier::kVtable = {
    .destroy = MockCarrier::Destroy,
    .activate = MockCarrier::Activate,
    .deactivate = MockCarrier::Deactivate,
    .query_send_budget = MockCarrier::QuerySendBudget,
    .send = MockCarrier::Send,
    .shutdown = MockCarrier::Shutdown,
};

void MockCarrier::Initialize() {
  iree_net_carrier_initialize(&kVtable, IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                              /*max_send_spans=*/8, iree_allocator_system(),
                              &base);
  IREE_ASSERT_OK(iree_net_carrier_set_handlers(
      &base, {.on_receive = MockCarrier::HandleReceive,
              .on_error = MockCarrier::HandleError,
              .user_data = this}));
}

struct SendCompletion {
  int count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
  iree_host_size_t bytes_transferred = 0;

  static void Handle(void* user_data, iree_status_t status,
                     iree_host_size_t bytes_transferred) {
    SendCompletion* completion = static_cast<SendCompletion*>(user_data);
    ++completion->count;
    completion->status_code = iree_status_code(status);
    completion->bytes_transferred = bytes_transferred;
    iree_status_free(status);
  }
};

class CarrierTest : public ::testing::Test {
 protected:
  void SetUp() override { carrier_.Initialize(); }
  void TearDown() override { carrier_.Deinitialize(); }

  MockCarrier carrier_;
};

TEST_F(CarrierTest, SendValidatesSpanListBeforeSubmission) {
  SendCompletion completion;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = SendCompletion::Handle,
              .user_data = &completion,
          },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(&carrier_.base, &params));

  params.generated_prefix = {
      .length = 1,
      .write = nullptr,
      .user_data = nullptr,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(&carrier_.base, &params));

  params.generated_prefix = {
      .length = 0,
      .write = iree_net_send_prefix_copy,
      .user_data = nullptr,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(&carrier_.base, &params));

  params.generated_prefix = iree_net_send_prefix_empty();

  params.data = {.values = nullptr, .count = 1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(&carrier_.base, &params));

  iree_async_span_t too_many_spans[9] = {};
  params.data = iree_async_span_list_make(too_many_spans, 9);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_send(&carrier_.base, &params));

  iree_async_span_t empty_span = iree_async_span_empty();
  params.data = iree_async_span_list_make(&empty_span, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(&carrier_.base, &params));

  uint8_t byte = 0;
  iree_async_span_t overflowing_spans[2] = {
      iree_async_span_from_ptr(&byte, IREE_HOST_SIZE_MAX),
      iree_async_span_from_ptr(&byte, 1),
  };
  params.data = iree_async_span_list_make(overflowing_spans, 2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_net_carrier_send(&carrier_.base, &params));
  EXPECT_EQ(carrier_.send_count, 0);
}

TEST_F(CarrierTest, AcceptedSendHasTerminalCompletion) {
  uint8_t payload[16] = {};
  iree_async_span_t span = iree_async_span_from_ptr(payload, sizeof(payload));
  SendCompletion completion;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_empty(),
      .data = iree_async_span_list_make(&span, 1),
      .completion_callback =
          {
              .fn = SendCompletion::Handle,
              .user_data = &completion,
          },
  };
  IREE_ASSERT_OK(iree_net_carrier_send(&carrier_.base, &params));
  EXPECT_EQ(carrier_.send_count, 1);
  EXPECT_EQ(completion.count, 0);

  carrier_.pending_completion.fn(carrier_.pending_completion.user_data,
                                 iree_ok_status(), sizeof(payload));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(completion.bytes_transferred, sizeof(payload));
}

TEST_F(CarrierTest, GeneratedPrefixHasTerminalCompletion) {
  uint8_t prefix[32];
  memset(prefix, 0xA5, sizeof(prefix));
  SendCompletion completion;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(prefix, sizeof(prefix))),
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = SendCompletion::Handle,
              .user_data = &completion,
          },
  };
  IREE_ASSERT_OK(iree_net_carrier_send(&carrier_.base, &params));
  EXPECT_EQ(carrier_.send_count, 1);
  EXPECT_EQ(carrier_.prefix_length, sizeof(prefix));
  EXPECT_EQ(memcmp(carrier_.prefix_storage, prefix, sizeof(prefix)), 0);
  EXPECT_EQ(completion.count, 0);

  carrier_.pending_completion.fn(carrier_.pending_completion.user_data,
                                 iree_ok_status(), sizeof(prefix));
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(completion.bytes_transferred, sizeof(prefix));
}

TEST_F(CarrierTest, GeneratedPrefixFailureCompletesAcceptedSend) {
  SendCompletion completion;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(nullptr, 1)),
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = SendCompletion::Handle,
              .user_data = &completion,
          },
  };
  IREE_ASSERT_OK(iree_net_carrier_send(&carrier_.base, &params));
  EXPECT_EQ(carrier_.send_count, 1);
  EXPECT_EQ(completion.count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(completion.bytes_transferred, 0u);
}

TEST_F(CarrierTest, TerminalErrorStopsAdmissionAndBudget) {
  EXPECT_TRUE(iree_net_carrier_report_terminal_error(
      &carrier_.base,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost")));
  EXPECT_EQ(carrier_.error_count, 1);
  EXPECT_EQ(carrier_.error_code, IREE_STATUS_UNAVAILABLE);

  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(&carrier_.base);
  EXPECT_EQ(budget.bytes, 0u);
  EXPECT_EQ(budget.slots, 0u);
  EXPECT_EQ(carrier_.query_budget_count, 0);

  uint8_t prefix = 0;
  SendCompletion completion;
  iree_net_send_params_t params = {
      .generated_prefix = iree_net_send_prefix_from_bytes(
          iree_make_const_byte_span(&prefix, 1)),
      .data = iree_async_span_list_empty(),
      .completion_callback =
          {
              .fn = SendCompletion::Handle,
              .user_data = &completion,
          },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_carrier_send(&carrier_.base, &params));
  EXPECT_EQ(carrier_.send_count, 0);
}

}  // namespace
}  // namespace net
}  // namespace iree
