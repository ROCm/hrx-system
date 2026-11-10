// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/resource.h>
#include <sys/socket.h>

#include <array>
#include <vector>

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/posix/api.h"
#include "iree/net/carrier/stream/carrier_test_suite.h"

namespace iree::net::testing {

namespace {

class ScopedLockedMemoryLimit {
 public:
  ScopedLockedMemoryLimit() = default;

  ~ScopedLockedMemoryLimit() {
    if (active_) {
      EXPECT_EQ(setrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
    }
  }

  void Exhaust() {
    ASSERT_EQ(getrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
    struct rlimit exhausted_limit = original_limit_;
    exhausted_limit.rlim_cur = 0;
    ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &exhausted_limit), 0);
    active_ = true;
  }

 private:
  struct rlimit original_limit_ = {};
  bool active_ = false;
};

}  // namespace

void StreamCarrierTest::ConstrainSendBuffer(iree_async_socket_t* socket) {
  int send_buffer_size = 1;
  ASSERT_EQ(setsockopt(socket->primitive.value.fd, SOL_SOCKET, SO_SNDBUF,
                       &send_buffer_size, sizeof(send_buffer_size)),
            0);
}

TEST_P(StreamCarrierTest, ZeroCopyResourceRefusalFallsBackToCopiedSend) {
  if (GetParam().socket_type != IREE_ASYNC_SOCKET_TYPE_TCP ||
      GetParam().zero_copy_min_send_size != 0) {
    GTEST_SKIP() << "requires the IPv4 zero-copy stream configuration";
  }
  ASSERT_NO_FATAL_FAILURE(CreateCarrierPair());
  if (!iree_any_bit_set(iree_net_carrier_capabilities(client_carrier_),
                        IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX)) {
    GTEST_SKIP() << "platform lacks native zero-copy sends";
  }

  ScopedLockedMemoryLimit locked_memory_limit;
  ASSERT_NO_FATAL_FAILURE(locked_memory_limit.Exhaust());

  std::vector<uint8_t> payload(64 * 1024, 0xA5);
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult send_result;
  send_result.is_polling = &is_polling_;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  PollUntil([&] {
    return send_result.completion_count == 1 &&
           server_context_.received_data.size() == payload.size();
  });

  EXPECT_EQ(send_result.status_code, IREE_STATUS_OK);
  EXPECT_EQ(send_result.bytes_transferred, payload.size());
  EXPECT_EQ(server_context_.received_data, payload);
  EXPECT_EQ(client_context_.error_count, 0);
  EXPECT_EQ(server_context_.error_count, 0);
}

TEST_P(StreamCarrierTest,
       InitialSubmissionFailureCompletesAcceptedSendOnProactor) {
  iree_async_proactor_release(proactor_);
  proactor_ = nullptr;
  iree_async_proactor_options_t proactor_options =
      iree_async_proactor_options_default();
  proactor_options.max_concurrent_operations = 1;
  IREE_ASSERT_OK(iree_async_proactor_create_posix(
      proactor_options, iree_allocator_system(), &proactor_));
  ASSERT_NO_FATAL_FAILURE(CreateCarrierPair(/*max_send_operations=*/1));

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::array<iree_async_notification_signal_operation_t, 64> signal_operations =
      {};
  std::array<AsyncOperationResult, 64> signal_results;
  iree_host_size_t accepted_signal_count = 0;
  for (; accepted_signal_count < signal_operations.size();
       ++accepted_signal_count) {
    iree_async_notification_signal_operation_t* signal_operation =
        &signal_operations[accepted_signal_count];
    iree_async_operation_initialize(
        &signal_operation->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
        IREE_ASYNC_OPERATION_FLAG_NONE, AsyncOperationCompleted,
        &signal_results[accepted_signal_count]);
    signal_operation->notification = notification;
    signal_operation->wake_count = 1;
    iree_status_t status =
        iree_async_proactor_submit_one(proactor_, &signal_operation->base);
    if (!iree_status_is_ok(status)) {
      EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
      iree_status_free(status);
      break;
    }
  }
  ASSERT_GT(accepted_signal_count, 0u);
  ASSERT_LT(accepted_signal_count, signal_operations.size());

  std::array<uint8_t, 8> payload = {};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  SendResult send_result;
  send_result.is_polling = &is_polling_;
  iree_net_send_params_t params = {
      iree_net_send_prefix_empty(),
      iree_async_span_list_make(&span, 1),
      {SendCompleted, &send_result},
  };
  IREE_ASSERT_OK(iree_net_carrier_send(client_carrier_, &params));
  EXPECT_EQ(send_result.completion_count, 0);
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_carrier_).slots, 0u);

  PollUntil([&] {
    if (send_result.completion_count != 1) {
      return false;
    }
    for (iree_host_size_t i = 0; i < accepted_signal_count; ++i) {
      if (!signal_results[i].completed) {
        return false;
      }
    }
    return true;
  });
  EXPECT_EQ(send_result.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(send_result.bytes_transferred, 0u);
  EXPECT_EQ(client_context_.error_count, 1);
  EXPECT_EQ(client_context_.error_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_net_carrier_query_send_budget(client_carrier_).slots, 0u);
  for (iree_host_size_t i = 0; i < accepted_signal_count; ++i) {
    EXPECT_EQ(signal_results[i].status_code, IREE_STATUS_OK);
  }
  iree_async_notification_release(notification);
}

}  // namespace iree::net::testing
