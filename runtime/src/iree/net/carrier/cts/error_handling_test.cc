// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Error handling tests for carrier implementations.
//
// Tests error propagation and graceful degradation behaviors. Note that some
// error handling behaviors (like sticky errors) are implementation-specific
// and tested in carrier-specific test files.

#include <atomic>
#include <cstring>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class ErrorHandlingTest : public CarrierTestBase<> {};

// Recv handler that returns an error after receiving data.
struct ErrorRecvContext {
  std::atomic<int> call_count{0};
  iree_status_code_t error_code = IREE_STATUS_INTERNAL;
};

static iree_status_t ErrorRecvHandler(void* user_data, iree_async_span_t data,
                                      iree_async_buffer_lease_t* lease) {
  auto* ctx = static_cast<ErrorRecvContext*>(user_data);
  ctx->call_count.fetch_add(1, std::memory_order_release);
  return iree_make_status(ctx->error_code, "test error from recv handler");
}

// When recv handler returns an error, the carrier should not crash or leak.
// The exact error propagation behavior is carrier-specific; this test only
// verifies the carrier remains in a valid state.
TEST_P(ErrorHandlingTest, RecvHandlerErrorDoesNotCrash) {
  ErrorRecvContext error_ctx;
  iree_net_carrier_recv_handler_t error_handler = {ErrorRecvHandler,
                                                   &error_ctx};

  ActivateBoth(MakeNullRecvHandler(), error_handler);

  // Send to server which has error handler.
  const char* msg = "trigger error";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  IREE_ASSERT_OK(SendWhenReady(client_, &params));

  // Poll to process any async completions.
  PollUntil([&] { return error_ctx.call_count.load() > 0; });

  // Verify handler was actually called.
  EXPECT_GT(error_ctx.call_count.load(), 0);

  // Carrier should still be in a valid state (not crashed).
  iree_net_carrier_state_t state = iree_net_carrier_state(client_);
  EXPECT_TRUE(state == IREE_NET_CARRIER_STATE_ACTIVE ||
              state == IREE_NET_CARRIER_STATE_DRAINING ||
              state == IREE_NET_CARRIER_STATE_DEACTIVATED);
}

// Send after local deactivation must fail synchronously. Peer deactivation is
// not a transport synchronization edge: a stream or shared-memory sender may
// successfully hand bytes to the transport before observing peer closure.
TEST_P(ErrorHandlingTest, SendAfterDeactivateFails) {
  ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());
  DeactivateAndDrain(client_, proactor_);

  const char* msg = "to deactivated peer";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);
  iree_status_t status = iree_net_carrier_send(client_, &params);
  EXPECT_TRUE(iree_status_is_failed_precondition(status));
  iree_status_free(status);
}

// After shutdown, send must not deliver data to the peer. The carrier can
// reject the send synchronously (FAILED_PRECONDITION) or defer the error to
// the completion callback. Either way, the server must not see the data.
TEST_P(ErrorHandlingTest, SendAfterShutdownFails) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);
  SendCompletionTracker client_completions;
  client_->callback = client_completions.AsCallback();

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  IREE_ASSERT_OK(iree_net_carrier_shutdown(client_));

  const char* msg = "after shutdown";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);
  iree::Status status(iree_net_carrier_send(client_, &params));
  if (!status.ok()) {
    // Synchronous rejection (e.g. SHM's EOS marker prevents ring writes).
    EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  } else {
    // Async carriers (TCP) may accept the submission but fail at completion.
    ASSERT_TRUE(
        PollUntil([&] { return client_completions.call_count.load() == 1; }));
    iree::Status completion_status(client_completions.ConsumeStatus());
    EXPECT_FALSE(completion_status.ok());
  }

  // Deactivation is the transport barrier proving no accepted receive remains
  // able to reach the peer handler.
  DeactivateAndDrain(client_, proactor_);
  DeactivateAndDrain(server_, proactor_);

  // The server must not have received the post-shutdown data.
  EXPECT_EQ(server_capture.total_bytes.load(), 0u);
}

// Multiple rapid sends to error handler should not cause double-free or crash.
TEST_P(ErrorHandlingTest, RapidSendsToErrorHandler) {
  ErrorRecvContext error_ctx;
  iree_net_carrier_recv_handler_t error_handler = {ErrorRecvHandler,
                                                   &error_ctx};

  ActivateBoth(MakeNullRecvHandler(), error_handler);

  const char* msg = "X";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  // Send multiple times rapidly. Once the error handler's failure begins
  // transport teardown, later sends may be rejected synchronously.
  int accepted_count = 0;
  for (int i = 0; i < 10; ++i) {
    iree_status_t status = iree_net_carrier_send(client_, &params);
    if (iree_status_is_ok(status)) {
      ++accepted_count;
    } else {
      iree_status_free(status);
      break;
    }
  }
  ASSERT_GT(accepted_count, 0);

  // Poll to drain completions.
  PollUntil([&] { return error_ctx.call_count.load() >= 1; });

  // Should not have crashed - carrier still valid.
  iree_net_carrier_state_t state = iree_net_carrier_state(client_);
  EXPECT_TRUE(state == IREE_NET_CARRIER_STATE_ACTIVE ||
              state == IREE_NET_CARRIER_STATE_DRAINING ||
              state == IREE_NET_CARRIER_STATE_DEACTIVATED);
}

CTS_REGISTER_TEST_SUITE(ErrorHandlingTest);

}  // namespace
}  // namespace iree::net::carrier::cts
