// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback-specific carrier tests.
//
// Tests loopback-specific behaviors that cannot be generalized across
// transports. Common carrier tests live in cts/.

#include "iree/net/carrier/loopback/carrier.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class LoopbackCarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
  }

  void TearDown() override {
    if (client_) {
      DeactivateAndDrain(client_);
      iree_net_carrier_release(client_);
    }
    if (server_) {
      DeactivateAndDrain(server_);
      iree_net_carrier_release(server_);
    }
    iree_async_proactor_release(proactor_);
  }

  void DeactivateAndDrain(iree_net_carrier_t* carrier) {
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_DEACTIVATED) return;
    if (state != IREE_NET_CARRIER_STATE_CREATED &&
        state != IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_abort(iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "an existing carrier deactivation must be joined through its "
          "original callback"));
    }
    std::atomic<bool> deactivated{false};
    iree_status_t status = iree_net_carrier_deactivate(
        carrier,
        [](void* user_data) {
          static_cast<std::atomic<bool>*>(user_data)->store(
              true, std::memory_order_release);
        },
        &deactivated);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    while (!deactivated.load(std::memory_order_acquire)) {
      iree_host_size_t completed = 0;
      status = iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                        &completed);
      if (!iree_status_is_ok(status)) iree_status_abort(status);
    }
    EXPECT_EQ(IREE_NET_CARRIER_STATE_DEACTIVATED,
              iree_net_carrier_state(carrier));
  }

  void PollUntil(std::function<bool()> condition) {
    while (!condition()) {
      iree_host_size_t completed = 0;
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), &completed));
    }
  }

  static iree_status_t NullRecvHandler(void* /*user_data*/,
                                       iree_async_span_t /*data*/,
                                       iree_async_buffer_lease_t* /*lease*/) {
    return iree_ok_status();
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_carrier_t* client_ = nullptr;
  iree_net_carrier_t* server_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

// Loopback delivery is synchronous within the NOP completion callback: the
// sender's NOP fires, checks if the peer is alive, and delivers data inline.
// When the peer departs between send() and the NOP completion, delivery is
// skipped and the send completion callback must report an error.
//
// This is loopback-specific because TCP and SHM have decoupled send/delivery:
// data reaches the transport layer (socket buffer / shared memory ring) before
// the peer link is cleared, so the send genuinely succeeds. The OS handles
// peer departure detection separately for those transports.
TEST_F(LoopbackCarrierTest, InFlightSendReportsErrorOnPeerDeparture) {
  // Track send completions.
  struct CompletionState {
    std::atomic<int> count{0};
    std::atomic<bool> had_error{false};
  } completion;

  iree_net_carrier_callback_t callback = {
      [](void* user_data, iree_net_carrier_completion_kind_t /*kind*/,
         uint64_t /*operation_user_data*/, iree_status_t status,
         iree_host_size_t /*bytes_transferred*/,
         iree_async_buffer_lease_t* /*recv_lease*/) {
        auto* state = static_cast<CompletionState*>(user_data);
        if (!iree_status_is_ok(status)) {
          state->had_error.store(true, std::memory_order_release);
        }
        iree_status_ignore(status);
        state->count.fetch_add(1, std::memory_order_release);
      },
      &completion};

  IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
      proactor_, callback, iree_allocator_system(), &client_, &server_));

  iree_net_carrier_recv_handler_t null_handler = {NullRecvHandler, nullptr};
  iree_net_carrier_set_recv_handler(client_, null_handler);
  iree_net_carrier_set_recv_handler(server_, null_handler);
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));
  IREE_ASSERT_OK(iree_net_carrier_activate(server_));

  // Send a message (queues a NOP, returns immediately).
  const char* message = "in flight when peer departs";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(message), strlen(message));
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  // Deactivate the server (peer) BEFORE polling. The NOP is queued but has not
  // fired yet. Deactivation detaches the server, so delivery is rejected.
  std::atomic<bool> deactivated{false};
  IREE_ASSERT_OK(iree_net_carrier_deactivate(
      server_, [](void* ud) { *static_cast<std::atomic<bool>*>(ud) = true; },
      &deactivated));

  // Poll until the send completion fires.
  PollUntil([&] { return completion.count.load() > 0; });
  ASSERT_GT(completion.count.load(), 0)
      << "Send completion callback never fired";

  // The completion must report an error: the data was never delivered.
  EXPECT_TRUE(completion.had_error.load())
      << "In-flight send should report error when peer departs before "
         "delivery";

  // Ensure deactivation completes before TearDown.
  PollUntil([&] { return deactivated.load(); });
}

TEST_F(LoopbackCarrierTest, SendBeforePeerActivation) {
  struct ReceiveState {
    std::atomic<int> count{0};
    std::atomic<bool> payload_matches{false};
  } receive_state;

  IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
      proactor_, /*callback=*/{}, iree_allocator_system(), &client_, &server_));

  iree_net_carrier_set_recv_handler(
      client_, {/*fn=*/NullRecvHandler, /*user_data=*/nullptr});
  iree_net_carrier_set_recv_handler(
      server_, {/*fn=*/[](void* user_data, iree_async_span_t data,
                          iree_async_buffer_lease_t* /*lease*/) {
                  auto* state = static_cast<ReceiveState*>(user_data);
                  constexpr char kExpected[] = "bootstrap";
                  state->payload_matches.store(
                      data.length == sizeof(kExpected) - 1 &&
                          memcmp(iree_async_span_ptr(data), kExpected,
                                 sizeof(kExpected) - 1) == 0,
                      std::memory_order_release);
                  state->count.fetch_add(1, std::memory_order_release);
                  return iree_ok_status();
                },
                /*user_data=*/&receive_state});
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));

  const char message[] = "bootstrap";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(message), sizeof(message) - 1);
  iree_net_send_params_t params = {};
  params.data = {/*values=*/&span, /*count=*/1};
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  IREE_ASSERT_OK(iree_net_carrier_activate(server_));
  PollUntil(
      [&] { return receive_state.count.load(std::memory_order_acquire); });

  EXPECT_TRUE(receive_state.payload_matches.load(std::memory_order_acquire));
}

TEST_F(LoopbackCarrierTest, DeactivationWaitsForAcceptedReceive) {
  struct ReceiveState {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool release = false;
  } receive_state;

  IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
      proactor_, /*callback=*/{}, iree_allocator_system(), &client_, &server_));

  iree_net_carrier_set_recv_handler(
      client_, {/*fn=*/NullRecvHandler, /*user_data=*/nullptr});
  iree_net_carrier_set_recv_handler(
      server_, {/*fn=*/[](void* user_data, iree_async_span_t /*data*/,
                          iree_async_buffer_lease_t* /*lease*/) {
                  auto* state = static_cast<ReceiveState*>(user_data);
                  std::unique_lock<std::mutex> lock(state->mutex);
                  state->entered = true;
                  state->condition.notify_all();
                  state->condition.wait(lock, [&] { return state->release; });
                  return iree_ok_status();
                },
                /*user_data=*/&receive_state});
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));
  IREE_ASSERT_OK(iree_net_carrier_activate(server_));

  const char* message = "receive blocks during deactivation";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(message), strlen(message));
  iree_net_send_params_t params = {};
  params.data = {/*values=*/&span, /*count=*/1};
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  std::thread poll_thread([&] {
    IREE_EXPECT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                            /*out_completed_count=*/nullptr));
  });

  {
    std::unique_lock<std::mutex> lock(receive_state.mutex);
    receive_state.condition.wait(lock, [&] { return receive_state.entered; });
  }

  std::atomic<bool> deactivated{false};
  IREE_ASSERT_OK(iree_net_carrier_deactivate(
      server_,
      [](void* user_data) {
        static_cast<std::atomic<bool>*>(user_data)->store(
            true, std::memory_order_release);
      },
      &deactivated));
  EXPECT_FALSE(deactivated.load(std::memory_order_acquire));

  {
    std::lock_guard<std::mutex> lock(receive_state.mutex);
    receive_state.release = true;
  }
  receive_state.condition.notify_all();
  poll_thread.join();

  EXPECT_TRUE(deactivated.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace iree
