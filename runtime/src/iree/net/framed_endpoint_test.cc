// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/framed_endpoint.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/loopback/carrier.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct ReceivedMessage {
  // Original payload view into the moved lease, without the wire header.
  iree_const_byte_span_t payload;
  // Storage kept alive until the test releases it after endpoint teardown.
  iree_async_buffer_lease_t lease;
};

struct SendCompletion {
  // Number of terminal callbacks for this accepted send.
  int count = 0;
  // Terminal result without ownership of the callback's status object.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  // Logical payload bytes completed, excluding the transport header.
  iree_host_size_t bytes_transferred = 0;
  // Optional next operation submitted directly from terminal completion.
  std::function<void()> then;

  static void Complete(void* user_data, iree_status_t status,
                       iree_host_size_t bytes_transferred) {
    auto* self = static_cast<SendCompletion*>(user_data);
    ++self->count;
    self->code = iree_status_code(status);
    self->bytes_transferred = bytes_transferred;
    iree_status_free(status);
    if (self->then) {
      self->then();
    }
  }

  iree_net_send_completion_callback_t callback() { return {Complete, this}; }
};

struct PrefixWriter {
  // Endpoint queried from inside the writer to check lock-free invocation.
  iree_net_message_endpoint_t endpoint = {};
  // Prefix source needed only until send returns.
  std::string source;
  // Number of synchronous invocations.
  int count = 0;
  // Result returned by the writer after initializing the prefix.
  iree_status_code_t result = IREE_STATUS_OK;

  static iree_status_t Write(void* user_data, iree_byte_span_t target) {
    auto* self = static_cast<PrefixWriter*>(user_data);
    ++self->count;
    EXPECT_EQ(target.data_length, self->source.size());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(target.data) %
                  IREE_NET_SEND_PREFIX_ALIGNMENT,
              0u);
    EXPECT_EQ(iree_net_message_endpoint_query_send_budget(self->endpoint).slots,
              0u);
    memcpy(target.data, self->source.data(), target.data_length);
    return self->result == IREE_STATUS_OK
               ? iree_ok_status()
               : iree_make_status(self->result, "prefix generation failed");
  }

  iree_net_send_prefix_t prefix() { return {source.size(), Write, this}; }
};

class FramedEndpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    auto options = iree_net_loopback_carrier_options_default();
    options.max_send_operations = 1;
    iree_net_carrier_t* carriers[2] = {};
    IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
        proactor_, proactor_, &options, iree_allocator_system(), &carriers[0],
        &carriers[1]));
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_net_framed_endpoint_allocate(
          carriers[i], proactor_, options.max_send_operations,
          /*connection_barrier=*/nullptr, iree_allocator_system(),
          &owners_[i]));
      endpoints_[i] = iree_net_framed_endpoint_as_message_endpoint(owners_[i]);
      iree_net_message_endpoint_set_callbacks(endpoints_[i],
                                              {OnMessage, OnError, this});
      IREE_ASSERT_OK(iree_net_message_endpoint_activate(endpoints_[i]));
    }
  }

  void TearDown() override {
    Shutdown();
    for (auto& message : received_) {
      iree_async_buffer_lease_release(&message.lease);
    }
  }

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<FramedEndpointTest*>(user_data);
    EXPECT_NE(lease, nullptr);
    self->received_.push_back({message, *lease});
    *lease = {};
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<FramedEndpointTest*>(user_data);
    if (self->closing_) {
      EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNAVAILABLE);
      iree_status_free(status);
    } else {
      IREE_EXPECT_OK(status);
    }
  }

  void PollUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), /*out_completed_count=*/nullptr));
    }
  }

  void Shutdown() {
    if (!proactor_) {
      return;
    }
    closing_ = true;
    int drained_count = 0;
    for (auto endpoint : endpoints_) {
      IREE_ASSERT_OK(iree_net_message_endpoint_deactivate(
          endpoint, [](void* count) { ++*static_cast<int*>(count); },
          &drained_count));
    }
    PollUntil([&] { return drained_count == 2; });
    for (auto* owner : owners_) {
      iree_net_framed_endpoint_free(owner);
    }
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  std::string ReceivedText(size_t index) {
    const auto& payload = received_[index].payload;
    return std::string(reinterpret_cast<const char*>(payload.data),
                       payload.data_length);
  }

  // Sole polling owner for both independently activated carrier stacks.
  iree_async_proactor_t* proactor_ = nullptr;
  // Owned framing stacks, drained and freed before retained message storage.
  iree_net_framed_endpoint_t* owners_[2] = {};
  // Borrowed message views into the stacks.
  iree_net_message_endpoint_t endpoints_[2] = {};
  // Moved receive storage, held across subsequent messages and teardown.
  std::vector<ReceivedMessage> received_;
  // Permits the peer EOF produced during coordinated teardown.
  bool closing_ = false;
};

TEST_F(FramedEndpointTest, GeneratedPrefixAndPayloadOutliveSourcesAndEndpoint) {
  PrefixWriter writer;
  writer.endpoint = endpoints_[0];
  writer.source.assign(65537, 'p');
  std::string payloads[2] = {std::string(4097, 'a'), std::string(8193, 'b')};
  const std::string expected = writer.source + payloads[0] + payloads[1];
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(payloads[0].data(), payloads[0].size()),
      iree_async_span_from_ptr(payloads[1].data(), payloads[1].size()),
  };
  SendCompletion completion;
  completion.then = [&] {
    for (auto& payload : payloads) {
      std::fill(payload.begin(), payload.end(), 'x');
    }
  };
  iree_net_message_endpoint_send_params_t params = {
      writer.prefix(), {spans, IREE_ARRAYSIZE(spans)}, completion.callback()};
  IREE_ASSERT_OK(iree_net_message_endpoint_send(endpoints_[0], &params));
  EXPECT_EQ(writer.count, 1);
  EXPECT_EQ(completion.count, 0);
  std::fill(writer.source.begin(), writer.source.end(), 'x');
  PollUntil([&] { return completion.count == 1 && received_.size() == 1; });
  EXPECT_EQ(completion.code, IREE_STATUS_OK);
  EXPECT_EQ(completion.bytes_transferred, expected.size());
  Shutdown();
  EXPECT_EQ(ReceivedText(0), expected);
}

TEST_F(FramedEndpointTest,
       CompletionReplenishesOneSlotWithAllReceivesRetained) {
  constexpr size_t kMessageCount = 32;
  std::vector<SendCompletion> completions(kMessageCount);
  std::function<void(size_t)> submit = [&](size_t index) {
    EXPECT_EQ(iree_net_message_endpoint_query_send_budget(endpoints_[0]).slots,
              1u);
    const std::string text = "message-" + std::to_string(index);
    iree_net_message_endpoint_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(text.data(), text.size())),
        .completion_callback = completions[index].callback(),
    };
    IREE_EXPECT_OK(iree_net_message_endpoint_send(endpoints_[0], &params));
  };
  for (size_t i = 0; i + 1 < kMessageCount; ++i) {
    completions[i].then = [&, i] { submit(i + 1); };
  }
  submit(0);
  EXPECT_EQ(iree_net_message_endpoint_query_send_budget(endpoints_[0]).slots,
            0u);
  PollUntil([&] {
    return completions.back().count == 1 && received_.size() == kMessageCount;
  });
  Shutdown();
  for (size_t i = 0; i < kMessageCount; ++i) {
    EXPECT_EQ(completions[i].count, 1);
    EXPECT_EQ(completions[i].code, IREE_STATUS_OK);
    const std::string expected = "message-" + std::to_string(i);
    EXPECT_EQ(completions[i].bytes_transferred, expected.size());
    EXPECT_EQ(ReceivedText(i), expected);
  }
}

TEST_F(FramedEndpointTest, PrefixFailureCompletesAsynchronouslyAndReturnsSlot) {
  PrefixWriter writer;
  writer.endpoint = endpoints_[0];
  writer.source = "not published";
  writer.result = IREE_STATUS_ABORTED;
  SendCompletion failed;
  SendCompletion succeeded;
  failed.then = [&] {
    EXPECT_EQ(iree_net_message_endpoint_query_send_budget(endpoints_[0]).slots,
              1u);
    writer.source = "next message";
    writer.result = IREE_STATUS_OK;
    iree_net_message_endpoint_send_params_t params = {
        writer.prefix(), {}, succeeded.callback()};
    IREE_EXPECT_OK(iree_net_message_endpoint_send(endpoints_[0], &params));
  };
  iree_net_message_endpoint_send_params_t params = {
      writer.prefix(), {}, failed.callback()};
  IREE_ASSERT_OK(iree_net_message_endpoint_send(endpoints_[0], &params));
  EXPECT_EQ(writer.count, 1);
  EXPECT_EQ(failed.count, 0);
  PollUntil([&] { return succeeded.count == 1 && received_.size() == 1; });
  EXPECT_EQ(failed.count, 1);
  EXPECT_EQ(failed.code, IREE_STATUS_ABORTED);
  EXPECT_EQ(failed.bytes_transferred, 0u);
  EXPECT_EQ(succeeded.code, IREE_STATUS_OK);
  EXPECT_EQ(succeeded.bytes_transferred, writer.source.size());
  EXPECT_EQ(writer.count, 2);
  EXPECT_EQ(ReceivedText(0), writer.source);
}

}  // namespace
}  // namespace iree
