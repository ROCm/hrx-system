// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/connection.h"
#include "iree/net/cts/transport_backend.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

struct Completion {
  // Number of terminal callbacks, including successive operation reuse.
  int count = 0;
  // Last terminal status code.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  // Body bytes returned by the latest callback.
  iree_host_size_t bytes = 0;
  // Optional work submitted from within the terminal callback.
  std::function<void()> then;

  static void Complete(void* user_data, iree_status_t status,
                       iree_host_size_t bytes) {
    auto* self = static_cast<Completion*>(user_data);
    ++self->count;
    self->code = iree_status_code(status);
    self->bytes = bytes;
    iree_status_free(status);
    if (self->then) {
      self->then();
    }
  }
};

class ReceiveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    const auto& backend = GetTransportBackend();
    iree_status_t status = backend.create_factory(allocator(), &factory_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << backend.name << " transport unavailable";
    }
    IREE_ASSERT_OK(status);
    std::string bind_address;
    IREE_ASSERT_OK(backend.make_bind_address(&bind_address));
    IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
        factory_,
        iree_make_string_view(bind_address.data(), bind_address.size()),
        proactor_, nullptr, {Connected, &connections_[1]}, allocator(),
        &listener_));
    std::array<char, 1024> address_storage;
    iree_string_view_t address;
    IREE_ASSERT_OK(iree_net_listener_query_bound_address(
        listener_, address_storage.size(), address_storage.data(), &address));
    iree_net_transport_connect_operation_initialize(&connect_operation_);
    connect_initialized_ = true;
    IREE_ASSERT_OK(iree_net_transport_factory_connect(
        factory_, address, proactor_, nullptr, {Connected, &connections_[0]},
        &connect_operation_));
    PollUntil([&] { return connections_[0] && connections_[1]; });
    for (int side = 0; side < 2; ++side) {
      IREE_ASSERT_OK(iree_net_connection_open_endpoint(
          connections_[side], {EndpointReady, &endpoints_[side]}));
    }
    PollUntil([&] { return endpoints_[0].self && endpoints_[1].self; });
    for (auto endpoint : endpoints_) {
      iree_net_message_endpoint_set_callbacks(endpoint,
                                              {OnMessage, OnError, this});
      IREE_ASSERT_OK(iree_net_message_endpoint_activate(endpoint));
    }
    // Complete a real round trip before relying on remote ACTIVE admission.
    const std::string hello = "bootstrap";
    Completion sent;
    Send(hello, "", &sent);
    PollUntil([&] { return sent.count && messages_.size() == 1; });
    ASSERT_EQ(sent.code, IREE_STATUS_OK);
    messages_.clear();
  }

  void TearDown() override {
    Drain();
    for (auto* channel : queue_channels_) {
      iree_net_queue_channel_free(channel);
    }
    if (listener_) {
      bool stopped = false;
      IREE_EXPECT_OK(iree_net_listener_stop(
          listener_,
          {[](void* state) { *static_cast<bool*>(state) = true; }, &stopped}));
      PollUntil([&] { return stopped; });
      iree_net_listener_free(listener_);
    }
    if (connect_initialized_) {
      iree_net_transport_connect_operation_deinitialize(&connect_operation_);
    }
    iree_net_transport_factory_release(factory_);
    iree_async_proactor_release(proactor_);
  }

  iree_allocator_t allocator() { return {this, Allocate}; }

  static iree_status_t Allocate(void* user_data,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<ReceiveTest*>(user_data);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      const auto* allocation =
          static_cast<const iree_allocator_alloc_params_t*>(params);
      self->maximum_allocation_ =
          std::max(self->maximum_allocation_, allocation->byte_length);
    }
    const auto system = iree_allocator_system();
    return system.ctl(system.self, command, params, inout_pointer);
  }

  void PollUntil(const std::function<bool()>& condition) {
    while (!condition()) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  void Drain() {
    int pending = 0;
    for (auto* connection : connections_) {
      if (!connection) {
        continue;
      }
      ++pending;
      iree_net_connection_deactivate(
          connection,
          {[](void* state) { --*static_cast<int*>(state); }, &pending});
    }
    PollUntil([&] { return pending == 0; });
    for (auto*& connection : connections_) {
      iree_net_connection_release(connection);
      connection = nullptr;
    }
  }

  static void Connected(void* user_data, iree_status_t status,
                        iree_net_connection_t* connection) {
    IREE_EXPECT_OK(status);
    *static_cast<iree_net_connection_t**>(user_data) = connection;
  }
  static void EndpointReady(void* user_data, iree_status_t status,
                            iree_net_message_endpoint_t endpoint) {
    IREE_EXPECT_OK(status);
    *static_cast<iree_net_message_endpoint_t*>(user_data) = endpoint;
  }
  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<ReceiveTest*>(user_data);
    self->messages_.emplace_back(reinterpret_cast<const char*>(message.data),
                                 message.data_length);
    return iree_ok_status();
  }
  static void OnError(void* user_data, iree_status_t status) {
    auto* self = static_cast<ReceiveTest*>(user_data);
    self->error_code_ = iree_status_code(status);
    iree_status_free(status);
  }

  iree_status_t Post(iree_net_message_receive_operation_t* operation,
                     const std::string& prefix, iree_byte_span_t target,
                     Completion* completion) {
    const iree_net_message_endpoint_receive_params_t params = {
        iree_make_const_byte_span(prefix.data(), prefix.size()),
        target,
        {Completion::Complete, completion}};
    return iree_net_message_endpoint_receive(endpoints_[1], operation, &params);
  }

  void Send(const std::string& prefix, const std::string& body,
            Completion* completion) {
    auto span =
        iree_async_span_from_ptr(const_cast<char*>(body.data()), body.size());
    const iree_net_message_endpoint_send_params_t params = {
        iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(prefix.data(), prefix.size())),
        iree_async_span_list_make(&span, 1),
        {Completion::Complete, completion}};
    IREE_ASSERT_OK(iree_net_message_endpoint_send(endpoints_[0], &params));
  }

  // Native polling owner shared by both independent connection endpoints.
  iree_async_proactor_t* proactor_ = nullptr;
  // Backend factory and listener, retained until both connections drain.
  iree_net_transport_factory_t* factory_ = nullptr;
  // Listening service whose stop receipt precedes destruction.
  iree_net_listener_t* listener_ = nullptr;
  // Cold connection attempt storage.
  iree_net_transport_connect_operation_t connect_operation_ = {};
  // Whether the connection attempt was initialized before setup exited.
  bool connect_initialized_ = false;
  // Two owned native connections.
  iree_net_connection_t* connections_[2] = {};
  // Borrowed endpoint views in corresponding connection namespaces.
  iree_net_message_endpoint_t endpoints_[2] = {};
  // Optional protocol adapters, freed after their endpoint callbacks retire.
  iree_net_queue_channel_t* queue_channels_[2] = {};
  // Ordinary unmatched messages delivered through the existing callback.
  std::vector<std::string> messages_;
  // Largest allocation requested by the transport allocator in an interval.
  iree_host_size_t maximum_allocation_ = 0;
  // Most recent terminal endpoint error.
  iree_status_code_t error_code_ = IREE_STATUS_OK;
};

TEST_F(ReceiveTest, ReusesChangingSlabRangesAndPreservesOrdinaryMessages) {
  constexpr size_t kMaximumLength = 40976 * 512;
  std::vector<uint8_t> slab(kMaximumLength + 128, 0xCC);
  iree_net_message_receive_operation_t operation = {};
  Completion received;
  for (size_t tokens : {1, 16, 256, 512, 1}) {
    const size_t length = 40976 * tokens;
    const size_t offset = 17 + tokens % 31;
    std::fill(slab.begin(), slab.end(), 0xCC);
    const std::string prefix = "stage001";
    const std::string body(length, static_cast<char>(tokens % 127));
    const int initial_count = received.count;
    maximum_allocation_ = 0;
    IREE_ASSERT_OK(Post(&operation, prefix,
                        iree_make_byte_span(slab.data() + offset, length),
                        &received));
    Completion sent;
    Send(prefix, body, &sent);
    PollUntil([&] { return sent.count && received.count > initial_count; });
    ASSERT_EQ(sent.code, IREE_STATUS_OK);
    ASSERT_EQ(received.code, IREE_STATUS_OK);
    EXPECT_EQ(received.bytes, length);
    EXPECT_LT(maximum_allocation_, length);
    EXPECT_EQ(memcmp(slab.data() + offset, body.data(), length), 0);
    EXPECT_TRUE(std::all_of(slab.begin(), slab.begin() + offset,
                            [](uint8_t byte) { return byte == 0xCC; }));
    EXPECT_TRUE(std::all_of(slab.begin() + offset + length, slab.end(),
                            [](uint8_t byte) { return byte == 0xCC; }));
    EXPECT_TRUE(messages_.empty());
  }
  Completion ordinary_sent;
  Send("control", "", &ordinary_sent);
  PollUntil([&] { return ordinary_sent.count && messages_.size() == 1; });
  EXPECT_EQ(messages_[0], "control");
}

TEST_F(ReceiveTest, CompletionCanImmediatelyRepostSameOperation) {
  iree_net_message_receive_operation_t operation = {};
  std::array<uint8_t, 4> target = {};
  Completion received;
  received.then = [&] {
    if (received.count == 1) {
      IREE_ASSERT_OK(Post(&operation, "next",
                          iree_make_byte_span(target.data(), 4), &received));
    }
  };
  IREE_ASSERT_OK(Post(&operation, "once", iree_make_byte_span(target.data(), 4),
                      &received));
  for (const std::string prefix : {"once", "next"}) {
    Completion sent;
    const std::string body = "body";
    const int expected = received.count + 1;
    Send(prefix, body, &sent);
    PollUntil([&] { return sent.count && received.count == expected; });
    EXPECT_EQ(received.code, IREE_STATUS_OK);
  }
  EXPECT_EQ(std::string(target.begin(), target.end()), "body");
}

TEST_F(ReceiveTest, PostsMatchOutOfOrderAndRejectDuplicateOwnership) {
  iree_net_message_receive_operation_t operations[2] = {};
  std::array<uint8_t, 4> targets[2] = {};
  Completion received[2];
  for (int i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Post(&operations[i], i ? "two!" : "one!",
                        iree_make_byte_span(targets[i].data(), 4),
                        &received[i]));
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      Post(&operations[0], "busy", iree_make_byte_span(targets[0].data(), 4),
           &received[0]));
  iree_net_message_receive_operation_t duplicate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      Post(&duplicate, "one!", iree_make_byte_span(targets[0].data(), 4),
           &received[0]));
  for (int i : {1, 0}) {
    Completion sent;
    const std::string body = i ? "2222" : "1111";
    Send(i ? "two!" : "one!", body, &sent);
    PollUntil([&] { return sent.count && received[i].count; });
    EXPECT_EQ(received[i].code, IREE_STATUS_OK);
    EXPECT_EQ(std::string(targets[i].begin(), targets[i].end()), body);
  }
}

TEST_F(ReceiveTest, WrongExtentFailsBeforeTargetWrite) {
  iree_net_message_receive_operation_t operation = {};
  std::array<uint8_t, 8> target;
  target.fill(0xCC);
  Completion received;
  IREE_ASSERT_OK(Post(&operation, "size", iree_make_byte_span(target.data(), 8),
                      &received));
  Completion sent;
  const std::string body = "too short";
  Send("size", body, &sent);
  PollUntil([&] { return received.count && sent.count; });
  EXPECT_EQ(received.code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(received.bytes, 0u);
  EXPECT_TRUE(std::all_of(target.begin(), target.end(),
                          [](uint8_t byte) { return byte == 0xCC; }));
  EXPECT_TRUE(messages_.empty());
}

TEST_F(ReceiveTest, QueueFrontiersSelectExactBodyWithoutCommandDelivery) {
  for (int side = 0; side < 2; ++side) {
    const iree_net_queue_channel_callbacks_t callbacks = {
        .on_command =
            [](void*, uint32_t, const iree_net_queue_frontier_view_t*,
               const iree_net_queue_frontier_view_t*, iree_const_byte_span_t,
               iree_async_buffer_lease_t*) {
              ADD_FAILURE() << "posted command reached ordinary delivery";
              return iree_ok_status();
            },
        .on_advance =
            [](void*, const iree_net_queue_frontier_view_t*,
               iree_const_byte_span_t,
               iree_async_buffer_lease_t*) { return iree_ok_status(); },
        .on_error = OnError,
        .user_data = this,
    };
    IREE_ASSERT_OK(iree_net_queue_channel_allocate(endpoints_[side], callbacks,
                                                   iree_allocator_system(),
                                                   &queue_channels_[side]));
    iree_net_queue_channel_attach(queue_channels_[side]);
  }
  iree_async_single_frontier_t wait = {.entry_count = 1, .entries = {{12, 3}}};
  iree_async_single_frontier_t signal = {
      .entry_count = 1,
      .entries = {{19, 7}},
  };
  iree_net_message_receive_operation_t operation = {};
  std::array<uint8_t, 40976> target = {};
  Completion received;
  IREE_ASYNC_FIXED_FRONTIER_TYPE(WaitFrontier, 3);
  const WaitFrontier oversized_wait = {
      .entry_count = 3,
      .entries = {{12, 3}, {13, 4}, {14, 5}},
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_queue_channel_receive_command(
          queue_channels_[1], 13,
          reinterpret_cast<const iree_async_frontier_t*>(&oversized_wait),
          iree_async_single_frontier_as_const_frontier(&signal), &operation,
          iree_make_byte_span(target.data(), target.size()),
          {Completion::Complete, &received}));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_queue_channel_receive_command(
          queue_channels_[1], 13,
          iree_async_single_frontier_as_const_frontier(&wait),
          iree_async_single_frontier_as_const_frontier(&signal), &operation,
          iree_byte_span_empty(), {Completion::Complete, &received}));
  IREE_ASSERT_OK(iree_net_queue_channel_receive_command(
      queue_channels_[1], 13,
      iree_async_single_frontier_as_const_frontier(&wait),
      iree_async_single_frontier_as_const_frontier(&signal), &operation,
      iree_make_byte_span(target.data(), target.size()),
      {Completion::Complete, &received}));
  const std::string body(target.size(), 'X');
  auto span =
      iree_async_span_from_ptr(const_cast<char*>(body.data()), body.size());
  Completion sent;
  const iree_net_queue_channel_send_params_t params = {
      .wait_frontier_count = 1,
      .signal_frontier_count = 1,
      .build =
          [](void*, const iree_net_queue_message_builder_t* builder) {
            iree_net_queue_frontier_builder_set(&builder->wait_frontier, 0,
                                                {12, 3});
            iree_net_queue_frontier_builder_set(&builder->signal_frontier, 0,
                                                {19, 7});
            return iree_ok_status();
          },
      .payload = iree_async_span_list_make(&span, 1),
      .completion_callback = {Completion::Complete, &sent},
  };
  IREE_ASSERT_OK(
      iree_net_queue_channel_send_command(queue_channels_[0], 13, &params));
  PollUntil([&] { return received.count && sent.count; });
  EXPECT_EQ(received.code, IREE_STATUS_OK);
  EXPECT_EQ(received.bytes, target.size());
  EXPECT_EQ(memcmp(target.data(), body.data(), body.size()), 0);
}

TEST_F(ReceiveTest, PrefixOnlyMessageCompletesEmptyDestination) {
  iree_net_message_receive_operation_t operation = {};
  Completion received;
  IREE_ASSERT_OK(Post(&operation, "empty", iree_byte_span_empty(), &received));
  Completion sent;
  Send("empty", "", &sent);
  PollUntil([&] { return received.count && sent.count; });
  EXPECT_EQ(received.code, IREE_STATUS_OK);
  EXPECT_EQ(received.bytes, 0u);
  EXPECT_TRUE(messages_.empty());
}

TEST_F(ReceiveTest, ConnectionDrainReturnsUnmatchedDestinations) {
  iree_net_message_receive_operation_t operation = {};
  uint8_t target = 0xCC;
  Completion received;
  IREE_ASSERT_OK(
      Post(&operation, "wait", iree_make_byte_span(&target, 1), &received));
  Drain();
  EXPECT_EQ(received.count, 1);
  EXPECT_EQ(received.code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(received.bytes, 0u);
  EXPECT_EQ(target, 0xCC);
}

}  // namespace
}  // namespace iree::net::cts
