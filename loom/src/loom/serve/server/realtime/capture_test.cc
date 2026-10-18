// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/capture.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/worker_gate.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/websocket/handshake.h"

namespace {

class CaptureSpan {
 public:
  ~CaptureSpan() {
    if (value.capture) {
      loom_serve_realtime_capture_release(&value);
    }
  }
  // Stable loan record retained through explicit release or scope exit.
  loom_serve_realtime_capture_span_t value = {};
};

class RealtimeCaptureTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    auto status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        2, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto transport_options = loom_serve_http_server_options_default();
    transport_options.connection_capacity = 2;
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &transport_options, iree_allocator_system(), &server_));
    loom_serve_http_server_set_observer(server_, Wake());
    auto listen = loom_serve_http_listener_options_default();
    listen.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listen, iree_allocator_system(), &listener_));
    const auto* address = loom_serve_http_listener_address(listener_);
    const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
    client_ = socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(client_, 0);
    ASSERT_EQ(connect(client_, native, address->length), 0);
    Write(
        "GET /v1/realtime HTTP/1.1\r\nHost: local\r\n"
        "Connection: Upgrade\r\nUpgrade: websocket\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n");
    const loom_serve_http_request_t* request = nullptr;
    Await([&] {
      connection_ = loom_serve_http_server_take_request(server_, &request);
      return connection_ != nullptr;
    });
    char storage[512];
    iree_string_builder_t response;
    iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                                &response);
    IREE_ASSERT_OK(
        loom_serve_websocket_handshake(request, iree_string_view_list_empty(),
                                       iree_string_view_empty(), &response));
    IREE_ASSERT_OK(loom_serve_websocket_upgrade(
        connection_, loom_serve_websocket_limits_default(),
        iree_allocator_system(), &socket_));
    IREE_ASSERT_OK(loom_serve_http_connection_send(
        connection_, iree_string_builder_view(&response)));
    std::string received(response.size, '\0');
    ASSERT_EQ(recv(client_, received.data(), received.size(), MSG_WAITALL),
              received.size());
    ASSERT_EQ(received, std::string(response.buffer, response.size));
  }

  void TearDown() override {
    if (capture_) {
      loom_serve_realtime_capture_close(capture_);
      Await([&] { return !loom_serve_realtime_capture_pending(capture_); });
      loom_serve_realtime_capture_free(capture_);
    }
    if (connection_) {
      loom_serve_http_connection_abort(connection_);
    }
    if (client_ >= 0) {
      EXPECT_EQ(close(client_), 0);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
    iree_task_executor_release(executor_);
    iree_notification_deinitialize(&notification_);
  }

  loom_serve_wake_callback_t Wake() {
    return {[](void* self) {
              iree_notification_post(static_cast<iree_notification_t*>(self),
                                     IREE_ALL_WAITERS);
            },
            &notification_};
  }

  template <typename Done>
  void Await(Done done) {
    while (true) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      if (capture_) {
        loom_serve_realtime_capture_advance(capture_, &progress);
      }
      if (done()) {
        iree_notification_cancel_wait(&notification_);
        return;
      }
      if (progress) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  void Open(size_t capacity) {
    IREE_ASSERT_OK(loom_serve_realtime_capture_allocate(
        executor_, socket_, capacity, Wake(), iree_allocator_system(),
        &capture_));
  }

  void Write(const std::string& bytes) {
    ASSERT_EQ(send(client_, bytes.data(), bytes.size(), MSG_NOSIGNAL),
              bytes.size());
  }

  loom_serve_websocket_message_t* Send(const std::string& text,
                                       iree_const_byte_span_t* payload) {
    std::string frame(1, '\x81');
    if (text.size() < 126) {
      frame += static_cast<char>(0x80 | text.size());
    } else {
      frame += '\xfe';
      frame += static_cast<char>(text.size() >> 8);
      frame += static_cast<char>(text.size());
    }
    frame += std::string(4, '\0');
    frame += text;
    Write(frame);
    loom_serve_websocket_message_t* message = nullptr;
    loom_serve_websocket_message_type_t type;
    Await([&] {
      message = loom_serve_websocket_take(socket_, &type, payload);
      return message != nullptr;
    });
    EXPECT_EQ(type, LOOM_SERVE_WEBSOCKET_TEXT);
    return message;
  }

  void Append(const std::string& encoded) {
    iree_const_byte_span_t payload;
    auto* message = Send(encoded, &payload);
    auto status = loom_serve_realtime_capture_append(
        capture_, message,
        iree_make_string_view(reinterpret_cast<const char*>(payload.data),
                              payload.data_length));
    if (!iree_status_is_ok(status)) {
      loom_serve_websocket_message_release(socket_, message);
    }
    IREE_ASSERT_OK(status);
  }

  void CheckRing(size_t capacity, size_t frames) {
    ASSERT_NO_FATAL_FAILURE(Open(capacity));
    std::vector<uint8_t> expected(frames * 2);
    for (size_t i = 0; i < frames; ++i) {
      iree_unaligned_store_le_u16(expected.data() + 2 * i,
                                  static_cast<uint16_t>(i * 107 + 3));
    }
    std::string encoded(iree_base64_encoded_size(expected.size()), '\0');
    iree_host_size_t length = 0;
    IREE_ASSERT_OK(iree_base64_encode(
        iree_make_const_byte_span(expected.data(), expected.size()),
        iree_make_mutable_string_view(encoded.data(), encoded.size()),
        &length));
    encoded.resize(length);
    ASSERT_NO_FATAL_FAILURE(Append(encoded));
    IREE_ASSERT_OK(loom_serve_realtime_capture_finish(capture_));
    Await(
        [&] { return loom_serve_realtime_capture_end(capture_) == capacity; });
    CaptureSpan first, second;
    loom_serve_realtime_capture_borrow(capture_, 0, 3, &first.value);
    loom_serve_realtime_capture_borrow(capture_, 3, capacity, &second.value);
    loom_serve_realtime_capture_discard_before(capture_, capacity);
    EXPECT_EQ(memcmp(second.value.samples.data, expected.data() + 6,
                     second.value.samples.data_length),
              0);
    loom_serve_realtime_capture_release(&second.value);
    bool progress = false;
    loom_serve_realtime_capture_advance(capture_, &progress);
    EXPECT_FALSE(progress);
    EXPECT_EQ(loom_serve_realtime_capture_end(capture_), capacity);
    EXPECT_EQ(memcmp(first.value.samples.data, expected.data(), 6), 0);
    loom_serve_realtime_capture_release(&first.value);
    uint64_t position = capacity;
    while (position < frames) {
      Await(
          [&] { return loom_serve_realtime_capture_end(capture_) > position; });
      CaptureSpan span;
      loom_serve_realtime_capture_borrow(capture_, position, capacity,
                                         &span.value);
      ASSERT_GT(span.value.samples.data_length, 0u);
      EXPECT_EQ(memcmp(span.value.samples.data, expected.data() + position * 2,
                       span.value.samples.data_length),
                0);
      position += span.value.samples.data_length / 2;
      loom_serve_realtime_capture_discard_before(capture_, position);
    }
    Await([&] { return loom_serve_realtime_capture_is_complete(capture_); });
    EXPECT_EQ(loom_serve_realtime_capture_end(capture_), frames);
    IREE_ASSERT_OK(loom_serve_realtime_capture_status(capture_));
  }

  // Shared asynchronous CPU executor, retained through all release callbacks.
  iree_task_executor_t* executor_ = nullptr;
  // Real transport proactors, independent of numerical worker availability.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Explicit transport/CPU publication readiness.
  iree_notification_t notification_;
  // HTTP owner of the upgraded connection and accepted receipts.
  loom_serve_http_server_t* server_ = nullptr;
  // Ephemeral local TCP listener.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Retained claim, released after capture and loan retirement.
  loom_serve_http_connection_t* connection_ = nullptr;
  // Connection-owned WebSocket transport.
  loom_serve_websocket_t* socket_ = nullptr;
  // Client socket; half-close leaves the server claim alive.
  int client_ = -1;
  // Production capture under test, including its real worker/decoder.
  loom_serve_realtime_capture_t* capture_ = nullptr;
};

TEST_P(RealtimeCaptureTest, WrapAndOutOfOrderLoanRelease) { CheckRing(7, 20); }

TEST_P(RealtimeCaptureTest, LargeMessageUsesFiniteBlocksAndBoundedStorage) {
  CheckRing(8192, 16387);
}

TEST_P(RealtimeCaptureTest, CloseRetainsLoanAfterDecodeCompletes) {
  ASSERT_NO_FATAL_FAILURE(Open(7));
  ASSERT_NO_FATAL_FAILURE(Append("AQAC\\u0041AMA"));
  IREE_ASSERT_OK(loom_serve_realtime_capture_finish(capture_));
  Await([&] { return loom_serve_realtime_capture_is_complete(capture_); });
  CaptureSpan span;
  loom_serve_realtime_capture_borrow(capture_, 0, 7, &span.value);
  loom_serve_realtime_capture_close(capture_);
  bool progress = false;
  loom_serve_realtime_capture_advance(capture_, &progress);
  EXPECT_TRUE(loom_serve_realtime_capture_pending(capture_));
  EXPECT_EQ(iree_unaligned_load_le_u16(span.value.samples.data + 4), 3u);
  loom_serve_realtime_capture_release(&span.value);
  EXPECT_FALSE(loom_serve_realtime_capture_pending(capture_));
}

TEST_P(RealtimeCaptureTest, ClearRetainsDecodeThroughControlAndDisconnect) {
  ASSERT_NO_FATAL_FAILURE(Open(7));
  ASSERT_NO_FATAL_FAILURE(Append("AAAAAA=="));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int i = 0; i < GetParam(); ++i) {
    gates.push_back(
        std::make_unique<loom_serve::testing::WorkerGate>(executor_));
  }
  bool progress = false;
  loom_serve_realtime_capture_advance(capture_, &progress);
  ASSERT_TRUE(progress);
  loom_serve_realtime_capture_close(capture_);
  loom_serve_realtime_capture_advance(capture_, &progress);
  EXPECT_TRUE(loom_serve_realtime_capture_pending(capture_));
  iree_const_byte_span_t payload;
  auto* control = Send("later control", &payload);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(payload.data),
                        payload.data_length),
            "later control");
  loom_serve_websocket_message_release(socket_, control);
  ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
  Await([&] { return loom_serve_http_connection_failed(connection_); });
  EXPECT_TRUE(loom_serve_realtime_capture_pending(capture_));
  gates.clear();
  Await([&] { return !loom_serve_realtime_capture_pending(capture_); });
  EXPECT_EQ(loom_serve_realtime_capture_end(capture_), 0u);
}

TEST_P(RealtimeCaptureTest, MalformedEncodedInputFailsBeforePublication) {
  for (const char* encoded : {"AA==", "!!!!!!==", "AAAAA"}) {
    SCOPED_TRACE(encoded);
    ASSERT_NO_FATAL_FAILURE(Open(7));
    ASSERT_NO_FATAL_FAILURE(Append(encoded));
    Await([&] {
      auto status = loom_serve_realtime_capture_status(capture_);
      const bool failed = !iree_status_is_ok(status);
      iree_status_ignore(status);
      return failed;
    });
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_realtime_capture_status(capture_));
    EXPECT_EQ(loom_serve_realtime_capture_end(capture_), 0u);
    loom_serve_realtime_capture_close(capture_);
    Await([&] { return !loom_serve_realtime_capture_pending(capture_); });
    loom_serve_realtime_capture_free(capture_);
    capture_ = nullptr;
  }
}

INSTANTIATE_TEST_SUITE_P(Workers, RealtimeCaptureTest, ::testing::Values(1, 8));

}  // namespace
