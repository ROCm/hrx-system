// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/transport/websocket/connection.h"

#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/worker_gate.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/websocket/handshake.h"

namespace {

struct loom_serve_websocket_message_deleter_t {
  // Borrowed protocol owner retained by the test's HTTP claim.
  loom_serve_websocket_t* socket;
  void operator()(loom_serve_websocket_message_t* message) const {
    loom_serve_websocket_message_release(socket, message);
  }
};
using Message = std::unique_ptr<loom_serve_websocket_message_t,
                                loom_serve_websocket_message_deleter_t>;

std::string ClientFrame(uint8_t type, const std::string& payload,
                        bool final = true) {
  std::string bytes(1, static_cast<char>((final ? 0x80 : 0) | type));
  if (payload.size() < 126) {
    bytes.push_back(static_cast<char>(0x80 | payload.size()));
  } else {
    const size_t width = payload.size() <= 65535 ? 2 : 8;
    bytes.push_back(static_cast<char>(0x80 | (width == 2 ? 126 : 127)));
    for (size_t i = 0; i < width; ++i) {
      bytes.push_back(static_cast<char>(static_cast<uint64_t>(payload.size()) >>
                                        (8 * (width - i - 1))));
    }
  }
  constexpr uint8_t mask[] = {0x37, 0xFA, 0x21, 0x3D};
  bytes.append(reinterpret_cast<const char*>(mask), sizeof(mask));
  for (size_t i = 0; i < payload.size(); ++i) {
    bytes.push_back(payload[i] ^ mask[i % 4]);
  }
  return bytes;
}

std::string Copy(iree_const_byte_span_t bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data),
                     bytes.data_length);
}

class WebSocketConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        2, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto options = loom_serve_http_server_options_default();
    options.connection_capacity = 2;
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &options, iree_allocator_system(), &server_));
    auto listen = loom_serve_http_listener_options_default();
    listen.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listen, iree_allocator_system(), &listener_));
  }

  void TearDown() override {
    if (connection_) {
      loom_serve_http_connection_abort(connection_);
    }
    if (client_ >= 0) {
      EXPECT_EQ(close(client_), 0);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
  }

  void Open(loom_serve_websocket_limits_t limits =
                loom_serve_websocket_limits_default(),
            bool half_close = false) {
    const auto* address = loom_serve_http_listener_address(listener_);
    const auto* native = reinterpret_cast<const sockaddr*>(address->storage);
    client_ = socket(native->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
    ASSERT_GE(client_, 0);
    ASSERT_EQ(connect(client_, native, address->length), 0);
    ASSERT_TRUE(
        Write("GET /stream HTTP/1.1\r\nHost: local\r\n"
              "Connection: Upgrade\r\nUpgrade: websocket\r\n"
              "Sec-WebSocket-Version: 13\r\n"
              "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"));
    const loom_serve_http_request_t* request = nullptr;
    auto* notification = loom_serve_http_server_notification(server_);
    while (!connection_) {
      const auto token = iree_notification_prepare_wait(notification);
      connection_ = loom_serve_http_server_take_request(server_, &request);
      if (connection_) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    char buffer[512];
    iree_string_builder_t response;
    iree_string_builder_initialize_with_storage(buffer, sizeof(buffer),
                                                &response);
    IREE_ASSERT_OK(
        loom_serve_websocket_handshake(request, iree_string_view_list_empty(),
                                       iree_string_view_empty(), &response));
    if (half_close) {
      ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
    }
    IREE_ASSERT_OK(loom_serve_websocket_upgrade(
        connection_, limits, iree_allocator_system(), &socket_));
    if (half_close) {
      return;
    }
    const auto bytes = iree_string_builder_view(&response);
    IREE_ASSERT_OK(loom_serve_http_connection_send(connection_, bytes));
    EXPECT_EQ(ReadBytes(bytes.size), std::string(bytes.data, bytes.size));
  }

  ::testing::AssertionResult Write(const std::string& bytes) {
    size_t position = 0;
    while (position < bytes.size()) {
      const ssize_t count = send(client_, bytes.data() + position,
                                 bytes.size() - position, MSG_NOSIGNAL);
      if (count <= 0) {
        return ::testing::AssertionFailure()
               << "send failed: " << strerror(errno);
      }
      position += count;
    }
    return ::testing::AssertionSuccess();
  }

  std::string ReadBytes(size_t length) {
    std::string bytes(length, '\0');
    size_t position = 0;
    while (position < length) {
      const ssize_t count =
          recv(client_, bytes.data() + position, length - position, 0);
      EXPECT_GT(count, 0);
      if (count <= 0) {
        break;
      }
      position += count;
    }
    bytes.resize(position);
    return bytes;
  }

  Message AwaitMessage(loom_serve_websocket_message_type_t* type,
                       iree_const_byte_span_t* payload) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (true) {
      const auto token = iree_notification_prepare_wait(notification);
      auto* message = loom_serve_websocket_take(socket_, type, payload);
      if (message || loom_serve_http_connection_failed(connection_)) {
        iree_notification_cancel_wait(notification);
        return Message(message, {socket_});
      }
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  void ExpectMessage(loom_serve_websocket_message_type_t expected_type,
                     const std::string& expected_payload) {
    loom_serve_websocket_message_type_t type;
    iree_const_byte_span_t payload;
    auto message = AwaitMessage(&type, &payload);
    ASSERT_NE(message, nullptr);
    EXPECT_EQ(type, expected_type);
    EXPECT_EQ(Copy(payload), expected_payload);
  }

  void AwaitFailure() {
    iree_notification_await(
        loom_serve_http_server_notification(server_),
        [](void* value) {
          return loom_serve_http_connection_failed(
              static_cast<loom_serve_http_connection_t*>(value));
        },
        connection_, iree_infinite_timeout());
  }

  // Shared I/O workers retained until all connections and callbacks retire.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Owner of both HTTP requests and upgraded receivers.
  loom_serve_http_server_t* server_ = nullptr;
  // Independently retired real TCP listener.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Application claim, relinquished explicitly before transport destruction.
  loom_serve_http_connection_t* connection_ = nullptr;
  // Protocol view borrowed through the application claim.
  loom_serve_websocket_t* socket_ = nullptr;
  // Actual client peer, closed independently to exercise failure ownership.
  int client_ = -1;
};

TEST_F(WebSocketConnectionTest, FragmentsPermitPingAndSplitUtf8Characters) {
  Open();
  ASSERT_TRUE(Write(ClientFrame(1, std::string("hello \xE2", 7), false) +
                    ClientFrame(9, "ping")));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PING, "ping");
  IREE_ASSERT_OK(
      loom_serve_websocket_send(socket_, LOOM_SERVE_WEBSOCKET_PONG,
                                iree_make_const_byte_span("ping", 4)));
  EXPECT_EQ(ReadBytes(6), std::string("\x8A\x04ping", 6));
  ASSERT_TRUE(Write(ClientFrame(0, std::string("\x82\xAC", 2))));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_TEXT, "hello €");
  ASSERT_TRUE(Write(ClientFrame(1, "", false) + ClientFrame(0, "") +
                    ClientFrame(10, "")));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_TEXT, "");
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PONG, "");
}

TEST_F(WebSocketConnectionTest, AllPayloadLengthEncodingsAndMaskOffsets) {
  Open();
  for (size_t length :
       {size_t{0}, size_t{125}, size_t{126}, size_t{65535}, size_t{65536}}) {
    SCOPED_TRACE(length);
    std::string payload(length, '\0');
    for (size_t i = 0; i < length; ++i) {
      payload[i] = static_cast<char>(i);
    }
    const std::string frame = ClientFrame(2, payload);
    // Separate small header writes and a payload larger than the receive pool
    // exercise the real ordered carrier across arbitrary delivery boundaries.
    for (size_t i = 0; i < 6; ++i) {
      ASSERT_TRUE(Write(frame.substr(i, 1)));
    }
    ASSERT_TRUE(Write(frame.substr(6)));
    ExpectMessage(LOOM_SERVE_WEBSOCKET_BINARY, payload);
  }
}

TEST_F(WebSocketConnectionTest,
       HeldMessageSurvivesLaterInputAndPeerRetirement) {
  Open();
  ASSERT_TRUE(Write(ClientFrame(1, "borrowed")));
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t held;
  auto message = AwaitMessage(&type, &held);
  ASSERT_NE(message, nullptr);
  ASSERT_TRUE(Write(ClientFrame(2, std::string(65536, 'x'))));
  ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
  AwaitFailure();
  IREE_ASSERT_OK(loom_serve_http_listener_free(listener_));
  listener_ = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  EXPECT_EQ(loom_serve_http_server_take_request(server_, &request), nullptr);
  EXPECT_EQ(Copy(held), "borrowed");
  ExpectMessage(LOOM_SERVE_WEBSOCKET_BINARY, std::string(65536, 'x'));
  EXPECT_EQ(Copy(held), "borrowed");
}

TEST_F(WebSocketConnectionTest,
       FinishDrainsCopiedFramesAfterCallerPayloadChanges) {
  Open();
  std::string payload(65536, 'z');
  IREE_ASSERT_OK(loom_serve_websocket_send(
      socket_, LOOM_SERVE_WEBSOCKET_BINARY,
      iree_make_const_byte_span(payload.data(), payload.size())));
  payload.assign(payload.size(), 'q');
  loom_serve_http_connection_finish(connection_);
  connection_ = nullptr;
  socket_ = nullptr;
  EXPECT_EQ(ReadBytes(10), std::string("\x82\x7F\0\0\0\0\0\x01\0\0", 10));
  EXPECT_EQ(ReadBytes(65536), std::string(65536, 'z'));
  char byte;
  EXPECT_EQ(recv(client_, &byte, 1, 0), 0);
}

TEST_F(WebSocketConnectionTest,
       OutboundLengthEncodingsAreUnmaskedAndCanonical) {
  Open();
  const std::pair<size_t, std::string> cases[] = {
      {0, std::string("\x82\x00", 2)},
      {125, std::string("\x82\x7D", 2)},
      {126, std::string("\x82\x7E\x00\x7E", 4)},
      {65535, std::string("\x82\x7E\xFF\xFF", 4)},
      {65536, std::string("\x82\x7F\0\0\0\0\0\x01\0\0", 10)},
  };
  for (const auto& [length, header] : cases) {
    SCOPED_TRACE(length);
    const std::string payload(length, 'a');
    IREE_ASSERT_OK(loom_serve_websocket_send(
        socket_, LOOM_SERVE_WEBSOCKET_BINARY,
        iree_make_const_byte_span(payload.data(), payload.size())));
    EXPECT_EQ(ReadBytes(header.size() + length), header + payload);
  }
}

TEST_F(WebSocketConnectionTest, HalfCloseDuringUpgradeStillRetiresTheReceiver) {
  Open(loom_serve_websocket_limits_default(), true);
  AwaitFailure();
  EXPECT_FALSE(loom_serve_websocket_peer_closed(socket_));
}

TEST_F(WebSocketConnectionTest,
       CloseCanInterruptFragmentsAndFinishTheHandshake) {
  Open();
  const std::string close_payload("\x03\xE8", 2);
  ASSERT_TRUE(Write(ClientFrame(1, "unfinished", false) +
                    ClientFrame(8, close_payload)));
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t payload;
  auto message = AwaitMessage(&type, &payload);
  ASSERT_NE(message, nullptr);
  ASSERT_EQ(type, LOOM_SERVE_WEBSOCKET_CLOSE);
  EXPECT_TRUE(loom_serve_websocket_peer_closed(socket_));
  IREE_ASSERT_OK(loom_serve_websocket_send(socket_, type, payload));
  message.reset();
  ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
  loom_serve_http_connection_finish(connection_);
  connection_ = nullptr;
  socket_ = nullptr;
  EXPECT_EQ(ReadBytes(4), std::string("\x88\x02\x03\xE8", 4));
  char byte;
  EXPECT_EQ(recv(client_, &byte, 1, 0), 0);
}

TEST_F(WebSocketConnectionTest, QueueBoundsCountEmptyMessages) {
  auto limits = loom_serve_websocket_limits_default();
  limits.message_capacity = 2;
  Open(limits);
  ASSERT_TRUE(
      Write(ClientFrame(9, "") + ClientFrame(9, "") + ClientFrame(9, "")));
  AwaitFailure();
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PING, "");
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PING, "");
}

TEST_F(WebSocketConnectionTest, QueueBoundsAllocatedBytesAcrossMessages) {
  auto limits = loom_serve_websocket_limits_default();
  limits.message_byte_capacity = limits.retained_byte_capacity = 256;
  Open(limits);
  ASSERT_TRUE(
      Write(ClientFrame(2, std::string(256, 'a')) + ClientFrame(2, "b")));
  AwaitFailure();
  ExpectMessage(LOOM_SERVE_WEBSOCKET_BINARY, std::string(256, 'a'));
}

TEST_F(WebSocketConnectionTest, FragmentedMessageCannotExceedItsOwnBudget) {
  auto limits = loom_serve_websocket_limits_default();
  limits.message_byte_capacity = 125;
  Open(limits);
  ASSERT_TRUE(Write(ClientFrame(1, std::string(80, 'a'), false) +
                    ClientFrame(0, std::string(46, 'b'))));
  AwaitFailure();
}

TEST_F(WebSocketConnectionTest, TakenMessagesKeepRecordCreditUntilRelease) {
  auto limits = loom_serve_websocket_limits_default();
  limits.message_capacity = 2;
  Open(limits);
  ASSERT_TRUE(Write(ClientFrame(1, "held")));
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t payload;
  auto held = AwaitMessage(&type, &payload);
  ASSERT_NE(held, nullptr);
  ASSERT_TRUE(Write(ClientFrame(9, "control")));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PING, "control");
  EXPECT_EQ(Copy(payload), "held");
  ASSERT_TRUE(Write(ClientFrame(9, "one") + ClientFrame(9, "two")));
  AwaitFailure();
  ExpectMessage(LOOM_SERVE_WEBSOCKET_PING, "one");
  EXPECT_EQ(Copy(payload), "held");
}

TEST_F(WebSocketConnectionTest, TakenMessagesKeepPayloadCreditUntilRelease) {
  auto limits = loom_serve_websocket_limits_default();
  limits.message_byte_capacity = limits.retained_byte_capacity = 256;
  Open(limits);
  ASSERT_TRUE(Write(ClientFrame(2, std::string(256, 'a'))));
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t payload;
  auto held = AwaitMessage(&type, &payload);
  ASSERT_NE(held, nullptr);
  ASSERT_TRUE(Write(ClientFrame(2, "b")));
  AwaitFailure();
  EXPECT_EQ(Copy(payload), std::string(256, 'a'));
}

// A real shared-worker borrower, joined after its release callback. The gate
// controlling worker admission is separate from the job and protocol owner.
class WebSocketReadJob {
 public:
  ~WebSocketReadJob() {
    if (scheduled_) {
      Wait();
    }
  }

  void Schedule(iree_task_executor_t* executor,
                iree_const_byte_span_t payload) {
    payload_ = payload;
    iree_task_process_initialize(
        [](iree_task_process_t* process, const iree_task_worker_context_t*,
           iree_task_process_drain_result_t* result) {
          auto* job = static_cast<WebSocketReadJob*>(process->user_data);
          job->copy_ = Copy(job->payload_);
          result->did_work = true;
          result->completed = true;
          return iree_ok_status();
        },
        0, 1, &process_);
    process_.user_data = this;
    process_.completion_fn = [](iree_task_process_t*, iree_status_t status) {
      IREE_EXPECT_OK(status);
    };
    process_.release_fn = [](iree_task_process_t* process) {
      auto* job = static_cast<WebSocketReadJob*>(process->user_data);
      std::lock_guard<std::mutex> lock(job->mutex_);
      job->released_ = true;
      job->condition_.notify_all();
    };
    scheduled_ = true;
    iree_task_executor_schedule_process(executor, &process_);
  }

  const std::string& Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return released_; });
    return copy_;
  }

 private:
  // Intrusive process storage retained until the final release callback.
  iree_task_process_t process_ = {};
  // Borrowed immutable network payload held by the application's receipt.
  iree_const_byte_span_t payload_ = iree_const_byte_span_empty();
  // Worker result visible to the caller only after release publication.
  std::string copy_;
  // Joins the release callback's last access with the outer test caller.
  std::mutex mutex_;
  // Explicit final release notification without wall-clock polling.
  std::condition_variable condition_;
  // Owner-only fact distinguishing an unused job from accepted work.
  bool scheduled_ = false;
  // Publication under mutex after the executor relinquishes job storage.
  bool released_ = false;
};

class WebSocketBorrowTest : public WebSocketConnectionTest,
                            public ::testing::WithParamInterface<size_t> {};

TEST_P(WebSocketBorrowTest, LaterControlsPassAnAcceptedCpuBorrower) {
  Open();
  const std::string audio(4096, 'a');
  ASSERT_TRUE(Write(ClientFrame(2, audio)));
  loom_serve_websocket_message_type_t type;
  iree_const_byte_span_t payload;
  auto held = AwaitMessage(&type, &payload);
  ASSERT_NE(held, nullptr);

  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(GetParam(), &topology);
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  iree_task_executor_t* raw_executor = nullptr;
  auto status = iree_task_executor_create(
      options, &topology, iree_allocator_system(), &raw_executor);
  iree_task_topology_deinitialize(&topology);
  IREE_ASSERT_OK(status);
  std::unique_ptr<iree_task_executor_t, decltype(&iree_task_executor_release)>
      executor(raw_executor, iree_task_executor_release);
  WebSocketReadJob job;
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (size_t i = 0; i < GetParam(); ++i) {
    gates.emplace_back(
        std::make_unique<loom_serve::testing::WorkerGate>(executor.get()));
  }
  job.Schedule(executor.get(), payload);

  const std::string cancel = R"({"type":"response.cancel"})";
  ASSERT_TRUE(Write(ClientFrame(1, cancel)));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_TEXT, cancel);
  ASSERT_EQ(shutdown(client_, SHUT_WR), 0);
  AwaitFailure();
  const loom_serve_http_request_t* request = nullptr;
  EXPECT_EQ(loom_serve_http_server_take_request(server_, &request), nullptr);

  for (const auto& gate : gates) {
    gate->Open();
  }
  EXPECT_EQ(job.Wait(), audio);
  held.reset();
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, WebSocketBorrowTest,
                         ::testing::Values(1u, 8u));

class WebSocketMalformedTest
    : public WebSocketConnectionTest,
      public ::testing::WithParamInterface<std::string> {};

TEST_P(WebSocketMalformedTest, FailsOnlyTheMalformedPeer) {
  Open();
  ASSERT_TRUE(Write(GetParam()));
  AwaitFailure();
  loom_serve_http_connection_abort(connection_);
  connection_ = nullptr;
  socket_ = nullptr;
  ASSERT_EQ(close(client_), 0);
  client_ = -1;
  Open();
  ASSERT_TRUE(Write(ClientFrame(1, "healthy")));
  ExpectMessage(LOOM_SERVE_WEBSOCKET_TEXT, "healthy");
}

std::vector<std::string> MalformedFrames() {
  std::vector<std::string> frames = {
      std::string("\x81\x00", 2),             // Unmasked.
      std::string("\xC1\x80", 2),             // Unnegotiated RSV.
      std::string("\x09\x80", 2),             // Fragmented control.
      std::string("\x89\xFE", 2),             // Oversized control.
      std::string("\x82\xFE\0\x01maskx", 9),  // Noncanonical 16-bit length.
      std::string("\x82\xFF\0\0\0\0\0\0\0\x01maskx", 15),
      std::string("\x82\xFF\x80\0\0\0\0\0\0\0mask", 14),
      ClientFrame(0, "unexpected"),
      ClientFrame(1, "start", false) + ClientFrame(2, "new"),
      ClientFrame(1, std::string("\xC0\x80", 2)),  // Overlong UTF-8.
      ClientFrame(1, std::string("\xE2", 1)),      // Incomplete UTF-8.
      ClientFrame(8, "x"),
      ClientFrame(8, std::string("\x03\xED", 2)),  // Reserved 1005.
      ClientFrame(8, std::string("\x03\xE8\xFF", 3)),
      ClientFrame(8, "") + ClientFrame(1, "after close"),
  };
  for (uint8_t opcode : {3, 4, 5, 6, 7, 11, 12, 13, 14, 15}) {
    frames.push_back(ClientFrame(opcode, ""));
  }
  return frames;
}

INSTANTIATE_TEST_SUITE_P(Protocol, WebSocketMalformedTest,
                         ::testing::ValuesIn(MalformedFrames()));

}  // namespace
