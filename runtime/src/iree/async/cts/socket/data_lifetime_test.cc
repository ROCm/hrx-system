// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for send descriptor and data lifetime during CQE processing.
//
// Span descriptor arrays are consumed during submit and may be stack-local.
// The buffer data they reference remains caller-owned until completion. These
// tests exercise both sides of that contract by destroying descriptor storage
// immediately after submit while keeping payload storage alive in the callback
// context.
//
// The test pattern:
//   - recv callback fires on the poll thread (CQE processing)
//   - Callback calls a helper function that creates a stack-local span list
//   - Helper submits a send referencing context-owned data, then returns
//   - The stack frame with the span descriptors is now dead
//   - The receiver verifies the data matches the expected pattern
//
// Backends must materialize native iovec/WSABUF descriptors during submit so
// deferred execution never dereferences the expired span list.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

// Context passed to the recv callback for the echo-back test.
struct EchoBackContext {
  // Proactor receiving and submitting the socket operations.
  iree_async_proactor_t* proactor;
  // Connected socket used to send the echo response.
  iree_async_socket_t* socket;
  // Completion state for the echo send.
  CompletionTracker send_tracker;
  // Set after the trigger receive callback begins.
  bool recv_callback_fired = false;
  // Trigger receive completion status owned by the context.
  iree_status_t recv_status = iree_ok_status();
  // Echo send submission status owned by the context.
  iree_status_t send_submit_status = iree_ok_status();

  // The send operation must outlive the submit call (completion references it).
  // Stored here so it's valid until the send completes.
  iree_async_socket_send_operation_t send_op;

  // Pattern byte for the response. The helper function creates stack-local
  // descriptors referencing buffers filled with this byte.
  uint8_t pattern;

  // Header payload retained until the echo send completes.
  uint8_t header[16];

  // Body payload retained until the echo send completes.
  uint8_t body[128];
};

// Helper function called from the recv callback. Creates a stack-local span
// list, submits a send, and returns, destroying the descriptor storage. The
// referenced payload remains alive in |context| until completion.
//
// NOINLINE prevents the compiler from inlining this into the callback, which
// would keep the descriptor storage alive for the callback's lifetime.
static IREE_ATTRIBUTE_NOINLINE void submit_echo_response(
    EchoBackContext* context) {
  memset(context->header, context->pattern, sizeof(context->header));
  memset(context->body, context->pattern, sizeof(context->body));

  // Build stack-local scatter-gather descriptors referencing retained data.
  iree_async_span_t spans[2];
  spans[0] = iree_async_span_from_ptr(context->header, sizeof(context->header));
  spans[1] = iree_async_span_from_ptr(context->body, sizeof(context->body));

  InitSendOperation(&context->send_op, context->socket, spans, 2,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &context->send_tracker);

  context->send_submit_status =
      iree_async_proactor_submit_one(context->proactor, &context->send_op.base);

  // spans[] dies here. The backend-native descriptors materialized during
  // submit and the context-owned payload remain valid until completion.
}

// Recv completion callback. Fires on the poll thread during CQE processing.
// Calls submit_echo_response(), which creates stack-local span descriptors.
static void echo_back_recv_callback(void* user_data,
                                    iree_async_operation_t* operation,
                                    iree_status_t status,
                                    iree_async_completion_flags_t flags) {
  auto* context = static_cast<EchoBackContext*>(user_data);
  context->recv_callback_fired = true;
  context->recv_status = status;

  if (iree_status_is_ok(status)) {
    submit_echo_response(context);
  }

  // submit_echo_response has returned and its span list is now dead.
}

// Tests span descriptor materialization on the normal send path. See
// DataLifetimeBackpressureTest below for descriptors submitted while the send
// itself is deferred.
class DataLifetimeTest : public SocketTestBase<> {
 protected:
  static constexpr iree_host_size_t kResponseSize = 16 + 128;  // header + body

  // Runs one iteration of the echo-back test:
  //   1. Client sends a trigger byte to the server.
  //   2. Server recv callback submits context data through a stack span list.
  //   3. Client receives the response and verifies the pattern.
  void RunEchoBack(iree_async_socket_t* client, iree_async_socket_t* server,
                   uint8_t pattern) {
    // Set up the echo-back context for the server's recv callback.
    EchoBackContext echo_context;
    echo_context.proactor = proactor_;
    echo_context.socket = server;
    echo_context.pattern = pattern;

    // Submit server recv with the echo-back callback.
    uint8_t server_recv_buffer[1];
    iree_async_span_t server_recv_span = iree_async_span_from_ptr(
        server_recv_buffer, sizeof(server_recv_buffer));
    iree_async_socket_recv_operation_t server_recv_op;
    InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                      echo_back_recv_callback, &echo_context);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &server_recv_op.base));

    // Submit client recv to capture the echo response.
    uint8_t response_buffer[kResponseSize];
    memset(response_buffer, 0, sizeof(response_buffer));
    iree_async_span_t client_recv_span =
        iree_async_span_from_ptr(response_buffer, sizeof(response_buffer));
    iree_async_socket_recv_operation_t client_recv_op;
    CompletionTracker client_recv_tracker;
    InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                      CompletionTracker::Callback, &client_recv_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &client_recv_op.base));

    // Send a trigger byte from client to server. This generates a recv CQE
    // on the server side, whose callback submits the echo response.
    uint8_t trigger = pattern;
    iree_async_span_t trigger_span =
        iree_async_span_from_ptr(&trigger, sizeof(trigger));
    iree_async_socket_send_operation_t trigger_send_op;
    CompletionTracker trigger_tracker;
    InitSendOperation(&trigger_send_op, client, &trigger_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &trigger_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &trigger_send_op.base));

    PollUntilCondition([&] { return echo_context.recv_callback_fired; },
                       "echo trigger receive");
    IREE_ASSERT_OK(echo_context.recv_status);
    IREE_ASSERT_OK(echo_context.send_submit_status);

    // Poll until the client receives the echo response.
    iree_host_size_t total_received = 0;
    while (total_received < kResponseSize) {
      PollUntilCondition([&] { return client_recv_tracker.call_count > 0; },
                         "echo response receive");
      IREE_ASSERT_OK(client_recv_tracker.ConsumeStatus());
      ASSERT_GT(client_recv_op.bytes_received, 0u);
      total_received += client_recv_op.bytes_received;

      // TCP may deliver partial data. If we haven't received everything,
      // submit another recv for the remainder.
      if (total_received < kResponseSize) {
        client_recv_tracker.Reset();
        iree_async_span_t remaining_span = iree_async_span_from_ptr(
            response_buffer + total_received, kResponseSize - total_received);
        InitRecvOperation(&client_recv_op, client, &remaining_span, 1,
                          CompletionTracker::Callback, &client_recv_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
      }
    }

    PollUntilCondition(
        [&] {
          return trigger_tracker.call_count > 0 &&
                 echo_context.send_tracker.call_count > 0;
        },
        "echo operation completions");
    IREE_ASSERT_OK(trigger_tracker.ConsumeStatus());
    IREE_ASSERT_OK(echo_context.send_tracker.ConsumeStatus());

    // Verify we received the full response.
    ASSERT_EQ(total_received, kResponseSize)
        << "Expected " << kResponseSize << " bytes but received "
        << total_received;

    // Verify EVERY byte matches the expected pattern. This is the core
    // assertion: if the send SQE referenced dead stack data, the kernel
    // read garbage and the received bytes won't match.
    for (iree_host_size_t i = 0; i < kResponseSize; ++i) {
      ASSERT_EQ(response_buffer[i], pattern)
          << "Data corruption at byte " << i << ": expected 0x" << std::hex
          << (int)pattern << " but got 0x" << (int)response_buffer[i]
          << std::dec
          << ". This indicates the send used invalid descriptor or payload "
             "storage.";
    }
  }
};

// Single echo-back: verifies the basic mechanism works.
TEST_P(DataLifetimeTest, SendFromRecvCallback_StackLocalSpanList) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  RunEchoBack(client, server, 0xAA);

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Repeated echo-backs with different patterns. Each iteration uses a different
// fill byte, making it increasingly unlikely that stale stack data accidentally
// matches the expected pattern.
TEST_P(DataLifetimeTest, RepeatedSendFromRecvCallback_VaryingPatterns) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // 50 iterations with varying patterns. The stack region used by
  // The context payload storage is overwritten on each iteration, making
  // stale descriptor or payload references detectable.
  for (int i = 0; i < 50; ++i) {
    uint8_t pattern = static_cast<uint8_t>(0x01 + i);
    SCOPED_TRACE(::testing::Message() << "iteration " << i << " pattern=0x"
                                      << std::hex << (int)pattern);
    RunEchoBack(client, server, pattern);
  }

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

CTS_REGISTER_TEST_SUITE(DataLifetimeTest);

//===----------------------------------------------------------------------===//
// Backpressure data lifetime tests
//===----------------------------------------------------------------------===//
// The normal-path tests above validate stack-local span descriptor
// materialization. These tests validate the deferred send path: when the
// socket buffer is full, the proactor cannot transmit data inline and must
// defer to the poll loop (POLLOUT-driven retry on POSIX, kernel-managed on
// io_uring, inherently async on IOCP). The caller's buffer remains valid until
// the completion callback fires.
//
// To exercise this, we shrink SO_SNDBUF to the platform minimum (~2KB on
// Linux) and send 8KB responses — well above what fits in a single writev.
// The response data is stored in the context struct (not on the stack), so it
// remains valid for the full duration of the deferred send. If the proactor
// reads from the wrong address or the caller freed the buffer too early, the
// received data would be corrupted.

// Context for backpressure echo-back tests. Unlike EchoBackContext, the
// response data lives IN the context (not on the stack of a NOINLINE helper),
// because the whole point is that the data must survive until the send
// completion callback fires — which may be arbitrarily delayed under
// backpressure.
struct BackpressureEchoBackContext {
  // Proactor receiving and submitting the socket operations.
  iree_async_proactor_t* proactor;
  // Connected socket used to send the echo response.
  iree_async_socket_t* socket;
  // Completion state for the echo send.
  CompletionTracker send_tracker;
  // Set after the trigger receive callback begins.
  bool recv_callback_fired = false;
  // Trigger receive completion status owned by the context.
  iree_status_t recv_status = iree_ok_status();
  // Echo send submission status owned by the context.
  iree_status_t send_submit_status = iree_ok_status();

  // Send operation retained until completion.
  iree_async_socket_send_operation_t send_op;

  // Byte pattern written into the response payload.
  uint8_t pattern;
  // Number of response bytes to send.
  iree_host_size_t response_size;

  // Response data stored in the context — lives until send completion.
  // This is the critical difference from the stack-local test: the data
  // survives the callback return because the context outlives the send.
  static constexpr iree_host_size_t kMaxResponseSize = 8192;
  uint8_t response_data[kMaxResponseSize];
};

// Fills the context's response_data with the pattern and submits the send.
// Unlike submit_echo_response() above, this does NOT need NOINLINE — the data
// is in the context, not on the stack. The test validates that the proactor
// correctly reads from the context buffer even when the send is deferred.
static void submit_backpressure_echo_response(
    BackpressureEchoBackContext* context) {
  memset(context->response_data, context->pattern, context->response_size);

  iree_async_span_t span =
      iree_async_span_from_ptr(context->response_data, context->response_size);

  InitSendOperation(&context->send_op, context->socket, &span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &context->send_tracker);

  context->send_submit_status =
      iree_async_proactor_submit_one(context->proactor, &context->send_op.base);
}

// Recv completion callback for backpressure tests. Fires on the poll thread
// during CQE processing and submits a large response through a constrained
// send buffer.
static void backpressure_echo_recv_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  auto* context = static_cast<BackpressureEchoBackContext*>(user_data);
  context->recv_callback_fired = true;
  context->recv_status = status;
  if (iree_status_is_ok(status)) {
    submit_backpressure_echo_response(context);
  }
}

class DataLifetimeBackpressureTest : public SocketTestBase<> {
 protected:
  // 8KB response — well above the ~2KB minimum SO_SNDBUF on Linux. This
  // guarantees the send cannot complete in a single writev and must be
  // deferred to the poll loop for POLLOUT-driven retry (on POSIX).
  static constexpr iree_host_size_t kResponseSize = 8192;

  // Shrinks SO_SNDBUF on |socket| to the platform minimum. The kernel rounds
  // the requested value of 1 up to its floor (SOCK_MIN_SNDBUF = ~2304 bytes
  // on Linux, varies on other platforms). This ensures that an 8KB send
  // overflows the buffer and exercises the deferred send path.
  void ShrinkSendBuffer(iree_async_socket_t* socket) {
    int send_buffer_size = 1;
#if defined(IREE_PLATFORM_WINDOWS)
    ASSERT_EQ(
        setsockopt(static_cast<SOCKET>(socket->primitive.value.win32_handle),
                   SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&send_buffer_size),
                   sizeof(send_buffer_size)),
        0)
        << "setsockopt SO_SNDBUF failed";
#else
    ASSERT_EQ(setsockopt(socket->primitive.value.fd, SOL_SOCKET, SO_SNDBUF,
                         &send_buffer_size, sizeof(send_buffer_size)),
              0)
        << "setsockopt SO_SNDBUF failed: errno=" << errno;
#endif  // IREE_PLATFORM_WINDOWS
  }

  // Runs one iteration of the backpressure echo-back test:
  //   1. Shrinks server SO_SNDBUF to force backpressure.
  //   2. Client sends a trigger byte to the server.
  //   3. Server recv callback fires, submits 8KB send-back from context data.
  //   4. Client receives the full response (handling partial recvs) and
  //      verifies every byte matches the pattern.
  void RunBackpressureEchoBack(iree_async_socket_t* client,
                               iree_async_socket_t* server, uint8_t pattern) {
    ShrinkSendBuffer(server);

    BackpressureEchoBackContext echo_context;
    echo_context.proactor = proactor_;
    echo_context.socket = server;
    echo_context.pattern = pattern;
    echo_context.response_size = kResponseSize;

    // Submit server recv with the backpressure echo callback.
    uint8_t server_recv_buffer[1];
    iree_async_span_t server_recv_span = iree_async_span_from_ptr(
        server_recv_buffer, sizeof(server_recv_buffer));
    iree_async_socket_recv_operation_t server_recv_op;
    InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                      backpressure_echo_recv_callback, &echo_context);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &server_recv_op.base));

    // Submit client recv for the large response.
    uint8_t response_buffer[kResponseSize];
    memset(response_buffer, 0, sizeof(response_buffer));
    iree_async_span_t client_recv_span =
        iree_async_span_from_ptr(response_buffer, kResponseSize);
    iree_async_socket_recv_operation_t client_recv_op;
    CompletionTracker client_recv_tracker;
    InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                      CompletionTracker::Callback, &client_recv_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &client_recv_op.base));

    // Send a trigger byte from client to server. This generates a recv CQE
    // on the server side, whose callback submits the backpressure response.
    uint8_t trigger = pattern;
    iree_async_span_t trigger_span =
        iree_async_span_from_ptr(&trigger, sizeof(trigger));
    iree_async_socket_send_operation_t trigger_send_op;
    CompletionTracker trigger_tracker;
    InitSendOperation(&trigger_send_op, client, &trigger_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &trigger_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &trigger_send_op.base));

    PollUntilCondition([&] { return echo_context.recv_callback_fired; },
                       "backpressure echo trigger receive");
    IREE_ASSERT_OK(echo_context.recv_status);
    IREE_ASSERT_OK(echo_context.send_submit_status);

    // Poll until the client receives the full 8KB response. Backpressure means
    // the server send is split across multiple POLLOUT-driven retries.
    iree_host_size_t total_received = 0;
    while (total_received < kResponseSize) {
      PollUntilCondition([&] { return client_recv_tracker.call_count > 0; },
                         "backpressure response receive");
      IREE_ASSERT_OK(client_recv_tracker.ConsumeStatus());
      ASSERT_GT(client_recv_op.bytes_received, 0u);
      total_received += client_recv_op.bytes_received;

      // TCP may deliver partial data. If we haven't received everything,
      // submit another recv for the remainder.
      if (total_received < kResponseSize) {
        client_recv_tracker.Reset();
        iree_async_span_t remaining_span = iree_async_span_from_ptr(
            response_buffer + total_received, kResponseSize - total_received);
        InitRecvOperation(&client_recv_op, client, &remaining_span, 1,
                          CompletionTracker::Callback, &client_recv_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
      }
    }

    PollUntilCondition(
        [&] {
          return trigger_tracker.call_count > 0 &&
                 echo_context.send_tracker.call_count > 0;
        },
        "backpressure echo operation completions");
    IREE_ASSERT_OK(trigger_tracker.ConsumeStatus());
    IREE_ASSERT_OK(echo_context.send_tracker.ConsumeStatus());

    // Verify we received the full response.
    ASSERT_EQ(total_received, kResponseSize)
        << "Expected " << kResponseSize << " bytes but received "
        << total_received;

    // Verify EVERY byte matches the expected pattern. Under backpressure the
    // send is deferred to the poll loop; if the proactor reads from a stale
    // or freed buffer, the received data will be corrupted.
    for (iree_host_size_t i = 0; i < kResponseSize; ++i) {
      ASSERT_EQ(response_buffer[i], pattern)
          << "Data corruption at byte " << i << ": expected 0x" << std::hex
          << (int)pattern << " but got 0x" << (int)response_buffer[i]
          << std::dec
          << ". This indicates the send deferred under backpressure and "
             "read from stale buffer data.";
    }
  }
};

// Single backpressure echo-back: verifies the deferred send path delivers
// correct data when the send buffer is saturated.
TEST_P(DataLifetimeBackpressureTest, BackpressureSend_ContextOwnedData) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  RunBackpressureEchoBack(client, server, 0xBB);

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Repeated backpressure echo-backs with different patterns. Each iteration
// fills the response with a distinct byte, making it progressively less likely
// that stale buffer contents accidentally match the expected pattern. Fewer
// iterations than the inline-path test (10 vs 50) because each iteration
// transfers 8KB through a ~2KB send buffer.
TEST_P(DataLifetimeBackpressureTest, RepeatedBackpressureSend_VaryingPatterns) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  for (int i = 0; i < 10; ++i) {
    uint8_t pattern = static_cast<uint8_t>(0x10 + i);
    SCOPED_TRACE(::testing::Message() << "iteration " << i << " pattern=0x"
                                      << std::hex << (int)pattern);
    RunBackpressureEchoBack(client, server, pattern);
  }

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

CTS_REGISTER_TEST_SUITE(DataLifetimeBackpressureTest);

}  // namespace iree::async::cts
