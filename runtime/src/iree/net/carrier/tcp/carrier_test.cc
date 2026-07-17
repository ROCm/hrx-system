// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP-specific carrier tests.
//
// Tests TCP carrier allocation options, validation, shutdown semantics, sticky
// error cloning, and connect/disconnect lifecycle stress. Generic carrier
// behavior (send/recv, backpressure, lifecycle, error handling) is tested by
// the CTS suite in carrier/cts/.

#include "iree/net/carrier/tcp/carrier.h"

#include <atomic>
#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/operations/net.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/async/socket.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture with proactor and buffer pool setup
//===----------------------------------------------------------------------===//

class TcpCarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CreateAsyncResources(/*max_concurrent_operations=*/0);
  }

  void TearDown() override { ReleaseAsyncResources(); }

  void ResetAsyncResources(iree_host_size_t max_concurrent_operations) {
    ReleaseAsyncResources();
    CreateAsyncResources(max_concurrent_operations);
  }

  void CreateAsyncResources(iree_host_size_t max_concurrent_operations) {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.max_concurrent_operations = max_concurrent_operations;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));

    // Create slab for recv buffers.
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));

    // Register slab with proactor for zero-copy recv.
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));

    // Create buffer pool over region.
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        region_, iree_allocator_system(), &recv_pool_));
  }

  void ReleaseAsyncResources() {
    iree_async_buffer_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_region_release(region_);
    region_ = nullptr;
    iree_async_slab_release(slab_);
    slab_ = nullptr;
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  // Creates a TCP listener bound to localhost on an ephemeral port.
  iree_async_socket_t* CreateListener(iree_async_address_t* out_address) {
    iree_async_socket_t* listener = nullptr;
    IREE_CHECK_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR, &listener));
    iree_async_address_t bind_address;
    IREE_CHECK_OK(iree_async_address_from_ipv4(
        iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
    IREE_CHECK_OK(iree_async_socket_bind(listener, &bind_address));
    IREE_CHECK_OK(iree_async_socket_listen(listener, 16));
    IREE_CHECK_OK(iree_async_socket_query_local_address(listener, out_address));
    return listener;
  }

  // Establishes a connected client/server pair.
  void EstablishConnection(iree_async_socket_t** out_client,
                           iree_async_socket_t** out_server,
                           iree_async_socket_t** out_listener) {
    iree_async_address_t listen_address;
    *out_listener = CreateListener(&listen_address);

    // Submit accept.
    iree_async_socket_accept_operation_t accept_op;
    memset(&accept_op, 0, sizeof(accept_op));
    iree_async_operation_initialize(
        &accept_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
        IREE_ASYNC_OPERATION_FLAG_NONE, AcceptCallback, this);
    accept_op.listen_socket = *out_listener;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

    // Create and connect client.
    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_NO_DELAY, out_client));
    iree_async_socket_connect_operation_t connect_op;
    memset(&connect_op, 0, sizeof(connect_op));
    iree_async_operation_initialize(
        &connect_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
        IREE_ASYNC_OPERATION_FLAG_NONE, ConnectCallback, this);
    connect_op.socket = *out_client;
    connect_op.address = listen_address;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

    // Poll until both complete.
    int completions = 0;
    while (completions < 2) {
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
      completions += (int)count;
    }

    ASSERT_NE(accept_op.accepted_socket, nullptr);
    *out_server = accept_op.accepted_socket;
  }

  static void AcceptCallback(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
    IREE_CHECK_OK(status);
  }

  static void ConnectCallback(void* user_data,
                              iree_async_operation_t* operation,
                              iree_status_t status,
                              iree_async_completion_flags_t flags) {
    IREE_CHECK_OK(status);
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Recv handler that accumulates received data.
struct RecvTestContext {
  std::vector<uint8_t> received_data;
  bool handler_called = false;
  iree_status_t handler_return_status = iree_ok_status();
};

static iree_status_t TestRecvHandler(void* user_data, iree_async_span_t data,
                                     iree_async_buffer_lease_t* lease) {
  auto* ctx = static_cast<RecvTestContext*>(user_data);
  ctx->handler_called = true;
  uint8_t* ptr = iree_async_span_ptr(data);
  ctx->received_data.insert(ctx->received_data.end(), ptr, ptr + data.length);
  iree_async_buffer_lease_release(lease);
  return ctx->handler_return_status;
}

// Test-only region with caller-owned storage.
struct TestRegion {
  // Base async region presented to the send path.
  iree_async_region_t base;

  // Counter incremented when the region's final reference is released.
  std::atomic<int>* destroy_count = nullptr;
};

static void TestRegionDestroy(iree_async_region_t* region) {
  auto* test_region = reinterpret_cast<TestRegion*>(region);
  ++(*test_region->destroy_count);
}

// Context for deactivate callback.
struct DeactivateContext {
  bool completed = false;
};

static void TestDeactivateCallback(void* user_data) {
  auto* ctx = static_cast<DeactivateContext*>(user_data);
  ctx->completed = true;
}

// Helper to deactivate carrier and wait for completion.
static void DeactivateAndWait(iree_async_proactor_t* proactor,
                              iree_net_carrier_t* carrier) {
  DeactivateContext deactivate_ctx;
  IREE_ASSERT_OK(iree_net_carrier_deactivate(carrier, TestDeactivateCallback,
                                             &deactivate_ctx));
  while (!deactivate_ctx.completed) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), &count));
  }
}

// Recv handler that returns an error.
static iree_status_t ErrorRecvHandler(void* user_data, iree_async_span_t data,
                                      iree_async_buffer_lease_t* lease) {
  (void)data;
  auto* error_count = static_cast<int*>(user_data);
  ++(*error_count);
  iree_async_buffer_lease_release(lease);
  return iree_make_status(IREE_STATUS_INTERNAL, "handler error for testing");
}

//===----------------------------------------------------------------------===//
// Options and allocation validation
//===----------------------------------------------------------------------===//

TEST(TcpCarrierOptionsTest, DefaultOptions) {
  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();

  EXPECT_EQ(options.send_slot_count, 32u);
  EXPECT_EQ(options.single_shot_recv_count, 8u);
  EXPECT_FALSE(options.prefer_multishot_recv);
  EXPECT_TRUE(options.prefer_zero_copy_send);
}

TEST_F(TcpCarrierTest, AllocateWithCustomOptions) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  options.send_slot_count = 16;
  options.single_shot_recv_count = 16;

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, options, {nullptr, nullptr},
      iree_allocator_system(), &carrier));
  ASSERT_NE(carrier, nullptr);

  // Verify capabilities and initial state.
  iree_net_carrier_capabilities_t caps = iree_net_carrier_capabilities(carrier);
  EXPECT_TRUE(caps & IREE_NET_CARRIER_CAPABILITY_RELIABLE);
  EXPECT_TRUE(caps & IREE_NET_CARRIER_CAPABILITY_ORDERED);
  EXPECT_EQ(iree_net_carrier_state(carrier), IREE_NET_CARRIER_STATE_CREATED);

  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

TEST_F(TcpCarrierTest, CommitSendQueuesWhenSubmitBackpressures) {
  ResetAsyncResources(/*max_concurrent_operations=*/4);

  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  options.send_slot_count = 8;
  options.single_shot_recv_count = 1;

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, options, {nullptr, nullptr},
      iree_allocator_system(), &carrier));
  ASSERT_NE(carrier, nullptr);

  RecvTestContext recv_ctx;
  iree_net_carrier_set_recv_handler(carrier, {TestRecvHandler, &recv_ctx});
  IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

  constexpr iree_host_size_t kPayloadSize = 16;
  constexpr int kSendCount = 8;
  for (int i = 0; i < kSendCount; ++i) {
    void* ptr = nullptr;
    iree_net_carrier_send_handle_t handle = 0;
    IREE_ASSERT_OK(
        iree_net_carrier_begin_send(carrier, kPayloadSize, &ptr, &handle));
    ASSERT_NE(ptr, nullptr);
    memset(ptr, i, kPayloadSize);
    IREE_ASSERT_OK(iree_net_carrier_commit_send(carrier, handle));
  }

  iree_host_size_t expected_bytes = kPayloadSize * kSendCount;
  while ((iree_host_size_t)iree_atomic_load(&carrier->bytes_sent,
                                            iree_memory_order_relaxed) <
         expected_bytes) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
  }
  EXPECT_GE((iree_host_size_t)iree_atomic_load(&carrier->bytes_sent,
                                               iree_memory_order_relaxed),
            expected_bytes);

  DeactivateAndWait(proactor_, carrier);
  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

TEST_F(TcpCarrierTest, QueuedSendRetainsRegisteredRegion) {
  ResetAsyncResources(/*max_concurrent_operations=*/4);

  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  options.send_slot_count = 8;
  options.single_shot_recv_count = 1;

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, options, {nullptr, nullptr},
      iree_allocator_system(), &carrier));
  ASSERT_NE(carrier, nullptr);

  RecvTestContext recv_ctx;
  iree_net_carrier_set_recv_handler(carrier, {TestRecvHandler, &recv_ctx});
  IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

  constexpr iree_host_size_t kPayloadSize = 16;
  constexpr int kBackpressureSendCount = 4;
  for (int i = 0; i < kBackpressureSendCount; ++i) {
    void* ptr = nullptr;
    iree_net_carrier_send_handle_t handle = 0;
    IREE_ASSERT_OK(
        iree_net_carrier_begin_send(carrier, kPayloadSize, &ptr, &handle));
    ASSERT_NE(ptr, nullptr);
    memset(ptr, i, kPayloadSize);
    IREE_ASSERT_OK(iree_net_carrier_commit_send(carrier, handle));
  }

  uint8_t registered_payload[kPayloadSize] = {0};
  std::atomic<int> destroy_count = 0;
  TestRegion test_region;
  memset(&test_region.base, 0, sizeof(test_region.base));
  iree_atomic_ref_count_init(&test_region.base.ref_count);
  test_region.base.proactor = proactor_;
  test_region.base.destroy_fn = TestRegionDestroy;
  test_region.base.type = IREE_ASYNC_REGION_TYPE_NONE;
  test_region.base.access_flags = IREE_ASYNC_BUFFER_ACCESS_FLAG_READ;
  test_region.base.base_ptr = registered_payload;
  test_region.base.length = sizeof(registered_payload);
  test_region.destroy_count = &destroy_count;

  iree_async_span_t span =
      iree_async_span_make(&test_region.base, 0, sizeof(registered_payload));
  iree_net_send_params_t params;
  memset(&params, 0, sizeof(params));
  params.data = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_carrier_send(carrier, &params));

  iree_async_region_release(&test_region.base);
  EXPECT_EQ(destroy_count.load(), 0);

  iree_host_size_t expected_bytes = kPayloadSize * (kBackpressureSendCount + 1);
  while ((iree_host_size_t)iree_atomic_load(&carrier->bytes_sent,
                                            iree_memory_order_relaxed) <
         expected_bytes) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
  }
  EXPECT_GE((iree_host_size_t)iree_atomic_load(&carrier->bytes_sent,
                                               iree_memory_order_relaxed),
            expected_bytes);
  EXPECT_EQ(destroy_count.load(), 1);

  DeactivateAndWait(proactor_, carrier);
  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

TEST_F(TcpCarrierTest, AllocateRejectsBadSendSlotCount) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  options.send_slot_count = 100;  // Not power of 2.

  iree_net_carrier_t* carrier = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_tcp_carrier_create(proactor_, server, recv_pool_, options,
                                  {nullptr, nullptr}, iree_allocator_system(),
                                  &carrier));
  EXPECT_EQ(carrier, nullptr);

  // On validation failure the socket is NOT consumed, so we release it.
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

TEST_F(TcpCarrierTest, AllocateRejectsBadRecvCount) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_tcp_carrier_options_t options =
      iree_net_tcp_carrier_options_default();
  options.single_shot_recv_count = 7;  // Not power of 2.

  iree_net_carrier_t* carrier = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_tcp_carrier_create(proactor_, server, recv_pool_, options,
                                  {nullptr, nullptr}, iree_allocator_system(),
                                  &carrier));
  EXPECT_EQ(carrier, nullptr);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// TCP shutdown tests (FIN semantics not in CTS)
//===----------------------------------------------------------------------===//

TEST_F(TcpCarrierTest, ShutdownBeforeActivate) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, iree_net_tcp_carrier_options_default(),
      {nullptr, nullptr}, iree_allocator_system(), &carrier));

  // Shutdown before activation should fail.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_carrier_shutdown(carrier));

  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

struct ShutdownRecvContext {
  bool recv_complete = false;
  iree_host_size_t bytes_received = 0;
};

TEST_F(TcpCarrierTest, GracefulShutdown) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, iree_net_tcp_carrier_options_default(),
      {nullptr, nullptr}, iree_allocator_system(), &carrier));

  RecvTestContext recv_ctx;
  iree_net_carrier_recv_handler_t handler = {TestRecvHandler, &recv_ctx};
  iree_net_carrier_set_recv_handler(carrier, handler);
  IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

  // Shutdown sends FIN to peer.
  IREE_ASSERT_OK(iree_net_carrier_shutdown(carrier));

  // Client should receive EOF (0 bytes).
  uint8_t recv_buffer[256];
  ShutdownRecvContext shutdown_ctx;
  iree_async_socket_recv_operation_t recv_op;
  memset(&recv_op, 0, sizeof(recv_op));
  iree_async_operation_initialize(
      &recv_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      [](void* user_data, iree_async_operation_t* op, iree_status_t status,
         iree_async_completion_flags_t) {
        IREE_CHECK_OK(status);
        auto* recv_op = (iree_async_socket_recv_operation_t*)op;
        auto* ctx = static_cast<ShutdownRecvContext*>(user_data);
        ctx->recv_complete = true;
        ctx->bytes_received = recv_op->bytes_received;
      },
      &shutdown_ctx);
  recv_op.socket = client;
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));
  recv_op.buffers = iree_async_span_list_make(&recv_span, 1);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  while (!shutdown_ctx.recv_complete) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
  }

  EXPECT_TRUE(shutdown_ctx.recv_complete);
  EXPECT_EQ(shutdown_ctx.bytes_received, 0u);

  DeactivateAndWait(proactor_, carrier);
  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

TEST_F(TcpCarrierTest, ShutdownThenSendFails) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, iree_net_tcp_carrier_options_default(),
      {nullptr, nullptr}, iree_allocator_system(), &carrier));

  RecvTestContext recv_ctx;
  iree_net_carrier_recv_handler_t handler = {TestRecvHandler, &recv_ctx};
  iree_net_carrier_set_recv_handler(carrier, handler);
  IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

  IREE_ASSERT_OK(iree_net_carrier_shutdown(carrier));

  // Send after shutdown: the send may succeed at submit time but fail at
  // completion. We verify the carrier handles this gracefully without crashing.
  const char* test_data = "After shutdown";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(test_data), strlen(test_data));
  iree_net_send_params_t params;
  memset(&params, 0, sizeof(params));
  params.data = iree_async_span_list_make(&span, 1);

  iree_status_t send_status = iree_net_carrier_send(carrier, &params);
  iree_status_free(send_status);

  // Deactivation is the completion barrier for any send accepted before the
  // shutdown state became visible to the transport.
  DeactivateAndWait(proactor_, carrier);
  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Sticky error clone safety
//===----------------------------------------------------------------------===//

TEST_F(TcpCarrierTest, StickyErrorClonedCorrectly) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  iree_net_carrier_t* carrier = nullptr;
  IREE_ASSERT_OK(iree_net_tcp_carrier_create(
      proactor_, server, recv_pool_, iree_net_tcp_carrier_options_default(),
      {nullptr, nullptr}, iree_allocator_system(), &carrier));

  // Set up recv handler that returns error.
  int error_count = 0;
  iree_net_carrier_recv_handler_t handler = {ErrorRecvHandler, &error_count};
  iree_net_carrier_set_recv_handler(carrier, handler);
  IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

  // Send data to trigger the error handler.
  const char* test_data = "X";
  iree_async_socket_send_operation_t send_op;
  memset(&send_op, 0, sizeof(send_op));
  iree_async_operation_initialize(
      &send_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      [](void*, iree_async_operation_t*, iree_status_t status,
         iree_async_completion_flags_t) { IREE_CHECK_OK(status); },
      nullptr);
  send_op.socket = client;
  iree_async_span_t send_span =
      iree_async_span_from_ptr(const_cast<char*>(test_data), strlen(test_data));
  send_op.buffers = iree_async_span_list_make(&send_span, 1);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Poll until error is captured.
  while (error_count == 0) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
  }

  // Get multiple copies of the sticky error — each should be independent.
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(test_data), strlen(test_data));
  iree_net_send_params_t params;
  memset(&params, 0, sizeof(params));
  params.data = iree_async_span_list_make(&span, 1);

  std::vector<iree::Status> errors;
  for (int i = 0; i < 5; ++i) {
    errors.push_back(iree::Status(iree_net_carrier_send(carrier, &params)));
  }

  // All errors should be INTERNAL.
  for (const auto& error : errors) {
    EXPECT_EQ(error.code(), iree::StatusCode::kInternal);
  }

  DeactivateAndWait(proactor_, carrier);
  iree_net_carrier_release(carrier);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Lifecycle stress
//===----------------------------------------------------------------------===//

TEST_F(TcpCarrierTest, RapidConnectDisconnect) {
  const int kCycles = 20;

  for (int cycle = 0; cycle < kCycles; ++cycle) {
    iree_async_socket_t* client = nullptr;
    iree_async_socket_t* server = nullptr;
    iree_async_socket_t* listener = nullptr;
    EstablishConnection(&client, &server, &listener);

    iree_net_carrier_t* carrier = nullptr;
    IREE_ASSERT_OK(iree_net_tcp_carrier_create(
        proactor_, server, recv_pool_, iree_net_tcp_carrier_options_default(),
        {nullptr, nullptr}, iree_allocator_system(), &carrier));

    RecvTestContext recv_ctx;
    iree_net_carrier_recv_handler_t handler = {TestRecvHandler, &recv_ctx};
    iree_net_carrier_set_recv_handler(carrier, handler);
    IREE_ASSERT_OK(iree_net_carrier_activate(carrier));

    // Quick send/recv.
    const char* msg = "Cycle test";
    iree_async_socket_send_operation_t send_op;
    memset(&send_op, 0, sizeof(send_op));
    iree_async_operation_initialize(
        &send_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        [](void*, iree_async_operation_t*, iree_status_t status,
           iree_async_completion_flags_t) { IREE_CHECK_OK(status); },
        nullptr);
    send_op.socket = client;
    iree_async_span_t span =
        iree_async_span_from_ptr(const_cast<char*>(msg), strlen(msg));
    send_op.buffers = iree_async_span_list_make(&span, 1);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

    while (!recv_ctx.handler_called) {
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(
          iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
    }

    DeactivateAndWait(proactor_, carrier);
    iree_net_carrier_release(carrier);
    iree_async_socket_release(client);
    iree_async_socket_release(listener);
  }

  // If we get here without leaks or crashes, the test passes.
  // ASan/TSan will catch any issues.
}

}  // namespace
}  // namespace iree
