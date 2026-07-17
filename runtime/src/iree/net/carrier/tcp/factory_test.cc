// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP-specific transport factory tests.
//
// Tests that exercise TCP-specific behavior which cannot be generalized across
// transports: address parsing, capability queries, ephemeral port assignment.
// Common factory tests live in cts/factory_test.cc.

#include "iree/net/carrier/tcp/factory.h"

#include <cstring>
#include <string>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture and helpers
//===----------------------------------------------------------------------===//

class TcpFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));

    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        region_, iree_allocator_system(), &recv_pool_));

    iree_net_tcp_carrier_options_t tcp_options =
        iree_net_tcp_carrier_options_default();
    tcp_options.max_endpoint_count = 3;
    IREE_ASSERT_OK(iree_net_tcp_factory_create(
        tcp_options, iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    iree_net_transport_factory_release(factory_);
    factory_ = nullptr;
    iree_async_buffer_pool_release(recv_pool_);
    recv_pool_ = nullptr;
    iree_async_region_release(region_);
    region_ = nullptr;
    iree_async_slab_release(slab_);
    slab_ = nullptr;
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  template <typename Fn>
  bool PollUntil(Fn condition) {
    while (!condition()) {
      iree_host_size_t count = 0;
      IREE_CHECK_OK(
          iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
    }
    return true;
  }

  struct ListenerInfo {
    iree_net_listener_t* listener = nullptr;
    std::string connect_address;
  };
  ListenerInfo CreateListener(
      iree_net_listener_accept_callback_t accept_callback, void* user_data) {
    ListenerInfo info;
    IREE_CHECK_OK(iree_net_transport_factory_create_listener(
        factory_, IREE_SV("127.0.0.1:0"), proactor_, recv_pool_,
        accept_callback, user_data, iree_allocator_system(), &info.listener));

    char address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t bound_address;
    IREE_CHECK_OK(iree_net_listener_query_bound_address(
        info.listener, sizeof(address_buffer), address_buffer, &bound_address));
    info.connect_address = std::string(bound_address.data, bound_address.size);
    return info;
  }

  void StopAndWait(iree_net_listener_t* listener) {
    bool stopped = false;
    IREE_ASSERT_OK(iree_net_listener_stop(
        listener,
        {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &stopped}));
    ASSERT_TRUE(PollUntil([&]() { return stopped; }))
        << "Listener stop timed out";
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
};

// Tracks the result of a connect callback.
struct ConnectResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_connection_t* connection = nullptr;
};

static void TrackConnectCallback(void* user_data, iree_status_t status,
                                 iree_net_connection_t* connection) {
  auto* result = static_cast<ConnectResult*>(user_data);
  result->fired = true;
  result->status_code = iree_status_code(status);
  result->connection = connection;
  iree_status_ignore(status);
}

struct EndpointResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_message_endpoint_t endpoint = {nullptr, nullptr};
};

static void TrackEndpointReady(void* user_data, iree_status_t status,
                               iree_net_message_endpoint_t endpoint) {
  auto* result = static_cast<EndpointResult*>(user_data);
  result->fired = true;
  result->status_code = iree_status_code(status);
  result->endpoint = endpoint;
  iree_status_ignore(status);
}

struct MessageCapture {
  bool message_fired = false;
  bool error_fired = false;
  iree_status_code_t error_code = IREE_STATUS_OK;
  std::string message;

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    auto* capture = static_cast<MessageCapture*>(user_data);
    capture->message_fired = true;
    capture->message.assign(reinterpret_cast<const char*>(message.data),
                            message.data_length);
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    auto* capture = static_cast<MessageCapture*>(user_data);
    capture->error_fired = true;
    capture->error_code = iree_status_code(status);
    iree_status_ignore(status);
  }

  iree_net_message_endpoint_callbacks_t callbacks() {
    return {
        /*.on_message=*/MessageCapture::OnMessage,
        /*.on_error=*/MessageCapture::OnError,
        /*.user_data=*/this,
    };
  }
};

struct DeactivateResult {
  bool fired = false;

  static void Callback(void* user_data) {
    static_cast<DeactivateResult*>(user_data)->fired = true;
  }
};

//===----------------------------------------------------------------------===//
// TCP-specific tests
//===----------------------------------------------------------------------===//

TEST_F(TcpFactoryTest, QueryCapabilities) {
  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory_);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
}

TEST_F(TcpFactoryTest, ConnectAsyncBadAddress) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("localhost"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncEmptyAddress) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, iree_string_view_empty(), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncBadPort) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("127.0.0.1:abc"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncPortOutOfRange) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("127.0.0.1:99999"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ListenerEphemeralPort) {
  struct AcceptCtx {
    bool fired = false;
  } accept_ctx;

  auto info = CreateListener(
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        iree_status_ignore(status);
        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx);

  // The bound address should have a non-zero port.
  EXPECT_FALSE(info.connect_address.empty());
  auto colon_position = info.connect_address.rfind(':');
  ASSERT_NE(colon_position, std::string::npos);
  std::string port_string = info.connect_address.substr(colon_position + 1);
  int port = std::stoi(port_string);
  EXPECT_GT(port, 0);
  EXPECT_LE(port, 65535);

  StopAndWait(info.listener);
  iree_net_listener_free(info.listener);
}

TEST_F(TcpFactoryTest, QueuesFrameForFutureStreamUntilActivation) {
  ConnectResult accept_result;
  auto info = CreateListener(TrackConnectCallback, &accept_result);

  ConnectResult connect_result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, iree_make_cstring_view(info.connect_address.c_str()), proactor_,
      recv_pool_, TrackConnectCallback, &connect_result));

  ASSERT_TRUE(PollUntil([&]() {
    return accept_result.fired && connect_result.fired;
  })) << "TCP connect/accept timed out";
  ASSERT_EQ(accept_result.status_code, IREE_STATUS_OK);
  ASSERT_EQ(connect_result.status_code, IREE_STATUS_OK);
  ASSERT_NE(accept_result.connection, nullptr);
  ASSERT_NE(connect_result.connection, nullptr);

  iree_net_connection_t* server_connection = accept_result.connection;
  iree_net_connection_t* client_connection = connect_result.connection;

  EndpointResult client_control_endpoint;
  EndpointResult server_control_endpoint;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      client_connection, {TrackEndpointReady, &client_control_endpoint}));
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      server_connection, {TrackEndpointReady, &server_control_endpoint}));
  ASSERT_TRUE(PollUntil([&]() {
    return client_control_endpoint.fired && server_control_endpoint.fired;
  })) << "Control endpoint open timed out";
  ASSERT_EQ(client_control_endpoint.status_code, IREE_STATUS_OK);
  ASSERT_EQ(server_control_endpoint.status_code, IREE_STATUS_OK);

  MessageCapture client_control_capture;
  MessageCapture server_control_capture;
  iree_net_message_endpoint_set_callbacks(client_control_endpoint.endpoint,
                                          client_control_capture.callbacks());
  iree_net_message_endpoint_set_callbacks(server_control_endpoint.endpoint,
                                          server_control_capture.callbacks());
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(client_control_endpoint.endpoint));
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(server_control_endpoint.endpoint));

  EndpointResult server_future_endpoint;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      server_connection, {TrackEndpointReady, &server_future_endpoint}));
  ASSERT_TRUE(PollUntil([&]() { return server_future_endpoint.fired; }))
      << "Server future endpoint open timed out";
  ASSERT_EQ(server_future_endpoint.status_code, IREE_STATUS_OK);

  MessageCapture server_future_capture;
  iree_net_message_endpoint_set_callbacks(server_future_endpoint.endpoint,
                                          server_future_capture.callbacks());
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(server_future_endpoint.endpoint));

  const char* early_payload = "early stream payload";
  iree_async_span_t early_span =
      iree_async_span_from_ptr((void*)early_payload, strlen(early_payload));
  iree_async_span_list_t early_span_list =
      iree_async_span_list_make(&early_span, 1);
  iree_net_message_endpoint_send_params_t early_send = {
      /*.data=*/early_span_list,
      /*.user_data=*/0,
  };
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_future_endpoint.endpoint,
                                                &early_send));

  EndpointResult client_future_endpoint;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      client_connection, {TrackEndpointReady, &client_future_endpoint}));
  ASSERT_TRUE(PollUntil([&]() { return client_future_endpoint.fired; }))
      << "Client future endpoint open timed out";
  ASSERT_EQ(client_future_endpoint.status_code, IREE_STATUS_OK);

  MessageCapture client_future_capture;
  iree_net_message_endpoint_set_callbacks(client_future_endpoint.endpoint,
                                          client_future_capture.callbacks());
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(client_future_endpoint.endpoint));

  ASSERT_TRUE(PollUntil([&]() { return client_future_capture.message_fired; }))
      << "Queued future-stream frame was not delivered";
  EXPECT_EQ(client_future_capture.message, early_payload);
  ASSERT_FALSE(client_control_capture.error_fired)
      << "Early future-stream frame caused control stream error "
      << client_control_capture.error_code;

  DeactivateResult client_deactivated;
  DeactivateResult server_deactivated;
  iree_net_connection_deactivate(
      client_connection, {DeactivateResult::Callback, &client_deactivated});
  iree_net_connection_deactivate(
      server_connection, {DeactivateResult::Callback, &server_deactivated});
  ASSERT_TRUE(PollUntil([&]() {
    return client_deactivated.fired && server_deactivated.fired;
  })) << "TCP connection deactivation timed out";

  iree_net_connection_release(client_connection);
  iree_net_connection_release(server_connection);
  StopAndWait(info.listener);
  iree_net_listener_free(info.listener);
}

}  // namespace
}  // namespace iree
