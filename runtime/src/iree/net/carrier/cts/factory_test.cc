// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS factory tests: transport factory → listener → connection → endpoint.
//
// These tests exercise the factory-level pipeline that is common across all
// transport backends. Transport-specific tests (address parsing, ordering
// guarantees, capability queries) remain in per-transport factory_test.cc
// files.
//
// Registered with the "factory" tag — only instantiated for backends that
// provide factory-level fields in their BackendInfo.

#include <cstring>
#include <string>
#include <vector>

#include "iree/net/carrier/cts/util/factory_test_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {
namespace {

using FactoryTest = FactoryTestBase;

//===----------------------------------------------------------------------===//
// Factory lifecycle
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, AllocateAndFree) { EXPECT_NE(factory_, nullptr); }

//===----------------------------------------------------------------------===//
// Connect and listener management
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, ConnectAsyncUnreachable) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string unreachable, MakeUnreachableAddress());
  iree_string_view_t address = iree_make_cstring_view(unreachable.c_str());

  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, address, proactor_, recv_pool_, ConnectResult::Callback,
      &result));

  // Callback must NOT fire synchronously.
  EXPECT_FALSE(result.fired);

  ASSERT_TRUE(PollUntil([&]() { return result.fired; }))
      << "Connect callback never fired";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(result.connection, nullptr);
}

TEST_P(FactoryTest, ConnectAsyncSuccess) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());
  ASSERT_NE(pair.client, nullptr);
  ASSERT_NE(pair.server, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(FactoryTest, CallbackNotFiredBeforePoll) {
  // Verify that connect returns with the callback still pending.
  // This is the fundamental async contract for all transports.
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  iree_net_listener_t* listener = nullptr;
  bool accept_fired = false;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        *static_cast<bool*>(user_data) = true;
        iree_status_ignore(status);
        iree_net_connection_release(connection);
      },
      &accept_fired, iree_allocator_system(), &listener));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener));
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());

  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
      &result));

  // No poll — neither callback should have fired.
  EXPECT_FALSE(accept_fired);
  EXPECT_FALSE(result.fired);

  // Now poll and verify they fire. Transport CM event ordering may let the
  // connector observe establishment before the listener observes the matching
  // accept completion, but both callbacks must be delivered by the proactor.
  ASSERT_TRUE(PollUntil([&]() { return result.fired && accept_fired; }));
  EXPECT_TRUE(result.fired);
  EXPECT_TRUE(accept_fired);

  iree_net_connection_release(result.connection);
  StopAndWait(listener);
  iree_net_listener_free(listener);
}

TEST_P(FactoryTest, MultipleConnections) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct MultiAcceptCtx {
    iree::Status status;
    std::vector<iree_net_connection_t*> connections;
  } accept_ctx;

  iree_net_listener_t* listener = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<MultiAcceptCtx*>(user_data);
        ctx->status = iree::Status(std::move(status));
        if (ctx->status.ok()) ctx->connections.push_back(connection);
      },
      &accept_ctx, iree_allocator_system(), &listener));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener));
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());

  ConnectResult connect_results[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_net_transport_factory_connect(
        factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
        &connect_results[i]));
  }

  ASSERT_TRUE(PollUntil([&]() {
    return !accept_ctx.status.ok() ||
           (accept_ctx.connections.size() >= 3 && connect_results[0].fired &&
            connect_results[1].fired && connect_results[2].fired);
  })) << "Not all connections completed";
  IREE_ASSERT_OK(accept_ctx.status);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(connect_results[i].status_code, IREE_STATUS_OK);
    ASSERT_NE(connect_results[i].connection, nullptr);
    iree_net_connection_release(connect_results[i].connection);
  }
  for (auto* connection : accept_ctx.connections) {
    iree_net_connection_release(connection);
  }

  StopAndWait(listener);
  iree_net_listener_free(listener);
}

TEST_P(FactoryTest, ListenerStop) {
  IREE_ASSERT_OK_AND_ASSIGN(std::string bind_str, MakeBindAddress());
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  iree_net_listener_t* listener = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        iree_status_ignore(status);
        iree_net_connection_release(connection);
      },
      nullptr, iree_allocator_system(), &listener));

  IREE_ASSERT_OK_AND_ASSIGN(std::string connect_str,
                            ResolveConnectAddress(bind_str, listener));

  // Stopped callback must be delivered asynchronously.
  bool stopped = false;
  IREE_ASSERT_OK(iree_net_listener_stop(
      listener, {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
                 &stopped}));
  EXPECT_FALSE(stopped);

  ASSERT_TRUE(PollUntil([&]() { return stopped; }));

  // Free the listener so the transport can clean up (TCP releases the socket,
  // which causes the kernel to RST new SYNs).
  iree_net_listener_free(listener);

  // After stop+free, connecting should fail asynchronously.
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());
  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
      &result));
  ASSERT_TRUE(PollUntil([&]() { return result.fired; }))
      << "Connect callback never fired after listener stop";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Endpoint opening
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, OpenEndpointFirst) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());

  // First open_endpoint returns a borrowed endpoint view.
  EndpointReadyResult endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.client, {EndpointReadyResult::Callback, &endpoint_result}));

  // Must be async.
  EXPECT_FALSE(endpoint_result.fired);

  ASSERT_TRUE(PollUntil([&]() { return endpoint_result.fired; }))
      << "Endpoint ready callback never fired";
  EXPECT_EQ(endpoint_result.status_code, IREE_STATUS_OK);
  ASSERT_NE(endpoint_result.endpoint.self, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(FactoryTest, ConnectionJoinsEndpointDeactivation) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());

  EndpointReadyResult client_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.client, {EndpointReadyResult::Callback, &client_endpoint_result}));
  EndpointReadyResult server_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.server, {EndpointReadyResult::Callback, &server_endpoint_result}));
  EndpointReadyResult client_second_endpoint_result;
  EndpointReadyResult server_second_endpoint_result;
  bool use_second_endpoint =
      iree_net_connection_max_endpoint_count(pair.client) >= 2 &&
      iree_net_connection_max_endpoint_count(pair.server) >= 2;
  if (use_second_endpoint) {
    IREE_ASSERT_OK(iree_net_connection_open_endpoint(
        pair.client,
        {EndpointReadyResult::Callback, &client_second_endpoint_result}));
    IREE_ASSERT_OK(iree_net_connection_open_endpoint(
        pair.server,
        {EndpointReadyResult::Callback, &server_second_endpoint_result}));
  }
  ASSERT_TRUE(PollUntil([&]() {
    return client_endpoint_result.fired && server_endpoint_result.fired &&
           (!use_second_endpoint || (client_second_endpoint_result.fired &&
                                     server_second_endpoint_result.fired));
  }));

  auto callbacks = iree_net_message_endpoint_callbacks_t{
      [](void*, iree_const_byte_span_t, iree_async_buffer_lease_t*)
          -> iree_status_t { return iree_ok_status(); },
      [](void*, iree_status_t status) { iree_status_ignore(status); }, nullptr};
  iree_net_message_endpoint_set_callbacks(client_endpoint_result.endpoint,
                                          callbacks);
  iree_net_message_endpoint_set_callbacks(server_endpoint_result.endpoint,
                                          callbacks);
  if (use_second_endpoint) {
    iree_net_message_endpoint_set_callbacks(
        client_second_endpoint_result.endpoint, callbacks);
    iree_net_message_endpoint_set_callbacks(
        server_second_endpoint_result.endpoint, callbacks);
  }
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(client_endpoint_result.endpoint));
  IREE_ASSERT_OK(
      iree_net_message_endpoint_activate(server_endpoint_result.endpoint));
  if (use_second_endpoint) {
    IREE_ASSERT_OK(iree_net_message_endpoint_activate(
        client_second_endpoint_result.endpoint));
    IREE_ASSERT_OK(iree_net_message_endpoint_activate(
        server_second_endpoint_result.endpoint));
  }

  // Keep a direct-write reservation pending so endpoint deactivation cannot
  // finish before connection deactivation joins it. Aborting after both
  // requests retires the operation without publishing a frame to the peer.
  void* send_ptr = nullptr;
  iree_net_carrier_send_handle_t send_handle = 0;
  IREE_ASSERT_OK(iree_net_message_endpoint_begin_send(
      client_endpoint_result.endpoint, /*size=*/16, &send_ptr, &send_handle));

  int endpoint_deactivate_count = 0;
  IREE_ASSERT_OK(iree_net_message_endpoint_deactivate(
      client_endpoint_result.endpoint,
      [](void* user_data) { ++*static_cast<int*>(user_data); },
      &endpoint_deactivate_count));

  struct ConnectionDeactivateResult {
    int callback_count = 0;
    iree_net_connection_t* connection = nullptr;

    static void Callback(void* user_data) {
      auto* result = static_cast<ConnectionDeactivateResult*>(user_data);
      ++result->callback_count;
      iree_net_connection_release(result->connection);
    }
  } client_deactivate{0, pair.client}, server_deactivate{0, pair.server};
  iree_net_connection_deactivate(
      pair.client, {ConnectionDeactivateResult::Callback, &client_deactivate});
  iree_net_message_endpoint_abort_send(client_endpoint_result.endpoint,
                                       send_handle);

  while (endpoint_deactivate_count == 0 ||
         client_deactivate.callback_count == 0) {
    iree_host_size_t completion_count = 0;
    IREE_ASSERT_OK(PollProactorOnce(proactor_, &completion_count));
  }
  iree_net_connection_deactivate(
      pair.server, {ConnectionDeactivateResult::Callback, &server_deactivate});
  while (server_deactivate.callback_count == 0) {
    iree_host_size_t completion_count = 0;
    IREE_ASSERT_OK(PollProactorOnce(proactor_, &completion_count));
  }
  EXPECT_EQ(endpoint_deactivate_count, 1);
  EXPECT_EQ(client_deactivate.callback_count, 1);
  EXPECT_EQ(server_deactivate.callback_count, 1);

  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(FactoryTest, ConnectionReleaseWithoutEndpoint) {
  // Create a connection but never open_endpoint — the transport stack should be
  // released when the connection is destroyed. ASan/LSan catch leaks.
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

//===----------------------------------------------------------------------===//
// End-to-end data transfer
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, BidirectionalSendRecv) {
  IREE_ASSERT_OK_AND_ASSIGN(auto pair, EstablishConnection());

  // Open endpoints on both sides.
  EndpointReadyResult client_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.client, {EndpointReadyResult::Callback, &client_endpoint_result}));
  EndpointReadyResult server_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.server, {EndpointReadyResult::Callback, &server_endpoint_result}));
  ASSERT_TRUE(PollUntil([&]() {
    return client_endpoint_result.fired && server_endpoint_result.fired;
  })) << "Endpoint ready callbacks never fired";
  ASSERT_NE(client_endpoint_result.endpoint.self, nullptr);
  ASSERT_NE(server_endpoint_result.endpoint.self, nullptr);

  iree_net_message_endpoint_t client_endpoint = client_endpoint_result.endpoint;
  iree_net_message_endpoint_t server_endpoint = server_endpoint_result.endpoint;

  // Set recv handlers on both sides.
  struct RecvContext {
    std::vector<uint8_t> data;
    bool received = false;
  };
  RecvContext client_recv;
  RecvContext server_recv;

  auto make_callbacks = [](RecvContext* ctx) {
    return iree_net_message_endpoint_callbacks_t{
        [](void* user_data, iree_const_byte_span_t message,
           iree_async_buffer_lease_t* lease) -> iree_status_t {
          auto* recv_ctx = static_cast<RecvContext*>(user_data);
          recv_ctx->data.insert(recv_ctx->data.end(), message.data,
                                message.data + message.data_length);
          recv_ctx->received = true;
          iree_async_buffer_lease_release(lease);
          return iree_ok_status();
        },
        nullptr,  // on_error
        ctx};
  };

  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          make_callbacks(&client_recv));
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          make_callbacks(&server_recv));

  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));

  // Client → server.
  char client_msg[] = "hello from client";
  iree_async_span_t client_span =
      iree_async_span_from_ptr(client_msg, strlen(client_msg));
  iree_net_message_endpoint_send_params_t params;
  memset(&params, 0, sizeof(params));
  params.data = iree_async_span_list_make(&client_span, 1);
  IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &params));

  // Poll until server receives.
  ASSERT_TRUE(PollUntil([&]() { return server_recv.received; }))
      << "Server never received data";
  EXPECT_EQ(std::string(server_recv.data.begin(), server_recv.data.end()),
            "hello from client");

  // Server → client.
  char server_msg[] = "hello from server";
  iree_async_span_t server_span =
      iree_async_span_from_ptr(server_msg, strlen(server_msg));
  params.data = iree_async_span_list_make(&server_span, 1);
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &params));

  ASSERT_TRUE(PollUntil([&]() { return client_recv.received; }))
      << "Client never received data";
  EXPECT_EQ(std::string(client_recv.data.begin(), client_recv.data.end()),
            "hello from server");

  // Cleanup.
  DeactivateEndpointAndWait(client_endpoint);
  DeactivateEndpointAndWait(server_endpoint);
  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

}  // namespace

CTS_REGISTER_TEST_SUITE_WITH_TAGS(FactoryTest, {"factory"}, {});

}  // namespace iree::net::carrier::cts
