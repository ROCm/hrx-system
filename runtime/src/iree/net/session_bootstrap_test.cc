// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/session_bootstrap.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::vector<uint8_t> SerializePeer(
    iree_net_bootstrap_type_t type,
    const iree_net_bootstrap_peer_info_t& peer) {
  iree_net_bootstrap_message_t message = {};
  message.type = type;
  if (type == IREE_NET_BOOTSTRAP_TYPE_HELLO) {
    message.value.hello = peer;
  } else {
    message.value.hello_ack = peer;
  }
  iree_host_size_t message_size = 0;
  IREE_CHECK_OK(
      iree_net_bootstrap_message_calculate_size(&message, &message_size));
  std::vector<uint8_t> wire(message_size);
  IREE_CHECK_OK(iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(wire.data(), wire.size())));
  return wire;
}

static iree_net_bootstrap_message_view_t Parse(iree_const_byte_span_t message) {
  iree_net_bootstrap_message_view_t parsed = {};
  IREE_CHECK_OK(iree_net_bootstrap_message_parse(message, &parsed));
  return parsed;
}

TEST(SessionBootstrapTest, CapturesLocalPeerInformationOnce) {
  std::array<iree_async_frontier_entry_t, 1> axes = {};
  axes[0].axis = iree_async_axis_make_queue(
      /*session_epoch=*/7, /*machine_index=*/3, /*device_index=*/1,
      /*queue_index=*/2, /*queue_incarnation=*/3);
  axes[0].epoch = 42;
  std::array<uint8_t, 4> application_data = {1, 2, 3, 4};
  iree_net_bootstrap_peer_info_t local_peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER,
      .application_endpoint_count = 2,
      .axes = axes.data(),
      .axis_count = static_cast<uint32_t>(axes.size()),
      .application_data = iree_make_const_byte_span(application_data.data(),
                                                    application_data.size()),
      .machine_index = 3,
      .session_epoch = 7,
  };

  iree_net_session_bootstrap_t bootstrap;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_CLIENT, &local_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER, iree_allocator_system(),
      &bootstrap));
  axes[0].epoch = 99;
  application_data.fill(0xFF);

  const iree_const_byte_span_t outbound =
      iree_net_session_bootstrap_outbound_message(&bootstrap);
  const iree_net_bootstrap_message_view_t parsed = Parse(outbound);
  ASSERT_EQ(parsed.type, IREE_NET_BOOTSTRAP_TYPE_HELLO);
  EXPECT_EQ(parsed.value.hello.capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);
  EXPECT_EQ(parsed.value.hello.application_endpoint_count, 2u);
  EXPECT_EQ(iree_net_bootstrap_axis_list_get(&parsed.value.hello.axes, 0).epoch,
            42u);
  ASSERT_EQ(parsed.value.hello.application_data.data_length, 4u);
  EXPECT_EQ(parsed.value.hello.application_data.data[0], 1u);

  iree_net_session_bootstrap_consume_outbound_message(&bootstrap);
  EXPECT_EQ(bootstrap.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_CLIENT_WAIT_ACK);
  EXPECT_EQ(bootstrap.outbound_message.data, nullptr);
  iree_net_session_bootstrap_deinitialize(&bootstrap);
}

TEST(SessionBootstrapTest, NegotiatesSymmetricCapabilities) {
  const iree_net_bootstrap_peer_info_t client_peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED,
      .application_endpoint_count = 3,
  };
  const iree_net_bootstrap_peer_info_t server_peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER,
      .application_endpoint_count = 3,
  };

  iree_net_session_bootstrap_t client;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_CLIENT, &client_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER, iree_allocator_system(),
      &client));
  iree_net_session_bootstrap_t server;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_SERVER, &server_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_NONE, iree_allocator_system(), &server));

  const iree_const_byte_span_t hello =
      iree_net_session_bootstrap_outbound_message(&client);
  iree_net_bootstrap_peer_info_view_t remote_client = {};
  iree_net_bootstrap_capabilities_t server_negotiated = 0;
  IREE_ASSERT_OK(iree_net_session_bootstrap_process_message(
      &server, hello, &remote_client, &server_negotiated));
  EXPECT_EQ(server.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_SERVER_SEND_ACK);
  EXPECT_EQ(remote_client.capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED);
  EXPECT_EQ(server_negotiated, IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);
  iree_net_session_bootstrap_consume_outbound_message(&client);

  const iree_const_byte_span_t ack =
      iree_net_session_bootstrap_outbound_message(&server);
  const iree_net_bootstrap_message_view_t parsed_ack = Parse(ack);
  EXPECT_EQ(parsed_ack.value.hello_ack.capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);
  iree_net_bootstrap_peer_info_view_t remote_server = {};
  iree_net_bootstrap_capabilities_t client_negotiated = 0;
  IREE_ASSERT_OK(iree_net_session_bootstrap_process_message(
      &client, ack, &remote_server, &client_negotiated));
  EXPECT_EQ(client.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_COMPLETE);
  EXPECT_EQ(remote_server.capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER);
  EXPECT_EQ(client_negotiated, server_negotiated);
  iree_net_session_bootstrap_consume_outbound_message(&server);

  iree_net_session_bootstrap_deinitialize(&server);
  iree_net_session_bootstrap_deinitialize(&client);
}

TEST(SessionBootstrapTest, RejectsEndpointCountMismatch) {
  iree_net_bootstrap_peer_info_t local_peer = {};
  local_peer.application_endpoint_count = 2;
  iree_net_bootstrap_peer_info_t remote_peer = {};
  remote_peer.application_endpoint_count = 3;
  const std::vector<uint8_t> hello =
      SerializePeer(IREE_NET_BOOTSTRAP_TYPE_HELLO, remote_peer);

  iree_net_session_bootstrap_t server;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_SERVER, &local_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_NONE, iree_allocator_system(), &server));
  iree_net_bootstrap_peer_info_view_t remote = {};
  iree_net_bootstrap_capabilities_t negotiated = 0;
  iree_status_t status = iree_net_session_bootstrap_process_message(
      &server, iree_make_const_byte_span(hello.data(), hello.size()), &remote,
      &negotiated);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  EXPECT_EQ(server.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_SERVER_WAIT_HELLO);
  EXPECT_EQ(remote.application_endpoint_count, 0u);
  EXPECT_EQ(negotiated, IREE_NET_BOOTSTRAP_CAPABILITY_NONE);
  iree_net_session_bootstrap_deinitialize(&server);
}

TEST(SessionBootstrapTest, RejectsMissingRequiredCapability) {
  const iree_net_bootstrap_peer_info_t local_peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED,
      .application_endpoint_count = 1,
  };
  const iree_net_bootstrap_peer_info_t remote_peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_RDMA,
      .application_endpoint_count = 1,
  };
  const std::vector<uint8_t> ack =
      SerializePeer(IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK, remote_peer);

  iree_net_session_bootstrap_t client;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_CLIENT, &local_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER, iree_allocator_system(),
      &client));
  iree_net_session_bootstrap_consume_outbound_message(&client);
  iree_net_bootstrap_peer_info_view_t remote = {};
  iree_net_bootstrap_capabilities_t negotiated = 0;
  iree_status_t status = iree_net_session_bootstrap_process_message(
      &client, iree_make_const_byte_span(ack.data(), ack.size()), &remote,
      &negotiated);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNAVAILABLE);
  iree_status_free(status);
  EXPECT_EQ(client.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_CLIENT_WAIT_ACK);
  iree_net_session_bootstrap_deinitialize(&client);
}

TEST(SessionBootstrapTest, ConvertsRemoteRejectToTerminalStatus) {
  const iree_net_bootstrap_peer_info_t local_peer = {};
  iree_net_session_bootstrap_t client;
  IREE_ASSERT_OK(iree_net_session_bootstrap_initialize(
      IREE_NET_SESSION_BOOTSTRAP_ROLE_CLIENT, &local_peer,
      IREE_NET_BOOTSTRAP_CAPABILITY_NONE, iree_allocator_system(), &client));
  iree_net_session_bootstrap_consume_outbound_message(&client);

  const std::string reason = "server policy rejected the session";
  iree_net_bootstrap_message_t reject = {};
  reject.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  reject.value.reject.status_code = IREE_STATUS_PERMISSION_DENIED;
  reject.value.reject.reason =
      iree_make_string_view(reason.data(), reason.size());
  iree_host_size_t reject_size = 0;
  IREE_ASSERT_OK(
      iree_net_bootstrap_message_calculate_size(&reject, &reject_size));
  std::vector<uint8_t> wire(reject_size);
  IREE_ASSERT_OK(iree_net_bootstrap_message_serialize(
      &reject, iree_make_byte_span(wire.data(), wire.size())));

  iree_net_bootstrap_peer_info_view_t remote = {};
  iree_net_bootstrap_capabilities_t negotiated = 0;
  iree_status_t status = iree_net_session_bootstrap_process_message(
      &client, iree_make_const_byte_span(wire.data(), wire.size()), &remote,
      &negotiated);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_PERMISSION_DENIED);
  iree_status_free(status);
  EXPECT_EQ(client.phase, IREE_NET_SESSION_BOOTSTRAP_PHASE_CLIENT_WAIT_ACK);
  iree_net_session_bootstrap_deinitialize(&client);
}

}  // namespace
