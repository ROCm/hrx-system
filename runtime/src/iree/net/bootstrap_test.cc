// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/bootstrap.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ByteVector = std::vector<uint8_t>;

static iree_const_byte_span_t AsSpan(const ByteVector& bytes) {
  return iree_make_const_byte_span(bytes.data(), bytes.size());
}

template <typename T>
static void Store(ByteVector& bytes, iree_host_size_t offset, const T& value) {
  ASSERT_LE(offset + sizeof(value), bytes.size());
  memcpy(bytes.data() + offset, &value, sizeof(value));
}

static ByteVector BuildHello(
    std::initializer_list<iree_net_bootstrap_axis_entry_t> axes = {},
    std::string_view application_data = {}) {
  iree_net_bootstrap_topology_layout_t layout;
  IREE_CHECK_OK(iree_net_bootstrap_topology_layout_calculate(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes.size(), application_data.size(),
      &layout));
  ByteVector bytes(layout.payload_length, 0);

  iree_net_bootstrap_hello_t hello = {};
  hello.header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO;
  hello.protocol_version = IREE_NET_BOOTSTRAP_PROTOCOL_VERSION;
  hello.capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER;
  hello.machine_index = 7;
  hello.session_epoch = 9;
  hello.axis_count = static_cast<uint16_t>(axes.size());
  hello.application_data_length = application_data.size();
  Store(bytes, 0, hello);

  iree_host_size_t offset = layout.axis_entries_offset;
  for (const auto& axis : axes) {
    Store(bytes, offset, axis);
    offset += sizeof(axis);
  }
  if (!application_data.empty()) {
    memcpy(bytes.data() + layout.application_data_offset,
           application_data.data(), application_data.size());
  }
  return bytes;
}

static ByteVector BuildHelloAck(
    std::initializer_list<iree_net_bootstrap_axis_entry_t> axes = {},
    std::string_view application_data = {}) {
  iree_net_bootstrap_topology_layout_t layout;
  IREE_CHECK_OK(iree_net_bootstrap_topology_layout_calculate(
      IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK, axes.size(), application_data.size(),
      &layout));
  ByteVector bytes(layout.payload_length, 0);

  iree_net_bootstrap_hello_ack_t hello_ack = {};
  hello_ack.header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK;
  hello_ack.session_id = 0x123456789ABCDEF0ull;
  hello_ack.application_data_length = application_data.size();
  hello_ack.negotiated_capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_RDMA;
  hello_ack.machine_index = 3;
  hello_ack.session_epoch = 5;
  hello_ack.axis_count = static_cast<uint16_t>(axes.size());
  Store(bytes, 0, hello_ack);

  iree_host_size_t offset = layout.axis_entries_offset;
  for (const auto& axis : axes) {
    Store(bytes, offset, axis);
    offset += sizeof(axis);
  }
  if (!application_data.empty()) {
    memcpy(bytes.data() + layout.application_data_offset,
           application_data.data(), application_data.size());
  }
  return bytes;
}

static ByteVector BuildReject(iree_status_code_t reason_code,
                              std::string_view reason) {
  ByteVector bytes(sizeof(iree_net_bootstrap_reject_t) + reason.size(), 0);
  iree_net_bootstrap_reject_t reject = {};
  reject.header.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  reject.reason_code = reason_code;
  Store(bytes, 0, reject);
  if (!reason.empty()) {
    memcpy(bytes.data() + sizeof(reject), reason.data(), reason.size());
  }
  return bytes;
}

static void ExpectParseFailure(const ByteVector& bytes) {
  iree_net_bootstrap_message_view_t message;
  memset(&message, 0xA5, sizeof(message));
  iree_status_t status =
      iree_net_bootstrap_message_parse(AsSpan(bytes), &message);
  IREE_EXPECT_NOT_OK(status);

  const uint8_t* message_bytes = reinterpret_cast<const uint8_t*>(&message);
  EXPECT_TRUE(std::all_of(message_bytes, message_bytes + sizeof(message),
                          [](uint8_t value) { return value == 0; }));
}

TEST(BootstrapLayoutTest, CanonicalTopologyExtent) {
  iree_net_bootstrap_topology_layout_t layout;
  IREE_ASSERT_OK(iree_net_bootstrap_topology_layout_calculate(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, 2, 3, &layout));
  EXPECT_EQ(layout.axis_entries_offset, 32u);
  EXPECT_EQ(layout.application_data_offset, 64u);
  EXPECT_EQ(layout.payload_length, 72u);
}

TEST(BootstrapLayoutTest, RejectHasNoTopologyLayout) {
  iree_net_bootstrap_topology_layout_t layout;
  memset(&layout, 0xA5, sizeof(layout));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_bootstrap_topology_layout_calculate(
                            IREE_NET_BOOTSTRAP_TYPE_REJECT, 0, 0, &layout));
  EXPECT_EQ(layout.axis_entries_offset, 0u);
  EXPECT_EQ(layout.application_data_offset, 0u);
  EXPECT_EQ(layout.payload_length, 0u);
}

TEST(BootstrapMessageTest, ParsesHello) {
  ByteVector bytes = BuildHello(
      {{0x0102030405060708ull, 11}, {0x1112131415161718ull, 22}}, "abc");
  iree_net_bootstrap_message_view_t message;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(AsSpan(bytes), &message));

  ASSERT_EQ(message.type, IREE_NET_BOOTSTRAP_TYPE_HELLO);
  const iree_net_bootstrap_hello_view_t& hello = message.value.hello;
  EXPECT_EQ(hello.fixed.machine_index, 7);
  EXPECT_EQ(hello.fixed.session_epoch, 9);
  ASSERT_EQ(hello.axes.count, 2u);
  iree_net_bootstrap_axis_entry_t axis0 =
      iree_net_bootstrap_axis_list_get(&hello.axes, 0);
  EXPECT_EQ(axis0.axis, 0x0102030405060708ull);
  EXPECT_EQ(axis0.current_epoch, 11u);
  iree_net_bootstrap_axis_entry_t axis1 =
      iree_net_bootstrap_axis_list_get(&hello.axes, 1);
  EXPECT_EQ(axis1.axis, 0x1112131415161718ull);
  EXPECT_EQ(axis1.current_epoch, 22u);
  EXPECT_EQ(hello.application_data.data_length, 3u);
  EXPECT_EQ(memcmp(hello.application_data.data, "abc", 3), 0);
}

TEST(BootstrapMessageTest, ParsesHelloAckWithZeroAxes) {
  ByteVector bytes = BuildHelloAck({}, "accepted");
  iree_net_bootstrap_message_view_t message;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(AsSpan(bytes), &message));

  ASSERT_EQ(message.type, IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK);
  const iree_net_bootstrap_hello_ack_view_t& hello_ack =
      message.value.hello_ack;
  EXPECT_EQ(hello_ack.fixed.session_id, 0x123456789ABCDEF0ull);
  EXPECT_EQ(hello_ack.axes.count, 0u);
  EXPECT_EQ(hello_ack.axes.encoded_entries.data_length, 0u);
  EXPECT_EQ(hello_ack.application_data.data_length, 8u);
  EXPECT_EQ(memcmp(hello_ack.application_data.data, "accepted", 8), 0);
}

TEST(BootstrapMessageTest, ParsesReject) {
  ByteVector bytes = BuildReject(IREE_STATUS_PERMISSION_DENIED, "no access");
  iree_net_bootstrap_message_view_t message;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(AsSpan(bytes), &message));

  ASSERT_EQ(message.type, IREE_NET_BOOTSTRAP_TYPE_REJECT);
  EXPECT_EQ(message.value.reject.fixed.reason_code,
            IREE_STATUS_PERMISSION_DENIED);
  EXPECT_TRUE(iree_string_view_equal(message.value.reject.reason,
                                     IREE_SV("no access")));
}

TEST(BootstrapMessageTest, PreservesLongRejectReason) {
  std::string reason(4096, 'r');
  ByteVector bytes = BuildReject(IREE_STATUS_RESOURCE_EXHAUSTED, reason);
  iree_net_bootstrap_message_view_t message;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(AsSpan(bytes), &message));

  ASSERT_EQ(message.type, IREE_NET_BOOTSTRAP_TYPE_REJECT);
  EXPECT_EQ(message.value.reject.reason.size, reason.size());
  EXPECT_EQ(
      memcmp(message.value.reject.reason.data, reason.data(), reason.size()),
      0);
}

TEST(BootstrapMessageTest, ParsesFromUnalignedStorage) {
  ByteVector aligned_bytes = BuildHello({{0xAABBCCDDEEFF0011ull, 44}}, "x");
  ByteVector storage(aligned_bytes.size() + 1, 0);
  memcpy(storage.data() + 1, aligned_bytes.data(), aligned_bytes.size());
  iree_const_byte_span_t unaligned_payload =
      iree_make_const_byte_span(storage.data() + 1, aligned_bytes.size());

  iree_net_bootstrap_message_view_t message;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(unaligned_payload, &message));
  iree_net_bootstrap_axis_entry_t axis =
      iree_net_bootstrap_axis_list_get(&message.value.hello.axes, 0);
  EXPECT_EQ(axis.axis, 0xAABBCCDDEEFF0011ull);
  EXPECT_EQ(axis.current_epoch, 44u);
}

TEST(BootstrapMessageTest, RejectsMalformedCommonHeader) {
  ExpectParseFailure(ByteVector(7, 0));

  ByteVector unknown_type(sizeof(iree_net_bootstrap_header_t), 0);
  unknown_type[0] = 0xFF;
  ExpectParseFailure(unknown_type);

  ByteVector reserved0 = BuildHello();
  reserved0[1] = 1;
  ExpectParseFailure(reserved0);

  ByteVector reserved1 = BuildHello();
  reserved1[4] = 1;
  ExpectParseFailure(reserved1);
}

TEST(BootstrapMessageTest, RejectsMalformedHello) {
  ByteVector truncated(sizeof(iree_net_bootstrap_hello_t) - 1, 0);
  truncated[0] = IREE_NET_BOOTSTRAP_TYPE_HELLO;
  ExpectParseFailure(truncated);

  ByteVector bad_version = BuildHello();
  bad_version[8] ^= 1;
  ExpectParseFailure(bad_version);

  ByteVector bad_capability = BuildHello();
  bad_capability[15] = 0x80;
  ExpectParseFailure(bad_capability);

  ByteVector reserved = BuildHello();
  reserved[20] = 1;
  ExpectParseFailure(reserved);

  ByteVector short_tail = BuildHello({{1, 2}}, "abc");
  short_tail.pop_back();
  ExpectParseFailure(short_tail);

  ByteVector trailing_byte = BuildHello();
  trailing_byte.push_back(0);
  ExpectParseFailure(trailing_byte);

  ByteVector nonzero_padding = BuildHello({}, "abc");
  nonzero_padding.back() = 1;
  ExpectParseFailure(nonzero_padding);

  ByteVector oversized_application_data = BuildHello();
  uint64_t application_data_length = UINT64_MAX;
  Store(oversized_application_data, 24, application_data_length);
  ExpectParseFailure(oversized_application_data);
}

TEST(BootstrapMessageTest, RejectsMalformedHelloAck) {
  ByteVector truncated(sizeof(iree_net_bootstrap_hello_ack_t) - 1, 0);
  truncated[0] = IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK;
  ExpectParseFailure(truncated);

  ByteVector zero_session_id = BuildHelloAck();
  uint64_t zero = 0;
  Store(zero_session_id, 8, zero);
  ExpectParseFailure(zero_session_id);

  ByteVector bad_capability = BuildHelloAck();
  bad_capability[27] = 0x80;
  ExpectParseFailure(bad_capability);

  ByteVector short_tail = BuildHelloAck({{1, 2}}, "abc");
  short_tail.pop_back();
  ExpectParseFailure(short_tail);

  ByteVector trailing_byte = BuildHelloAck();
  trailing_byte.push_back(0);
  ExpectParseFailure(trailing_byte);

  ByteVector nonzero_padding = BuildHelloAck({}, "abc");
  nonzero_padding.back() = 1;
  ExpectParseFailure(nonzero_padding);
}

TEST(BootstrapMessageTest, RejectsMalformedReject) {
  ByteVector truncated(sizeof(iree_net_bootstrap_reject_t) - 1, 0);
  truncated[0] = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  ExpectParseFailure(truncated);

  ExpectParseFailure(BuildReject(IREE_STATUS_OK, "not an error"));
  ExpectParseFailure(
      BuildReject(static_cast<iree_status_code_t>(IREE_STATUS_CODE_MASK + 1),
                  "invalid code"));

  ByteVector reserved = BuildReject(IREE_STATUS_ABORTED, "aborted");
  reserved[12] = 1;
  ExpectParseFailure(reserved);
}

}  // namespace
