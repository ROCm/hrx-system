// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/bootstrap.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static std::vector<uint8_t> Serialize(
    const iree_net_bootstrap_message_t& message) {
  iree_host_size_t size = 0;
  IREE_CHECK_OK(iree_net_bootstrap_message_calculate_size(&message, &size));
  std::vector<uint8_t> wire(size);
  IREE_CHECK_OK(iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(wire.data(), wire.size())));
  return wire;
}

static iree_net_bootstrap_message_view_t Parse(
    const std::vector<uint8_t>& wire) {
  iree_net_bootstrap_message_view_t message;
  std::memset(&message, 0, sizeof(message));
  IREE_CHECK_OK(iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(wire.data(), wire.size()), &message));
  return message;
}

static void ExpectInvalid(const std::vector<uint8_t>& wire) {
  iree_net_bootstrap_message_view_t message;
  std::memset(&message, 0xCD, sizeof(message));
  iree_status_t status = iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(wire.data(), wire.size()), &message);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  iree_net_bootstrap_message_view_t zero_message;
  std::memset(&zero_message, 0, sizeof(zero_message));
  EXPECT_EQ(std::memcmp(&message, &zero_message, sizeof(message)), 0);
}

static iree_net_bootstrap_message_t MakePeerMessage(
    iree_net_bootstrap_type_t type,
    const std::vector<iree_async_frontier_entry_t>& axes,
    iree_const_byte_span_t application_data) {
  iree_net_bootstrap_peer_info_t peer = {
      .capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER |
                      IREE_NET_BOOTSTRAP_CAPABILITY_RDMA,
      .application_endpoint_count = 2,
      .axes = axes.data(),
      .axis_count = static_cast<uint32_t>(axes.size()),
      .application_data = application_data,
      .machine_index = 3,
      .session_epoch = 7,
  };
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = type;
  if (type == IREE_NET_BOOTSTRAP_TYPE_HELLO) {
    message.value.hello = peer;
  } else {
    message.value.hello_ack = peer;
  }
  return message;
}

static std::vector<iree_async_frontier_entry_t> MakeAxes() {
  return {
      {
          .axis = iree_async_axis_make_queue(
              /*session_epoch=*/7, /*machine_index=*/3, /*device_index=*/1,
              /*queue_index=*/2, /*queue_incarnation=*/3),
          .epoch = 123,
      },
      {
          .axis = iree_async_axis_make(
              /*session_epoch=*/7, /*machine_index=*/3,
              IREE_ASYNC_CAUSAL_DOMAIN_HOST, /*ordinal=*/42),
          .epoch = UINT64_C(0x0102030405060708),
      },
  };
}

TEST(BootstrapTest, RoundTripsHelloWithExactWireLayout) {
  const std::vector<iree_async_frontier_entry_t> axes = MakeAxes();
  const uint8_t application_data[] = {0x00, 0x11, 0x22, 0x33, 0x44};
  const iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
      iree_make_const_byte_span(application_data, sizeof(application_data)));

  const std::vector<uint8_t> wire = Serialize(message);
  ASSERT_EQ(wire.size(), 72u);
  EXPECT_EQ(wire[0], IREE_NET_BOOTSTRAP_TYPE_HELLO);
  EXPECT_EQ(wire[1], IREE_NET_BOOTSTRAP_PROTOCOL_VERSION);
  EXPECT_EQ(iree_unaligned_load_le_u16(wire.data() + 2), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 4), wire.size());
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 8),
            IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 12), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 16), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 20), 5u);
  EXPECT_EQ(wire[24], 3u);
  EXPECT_EQ(wire[25], 7u);
  EXPECT_EQ(iree_unaligned_load_le_u16(wire.data() + 26), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 28), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u64(wire.data() + 32), axes[0].axis);
  EXPECT_EQ(iree_unaligned_load_le_u64(wire.data() + 40), axes[0].epoch);
  EXPECT_EQ(iree_unaligned_load_le_u64(wire.data() + 48), axes[1].axis);
  EXPECT_EQ(iree_unaligned_load_le_u64(wire.data() + 56), axes[1].epoch);
  EXPECT_EQ(
      std::memcmp(wire.data() + 64, application_data, sizeof(application_data)),
      0);
  EXPECT_EQ(wire[69], 0u);
  EXPECT_EQ(wire[70], 0u);
  EXPECT_EQ(wire[71], 0u);

  const iree_net_bootstrap_message_view_t parsed = Parse(wire);
  EXPECT_EQ(parsed.type, IREE_NET_BOOTSTRAP_TYPE_HELLO);
  const iree_net_bootstrap_peer_info_view_t& peer = parsed.value.hello;
  EXPECT_EQ(peer.capabilities, IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED);
  EXPECT_EQ(peer.application_endpoint_count, 2u);
  EXPECT_EQ(peer.machine_index, 3u);
  EXPECT_EQ(peer.session_epoch, 7u);
  ASSERT_EQ(peer.axes.count, 2u);
  for (uint32_t i = 0; i < peer.axes.count; ++i) {
    const iree_async_frontier_entry_t entry =
        iree_net_bootstrap_axis_list_get(&peer.axes, i);
    EXPECT_EQ(entry.axis, axes[i].axis);
    EXPECT_EQ(entry.epoch, axes[i].epoch);
  }
  ASSERT_EQ(peer.application_data.data_length, sizeof(application_data));
  EXPECT_EQ(std::memcmp(peer.application_data.data, application_data,
                        sizeof(application_data)),
            0);
}

TEST(BootstrapTest, RoundTripsHelloAckWithSymmetricPeerLayout) {
  const std::vector<iree_async_frontier_entry_t> axes = MakeAxes();
  const iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK, axes, iree_const_byte_span_empty());

  const std::vector<uint8_t> wire = Serialize(message);
  ASSERT_EQ(wire.size(), 64u);
  EXPECT_EQ(wire[0], IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK);

  const iree_net_bootstrap_message_view_t parsed = Parse(wire);
  EXPECT_EQ(parsed.type, IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK);
  EXPECT_EQ(parsed.value.hello_ack.application_endpoint_count, 2u);
  EXPECT_EQ(parsed.value.hello_ack.axes.count, 2u);
  EXPECT_EQ(parsed.value.hello_ack.application_data.data_length, 0u);
}

TEST(BootstrapTest, RoundTripsEmptyPeerInformation) {
  const std::vector<iree_async_frontier_entry_t> axes;
  iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty());
  message.value.hello.capabilities = IREE_NET_BOOTSTRAP_CAPABILITY_NONE;
  message.value.hello.application_endpoint_count = 0;

  const std::vector<uint8_t> wire = Serialize(message);
  ASSERT_EQ(wire.size(), IREE_NET_BOOTSTRAP_PEER_INFO_SIZE);
  const iree_net_bootstrap_message_view_t parsed = Parse(wire);
  EXPECT_EQ(parsed.value.hello.capabilities,
            IREE_NET_BOOTSTRAP_CAPABILITY_NONE);
  EXPECT_EQ(parsed.value.hello.application_endpoint_count, 0u);
  EXPECT_EQ(parsed.value.hello.axes.count, 0u);
  EXPECT_EQ(parsed.value.hello.application_data.data_length, 0u);
}

TEST(BootstrapTest, RoundTripsReject) {
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  message.value.reject.status_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  message.value.reject.reason = IREE_SV("at capacity");

  const std::vector<uint8_t> wire = Serialize(message);
  ASSERT_EQ(wire.size(), 32u);
  EXPECT_EQ(wire[0], IREE_NET_BOOTSTRAP_TYPE_REJECT);
  EXPECT_EQ(wire[8], IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_unaligned_load_le_u32(wire.data() + 12), 11u);
  EXPECT_EQ(std::memcmp(wire.data() + 16, "at capacity", 11), 0);

  const iree_net_bootstrap_message_view_t parsed = Parse(wire);
  EXPECT_EQ(parsed.type, IREE_NET_BOOTSTRAP_TYPE_REJECT);
  EXPECT_EQ(parsed.value.reject.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(std::string(parsed.value.reject.reason.data,
                        parsed.value.reject.reason.size),
            "at capacity");
}

TEST(BootstrapTest, RoundTripsRejectWithoutDiagnostic) {
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  message.value.reject.status_code = IREE_STATUS_UNAVAILABLE;

  const std::vector<uint8_t> wire = Serialize(message);
  ASSERT_EQ(wire.size(), IREE_NET_BOOTSTRAP_REJECT_SIZE);
  const iree_net_bootstrap_message_view_t parsed = Parse(wire);
  EXPECT_EQ(parsed.value.reject.status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_TRUE(iree_string_view_is_empty(parsed.value.reject.reason));
}

TEST(BootstrapTest, ParsesUnalignedInput) {
  const std::vector<iree_async_frontier_entry_t> axes = MakeAxes();
  const std::vector<uint8_t> wire = Serialize(MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty()));
  std::vector<uint8_t> unaligned_wire(wire.size() + 1, 0);
  std::memcpy(unaligned_wire.data() + 1, wire.data(), wire.size());

  iree_net_bootstrap_message_view_t parsed;
  IREE_ASSERT_OK(iree_net_bootstrap_message_parse(
      iree_make_const_byte_span(unaligned_wire.data() + 1, wire.size()),
      &parsed));
  EXPECT_EQ(parsed.value.hello.axes.count, 2u);
  EXPECT_EQ(iree_net_bootstrap_axis_list_get(&parsed.value.hello.axes, 1).epoch,
            axes[1].epoch);
}

TEST(BootstrapTest, LeavesExcessOutputCapacityUnmodified) {
  const std::vector<iree_async_frontier_entry_t> axes;
  const iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty());
  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(
      iree_net_bootstrap_message_calculate_size(&message, &required_size));
  std::vector<uint8_t> buffer(required_size + 4, 0xA5);
  IREE_ASSERT_OK(iree_net_bootstrap_message_serialize(
      &message, iree_make_byte_span(buffer.data(), buffer.size())));
  for (iree_host_size_t i = required_size; i < buffer.size(); ++i) {
    EXPECT_EQ(buffer[i], 0xA5);
  }
}

TEST(BootstrapTest, RejectsInvalidHostValues) {
  iree_host_size_t size = 123;
  iree_net_bootstrap_message_t empty_message;
  std::memset(&empty_message, 0, sizeof(empty_message));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&empty_message, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(nullptr, &size));
  EXPECT_EQ(size, 0u);
  const std::vector<iree_async_frontier_entry_t> axes = MakeAxes();
  iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty());

  message.type = static_cast<iree_net_bootstrap_type_t>(0xFF);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

#if UINTPTR_MAX > UINT32_MAX
  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  const uint8_t unused_byte = 0;
  message.value.hello.application_data = iree_make_const_byte_span(
      &unused_byte, static_cast<iree_host_size_t>(UINT32_MAX) + 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_bootstrap_message_calculate_size(&message, &size));
#endif  // UINTPTR_MAX > UINT32_MAX

  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  message.value.hello.capabilities = 1u << 31;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  message.value.hello.axes = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  message.value.hello.application_data = iree_make_const_byte_span(nullptr, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  message.value.hello.machine_index++;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                            iree_const_byte_span_empty());
  message.value.hello.session_epoch++;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  std::vector<iree_async_frontier_entry_t> invalid_axes = axes;
  invalid_axes[0].axis = iree_async_axis_make(
      /*session_epoch=*/7, /*machine_index=*/3,
      static_cast<iree_async_causal_domain_t>(0xFF), /*ordinal=*/0);
  message = MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, invalid_axes,
                            iree_const_byte_span_empty());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));
}

TEST(BootstrapTest, RejectsInvalidRejectionHostValues) {
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  message.value.reject.status_code = IREE_STATUS_OK;
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message.value.reject.status_code = static_cast<iree_status_code_t>(-1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  message.value.reject.status_code = IREE_STATUS_INTERNAL;
  message.value.reject.reason = iree_make_string_view(nullptr, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));

  const char reason_with_nul[] = {'a', 0, 'b'};
  message.value.reject.reason =
      iree_make_string_view(reason_with_nul, sizeof(reason_with_nul));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_calculate_size(&message, &size));
}

TEST(BootstrapTest, RejectsMissingOrSmallOutputBuffer) {
  const std::vector<iree_async_frontier_entry_t> axes;
  const iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_serialize(&message, iree_byte_span_empty()));
  uint8_t buffer[IREE_NET_BOOTSTRAP_PEER_INFO_SIZE - 1] = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_bootstrap_message_serialize(
          &message, iree_make_byte_span(buffer, sizeof(buffer))));
}

TEST(BootstrapTest, RejectsMalformedHeader) {
  iree_net_bootstrap_message_view_t parsed;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_parse(iree_const_byte_span_empty(), nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_bootstrap_message_parse(
          iree_make_const_byte_span(nullptr, IREE_NET_BOOTSTRAP_HEADER_SIZE),
          &parsed));

  std::vector<uint8_t> wire(IREE_NET_BOOTSTRAP_HEADER_SIZE - 1, 0);
  ExpectInvalid(wire);

  const std::vector<iree_async_frontier_entry_t> axes;
  wire = Serialize(MakePeerMessage(IREE_NET_BOOTSTRAP_TYPE_HELLO, axes,
                                   iree_const_byte_span_empty()));
  wire[1]++;
  ExpectInvalid(wire);

  wire[1] = IREE_NET_BOOTSTRAP_PROTOCOL_VERSION;
  wire[2] = 1;
  ExpectInvalid(wire);

  wire[2] = 0;
  iree_unaligned_store_le_u32(wire.data() + 4,
                              static_cast<uint32_t>(wire.size() - 8));
  ExpectInvalid(wire);

  iree_unaligned_store_le_u32(wire.data() + 4,
                              static_cast<uint32_t>(wire.size()));
  wire[0] = 0xFF;
  ExpectInvalid(wire);
}

TEST(BootstrapTest, RejectsTrailingAndMisalignedMessages) {
  const std::vector<iree_async_frontier_entry_t> axes;
  std::vector<uint8_t> wire = Serialize(MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_const_byte_span_empty()));
  wire.push_back(0);
  ExpectInvalid(wire);

  iree_unaligned_store_le_u32(wire.data() + 4,
                              static_cast<uint32_t>(wire.size()));
  ExpectInvalid(wire);
}

TEST(BootstrapTest, RejectsMalformedPeerInformation) {
  const std::vector<iree_async_frontier_entry_t> axes = MakeAxes();
  const iree_net_bootstrap_message_t message = MakePeerMessage(
      IREE_NET_BOOTSTRAP_TYPE_HELLO, axes, iree_make_const_byte_span("x", 1));
  const std::vector<uint8_t> valid_wire = Serialize(message);
  std::vector<uint8_t> wire = valid_wire;

  wire.resize(IREE_NET_BOOTSTRAP_PEER_INFO_SIZE - 8);
  iree_unaligned_store_le_u32(wire.data() + 4,
                              static_cast<uint32_t>(wire.size()));
  ExpectInvalid(wire);

  wire = valid_wire;
  iree_unaligned_store_le_u32(wire.data() + 8, 1u << 31);
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[26] = 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[28] = 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  iree_unaligned_store_le_u32(wire.data() + 16, UINT32_MAX);
  ExpectInvalid(wire);

  wire = valid_wire;
  iree_unaligned_store_le_u32(wire.data() + 20, UINT32_MAX);
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[32 + 6] ^= 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[32 + 7] ^= 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[32 + 5] = 0xFF;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire.back() = 1;
  ExpectInvalid(wire);
}

TEST(BootstrapTest, RejectsMalformedRejection) {
  iree_net_bootstrap_message_t message;
  std::memset(&message, 0, sizeof(message));
  message.type = IREE_NET_BOOTSTRAP_TYPE_REJECT;
  message.value.reject.status_code = IREE_STATUS_ABORTED;
  message.value.reject.reason = IREE_SV("reason");
  const std::vector<uint8_t> valid_wire = Serialize(message);
  std::vector<uint8_t> wire = valid_wire;

  wire.resize(IREE_NET_BOOTSTRAP_REJECT_SIZE - 8);
  iree_unaligned_store_le_u32(wire.data() + 4,
                              static_cast<uint32_t>(wire.size()));
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[8] = IREE_STATUS_OK;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[8] = IREE_STATUS_CODE_MASK + 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[9] = 1;
  ExpectInvalid(wire);

  wire = valid_wire;
  iree_unaligned_store_le_u32(wire.data() + 12, UINT32_MAX);
  ExpectInvalid(wire);

  wire = valid_wire;
  wire[17] = 0;
  ExpectInvalid(wire);

  wire = valid_wire;
  wire.back() = 1;
  ExpectInvalid(wire);
}

}  // namespace
