// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/blocking_printf_protocol.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum FragmentFlags : uint64_t {
  kFragmentBegin = UINT64_C(1) << 0,
  kFragmentEnd = UINT64_C(1) << 1,
};

static uint64_t MakeDescriptor(uint64_t flags, iree_host_size_t length) {
  return flags | ((uint64_t)length << 5);
}

class BlockingPrintfProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_calculate_layout(
        /*compute_unit_count=*/1, /*maximum_waves_per_compute_unit=*/1,
        &layout_));
    storage_.resize((layout_.allocation_size + sizeof(uint64_t) - 1) /
                    sizeof(uint64_t));
    iree_hip_blocking_printf_protocol_initialize(
        storage_.data(), kDeviceAddress, kDoorbellToken, &layout_, &protocol_);
    iree_hip_hostcall_message_table_initialize(iree_allocator_system(),
                                               &message_table_);
    output_stream_ = tmpfile();
    error_stream_ = tmpfile();
    ASSERT_NE(nullptr, output_stream_);
    ASSERT_NE(nullptr, error_stream_);
  }

  void TearDown() override {
    if (output_stream_) fclose(output_stream_);
    if (error_stream_) fclose(error_stream_);
    iree_hip_hostcall_message_table_deinitialize(&message_table_);
  }

  uint64_t PacketTag(uint32_t packet_index) const {
    return packet_index == 0 ? protocol_.packet_count : packet_index;
  }

  void PreparePacket(uint32_t packet_index, uint64_t next, uint32_t service,
                     uint64_t activemask) {
    iree_hip_hostcall_packet_header_t* header =
        &protocol_.packet_headers[packet_index];
    header->next = next;
    header->activemask = activemask;
    header->service = service;
    iree_atomic_store(&header->control, IREE_HIP_HOSTCALL_PACKET_CONTROL_READY,
                      iree_memory_order_relaxed);
  }

  void SetCompletePrintfFragment(uint32_t packet_index, uint32_t lane,
                                 uint64_t control, const char* format,
                                 uint64_t argument) {
    std::array<uint64_t, 4> message = {control, 0, 0, argument};
    ASSERT_LE(strlen(format) + 1, 2 * sizeof(uint64_t));
    memcpy(&message[1], format, strlen(format) + 1);

    uint64_t* payload = protocol_.packet_payloads[packet_index].slots[lane];
    payload[0] = MakeDescriptor(kFragmentBegin | kFragmentEnd, message.size());
    memcpy(payload + 1, message.data(), message.size() * sizeof(message[0]));
  }

  std::string ReadStream(FILE* stream) {
    const long end = ftell(stream);
    EXPECT_GE(end, 0);
    EXPECT_EQ(0, fseek(stream, 0, SEEK_SET));
    std::string value(end > 0 ? (size_t)end : 0, '\0');
    if (!value.empty()) {
      EXPECT_EQ(value.size(), fread(value.data(), 1, value.size(), stream));
    }
    return value;
  }

  static constexpr uint64_t kDeviceAddress = UINT64_C(0x10000000);
  static constexpr uint64_t kDoorbellToken = UINT64_C(0x12345678);
  iree_hip_blocking_printf_protocol_layout_t layout_ = {};
  std::vector<uint64_t> storage_;
  iree_hip_blocking_printf_protocol_t protocol_ = {};
  iree_hip_hostcall_message_table_t message_table_ = {};
  FILE* output_stream_ = nullptr;
  FILE* error_stream_ = nullptr;
};

TEST_F(BlockingPrintfProtocolTest, InitializesTaggedFreeStack) {
  ASSERT_EQ(2u, protocol_.packet_count);
  EXPECT_EQ(kDeviceAddress + layout_.packet_headers_offset,
            protocol_.buffer_header->headers);
  EXPECT_EQ(kDeviceAddress + layout_.packet_payloads_offset,
            protocol_.buffer_header->payloads);
  EXPECT_EQ(kDoorbellToken, protocol_.buffer_header->doorbell);
  EXPECT_EQ(1u, protocol_.buffer_header->index_mask);
  EXPECT_EQ(1u, protocol_.index_mask);
  EXPECT_EQ(PacketTag(1), protocol_.buffer_header->free_stack);
  EXPECT_EQ(PacketTag(0), protocol_.packet_headers[1].next);
  EXPECT_EQ(0u, protocol_.packet_headers[0].next);
}

TEST(BlockingPrintfProtocolLayoutTest, CoversEveryResidentWave) {
  iree_hip_blocking_printf_protocol_layout_t layout;
  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_calculate_layout(
      /*compute_unit_count=*/120, /*maximum_waves_per_compute_unit=*/40,
      &layout));
  EXPECT_EQ(4800u, layout.packet_count);
  EXPECT_EQ(8191u, layout.index_mask);
  EXPECT_GE(layout.packet_headers_offset,
            sizeof(iree_hip_hostcall_buffer_header_t));
  EXPECT_GE(
      layout.packet_payloads_offset,
      layout.packet_headers_offset +
          layout.packet_count * sizeof(iree_hip_hostcall_packet_header_t));
  EXPECT_EQ(
      layout.packet_payloads_offset +
          layout.packet_count * sizeof(iree_hip_hostcall_packet_payload_t),
      layout.allocation_size);
}

TEST(BlockingPrintfProtocolLayoutTest, InitializesOnlyResidentWavePackets) {
  iree_hip_blocking_printf_protocol_layout_t layout;
  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_calculate_layout(
      /*compute_unit_count=*/3, /*maximum_waves_per_compute_unit=*/1, &layout));
  ASSERT_EQ(3u, layout.packet_count);
  ASSERT_EQ(3u, layout.index_mask);

  std::vector<uint64_t> storage(
      (layout.allocation_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));
  iree_hip_blocking_printf_protocol_t protocol;
  iree_hip_blocking_printf_protocol_initialize(
      storage.data(), /*device_address=*/0, /*doorbell_token=*/0, &layout,
      &protocol);

  EXPECT_EQ(2u, protocol.buffer_header->free_stack);
  EXPECT_EQ(1u, protocol.packet_headers[2].next);
  EXPECT_EQ(4u, protocol.packet_headers[1].next);
  EXPECT_EQ(0u, protocol.packet_headers[0].next);

  iree_atomic_store(&protocol.buffer_header->ready_stack, 3,
                    iree_memory_order_release);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_blocking_printf_protocol_fail_ready(&protocol));
}

TEST(BlockingPrintfProtocolLayoutTest, RejectsInvalidCapacity) {
  iree_hip_blocking_printf_protocol_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hip_blocking_printf_protocol_calculate_layout(
                            /*compute_unit_count=*/0,
                            /*maximum_waves_per_compute_unit=*/1, &layout));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hip_blocking_printf_protocol_calculate_layout(
                            UINT32_MAX, UINT32_MAX, &layout));
}

TEST_F(BlockingPrintfProtocolTest, FormatsAndCompletesPublishedPacket) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "value=%d\n", /*argument=*/42);
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(1),
                    iree_memory_order_release);

  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_process_ready(
      &protocol_, &message_table_, output_stream_, error_stream_));

  EXPECT_EQ("value=42\n", ReadStream(output_stream_));
  EXPECT_TRUE(ReadStream(error_stream_).empty());
  EXPECT_EQ(9u, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_EQ(0u, protocol_.packet_payloads[1].slots[0][1]);
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[1].control,
                                 iree_memory_order_acquire));
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.buffer_header->ready_stack,
                                 iree_memory_order_relaxed));
}

TEST_F(BlockingPrintfProtocolTest, ContinuesThenCompletesPublishedMessage) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  uint64_t* payload = protocol_.packet_payloads[1].slots[0];
  payload[0] = MakeDescriptor(kFragmentBegin, /*length=*/1);
  payload[1] = 0;
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(1),
                    iree_memory_order_release);
  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_process_ready(
      &protocol_, &message_table_, output_stream_, error_stream_));
  const uint64_t continuation_descriptor = payload[0];
  EXPECT_EQ(0u, continuation_descriptor & kFragmentBegin);
  EXPECT_EQ(0, ftell(output_stream_));

  std::array<uint64_t, 3> tail = {0, 0, 42};
  const char format[] = "value=%d\n";
  memcpy(tail.data(), format, sizeof(format));
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  payload[0] = (continuation_descriptor & ~(UINT64_C(7) << 5)) |
               MakeDescriptor(kFragmentEnd, tail.size());
  memcpy(payload + 1, tail.data(), sizeof(tail));
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(1),
                    iree_memory_order_release);
  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_process_ready(
      &protocol_, &message_table_, output_stream_, error_stream_));

  EXPECT_EQ("value=42\n", ReadStream(output_stream_));
  EXPECT_EQ(9u, payload[0]);
}

TEST_F(BlockingPrintfProtocolTest, RoutesControlBitToStandardError) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/1,
                            "error=%u", /*argument=*/7);
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(1),
                    iree_memory_order_release);

  IREE_ASSERT_OK(iree_hip_blocking_printf_protocol_process_ready(
      &protocol_, &message_table_, output_stream_, error_stream_));

  EXPECT_TRUE(ReadStream(output_stream_).empty());
  EXPECT_EQ("error=7", ReadStream(error_stream_));
}

TEST_F(BlockingPrintfProtocolTest, RejectsUnsupportedServiceAfterCompletion) {
  PreparePacket(/*packet_index=*/1, /*next=*/0, /*service=*/99,
                /*activemask=*/UINT64_C(0x3));
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(1),
                    iree_memory_order_release);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hip_blocking_printf_protocol_process_ready(
          &protocol_, &message_table_, output_stream_, error_stream_));

  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[1].slots[1][0]);
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[1].control,
                                 iree_memory_order_acquire));
}

TEST_F(BlockingPrintfProtocolTest,
       FailureStopsOutputButCompletesDetachedChain) {
  PreparePacket(/*packet_index=*/0, /*next=*/PacketTag(1),
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  // A zero-length complete fragment has no required control qword.
  protocol_.packet_payloads[0].slots[0][0] =
      MakeDescriptor(kFragmentBegin | kFragmentEnd, /*length=*/0);

  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "must not print", /*argument=*/0);
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(0),
                    iree_memory_order_release);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_blocking_printf_protocol_process_ready(
          &protocol_, &message_table_, output_stream_, error_stream_));

  EXPECT_TRUE(ReadStream(output_stream_).empty());
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[0].slots[0][0]);
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[0].control,
                                 iree_memory_order_acquire));
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[1].control,
                                 iree_memory_order_acquire));
}

TEST_F(BlockingPrintfProtocolTest, BoundsCyclicReadyStack) {
  PreparePacket(/*packet_index=*/0, /*next=*/PacketTag(1), /*service=*/99,
                /*activemask=*/1);
  PreparePacket(/*packet_index=*/1, /*next=*/PacketTag(0), /*service=*/99,
                /*activemask=*/1);
  iree_atomic_store(&protocol_.buffer_header->ready_stack, PacketTag(0),
                    iree_memory_order_release);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_blocking_printf_protocol_fail_ready(&protocol_));
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[0].control,
                                 iree_memory_order_acquire));
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[1].control,
                                 iree_memory_order_acquire));
}

}  // namespace
