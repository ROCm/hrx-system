// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/blocking_printf_protocol.h"

#include <array>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
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

struct OutputEvent {
  // Legacy output stream selected by the message control qword.
  iree_hip_blocking_printf_stream_t stream;
  // Complete formatted message copied during the callback.
  std::string text;
};

struct ServiceRecorder {
  // Complete messages observed by the output callback.
  std::vector<OutputEvent> output_events;
  // Status code consumed by the structural error callback.
  iree_status_code_t failure_code = IREE_STATUS_OK;
  // Number of structural error callbacks observed.
  int failure_count = 0;
  // Packet header inspected while a callback is active.
  iree_hip_hostcall_packet_header_t* observed_header = nullptr;
  // Whether the output callback observed its packet still READY.
  bool output_observed_ready = false;
  // Whether the error callback observed its packet still READY.
  bool failure_observed_ready = false;
  // Whether the output callback waits for |output_release|.
  bool gate_output = false;
  // Set after the gated output callback has been entered.
  std::atomic<bool> output_entered{false};
  // Set to release the gated output callback.
  std::atomic<bool> output_release{false};
};

static bool HeaderIsReady(
    const iree_hip_hostcall_packet_header_t* packet_header) {
  return packet_header &&
         iree_any_bit_set(iree_atomic_load(&packet_header->control,
                                           iree_memory_order_acquire),
                          IREE_HIP_HOSTCALL_PACKET_CONTROL_READY);
}

static void CaptureOutput(void* user_data,
                          iree_hip_blocking_printf_stream_t stream,
                          iree_string_view_t text) {
  ServiceRecorder* recorder = (ServiceRecorder*)user_data;
  OutputEvent event;
  event.stream = stream;
  if (text.size != 0) event.text.assign(text.data, text.size);
  recorder->output_events.push_back(std::move(event));
  recorder->output_observed_ready = HeaderIsReady(recorder->observed_header);
  recorder->output_entered.store(true, std::memory_order_release);
  while (recorder->gate_output &&
         !recorder->output_release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

static void CaptureFailure(void* user_data, iree_status_t status) {
  ServiceRecorder* recorder = (ServiceRecorder*)user_data;
  recorder->failure_code = iree_status_code(status);
  ++recorder->failure_count;
  recorder->failure_observed_ready = HeaderIsReady(recorder->observed_header);
  iree_status_free(status);
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
    iree_hip_blocking_printf_output_sink_t output_sink = {
        /*.fn=*/CaptureOutput,
        /*.user_data=*/&recorder_,
    };
    iree_hal_hostcall_error_callback_t error_callback = {
        /*.fn=*/CaptureFailure,
        /*.user_data=*/&recorder_,
    };
    iree_hip_blocking_printf_service_initialize(
        &protocol_, output_sink, error_callback, iree_allocator_system(),
        &service_);
  }

  void TearDown() override {
    iree_hip_blocking_printf_service_deinitialize(&service_);
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

  void PublishPacket(uint32_t packet_index) {
    iree_atomic_store(&protocol_.buffer_header->ready_stack,
                      PacketTag(packet_index), iree_memory_order_release);
  }

  void ProcessReady() {
    iree_hip_blocking_printf_service_process_ready(&service_);
  }

  static constexpr uint64_t kDeviceAddress = UINT64_C(0x10000000);
  static constexpr uint64_t kDoorbellToken = UINT64_C(0x12345678);
  iree_hip_blocking_printf_protocol_layout_t layout_ = {};
  std::vector<uint64_t> storage_;
  iree_hip_blocking_printf_protocol_t protocol_ = {};
  ServiceRecorder recorder_;
  iree_hip_blocking_printf_service_t service_ = {};
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

TEST_F(BlockingPrintfProtocolTest,
       PublishesCompleteOutputBeforeCompletingPacket) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "value=%d\n", /*argument=*/42);
  recorder_.observed_header = &protocol_.packet_headers[1];
  PublishPacket(1);

  ProcessReady();

  ASSERT_EQ(1u, recorder_.output_events.size());
  EXPECT_EQ(IREE_HIP_BLOCKING_PRINTF_STREAM_STDOUT,
            recorder_.output_events[0].stream);
  EXPECT_EQ("value=42\n", recorder_.output_events[0].text);
  EXPECT_TRUE(recorder_.output_observed_ready);
  EXPECT_EQ(9u, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_EQ(0u, protocol_.packet_payloads[1].slots[0][1]);
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.packet_headers[1].control,
                                 iree_memory_order_acquire));
  EXPECT_EQ(0u, iree_atomic_load(&protocol_.buffer_header->ready_stack,
                                 iree_memory_order_relaxed));
}

TEST_F(BlockingPrintfProtocolTest, HoldsCompletionUntilOutputReturns) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "value=%d\n", /*argument=*/42);
  recorder_.observed_header = &protocol_.packet_headers[1];
  recorder_.gate_output = true;
  PublishPacket(1);

  std::atomic<bool> process_returned{false};
  std::thread service_thread([&]() {
    ProcessReady();
    process_returned.store(true, std::memory_order_release);
  });
  while (!recorder_.output_entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  EXPECT_TRUE(HeaderIsReady(&protocol_.packet_headers[1]));
  EXPECT_FALSE(process_returned.load(std::memory_order_acquire));
  EXPECT_NE(9u, protocol_.packet_payloads[1].slots[0][0]);

  recorder_.output_release.store(true, std::memory_order_release);
  service_thread.join();

  EXPECT_TRUE(process_returned.load(std::memory_order_acquire));
  EXPECT_EQ(9u, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));
}

TEST_F(BlockingPrintfProtocolTest, ContinuesThenCompletesPublishedMessage) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  uint64_t* payload = protocol_.packet_payloads[1].slots[0];
  payload[0] = MakeDescriptor(kFragmentBegin, /*length=*/1);
  payload[1] = 0;
  PublishPacket(1);
  ProcessReady();
  const uint64_t continuation_descriptor = payload[0];
  EXPECT_EQ(0u, continuation_descriptor & kFragmentBegin);
  EXPECT_TRUE(recorder_.output_events.empty());

  std::array<uint64_t, 3> tail = {0, 0, 42};
  const char format[] = "value=%d\n";
  memcpy(tail.data(), format, sizeof(format));
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  payload[0] = (continuation_descriptor & ~(UINT64_C(7) << 5)) |
               MakeDescriptor(kFragmentEnd, tail.size());
  memcpy(payload + 1, tail.data(), sizeof(tail));
  PublishPacket(1);
  ProcessReady();

  ASSERT_EQ(1u, recorder_.output_events.size());
  EXPECT_EQ("value=42\n", recorder_.output_events[0].text);
  EXPECT_EQ(9u, payload[0]);
}

TEST_F(BlockingPrintfProtocolTest, RoutesControlBitToStandardError) {
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/1,
                            "error=%u", /*argument=*/7);
  PublishPacket(1);

  ProcessReady();

  ASSERT_EQ(1u, recorder_.output_events.size());
  EXPECT_EQ(IREE_HIP_BLOCKING_PRINTF_STREAM_STDERR,
            recorder_.output_events[0].stream);
  EXPECT_EQ("error=7", recorder_.output_events[0].text);
}

TEST_F(BlockingPrintfProtocolTest, LocalFailureDoesNotPoisonService) {
  PreparePacket(/*packet_index=*/0, /*next=*/PacketTag(1),
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  // A zero-length complete fragment has no required control qword.
  protocol_.packet_payloads[0].slots[0][0] =
      MakeDescriptor(kFragmentBegin | kFragmentEnd, /*length=*/0);

  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "still works", /*argument=*/0);
  PublishPacket(0);

  ProcessReady();

  EXPECT_EQ(0, recorder_.failure_count);
  ASSERT_EQ(1u, recorder_.output_events.size());
  EXPECT_EQ("still works", recorder_.output_events[0].text);
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[0].slots[0][0]);
  EXPECT_EQ(11u, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[0]));
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));
}

TEST_F(BlockingPrintfProtocolTest, FormatsLargeMessage) {
  constexpr iree_host_size_t kFragmentQwordCount =
      IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT - 1;
  const std::string expected_output(1024 * 1024 + 1, 'x');
  std::vector<uint64_t> message(
      1 +
      (expected_output.size() + 1 + sizeof(uint64_t) - 1) / sizeof(uint64_t));
  memcpy(message.data() + 1, expected_output.data(), expected_output.size());

  uint64_t* payload = protocol_.packet_payloads[1].slots[0];
  uint64_t continuation_descriptor = 0;
  for (iree_host_size_t offset = 0; offset < message.size();) {
    const iree_host_size_t fragment_count =
        iree_min(kFragmentQwordCount, message.size() - offset);
    const bool is_first = offset == 0;
    const bool is_last = offset + fragment_count == message.size();
    const uint64_t flags =
        (is_first ? kFragmentBegin : 0) | (is_last ? kFragmentEnd : 0);
    PreparePacket(/*packet_index=*/1, /*next=*/0,
                  IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
    payload[0] = (continuation_descriptor & ~(UINT64_C(7) << 5)) |
                 MakeDescriptor(flags, fragment_count);
    memcpy(payload + 1, message.data() + offset,
           fragment_count * sizeof(message[0]));
    PublishPacket(1);
    ProcessReady();
    continuation_descriptor = payload[0];
    offset += fragment_count;
  }

  EXPECT_EQ(0, recorder_.failure_count);
  ASSERT_EQ(1u, recorder_.output_events.size());
  EXPECT_EQ(expected_output, recorder_.output_events[0].text);
  EXPECT_EQ(expected_output.size(), payload[0]);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));
}

TEST_F(BlockingPrintfProtocolTest,
       StructuralFailurePrecedesReleaseAndFailCompletesLaterPackets) {
  PreparePacket(/*packet_index=*/0, /*next=*/PacketTag(1), /*service=*/99,
                /*activemask=*/UINT64_C(0x3));
  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "must not print", /*argument=*/0);
  recorder_.observed_header = &protocol_.packet_headers[0];
  PublishPacket(0);

  ProcessReady();

  EXPECT_EQ(1, recorder_.failure_count);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, recorder_.failure_code);
  EXPECT_TRUE(recorder_.failure_observed_ready);
  EXPECT_TRUE(recorder_.output_events.empty());
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[0].slots[0][0]);
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[0].slots[1][0]);
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[0]));
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));

  PreparePacket(/*packet_index=*/1, /*next=*/0,
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "not printed", /*argument=*/0);
  PublishPacket(1);
  ProcessReady();

  EXPECT_EQ(1, recorder_.failure_count);
  EXPECT_TRUE(recorder_.output_events.empty());
  EXPECT_EQ(UINT64_MAX, protocol_.packet_payloads[1].slots[0][0]);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));
}

TEST_F(BlockingPrintfProtocolTest, BoundsCyclicReadyStack) {
  PreparePacket(/*packet_index=*/0, /*next=*/PacketTag(1),
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/0, /*lane=*/0, /*control=*/0,
                            "first", /*argument=*/0);
  PreparePacket(/*packet_index=*/1, /*next=*/PacketTag(0),
                IREE_HIP_HOSTCALL_SERVICE_PRINTF, /*activemask=*/1);
  SetCompletePrintfFragment(/*packet_index=*/1, /*lane=*/0, /*control=*/0,
                            "second", /*argument=*/0);
  PublishPacket(0);

  ProcessReady();

  EXPECT_EQ(1, recorder_.failure_count);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, recorder_.failure_code);
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[0]));
  EXPECT_FALSE(HeaderIsReady(&protocol_.packet_headers[1]));
}

}  // namespace
