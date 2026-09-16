// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/submission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"

namespace {

uint32_t ReadU32(const void* bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

uint64_t ReadU64(const void* bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

class WindowsXdnaSubmissionProtocolTest
    : public ::testing::TestWithParam<amdf_windows_xdna_protocol_t> {};

TEST_P(WindowsXdnaSubmissionProtocolTest, BuildsContextLifecycleRecords) {
  const bool direct = GetParam() == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  const uint32_t header_length = direct ? 120 : 104;
  const size_t response_offset = direct ? 0x38 : 0x30;
  std::array<uint8_t, 4096> command_bytes = {};
  for (size_t i = 0; i < 512; ++i) {
    command_bytes[i] = static_cast<uint8_t>(i);
  }
  amdf_windows_xdna_private_allocation_t instruction = {};
  instruction.allocation = 0x10;
  instruction.descriptor.allocation_byte_length = UINT64_C(0x4000000);
  amdf_windows_xdna_private_allocation_t command = {};
  command.allocation = 0x20;
  command.device_address = UINT64_C(0x100000000);
  command.host_pointer = command_bytes.data();

  amdf_windows_xdna_submission_t aperture = {};
  amdf_windows_xdna_submission_build_aperture(GetParam(), &instruction,
                                              &aperture);
  EXPECT_EQ(aperture.byte_length, header_length);
  EXPECT_EQ(ReadU64(aperture.bytes, 0x00), 2u);
  EXPECT_EQ(ReadU64(aperture.bytes, 0x08), 0x10u);
  EXPECT_EQ(ReadU64(aperture.bytes, 0x10), UINT64_C(0x4000000));

  amdf_windows_xdna_submission_t initialize = {};
  amdf_windows_xdna_submission_build_context_initialize(GetParam(), &command,
                                                        &initialize);
  EXPECT_EQ(initialize.byte_length, header_length + 520u);
  EXPECT_EQ(ReadU64(initialize.bytes, 0x00), 5u);
  EXPECT_EQ(ReadU64(initialize.bytes, 0x28), 0x20u);
  if (direct)
    EXPECT_EQ(ReadU64(initialize.bytes, 0x30), command.device_address);
  EXPECT_EQ(ReadU64(initialize.bytes, 0x48), 0u);
  EXPECT_EQ(ReadU32(initialize.bytes, response_offset), 0u);
  EXPECT_EQ(ReadU32(initialize.bytes, response_offset + 4), 8u);
  EXPECT_EQ(ReadU64(initialize.bytes, response_offset + 8),
            reinterpret_cast<uintptr_t>(command_bytes.data()));
  EXPECT_EQ(
      std::memcmp(initialize.bytes + header_length, command_bytes.data(), 512),
      0);

  amdf_windows_xdna_submission_t accounting = {};
  amdf_windows_xdna_submission_build_accounting(GetParam(), &instruction,
                                                UINT64_C(0x20000), &accounting);
  EXPECT_EQ(accounting.byte_length, header_length);
  EXPECT_EQ(ReadU64(accounting.bytes, 0x00), 9u);
  EXPECT_EQ(ReadU64(accounting.bytes, 0x08), direct ? 0u : 0x10u);
  EXPECT_EQ(ReadU64(accounting.bytes, 0x10), UINT64_C(0x20000));
}

TEST_P(WindowsXdnaSubmissionProtocolTest,
       BuildsInstructionRangeExecutionRecords) {
  const bool direct = GetParam() == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  const uint32_t header_length = direct ? 120 : 104;
  const size_t response_offset = direct ? 0x38 : 0x30;
  amdf_xdna_transaction_interpreter_packet_t packet = {};
  amdf_xdna_transaction_interpreter_packet_build(UINT64_C(0x04008000), 300,
                                                 &packet);
  EXPECT_EQ(ReadU32(packet.bytes, 0x00), 0x30010001u);
  EXPECT_EQ(ReadU32(packet.bytes, 0x04), 1u);
  EXPECT_EQ(ReadU32(packet.bytes, 0x08), 3u);
  EXPECT_EQ(ReadU64(packet.bytes, 0x10), UINT64_C(0x04008000));
  EXPECT_EQ(ReadU32(packet.bytes, 0x18), 75u);
  for (size_t i = 0x1C; i < sizeof(packet); ++i) EXPECT_EQ(packet.bytes[i], 0);

  std::array<uint8_t, 4096> command_bytes = {};
  amdf_windows_xdna_private_allocation_t execution = {};
  execution.allocation = 0x30;
  amdf_windows_xdna_private_allocation_t command = {};
  command.allocation = 0x20;
  command.device_address = UINT64_C(0x100000000);
  command.host_pointer = command_bytes.data();
  amdf_windows_xdna_submission_t submission = {};
  amdf_windows_xdna_submission_build_execute(GetParam(), &execution, &command,
                                             &packet, &submission);
  EXPECT_EQ(submission.byte_length, header_length + 512u);
  EXPECT_EQ(ReadU64(submission.bytes, 0x00), 3u);
  EXPECT_EQ(ReadU64(submission.bytes, 0x08), 0x30u);
  EXPECT_EQ(ReadU64(submission.bytes, 0x10), 68u);
  EXPECT_EQ(ReadU64(submission.bytes, 0x28), 0x20u);
  if (direct)
    EXPECT_EQ(ReadU64(submission.bytes, 0x30), command.device_address);
  EXPECT_EQ(ReadU64(submission.bytes, 0x48), 0u);
  EXPECT_EQ(ReadU32(submission.bytes, response_offset), 8u);
  EXPECT_EQ(ReadU32(submission.bytes, response_offset + 4), 8u);
  EXPECT_EQ(ReadU64(submission.bytes, response_offset + 8),
            reinterpret_cast<uintptr_t>(command_bytes.data() + 8));
  EXPECT_EQ(std::memcmp(submission.bytes + header_length, packet.bytes,
                        sizeof(packet.bytes)),
            0);
}

INSTANTIATE_TEST_SUITE_P(
    NativeInterfaces, WindowsXdnaSubmissionProtocolTest,
    ::testing::Values(AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT,
                      AMDF_WINDOWS_XDNA_PROTOCOL_METADATA));

TEST(WindowsXdnaSubmissionLayoutTest, ReadsInitializationBooleanResponse) {
  // The driver writes only the low 32 bits of the eight-byte response cell.
  std::array<uint32_t, 4> command_words = {1, 0xA5A5A5A5u, 4, 0};
  amdf_windows_xdna_private_allocation_t command = {};
  command.host_pointer = command_words.data();
  EXPECT_EQ(amdf_windows_xdna_submission_query_initialize_result(&command),
            AMDF_STATUS_OK);
  command_words[0] = 0;
  EXPECT_EQ(amdf_windows_xdna_submission_query_initialize_result(&command),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  command_words[0] = 2;
  EXPECT_EQ(amdf_windows_xdna_submission_query_initialize_result(&command),
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(command_words[1], 0xA5A5A5A5u);
  EXPECT_EQ(command_words[2], 4u);
}

TEST(WindowsXdnaSubmissionLayoutTest, ReadsExecutionResponseState) {
  std::array<uint32_t, 4> command_words = {1, 0, 0, 0xA5A5A5A5u};
  amdf_windows_xdna_private_allocation_t command = {};
  command.host_pointer = command_words.data();
  for (uint32_t state = 0; state < 16; ++state) {
    SCOPED_TRACE(state);
    // Non-state header bits and the untouched high word are not error codes.
    const uint32_t response = 0x30010000u | state;
    command_words[2] = response;
    const amdf_status_t status =
        amdf_windows_xdna_submission_query_execute_result(&command);
    if (state == 0) {
      EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
    } else if (state == 4) {
      EXPECT_EQ(status, AMDF_STATUS_OK);
    } else {
      EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_FIRMWARE);
      EXPECT_EQ(amdf_status_code(status), state);
    }
    EXPECT_EQ(command_words[0], 1u);
    EXPECT_EQ(command_words[2], response);
    EXPECT_EQ(command_words[3], 0xA5A5A5A5u);
  }
}

}  // namespace
