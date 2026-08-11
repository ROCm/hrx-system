// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/protocol/commands.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/hal/command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void InitializeHeader(iree_hal_remote_cmd_type_t type,
                             std::vector<uint8_t>* command_bytes) {
  auto* header =
      reinterpret_cast<iree_hal_remote_cmd_header_t*>(command_bytes->data());
  memset(header, 0, sizeof(*header));
  header->type = static_cast<uint16_t>(type);
  header->length = static_cast<uint32_t>(command_bytes->size());
}

static void ExpectInvalid(std::vector<uint8_t> command_bytes) {
  iree_hal_remote_command_view_t command;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_remote_command_parse(
          iree_make_const_byte_span(command_bytes.data(), command_bytes.size()),
          &command));
  EXPECT_EQ(command.bytes.data, nullptr);
  EXPECT_EQ(command.bytes.data_length, 0u);
}

TEST(CommandParseTest, RejectsMalformedCommonHeader) {
  ExpectInvalid(std::vector<uint8_t>(sizeof(iree_hal_remote_cmd_header_t) - 1));

  std::vector<uint8_t> command_bytes(
      sizeof(iree_hal_remote_debug_group_end_cmd_t));
  InitializeHeader(IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END, &command_bytes);

  auto* header =
      reinterpret_cast<iree_hal_remote_cmd_header_t*>(command_bytes.data());
  header->reserved = 1;
  ExpectInvalid(command_bytes);

  header->reserved = 0;
  header->length = sizeof(*header) - 1;
  ExpectInvalid(command_bytes);

  header->length = sizeof(*header) + 1;
  ExpectInvalid(command_bytes);

  header->length = sizeof(*header) * 2;
  ExpectInvalid(command_bytes);
}

TEST(CommandParseTest, RejectsNoncanonicalFixedRecord) {
  std::vector<uint8_t> command_bytes(
      sizeof(iree_hal_remote_debug_group_end_cmd_t) + 8);
  InitializeHeader(IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END, &command_bytes);
  ExpectInvalid(std::move(command_bytes));
}

TEST(CommandParseTest, ParsesMaximumUpdateAndFollowingCommand) {
  constexpr iree_host_size_t kUpdateLength =
      IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE;
  constexpr iree_host_size_t kUpdateCommandLength =
      sizeof(iree_hal_remote_buffer_update_cmd_t) + kUpdateLength;
  static_assert(kUpdateCommandLength == 65576);

  std::vector<uint8_t> stream_bytes(
      kUpdateCommandLength + sizeof(iree_hal_remote_debug_group_end_cmd_t));
  auto* update = reinterpret_cast<iree_hal_remote_buffer_update_cmd_t*>(
      stream_bytes.data());
  memset(update, 0, sizeof(*update));
  update->header.type = IREE_HAL_REMOTE_CMD_BUFFER_UPDATE;
  update->header.length = static_cast<uint32_t>(kUpdateCommandLength);
  update->target_length = kUpdateLength;

  auto* debug_end = reinterpret_cast<iree_hal_remote_debug_group_end_cmd_t*>(
      stream_bytes.data() + kUpdateCommandLength);
  memset(debug_end, 0, sizeof(*debug_end));
  debug_end->header.type = IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END;
  debug_end->header.length = sizeof(*debug_end);

  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream_bytes.data(), stream_bytes.size()),
      &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_BUFFER_UPDATE);
  EXPECT_EQ(command.header.length, kUpdateCommandLength);
  EXPECT_EQ(command.bytes.data_length, kUpdateCommandLength);

  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(
          stream_bytes.data() + command.bytes.data_length,
          stream_bytes.size() - command.bytes.data_length),
      &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END);
  EXPECT_EQ(command.bytes.data_length, sizeof(*debug_end));
}

TEST(CommandParseTest, RejectsUpdatePayloadMismatch) {
  std::vector<uint8_t> command_bytes(
      sizeof(iree_hal_remote_buffer_update_cmd_t));
  InitializeHeader(IREE_HAL_REMOTE_CMD_BUFFER_UPDATE, &command_bytes);
  auto* update = reinterpret_cast<iree_hal_remote_buffer_update_cmd_t*>(
      command_bytes.data());
  update->target_length = 1;
  ExpectInvalid(std::move(command_bytes));
}

TEST(CommandParseTest, RejectsVariableTailMismatch) {
  std::vector<uint8_t> command_bytes(
      sizeof(iree_hal_remote_debug_group_begin_cmd_t));
  InitializeHeader(IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN, &command_bytes);
  auto* debug_begin =
      reinterpret_cast<iree_hal_remote_debug_group_begin_cmd_t*>(
          command_bytes.data());
  debug_begin->label_length = 1;
  ExpectInvalid(command_bytes);

  command_bytes.assign(sizeof(iree_hal_remote_execution_barrier_cmd_t), 0);
  InitializeHeader(IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER, &command_bytes);
  auto* barrier = reinterpret_cast<iree_hal_remote_execution_barrier_cmd_t*>(
      command_bytes.data());
  barrier->memory_barrier_count = 1;
  ExpectInvalid(command_bytes);

  command_bytes.assign(sizeof(iree_hal_remote_dispatch_cmd_t), 0);
  InitializeHeader(IREE_HAL_REMOTE_CMD_DISPATCH, &command_bytes);
  auto* dispatch =
      reinterpret_cast<iree_hal_remote_dispatch_cmd_t*>(command_bytes.data());
  dispatch->constant_count = 1;
  ExpectInvalid(std::move(command_bytes));
}

TEST(CommandParseTest, RejectsCommandSpecificInvalidFields) {
  std::vector<uint8_t> command_bytes(
      sizeof(iree_hal_remote_event_signal_cmd_t));
  InitializeHeader(IREE_HAL_REMOTE_CMD_EVENT_SIGNAL, &command_bytes);
  auto* event_signal = reinterpret_cast<iree_hal_remote_event_signal_cmd_t*>(
      command_bytes.data());
  event_signal->reserved = 1;
  ExpectInvalid(command_bytes);

  command_bytes.assign(sizeof(iree_hal_remote_buffer_fill_cmd_t), 0);
  InitializeHeader(IREE_HAL_REMOTE_CMD_BUFFER_FILL, &command_bytes);
  auto* fill = reinterpret_cast<iree_hal_remote_buffer_fill_cmd_t*>(
      command_bytes.data());
  fill->pattern_length = 3;
  ExpectInvalid(command_bytes);

  command_bytes.assign(sizeof(iree_hal_remote_buffer_copy_cmd_t), 0);
  InitializeHeader(IREE_HAL_REMOTE_CMD_BUFFER_COPY, &command_bytes);
  auto* copy = reinterpret_cast<iree_hal_remote_buffer_copy_cmd_t*>(
      command_bytes.data());
  copy->reserved = 1;
  ExpectInvalid(std::move(command_bytes));
}

TEST(CommandParseTest, ParsesDispatchBeyondSixteenBitLength) {
  constexpr uint16_t kConstantCount = 16383;
  constexpr iree_host_size_t kConstantsLength =
      kConstantCount * sizeof(uint32_t);
  constexpr iree_host_size_t kCommandLength = 65632;
  static_assert(kConstantsLength == 65532);

  std::vector<uint8_t> command_bytes(kCommandLength);
  InitializeHeader(IREE_HAL_REMOTE_CMD_DISPATCH, &command_bytes);
  auto* dispatch =
      reinterpret_cast<iree_hal_remote_dispatch_cmd_t*>(command_bytes.data());
  dispatch->constant_count = kConstantCount;

  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(command_bytes.data(), command_bytes.size()),
      &command));
  EXPECT_EQ(command.header.length, kCommandLength);
  EXPECT_EQ(command.bytes.data_length, kCommandLength);
}

TEST(CommandParseTest, RejectsNestedReservedFields) {
  constexpr iree_host_size_t kBarrierCommandLength =
      sizeof(iree_hal_remote_execution_barrier_cmd_t) +
      sizeof(iree_hal_remote_buffer_barrier_t);
  std::vector<uint8_t> command_bytes(kBarrierCommandLength);
  InitializeHeader(IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER, &command_bytes);
  auto* barrier = reinterpret_cast<iree_hal_remote_execution_barrier_cmd_t*>(
      command_bytes.data());
  barrier->buffer_barrier_count = 1;
  auto* buffer_barrier = reinterpret_cast<iree_hal_remote_buffer_barrier_t*>(
      command_bytes.data() + sizeof(*barrier));
  buffer_barrier->reserved = 1;
  ExpectInvalid(command_bytes);

  constexpr iree_host_size_t kDispatchCommandLength =
      sizeof(iree_hal_remote_dispatch_cmd_t) +
      sizeof(iree_hal_remote_binding_t);
  command_bytes.assign(kDispatchCommandLength, 0);
  InitializeHeader(IREE_HAL_REMOTE_CMD_DISPATCH, &command_bytes);
  auto* dispatch =
      reinterpret_cast<iree_hal_remote_dispatch_cmd_t*>(command_bytes.data());
  dispatch->binding_count = 1;
  auto* binding = reinterpret_cast<iree_hal_remote_binding_t*>(
      command_bytes.data() + sizeof(*dispatch));
  binding->reserved = 1;
  ExpectInvalid(std::move(command_bytes));
}

TEST(CommandParseTest, ParsesMaximumDebugLabel) {
  constexpr iree_host_size_t kLabelLength = UINT16_MAX;
  constexpr iree_host_size_t kCommandLength = 65552;
  std::vector<uint8_t> command_bytes(kCommandLength);
  InitializeHeader(IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN, &command_bytes);
  auto* debug_begin =
      reinterpret_cast<iree_hal_remote_debug_group_begin_cmd_t*>(
          command_bytes.data());
  debug_begin->label_length = kLabelLength;

  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(command_bytes.data(), command_bytes.size()),
      &command));
  EXPECT_EQ(command.bytes.data_length, kCommandLength);
}

}  // namespace
