// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/hostcall_message.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum FragmentFlags : uint64_t {
  kFragmentBegin = UINT64_C(1) << 0,
  kFragmentEnd = UINT64_C(1) << 1,
};

static uint64_t MakeDescriptor(uint64_t flags, iree_host_size_t length,
                               uint64_t message_id = 0) {
  return flags | ((uint64_t)length << 5) | (message_id << 8);
}

static std::array<uint64_t, IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT>
MakeFragment(uint64_t descriptor, const uint64_t* data,
             iree_host_size_t data_count) {
  std::array<uint64_t, IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT> fragment = {};
  fragment[0] = descriptor;
  if (data_count != 0) {
    memcpy(fragment.data() + 1, data, data_count * sizeof(data[0]));
  }
  return fragment;
}

static std::vector<uint64_t> CopyMessage(
    const iree_hip_hostcall_message_table_t* table,
    const iree_hip_hostcall_message_result_t& result) {
  std::vector<uint64_t> data(result.count);
  iree_hip_hostcall_message_copy(
      table, result.message_id,
      iree_make_byte_span(data.data(), data.size() * sizeof(data[0])));
  return data;
}

class HostcallMessageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hip_hostcall_message_table_initialize(iree_allocator_system(),
                                               &table_);
  }

  void TearDown() override {
    iree_hip_hostcall_message_table_deinitialize(&table_);
  }

  iree_hip_hostcall_message_table_t table_ = {};
};

TEST_F(HostcallMessageTest, CompletesSingleFragmentMessage) {
  const uint64_t data[] = {1, 2, 3};
  const auto fragment = MakeFragment(
      MakeDescriptor(kFragmentBegin | kFragmentEnd, IREE_ARRAYSIZE(data)), data,
      IREE_ARRAYSIZE(data));

  iree_hip_hostcall_message_result_t result;
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &result));
  EXPECT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_COMPLETE, result.type);
  EXPECT_EQ(std::vector<uint64_t>(std::begin(data), std::end(data)),
            CopyMessage(&table_, result));
  EXPECT_EQ(IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE,
            table_.messages[result.message_id].state);

  iree_hip_hostcall_message_release(&table_, result.message_id);
  EXPECT_EQ(IREE_HIP_HOSTCALL_MESSAGE_STATE_FREE,
            table_.messages[result.message_id].state);
}

TEST_F(HostcallMessageTest, CompletedMessageRejectsFurtherFragments) {
  const uint64_t data[] = {1, 2, 3};
  auto fragment = MakeFragment(
      MakeDescriptor(kFragmentBegin | kFragmentEnd, IREE_ARRAYSIZE(data)), data,
      IREE_ARRAYSIZE(data));
  iree_hip_hostcall_message_result_t result;
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &result));
  ASSERT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_COMPLETE, result.type);
  const iree_hip_hostcall_message_result_t completed_result = result;
  const iree_host_size_t message_id = result.message_id;
  const std::vector<uint64_t> expected_data(std::begin(data), std::end(data));

  const uint64_t late_data[] = {4};
  fragment = MakeFragment(
      MakeDescriptor(kFragmentEnd, IREE_ARRAYSIZE(late_data), message_id),
      late_data, IREE_ARRAYSIZE(late_data));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_hostcall_message_consume_fragment(
                            &table_, fragment.data(), &result));
  EXPECT_EQ(expected_data, CopyMessage(&table_, completed_result));
  EXPECT_EQ(IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE,
            table_.messages[message_id].state);

  iree_hip_hostcall_message_release(&table_, message_id);
}

TEST_F(HostcallMessageTest, AssemblesInterleavedMessagesAndReusesIds) {
  const uint64_t first_data[] = {10, 11};
  const uint64_t second_data[] = {20, 21};
  iree_hip_hostcall_message_result_t first_result;
  iree_hip_hostcall_message_result_t second_result;
  auto fragment =
      MakeFragment(MakeDescriptor(kFragmentBegin, IREE_ARRAYSIZE(first_data)),
                   first_data, IREE_ARRAYSIZE(first_data));
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &first_result));
  fragment =
      MakeFragment(MakeDescriptor(kFragmentBegin, IREE_ARRAYSIZE(second_data)),
                   second_data, IREE_ARRAYSIZE(second_data));
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &second_result));
  ASSERT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE, first_result.type);
  ASSERT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE, second_result.type);
  const uint64_t first_id = first_result.continuation_descriptor >> 8;
  const uint64_t second_id = second_result.continuation_descriptor >> 8;
  EXPECT_NE(first_id, second_id);

  const uint64_t first_tail[] = {12};
  fragment = MakeFragment(
      MakeDescriptor(kFragmentEnd, IREE_ARRAYSIZE(first_tail), first_id),
      first_tail, IREE_ARRAYSIZE(first_tail));
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &first_result));
  const uint64_t expected_first[] = {10, 11, 12};
  EXPECT_EQ(std::vector<uint64_t>(std::begin(expected_first),
                                  std::end(expected_first)),
            CopyMessage(&table_, first_result));
  iree_hip_hostcall_message_release(&table_, first_result.message_id);

  fragment = MakeFragment(MakeDescriptor(kFragmentBegin, 0), nullptr, 0);
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &first_result));
  EXPECT_EQ(first_id, first_result.continuation_descriptor >> 8);
}

TEST_F(HostcallMessageTest, ReusesReleasedPayloadBlocks) {
  const uint64_t data[] = {1, 2, 3};
  const auto fragment = MakeFragment(
      MakeDescriptor(kFragmentBegin | kFragmentEnd, IREE_ARRAYSIZE(data)), data,
      IREE_ARRAYSIZE(data));

  iree_hip_hostcall_message_result_t first_result;
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &first_result));
  iree_arena_block_t* first_block =
      table_.messages[first_result.message_id].block_head;
  ASSERT_NE(nullptr, first_block);
  iree_hip_hostcall_message_release(&table_, first_result.message_id);

  iree_hip_hostcall_message_result_t second_result;
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &second_result));
  EXPECT_EQ(first_block, table_.messages[second_result.message_id].block_head);
  iree_hip_hostcall_message_release(&table_, second_result.message_id);
}

TEST_F(HostcallMessageTest, RejectsUnknownMessageId) {
  const auto fragment = MakeFragment(MakeDescriptor(/*flags=*/0, /*length=*/0,
                                                    /*message_id=*/42),
                                     nullptr, 0);

  iree_hip_hostcall_message_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_hostcall_message_consume_fragment(
                            &table_, fragment.data(), &result));
  EXPECT_EQ(0u, result.count);
}

TEST_F(HostcallMessageTest, ReservedBitsDiscardContinuedMessage) {
  auto fragment = MakeFragment(MakeDescriptor(kFragmentBegin, 0), nullptr, 0);
  iree_hip_hostcall_message_result_t result;
  IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
      &table_, fragment.data(), &result));
  const uint64_t message_id = result.continuation_descriptor >> 8;

  fragment =
      MakeFragment(MakeDescriptor(UINT64_C(1) << 2, 0, message_id), nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_hostcall_message_consume_fragment(
                            &table_, fragment.data(), &result));
  EXPECT_EQ(IREE_HIP_HOSTCALL_MESSAGE_STATE_FREE,
            table_.messages[message_id].state);
}

TEST_F(HostcallMessageTest, AssemblesLargeMessage) {
  constexpr iree_host_size_t kFragmentQwordCount =
      IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT - 1;
  constexpr iree_host_size_t kQwordCount = 128 * 1024 + 17;
  std::vector<uint64_t> expected_data(kQwordCount);
  for (iree_host_size_t i = 0; i < expected_data.size(); ++i) {
    expected_data[i] = i;
  }

  std::array<uint64_t, IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT> fragment = {};
  iree_hip_hostcall_message_result_t result;
  uint64_t message_id = 0;
  for (iree_host_size_t offset = 0; offset < expected_data.size();) {
    const iree_host_size_t fragment_count =
        iree_min(kFragmentQwordCount, expected_data.size() - offset);
    const bool is_first = offset == 0;
    const bool is_last = offset + fragment_count == expected_data.size();
    const uint64_t flags =
        (is_first ? kFragmentBegin : 0) | (is_last ? kFragmentEnd : 0);
    fragment = MakeFragment(MakeDescriptor(flags, fragment_count, message_id),
                            expected_data.data() + offset, fragment_count);
    IREE_ASSERT_OK(iree_hip_hostcall_message_consume_fragment(
        &table_, fragment.data(), &result));
    if (!is_last) {
      ASSERT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE, result.type);
      message_id = result.continuation_descriptor >> 8;
    }
    offset += fragment_count;
  }

  ASSERT_EQ(IREE_HIP_HOSTCALL_MESSAGE_RESULT_COMPLETE, result.type);
  EXPECT_EQ(expected_data, CopyMessage(&table_, result));
  iree_hip_hostcall_message_release(&table_, result.message_id);
}

}  // namespace
