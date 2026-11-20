// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_broadcast.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ProgramBroadcastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Independently model the tile-control destination. Guards make writes
  // outside a receiver's individual code extent observable.
  using Memory = std::array<uint8_t, 96>;
  static void Write(const loom_aie2p_worker_code_stream_t& stream,
                    Memory& memory) {
    for (size_t i = 0; i < stream.word_count;) {
      ASSERT_GE(stream.word_count - i, 3u);
      EXPECT_EQ(stream.words[i++], 0x80000000u);
      const uint32_t command = stream.words[i++];
      const size_t address = command & 0xfffffu;
      const size_t words = 1 + ((command >> 20) & 3u);
      ASSERT_LE(words, stream.word_count - i);
      ASSERT_LE(address + words * 4, memory.size());
      ASSERT_LE((address & 15u) + words * 4, 16u);
      for (size_t j = 0; j < words; ++j) {
        const uint32_t word = stream.words[i++];
        for (size_t byte = 0; byte < 4; ++byte) {
          memory[address + j * 4 + byte] =
              static_cast<uint8_t>(word >> (8 * byte));
        }
      }
    }
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(ProgramBroadcastTest, BroadcastsMostFrequentRelocatedWords) {
  // In block zero two workers agree with the first and two agree with the
  // second. In block one the last three workers agree. Block two is private.
  std::array<std::array<uint8_t, 48>, 4> programs;
  loom_aie2p_worker_code_t workers[4];
  for (size_t i = 0; i < programs.size(); ++i) {
    auto& bytes = programs[i];
    std::memset(bytes.data(), (i % 2) ? 1 : 8, 16);
    std::memset(bytes.data() + 16, i ? 5 : 2, 16);
    std::memset(bytes.data() + 32, static_cast<int>(i), 16);
    workers[i] = {16, iree_make_const_byte_span(bytes.data(), bytes.size())};
  }
  loom_aie2p_program_broadcast_t plan;
  IREE_ASSERT_OK(
      loom_aie2p_program_broadcast_build(workers, 4, &arena_, &plan));
  EXPECT_EQ(plan.shared.word_count, 12u);
  Memory shared;
  shared.fill(0xee);
  Write(plan.shared, shared);
  EXPECT_EQ(shared[16], 8u);
  EXPECT_EQ(shared[32], 5u);
  EXPECT_EQ(shared[48], 0xeeu);
  for (size_t i = 0; i < programs.size(); ++i) {
    Memory result = shared, expected;
    expected.fill(0xee);
    std::memcpy(expected.data() + 16, programs[i].data(), programs[i].size());
    Write(plan.individual[i], result);
    EXPECT_EQ(result, expected);
  }
}

TEST_F(ProgramBroadcastTest, PreservesOffsetsAndShorterCodeExtents) {
  std::array<uint8_t, 64> contents;
  for (size_t i = 0; i < contents.size(); ++i) {
    contents[i] = static_cast<uint8_t>(i);
  }
  const loom_aie2p_worker_code_t workers[] = {
      {12, iree_make_const_byte_span(contents.data() + 12, 38)},
      {16, iree_make_const_byte_span(contents.data() + 16, 22)},
      {20, iree_make_const_byte_span(contents.data() + 20, 26)},
  };
  loom_aie2p_program_broadcast_t plan;
  IREE_ASSERT_OK(loom_aie2p_program_broadcast_build(
      workers, IREE_ARRAYSIZE(workers), &arena_, &plan));
  EXPECT_EQ(plan.shared.word_count, 0u);
  for (size_t i = 0; i < IREE_ARRAYSIZE(workers); ++i) {
    const auto& worker = workers[i];
    Memory result, expected;
    result.fill(0xcc);
    expected.fill(0xcc);
    const size_t length = worker.contents.data_length;
    std::memset(expected.data() + worker.address, 0, (length + 3) / 4 * 4);
    std::memcpy(expected.data() + worker.address, worker.contents.data, length);
    Write(plan.individual[i], result);
    EXPECT_EQ(result, expected);
  }
}

TEST_F(ProgramBroadcastTest, IdenticalProgramsNeedOnlyOneDelivery) {
  const uint8_t bytes[] = {0x71, 0x32, 0x44, 0xa0, 0x9b, 0xc2};
  const loom_aie2p_worker_code_t code = {
      28, iree_make_const_byte_span(bytes, sizeof(bytes))};
  const loom_aie2p_worker_code_t workers[] = {code, code, code};
  loom_aie2p_program_broadcast_t plan;
  IREE_ASSERT_OK(loom_aie2p_program_broadcast_build(
      workers, IREE_ARRAYSIZE(workers), &arena_, &plan));
  EXPECT_EQ(plan.shared.word_count, 6u);
  for (size_t i = 0; i < IREE_ARRAYSIZE(workers); ++i) {
    EXPECT_EQ(plan.individual[i].word_count, 0u);
  }
  Memory result, expected;
  result.fill(0xcc);
  expected.fill(0xcc);
  std::memcpy(expected.data() + 28, bytes, sizeof(bytes));
  expected[34] = expected[35] = 0;
  Write(plan.shared, result);
  EXPECT_EQ(result, expected);
}

}  // namespace
