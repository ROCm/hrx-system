// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(Aie2pStackFrameTest, PreservesSlotDepthsAcrossRepairAndRecompilation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_aie2p_stack_layout_t layout = {.arena = &arena};
  std::array<loom_low_storage_layout_record_t, 5> records = {};
  loom_low_storage_layout_t source = {.records = records.data()};
  const struct {
    // Generic placement domain.
    loom_storage_space_t space;
    // Native payload extent in bytes.
    uint64_t size;
    // Native slot alignment in bytes.
    uint64_t alignment;
  } shapes[] = {
      {LOOM_STORAGE_SPACE_STACK, 72, 64},
      {LOOM_STORAGE_SPACE_PRIVATE, 1024, 256},
      {LOOM_STORAGE_SPACE_STACK, 4, 4},
      {LOOM_STORAGE_SPACE_STACK, 144, 64},
      {LOOM_STORAGE_SPACE_STACK, 32, 32},
  };
  for (size_t i = 0; i < records.size(); ++i) {
    IREE_ASSERT_OK(loom_low_storage_layout_pack_reservation(
        shapes[i].space, shapes[i].size, shapes[i].alignment,
        &source.space_sizes, &records[i].reservation));
  }
  source.record_count = 2;
  IREE_ASSERT_OK(loom_aie2p_stack_layout_extend(&source, &layout));
  EXPECT_EQ(layout.depths[0], 128u);
  EXPECT_EQ(layout.depths[1], 0u);
  // Plan the next spill before its storage op has been inserted. The next
  // shared snapshot includes that same slot and must not reserve it twice.
  uint64_t depth = 0;
  IREE_ASSERT_OK(loom_aie2p_stack_layout_append(LOOM_STORAGE_SPACE_STACK, 4, 4,
                                                &layout, &depth));
  EXPECT_EQ(depth, 132u);
  source.record_count = 3;
  IREE_ASSERT_OK(loom_aie2p_stack_layout_extend(&source, &layout));
  EXPECT_EQ(layout.count, 3u);
  source.record_count = records.size();
  IREE_ASSERT_OK(loom_aie2p_stack_layout_extend(&source, &layout));
  EXPECT_EQ(layout.depths[0], 128u);
  EXPECT_EQ(layout.depths[2], 132u);
  EXPECT_EQ(layout.depths[3], 320u);
  EXPECT_EQ(layout.depths[4], 352u);

  bool accepted = false;
  loom_aie2p_stack_frame_t worker = {};
  IREE_ASSERT_OK(loom_aie2p_stack_frame_build(
      nullptr, nullptr, &source, &layout, /*callee_byte_length=*/128,
      /*is_helper=*/false, {}, &accepted, &worker));
  ASSERT_TRUE(accepted);
  EXPECT_EQ(worker.byte_length, 384u);
  EXPECT_EQ(worker.storage.byte_length, 512u);
  EXPECT_EQ(worker.storage.minimum_alignment, 64u);
  EXPECT_EQ(worker.storage_layout.records[0].reservation.byte_offset, 256u);
  EXPECT_EQ(worker.storage_layout.records[1].reservation.byte_offset, 0u);
  EXPECT_EQ(worker.storage_layout.records[2].reservation.byte_offset, 252u);
  EXPECT_EQ(worker.storage_layout.records[3].reservation.byte_offset, 64u);
  EXPECT_EQ(worker.storage_layout.records[4].reservation.byte_offset, 32u);

  // A new compilation sees the repaired reservation inventory, independently
  // of the first compilation's transient layout and materialization context.
  loom_aie2p_stack_layout_t replay = {.arena = &arena};
  loom_aie2p_stack_frame_t helper = {};
  IREE_ASSERT_OK(loom_aie2p_stack_frame_build(
      nullptr, nullptr, &source, &replay, /*callee_byte_length=*/0,
      /*is_helper=*/true, {}, &accepted, &helper));
  ASSERT_TRUE(accepted);
  EXPECT_EQ(helper.byte_length, worker.byte_length);
  EXPECT_EQ(helper.storage.byte_length, 0u);
  for (size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(helper.storage_layout.records[i].reservation.byte_offset,
              worker.storage_layout.records[i].reservation.byte_offset);
  }
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace loom
