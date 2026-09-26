// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleBlockOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("block-order"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    block_ = loom_module_block(module_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_op_t* AllocateOp() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(
        iree_arena_allocate(&module_->arena, sizeof(*op), (void**)&op));
    std::memset(op, 0, sizeof(*op));
    op->kind = 0x0100;
    return op;
  }

  loom_op_t* AppendOp() {
    loom_op_t* op = AllocateOp();
    IREE_CHECK_OK(loom_block_append_op(module_, block_, op));
    return op;
  }

  void CheckOrder() {
    uint32_t count = 0;
    uint64_t previous_ordinal = 0;
    loom_op_t* previous_op = nullptr;
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block_, op) {
      EXPECT_EQ(op->parent_block, block_);
      EXPECT_EQ(op->prev_op, previous_op);
      EXPECT_GT(op->block_ordinal, previous_ordinal);
      previous_op = op;
      previous_ordinal = op->block_ordinal;
      ++count;
    }
    EXPECT_EQ(previous_op, block_->last_op);
    EXPECT_EQ(count, block_->op_count);
    if (block_->last_op) {
      EXPECT_LT(block_->last_op->block_ordinal, UINT64_MAX);
    }
  }

  // Recycled backing storage for the module arena.
  iree_arena_block_pool_t block_pool_ = {};
  // Minimal context; block ordering does not require registered operations.
  loom_context_t context_ = {};
  // Module owning the operations under test.
  loom_module_t* module_ = nullptr;
  // Module body block whose linked order and ordinals are checked together.
  loom_block_t* block_ = nullptr;
};

TEST_F(ModuleBlockOrderTest, RepeatedAfterStableOpRetainsSparseCapacity) {
  constexpr uint32_t kInsertionCount = 4096;
  loom_op_t* anchor = AppendOp();
  AppendOp();

  for (uint32_t i = 0; i < kInsertionCount; ++i) {
    IREE_ASSERT_OK(loom_block_insert_before_op(module_, block_, anchor->next_op,
                                               AllocateOp()));
  }

  CheckOrder();
  ASSERT_NE(anchor->next_op, nullptr);
  EXPECT_GT(anchor->next_op->block_ordinal - anchor->block_ordinal,
            kInsertionCount);
}

TEST_F(ModuleBlockOrderTest, RepeatedBeforeStableOpRetainsSparseCapacity) {
  constexpr uint32_t kInsertionCount = 4096;
  loom_op_t* predecessor = AppendOp();
  loom_op_t* anchor = AppendOp();

  for (uint32_t i = 0; i < kInsertionCount; ++i) {
    IREE_ASSERT_OK(
        loom_block_insert_before_op(module_, block_, anchor, AllocateOp()));
  }

  CheckOrder();
  ASSERT_NE(anchor->prev_op, nullptr);
  EXPECT_GT(anchor->block_ordinal - anchor->prev_op->block_ordinal,
            kInsertionCount);
  EXPECT_EQ(predecessor, block_->first_op);
}

TEST_F(ModuleBlockOrderTest, AlternatingInsertionPreservesStrictOrder) {
  constexpr uint32_t kInsertionCount = 4096;
  loom_op_t* anchor = AppendOp();
  AppendOp();

  for (uint32_t i = 0; i < kInsertionCount; ++i) {
    loom_op_t* before_op = i & 1 ? anchor : anchor->next_op;
    IREE_ASSERT_OK(
        loom_block_insert_before_op(module_, block_, before_op, AllocateOp()));
  }

  CheckOrder();
}

TEST_F(ModuleBlockOrderTest, BoundaryInsertionPreservesStrictOrder) {
  constexpr uint32_t kInsertionCount = 256;
  loom_op_t* stable_first_op = AppendOp();
  for (uint32_t i = 0; i < kInsertionCount; ++i) {
    IREE_ASSERT_OK(loom_block_insert_before_op(module_, block_, stable_first_op,
                                               AllocateOp()));
  }
  for (uint32_t i = 0; i < kInsertionCount; ++i) {
    IREE_ASSERT_OK(loom_block_insert_before_op(module_, block_,
                                               block_->first_op, AllocateOp()));
  }

  CheckOrder();
}

TEST_F(ModuleBlockOrderTest, RepairsForwardFromAnExhaustedGap) {
  loom_op_t* ops[] = {AppendOp(), AppendOp(), AppendOp(), AppendOp()};
  // Model a valid sparse-label state with no room in the first gap. Reaching
  // the same state through insertions would require billions of test ops.
  ops[0]->block_ordinal = 1;
  ops[1]->block_ordinal = 2;
  ops[2]->block_ordinal = 100;
  ops[3]->block_ordinal = 200;

  loom_op_t* inserted = AllocateOp();
  IREE_ASSERT_OK(
      loom_block_insert_before_op(module_, block_, ops[1], inserted));

  CheckOrder();
  EXPECT_EQ(inserted->prev_op, ops[0]);
  EXPECT_EQ(inserted->next_op, ops[1]);
  EXPECT_NE(ops[1]->block_ordinal, 2u);
}

TEST_F(ModuleBlockOrderTest, RepairsBackwardFromAnExhaustedGap) {
  loom_op_t* ops[] = {AppendOp(), AppendOp(), AppendOp(), AppendOp()};
  // Mirror the exhausted gap in the upper half of the ordinal arena so the
  // backward density search and forward redistribution are both exercised.
  ops[0]->block_ordinal = UINT64_MAX - 200;
  ops[1]->block_ordinal = UINT64_MAX - 100;
  ops[2]->block_ordinal = UINT64_MAX - 2;
  ops[3]->block_ordinal = UINT64_MAX - 1;

  loom_op_t* inserted = AllocateOp();
  IREE_ASSERT_OK(
      loom_block_insert_before_op(module_, block_, ops[3], inserted));

  CheckOrder();
  EXPECT_EQ(inserted->prev_op, ops[2]);
  EXPECT_EQ(inserted->next_op, ops[3]);
  EXPECT_NE(ops[2]->block_ordinal, UINT64_MAX - 2);
}

TEST_F(ModuleBlockOrderTest, AppendUsesRemainingTailWithoutRelabeling) {
  loom_op_t* ops[] = {AppendOp(), AppendOp(), AppendOp()};
  ops[0]->block_ordinal = 1;
  ops[1]->block_ordinal = 2;
  ops[2]->block_ordinal = UINT64_MAX - 100;

  const uint64_t old_ordinals[] = {
      ops[0]->block_ordinal,
      ops[1]->block_ordinal,
      ops[2]->block_ordinal,
  };
  loom_op_t* appended = AppendOp();

  CheckOrder();
  EXPECT_EQ(ops[0]->block_ordinal, old_ordinals[0]);
  EXPECT_EQ(ops[1]->block_ordinal, old_ordinals[1]);
  EXPECT_EQ(ops[2]->block_ordinal, old_ordinals[2]);
  EXPECT_EQ(appended->block_ordinal, UINT64_MAX - 99);
}

TEST_F(ModuleBlockOrderTest, AppendRepairsAnExhaustedTail) {
  loom_op_t* ops[] = {AppendOp(), AppendOp(), AppendOp()};
  ops[0]->block_ordinal = UINT64_MAX - 3;
  ops[1]->block_ordinal = UINT64_MAX - 2;
  ops[2]->block_ordinal = UINT64_MAX - 1;

  const uint64_t old_first_ordinal = ops[0]->block_ordinal;
  loom_op_t* appended = AppendOp();

  CheckOrder();
  EXPECT_NE(ops[0]->block_ordinal, old_first_ordinal);
  EXPECT_EQ(appended, block_->last_op);
}

TEST_F(ModuleBlockOrderTest, UnlinkedOpsCanBeReinsertedAroundOneAnchor) {
  constexpr uint32_t kOpCount = 256;
  std::vector<loom_op_t*> ops;
  ops.reserve(kOpCount);
  for (uint32_t i = 0; i < kOpCount; ++i) {
    ops.push_back(AppendOp());
  }

  loom_op_t* anchor = ops.front();
  for (uint32_t i = kOpCount; i-- > 1;) {
    loom_op_t* op = ops[i];
    loom_block_unlink_op(module_, op);
    IREE_ASSERT_OK(
        loom_block_insert_before_op(module_, block_, anchor->next_op, op));
  }

  CheckOrder();
  EXPECT_EQ(block_->op_count, kOpCount);
  EXPECT_EQ(block_->first_op, anchor);
}

}  // namespace
}  // namespace loom
