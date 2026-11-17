// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_loop_reads.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactLoopReadsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(sizeof(iree_arena_block_t),
                                     {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena_);
    owners_ = {values_.data(),
               {1024, values_.size()},
               {1024, values_.size()},
               kGeneration};
    reads_.owners = &owners_;
    values_[0] = loom_value_fact_loop_owner_make(kGeneration, 1);
    values_[1] = loom_value_fact_loop_owner_make(kGeneration, 4);
    values_[63] = loom_value_fact_loop_owner_make(kGeneration, 5);
    values_[64] = loom_value_fact_loop_owner_make(kGeneration, 6);
    values_[128] = loom_value_fact_loop_owner_make(kGeneration, 2);
    values_[192] = loom_value_fact_loop_owner_make(kGeneration, 3);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactLoopReadsTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      if (!test->allocations_remaining_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected fact read allocation failure");
      }
      --test->allocations_remaining_;
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  // Separate backing allocations expose both collector preparation failures.
  iree_arena_block_pool_t pool_ = {};
  // Owns all collection buffers until the outer solve ends.
  iree_arena_allocator_t arena_ = {};
  // Root ownership includes a parent, siblings, a descendant, and absent IDs.
  std::array<loom_value_fact_loop_owner_t, 193> values_ = {};
  // Borrowed immutable root ownership shared by nested collectors.
  loom_value_fact_loop_owners_t owners_ = {};
  // Current loop and its one descendant.
  loom_value_fact_loop_subtree_t subtree_ = {4, 6};
  // One nesting depth's lazily allocated collection buffers.
  loom_value_fact_loop_reads_t reads_ = {};
  // Successful backing allocations permitted before injecting failure.
  size_t allocations_remaining_ = SIZE_MAX;
  // Active root generation assigned to fixture ownership entries.
  static constexpr uint32_t kGeneration = 7;
};

TEST_F(FactLoopReadsTest, QueriesRecordOnlyExternalMutableDefinitions) {
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &subtree_, &arena_));
  loom_value_fact_table_t table = {};
  table.replay.reads = &reads_;
  const auto used_bytes = arena_.used_allocation_size;
  // Queries also observe absent numeric publications. Immutable captures,
  // holes, and definitions produced inside the solve are not dependencies.
  for (auto value : {1216u, 1024u, 1088u, 1152u, 1025u, 1087u, 1089u, 1023u,
                     1217u, 1216u, 1024u}) {
    (void)loom_value_fact_table_lookup(&table, value);
  }
  EXPECT_EQ(reads_.word_count, 4u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(reads_.bits[i], 1u);
  }
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);

  loom_value_fact_loop_reads_reset(&reads_);
  EXPECT_EQ(reads_.word_count, 0u);
  EXPECT_EQ(reads_.subtree, nullptr);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(reads_.bits[i], 0u);
  }
}

TEST_F(FactLoopReadsTest, SiblingScopesReuseClearedBuffersWithoutAllocating) {
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &subtree_, &arena_));
  loom_value_fact_loop_record_read(&reads_, 1216);
  auto* bits = reads_.bits;
  auto* words = reads_.words;
  const auto used_bytes = arena_.used_allocation_size;
  loom_value_fact_loop_reads_reset(&reads_);

  allocations_remaining_ = 0;
  const loom_value_fact_loop_subtree_t sibling = {6, 8};
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &sibling, &arena_));
  for (auto value : {1025u, 1087u, 1088u, 1025u}) {
    loom_value_fact_loop_record_read(&reads_, value);
  }
  EXPECT_EQ(reads_.bits, bits);
  EXPECT_EQ(reads_.words, words);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(reads_.word_count, 1u);
  EXPECT_EQ(reads_.words[0], 0u);
  EXPECT_EQ(reads_.bits[0], UINT64_C(2) | (UINT64_C(1) << 63));
  for (size_t i = 1; i < 4; ++i) {
    EXPECT_EQ(reads_.bits[i], 0u);
  }
  loom_value_fact_loop_reads_reset(&reads_);
  EXPECT_EQ(reads_.bits[0], 0u);
}

TEST_F(FactLoopReadsTest, NestedScopesObserveIndependentInputSets) {
  const loom_value_fact_loop_subtree_t parent = {2, 6};
  loom_value_fact_loop_reads_t child = {.owners = &owners_};
  reads_.child = &child;
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &parent, &arena_));
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&child, &subtree_, &arena_));
  for (auto* collector : {&reads_, &child}) {
    loom_value_fact_loop_record_read(collector, 1024);
    loom_value_fact_loop_record_read(collector, 1216);
  }
  EXPECT_EQ(reads_.word_count, 1u);
  EXPECT_EQ(reads_.bits[0], 1u);
  EXPECT_EQ(reads_.bits[3], 0u);
  EXPECT_EQ(child.word_count, 2u);
  EXPECT_EQ(child.bits[0], 1u);
  EXPECT_EQ(child.bits[3], 1u);
  loom_value_fact_loop_reads_reset(&child);
  EXPECT_EQ(reads_.word_count, 1u);
  EXPECT_EQ(reads_.bits[0], 1u);
  EXPECT_EQ(child.bits[0], 0u);
  EXPECT_EQ(child.bits[3], 0u);
}

TEST_F(FactLoopReadsTest, PreviousRootEntriesAreNotCurrentDefinitions) {
  ++owners_.generation;
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &subtree_, &arena_));
  for (auto value : {1024u, 1088u, 1152u, 1216u}) {
    loom_value_fact_loop_record_read(&reads_, value);
  }
  EXPECT_EQ(reads_.word_count, 0u);

  values_[64] = loom_value_fact_loop_owner_make(owners_.generation, 6);
  loom_value_fact_loop_record_read(&reads_, 1088);
  EXPECT_EQ(reads_.word_count, 1u);
  EXPECT_EQ(reads_.words[0], 1u);
  EXPECT_EQ(reads_.bits[1], 1u);
}

TEST_F(FactLoopReadsTest, SparseHighIdsUseOnlyTheRootRange) {
  owners_.storage_range = {UINT32_MAX - 128, 128};
  owners_.root_range = owners_.storage_range;
  values_[127] = loom_value_fact_loop_owner_make(kGeneration, 1);
  IREE_ASSERT_OK(loom_value_fact_loop_reads_begin(&reads_, &subtree_, &arena_));
  for (auto value :
       {0u, UINT32_MAX - 129, UINT32_MAX - 128, UINT32_MAX - 1, UINT32_MAX}) {
    loom_value_fact_loop_record_read(&reads_, value);
  }
  EXPECT_EQ(reads_.word_count, 2u);
  EXPECT_EQ(reads_.bits[0], 1u);
  EXPECT_EQ(reads_.bits[1], UINT64_C(1) << 63);
  EXPECT_LT(arena_.used_allocation_size, 256u);
}

class FactLoopReadsFailureTest : public FactLoopReadsTest,
                                 public ::testing::WithParamInterface<size_t> {
};

TEST_P(FactLoopReadsFailureTest, FailedPreparationDoesNotActivateCollection) {
  allocations_remaining_ = GetParam();
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_value_fact_loop_reads_begin(&reads_, &subtree_, &arena_));
  EXPECT_EQ(reads_.subtree, nullptr);
  EXPECT_EQ(reads_.word_count, 0u);
  loom_value_fact_loop_reads_reset(&reads_);
}
INSTANTIATE_TEST_SUITE_P(BackingAllocations, FactLoopReadsFailureTest,
                         ::testing::Values(0u, 1u));

}  // namespace
}  // namespace loom
