// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/buffer/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class BufferFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 1));
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_facts_t InferLength(
      loom_value_fact_reference_nullability_t nullability,
      loom_value_facts_t maximum_byte_extent) {
    loom_value_fact_buffer_reference_t reference = {
        /*.maximum_byte_extent=*/maximum_byte_extent,
        /*.minimum_alignment=*/1,
        /*.memory_space=*/LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
        /*.root_value_id=*/LOOM_VALUE_ID_INVALID,
        /*.alias_scope_id=*/LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
        /*.nullability=*/nullability,
    };
    loom_value_facts_t operand_facts = loom_value_facts_unknown();
    IREE_EXPECT_OK(loom_value_facts_make_buffer_reference(
        &fact_table_.context, reference, &operand_facts));
    loom_value_facts_t result_facts = loom_value_facts_unknown();
    IREE_EXPECT_OK(loom_buffer_length_facts(&fact_table_.context,
                                            /*module=*/nullptr, /*op=*/nullptr,
                                            &operand_facts, &result_facts));
    return result_facts;
  }

  // Block pool backing fact extension storage.
  iree_arena_block_pool_t block_pool_;
  // Arena retaining fact extension payloads for each test.
  iree_arena_allocator_t analysis_arena_;
  // Fact table providing the context used by buffer fact inference.
  loom_value_fact_table_t fact_table_;
};

TEST_F(BufferFactsTest, LengthAccountsForNullability) {
  const loom_value_facts_t extent = loom_value_facts_exact_i64(48);

  loom_value_facts_t non_null_length =
      InferLength(LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL, extent);
  EXPECT_TRUE(loom_value_facts_is_exact(non_null_length));
  EXPECT_EQ(non_null_length.range_lo, 48);
  EXPECT_EQ(non_null_length.range_hi, 48);

  loom_value_facts_t null_length =
      InferLength(LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL, extent);
  EXPECT_TRUE(loom_value_facts_is_exact(null_length));
  EXPECT_EQ(null_length.range_lo, 0);
  EXPECT_EQ(null_length.range_hi, 0);

  loom_value_facts_t maybe_null_length =
      InferLength(LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN, extent);
  EXPECT_FALSE(loom_value_facts_is_exact(maybe_null_length));
  EXPECT_EQ(maybe_null_length.range_lo, 0);
  EXPECT_EQ(maybe_null_length.range_hi, 48);
  EXPECT_EQ(maybe_null_length.known_divisor, 48);
}

}  // namespace
}  // namespace loom
