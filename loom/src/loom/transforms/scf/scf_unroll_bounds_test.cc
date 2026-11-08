// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll_bounds.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

class ScfUnrollBoundsTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source_domains"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &scratch_arena_);
    loom_builder_set_block(&rewriter_.builder, loom_module_block(module_));
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t Constant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &rewriter_.builder, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  // Backing blocks for the IR and transient refinement storage.
  iree_arena_block_pool_t block_pool_;
  // Transient refinement storage, separate from module-owned allocations.
  iree_arena_allocator_t scratch_arena_;
  // Registered index vocabulary for construction.
  loom_context_t context_;
  // Owner of the index identities being refined.
  loom_module_t* module_ = nullptr;
  // Production builder and name-transfer owner.
  loom_rewriter_t rewriter_ = {};
};

TEST_P(ScfUnrollBoundsTest, RetainsEveryIdentityAcrossSegmentBoundaries) {
  const uint32_t count = GetParam();
  const auto upper = Constant(4 * (int64_t)count);
  std::vector<loom_value_id_t> indices;
  indices.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    indices.push_back(Constant(4 * i));
  }
  const auto originals = indices;
  // Each constructed coordinate is a member of this source iteration grid.
  // The API must transport all three predicates even when their count exceeds
  // the representable predicate and operand segments of a single operation.
  const auto source = loom_value_facts_make(0, 4 * (int64_t)count - 4, 4);
  IREE_ASSERT_OK(loom_scf_unroll_refine_indices(
      &rewriter_, source, loom_value_facts_unknown(), LOOM_VALUE_ID_INVALID,
      upper, count, indices.data(), LOOM_LOCATION_UNKNOWN, &scratch_arena_));

  uint32_t checked = 0;
  for (auto* op = loom_module_block(module_)->first_op; op; op = op->next_op) {
    if (!loom_index_assume_isa(op)) {
      continue;
    }
    const auto values = loom_index_assume_values(op);
    const auto results = loom_index_assume_results(op);
    const auto predicates = loom_index_assume_predicates(op);
    const uint16_t per_value = count == 1 ? 2 : 3;
    ASSERT_EQ(results.count, values.count);
    ASSERT_EQ(predicates.count, per_value * values.count);
    ASSERT_LE(checked + values.count, count);
    for (uint16_t i = 0; i < values.count; ++i, ++checked) {
      EXPECT_EQ(values.values[i], originals[checked]);
      EXPECT_EQ(results.values[i], indices[checked]);
      EXPECT_NE(indices[checked], originals[checked]);
      const auto* refinements = predicates.predicate_list + per_value * i;
      for (uint16_t j = 0; j < per_value; ++j) {
        EXPECT_EQ(refinements[j].args[0], originals[checked]);
      }
      EXPECT_EQ(refinements[per_value - 1].kind, LOOM_PREDICATE_LT);
      EXPECT_EQ(refinements[per_value - 1].args[1], upper);
    }
  }
  EXPECT_EQ(checked, count);
}

INSTANTIATE_TEST_SUITE_P(SingletonAndLargeTile, ScfUnrollBoundsTest,
                         ::testing::Values(1u, 65536u));

}  // namespace
}  // namespace loom
