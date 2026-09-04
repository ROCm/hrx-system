// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_relation.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class LowStorageRelationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("storage_relation_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t DefineRegister(uint32_t unit_count = 1) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module_,
        loom_low_register_type(/*descriptor_set_stable_id=*/1,
                               /*register_class_id=*/0, unit_count),
        &value_id));
    return value_id;
  }

  std::vector<loom_low_storage_relation_t> Collect(const loom_op_t* op) {
    loom_low_storage_relation_iterator_t iterator;
    loom_low_storage_relation_iterator_initialize(module_, op, &iterator);
    std::vector<loom_low_storage_relation_t> relations;
    loom_low_storage_relation_t relation;
    while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
      relations.push_back(relation);
    }
    EXPECT_EQ(loom_low_storage_relation_count(module_, op), relations.size());
    return relations;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(LowStorageRelationTest, ProjectsSharedLoopEdgesWithoutBypassAffinity) {
  const loom_value_id_t lower_bound = DefineRegister();
  const loom_value_id_t upper_bound = DefineRegister();
  const loom_value_id_t step = DefineRegister();
  const loom_value_id_t initial = DefineRegister(2);
  const loom_value_id_t yielded = DefineRegister(2);

  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_low_scf_for_build(
      &builder_, /*build_flags=*/0, lower_bound, upper_bound, step, &initial, 1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, LOOM_LOCATION_UNKNOWN, &loop));
  const loom_value_id_t body_argument =
      loom_region_entry_arg_id(loom_low_scf_for_body(loop), 1);
  const loom_value_id_t result = loom_low_scf_for_results(loop).values[0];

  const loom_builder_ip_t saved =
      loom_builder_enter_region(&builder_, loop, loom_low_scf_for_body(loop));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_low_scf_yield_build(&builder_, &yielded, 1,
                                          LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  const std::vector<loom_low_storage_relation_t> loop_relations = Collect(loop);
  ASSERT_EQ(loop_relations.size(), 1u);
  EXPECT_EQ(loop_relations[0].source_value_id, initial);
  EXPECT_EQ(loop_relations[0].destination_value_id, body_argument);
  EXPECT_EQ(loop_relations[0].unit_count, 2u);
  EXPECT_EQ(loop_relations[0].cause,
            LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_LOOP_ENTRY);
  EXPECT_EQ(loop_relations[0].flags, LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED);

  const std::vector<loom_low_storage_relation_t> yield_relations =
      Collect(yield);
  ASSERT_EQ(yield_relations.size(), 2u);
  EXPECT_EQ(yield_relations[0].source_value_id, yielded);
  EXPECT_EQ(yield_relations[0].destination_value_id, body_argument);
  EXPECT_EQ(yield_relations[1].source_value_id, yielded);
  EXPECT_EQ(yield_relations[1].destination_value_id, result);
  EXPECT_EQ(yield_relations[0].cause,
            LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD);
  EXPECT_EQ(yield_relations[1].cause,
            LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD);
}

}  // namespace
}  // namespace loom
