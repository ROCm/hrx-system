// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/assignment_map.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationAssignmentMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  loom_value_id_t DefineIndexValue(loom_module_t* module) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    return value_id;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(LowAllocationAssignmentMapTest, LooksUpAssignedLocalValues) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t first_value = DefineIndexValue(module);
  const loom_value_id_t second_value = DefineIndexValue(module);
  const loom_value_id_t external_value = DefineIndexValue(module);

  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, first_value,
                                        /*ordinal=*/0);
  loom_module_value_ordinal_scratch_set(module, second_value,
                                        /*ordinal=*/1);

  const loom_value_id_t value_ids[] = {first_value, second_value};
  loom_liveness_analysis_t liveness = {};
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);

  loom_low_allocation_assignment_t assignments[1] = {};
  assignments[0].value_id = second_value;

  uint32_t assignment_indices_by_value_ordinal[] = {
      UINT32_MAX,
      0,
  };
  loom_low_allocation_assignment_map_t map = {};
  map.module = module;
  map.liveness = &liveness;
  map.assignments = assignments;
  map.assignment_count = IREE_ARRAYSIZE(assignments);
  map.assignment_indices_by_value_ordinal = assignment_indices_by_value_ordinal;

  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  EXPECT_TRUE(loom_low_allocation_assignment_map_value_ordinal_for_value(
      &map, first_value, &value_ordinal));
  EXPECT_EQ(value_ordinal, 0u);
  EXPECT_FALSE(loom_low_allocation_assignment_map_value_ordinal_for_value(
      &map, external_value, &value_ordinal));

  uint32_t assignment_index = 99;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_map_assignment_for_value_ordinal(
          &map, /*value_ordinal=*/0, &assignment_index);
  EXPECT_EQ(assignment, nullptr);
  EXPECT_EQ(assignment_index, UINT32_MAX);

  assignment = loom_low_allocation_assignment_map_assignment_for_value_ordinal(
      &map, /*value_ordinal=*/1, &assignment_index);
  ASSERT_NE(assignment, nullptr);
  EXPECT_EQ(assignment, &assignments[0]);
  EXPECT_EQ(assignment->value_id, second_value);
  EXPECT_EQ(assignment_index, 0u);

  assignment_index = 99;
  assignment = nullptr;
  IREE_ASSERT_OK(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &map, second_value, &assignment_index, &assignment));
  EXPECT_EQ(assignment, &assignments[0]);
  EXPECT_EQ(assignment_index, 0u);
  IREE_ASSERT_OK(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &map, second_value, nullptr, nullptr));

  assignment_index = 99;
  assignment = &assignments[0];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &map, first_value, &assignment_index, &assignment));
  EXPECT_EQ(assignment, nullptr);
  EXPECT_EQ(assignment_index, UINT32_MAX);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &map, external_value, nullptr, nullptr));

  loom_module_value_ordinal_scratch_clear(module, first_value);
  loom_module_value_ordinal_scratch_clear(module, second_value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
