// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_demand.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

TEST(ScheduleResourceDemandTest, PricesRequiredWorkAcrossResourceAliases) {
  const auto* descriptors = loom_test_low_core_descriptor_set();
  const auto* model = loom_low_descriptor_set_lookup_schedule_model_by_id(
      descriptors, UINT64_C(0x235c515ddcc8fd92));
  ASSERT_NE(model, nullptr);
  const auto schedule_class = [&](uint32_t descriptor_ref) {
    return loom_low_schedule_model_schedule_class(
        model, loom_low_descriptor_set_descriptor_view_at(descriptors,
                                                          descriptor_ref));
  };
  const auto* two_cycles =
      schedule_class(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);
  const auto* alias =
      schedule_class(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_REQUIRED_I32);
  const auto* reserved =
      schedule_class(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_RESERVED_I32);
  const auto* scalar =
      schedule_class(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_low_schedule_resource_demand_t demand = {};
  IREE_ASSERT_OK(
      loom_low_schedule_resource_demand_begin(model, &arena, &demand));
  loom_low_schedule_resource_demand_add(&demand, model, two_cycles);
  loom_low_schedule_resource_demand_add(&demand, model, two_cycles);
  loom_low_schedule_resource_demand_add(&demand, model, reserved);
  loom_low_schedule_resource_demand_add(&demand, model, reserved);
  // Four required unit-cycles share two units per cycle. Both resource
  // names see that work, and reservations contribute to neither side.
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, two_cycles),
            4 * LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE);
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, alias),
            2 * LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE);
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, reserved),
            0u);
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, scalar), 0u);

  const auto allocated = arena.used_allocation_size;
  IREE_ASSERT_OK(
      loom_low_schedule_resource_demand_begin(model, &arena, &demand));
  loom_low_schedule_resource_demand_add(&demand, model, scalar);
  loom_low_schedule_resource_demand_add(&demand, model, alias);
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, alias),
            LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE / 2);
  EXPECT_EQ(loom_low_schedule_resource_demand_cost(&demand, model, scalar),
            LOOM_LOW_SCHEDULE_RESOURCE_COST_ONE);
  EXPECT_EQ(arena.used_allocation_size, allocated);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
