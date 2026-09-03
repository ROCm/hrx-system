// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_cost.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowDescriptorCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    descriptor_set_ = loom_test_low_core_descriptor_set();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint32_t DescriptorOrdinal(iree_string_view_t key) const {
    const uint32_t ordinal =
        loom_low_descriptor_set_lookup_descriptor(descriptor_set_, key);
    EXPECT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    return ordinal;
  }

  static const loom_low_descriptor_resource_cost_t* FindResourceCost(
      const loom_low_descriptor_cost_t& cost, iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < cost.resource_cost_count; ++i) {
      if (iree_string_view_equal(cost.resource_costs[i].resource_name, name)) {
        return &cost.resource_costs[i];
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
};

TEST_F(LowDescriptorCostTest, ResourceCostNormalizesCapacityAndMultiplicity) {
  loom_low_descriptor_resource_cost_t resource_cost = {
      .resource_id = 7,
      .capacity_per_cycle = 4,
  };
  const loom_low_issue_use_t issue_use = {
      .resource_id = 7,
      .cycles = 2,
      .units = 3,
  };

  IREE_ASSERT_OK(loom_low_descriptor_resource_cost_accumulate(
      &issue_use, /*occurrence_count=*/3, &resource_cost));

  EXPECT_EQ(resource_cost.use_count, 3u);
  EXPECT_EQ(resource_cost.total_occupied_cycles, 6u);
  EXPECT_EQ(resource_cost.total_unit_cycles, 18u);
  EXPECT_EQ(resource_cost.estimated_min_cycles, 5u);
  EXPECT_EQ(resource_cost.peak_units_per_cycle, 3u);
}

TEST_F(LowDescriptorCostTest,
       RecipeAggregatesMultiplicityMemoryAndDependencyPath) {
  const loom_low_descriptor_recipe_entry_t entries[] = {
      {DescriptorOrdinal(IREE_SV("test.add.i32")), 4},
      {DescriptorOrdinal(IREE_SV("test.load.v4i32")), 2},
      {DescriptorOrdinal(IREE_SV("test.store.v4i32")), 1},
  };
  const loom_low_descriptor_recipe_dependency_t dependencies[] = {
      {/*.source_entry=*/0, /*.target_entry=*/1},
      {/*.source_entry=*/1, /*.target_entry=*/2},
  };
  const loom_low_descriptor_recipe_t recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
      .dependencies = dependencies,
      .dependency_count = IREE_ARRAYSIZE(dependencies),
  };

  loom_low_descriptor_cost_t cost = {};
  IREE_ASSERT_OK(loom_low_descriptor_cost_compute(descriptor_set_, &recipe,
                                                  &arena_, &cost));

  EXPECT_EQ(cost.descriptor_set, descriptor_set_);
  EXPECT_EQ(cost.model_quality, LOOM_LOW_MODEL_QUALITY_FALLBACK);
  EXPECT_EQ(cost.instruction_count, 7u);
  EXPECT_EQ(cost.maximum_resource_cycles, 4u);
  EXPECT_EQ(cost.total_resource_cycles, 10u);
  EXPECT_EQ(cost.critical_path_cycles, 7u);
  EXPECT_EQ(cost.memory.read_operation_count, 2u);
  EXPECT_EQ(cost.memory.read_byte_count, 32u);
  EXPECT_EQ(cost.memory.read_unknown_width_count, 0u);
  EXPECT_EQ(cost.memory.write_operation_count, 1u);
  EXPECT_EQ(cost.memory.write_byte_count, 16u);
  EXPECT_EQ(cost.memory.write_unknown_width_count, 0u);

  const loom_low_descriptor_resource_cost_t* scalar_cost =
      FindResourceCost(cost, IREE_SV("test.scalar"));
  ASSERT_NE(scalar_cost, nullptr);
  EXPECT_EQ(scalar_cost->use_count, 4u);
  EXPECT_EQ(scalar_cost->estimated_min_cycles, 4u);
  const loom_low_descriptor_resource_cost_t* address_cost =
      FindResourceCost(cost, IREE_SV("test.address"));
  ASSERT_NE(address_cost, nullptr);
  EXPECT_EQ(address_cost->use_count, 3u);
  EXPECT_EQ(address_cost->estimated_min_cycles, 3u);
  const loom_low_descriptor_resource_cost_t* load_cost =
      FindResourceCost(cost, IREE_SV("test.load"));
  ASSERT_NE(load_cost, nullptr);
  EXPECT_EQ(load_cost->use_count, 2u);
  const loom_low_descriptor_resource_cost_t* store_cost =
      FindResourceCost(cost, IREE_SV("test.store"));
  ASSERT_NE(store_cost, nullptr);
  EXPECT_EQ(store_cost->use_count, 1u);
}

TEST_F(LowDescriptorCostTest, UnknownEvidenceCannotDisplaceCanonical) {
  std::vector<loom_low_schedule_class_t> schedule_classes(
      descriptor_set_->schedule_classes,
      descriptor_set_->schedule_classes +
          descriptor_set_->schedule_class_count);
  loom_low_descriptor_set_t unknown_descriptor_set = *descriptor_set_;
  unknown_descriptor_set.schedule_classes = schedule_classes.data();
  const uint32_t descriptor_ordinal =
      DescriptorOrdinal(IREE_SV("test.add.i32"));
  const uint16_t schedule_class_id =
      descriptor_set_->descriptor_views[descriptor_ordinal].schedule_class_id;
  schedule_classes[schedule_class_id].model_quality =
      LOOM_LOW_MODEL_QUALITY_UNKNOWN;
  const loom_low_descriptor_recipe_entry_t entries[] = {
      {descriptor_ordinal, 1},
  };
  const loom_low_descriptor_recipe_t recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
  };
  loom_low_descriptor_cost_t cost = {};
  IREE_ASSERT_OK(loom_low_descriptor_cost_compute(&unknown_descriptor_set,
                                                  &recipe, &arena_, &cost));
  ASSERT_EQ(cost.model_quality, LOOM_LOW_MODEL_QUALITY_UNKNOWN);

  const loom_low_descriptor_cost_candidate_t alternate = {
      .cost = &cost,
      .stable_key = 1,
      .is_canonical = false,
  };
  const loom_low_descriptor_cost_candidate_t canonical = {
      .cost = &cost,
      .stable_key = 9,
      .is_canonical = true,
  };
  EXPECT_EQ(loom_low_descriptor_cost_compare(&alternate, &canonical),
            LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT);
}

TEST_F(LowDescriptorCostTest, DominanceCanSelectNoncanonicalCandidate) {
  const uint32_t add_ordinal = DescriptorOrdinal(IREE_SV("test.add.i32"));
  const loom_low_descriptor_recipe_entry_t entries[] = {{add_ordinal, 1}};
  const loom_low_pressure_delta_t lower_pressure[] = {{0, 1}};
  const loom_low_pressure_delta_t higher_pressure[] = {{0, 3}};
  const loom_low_descriptor_recipe_t lower_recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
      .durable_pressure_deltas = lower_pressure,
      .durable_pressure_delta_count = IREE_ARRAYSIZE(lower_pressure),
  };
  const loom_low_descriptor_recipe_t higher_recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
      .durable_pressure_deltas = higher_pressure,
      .durable_pressure_delta_count = IREE_ARRAYSIZE(higher_pressure),
  };
  loom_low_descriptor_cost_t lower_cost = {};
  loom_low_descriptor_cost_t higher_cost = {};
  IREE_ASSERT_OK(loom_low_descriptor_cost_compute(
      descriptor_set_, &lower_recipe, &arena_, &lower_cost));
  IREE_ASSERT_OK(loom_low_descriptor_cost_compute(
      descriptor_set_, &higher_recipe, &arena_, &higher_cost));

  EXPECT_TRUE(loom_low_descriptor_cost_dominates(&lower_cost, &higher_cost));
  EXPECT_FALSE(loom_low_descriptor_cost_dominates(&higher_cost, &lower_cost));
  const loom_low_descriptor_cost_candidate_t alternate = {
      .cost = &lower_cost,
      .stable_key = 99,
      .is_canonical = false,
  };
  const loom_low_descriptor_cost_candidate_t canonical = {
      .cost = &higher_cost,
      .stable_key = 1,
      .is_canonical = true,
  };
  EXPECT_EQ(loom_low_descriptor_cost_compare(&alternate, &canonical),
            LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT);
}

TEST_F(LowDescriptorCostTest, CanonicalThenStableIdentityBreakExactTies) {
  const loom_low_descriptor_recipe_entry_t entries[] = {
      {DescriptorOrdinal(IREE_SV("test.add.i32")), 1},
  };
  const loom_low_descriptor_recipe_t recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
  };
  loom_low_descriptor_cost_t cost = {};
  IREE_ASSERT_OK(loom_low_descriptor_cost_compute(descriptor_set_, &recipe,
                                                  &arena_, &cost));

  loom_low_descriptor_cost_candidate_t left = {
      .cost = &cost,
      .stable_key = 1,
      .is_canonical = false,
  };
  loom_low_descriptor_cost_candidate_t right = {
      .cost = &cost,
      .stable_key = 9,
      .is_canonical = true,
  };
  EXPECT_EQ(loom_low_descriptor_cost_compare(&left, &right),
            LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT);
  right.is_canonical = false;
  EXPECT_EQ(loom_low_descriptor_cost_compare(&left, &right),
            LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT);
}

TEST_F(LowDescriptorCostTest, RejectsNonTopologicalDependencyRows) {
  const uint32_t add_ordinal = DescriptorOrdinal(IREE_SV("test.add.i32"));
  const loom_low_descriptor_recipe_entry_t entries[] = {
      {add_ordinal, 1},
      {add_ordinal, 1},
      {add_ordinal, 1},
  };
  const loom_low_descriptor_recipe_dependency_t dependencies[] = {
      {/*.source_entry=*/1, /*.target_entry=*/2},
      {/*.source_entry=*/0, /*.target_entry=*/1},
  };
  const loom_low_descriptor_recipe_t recipe = {
      .entries = entries,
      .entry_count = IREE_ARRAYSIZE(entries),
      .dependencies = dependencies,
      .dependency_count = IREE_ARRAYSIZE(dependencies),
  };

  loom_low_descriptor_cost_t cost = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_descriptor_cost_compute(
                            descriptor_set_, &recipe, &arena_, &cost));
}

}  // namespace
}  // namespace loom
