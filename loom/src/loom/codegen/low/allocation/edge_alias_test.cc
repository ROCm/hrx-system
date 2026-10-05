// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_alias.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class LowAllocationEdgeAliasTest : public ::testing::Test {
 protected:
  static constexpr uint16_t kDestinationBlock = 2;
  static constexpr uint16_t kSourceBlock = 3;

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const auto* vtables = loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("edge_alias"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(loom_module_allocate_region(module_, 4, &region_));
    region_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t AddArgument(uint16_t block_index, uint32_t unit_count) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module_, loom_low_register_type(1, 0, unit_count), &value_id));
    IREE_CHECK_OK(loom_block_add_arg(
        module_, loom_region_block(region_, block_index), value_id));
    return value_id;
  }

  loom_op_t* Branch(loom_value_id_t source) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_block(region_, kSourceBlock), &builder);
    loom_op_t* branch = nullptr;
    IREE_CHECK_OK(loom_low_br_build(
        &builder, loom_region_block(region_, kDestinationBlock), &source, 1,
        LOOM_LOCATION_UNKNOWN, &branch));
    return branch;
  }

  loom_liveness_interval_t Interval(loom_value_id_t value_id, uint32_t start,
                                    uint32_t end, uint32_t definition_point) {
    loom_liveness_interval_t interval = {};
    interval.value_id = value_id;
    interval.value_class.type_kind = LOOM_TYPE_REGISTER;
    interval.value_class.register_descriptor_set_stable_id = 1;
    interval.value_class.register_class_id = 0;
    interval.start_point = start;
    interval.end_point = end;
    interval.unit_count = loom_low_register_type_unit_count(
        loom_module_value_type(module_, value_id));
    interval.definition_point = definition_point;
    return interval;
  }

  template <size_t N>
  bool AllowsOverlap(const loom_liveness_interval_t (&intervals)[N],
                     uint32_t candidate_ordinal, uint32_t counterpart_ordinal,
                     uint32_t counterpart_acquisition_start,
                     loom_op_t* branch) {
    loom_value_id_t value_ids[N];
    uint32_t interval_indices[N];
    loom_liveness_segment_t segments[N];
    loom_liveness_segment_range_t segment_ranges[N];
    loom_low_placement_relation_range_t result_ranges[N] = {};
    for (uint32_t i = 0; i < N; ++i) {
      value_ids[i] = intervals[i].value_id;
      interval_indices[i] = i;
      segments[i] = {intervals[i].start_point, intervals[i].end_point};
      segment_ranges[i] = {i, 1};
    }
    loom_liveness_analysis_t liveness = {};
    liveness.value_ids = value_ids;
    liveness.value_count = N;
    liveness.intervals = intervals;
    liveness.interval_count = N;
    liveness.value_interval_indices = interval_indices;
    liveness.segments = segments;
    liveness.segment_count = N;
    liveness.value_segment_ranges = segment_ranges;

    loom_low_placement_relation_t relation = {};
    relation.op = branch;
    relation.source_ordinal = 0;
    relation.result_ordinal = 1;
    relation.unit_count = intervals[0].unit_count;
    relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH;
    relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
    result_ranges[1] = {0, 1};
    loom_low_placement_table_t placement = {};
    placement.module = module_;
    placement.region = region_;
    placement.value_ids = value_ids;
    placement.value_count = N;
    placement.relations = &relation;
    placement.relation_count = 1;
    placement.ranges_by_result_ordinal = result_ranges;

    loom_low_allocation_assignment_t counterpart = {};
    const auto& counterpart_interval = intervals[counterpart_ordinal];
    counterpart.value_id = counterpart_interval.value_id;
    counterpart.descriptor_reg_class_id =
        counterpart_interval.value_class.register_class_id;
    counterpart.start_point = counterpart_acquisition_start;
    counterpart.end_point = counterpart_interval.end_point;
    counterpart.unit_count = counterpart_interval.unit_count;
    counterpart.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    counterpart.location_count = counterpart_interval.unit_count;
    loom_low_allocation_unit_liveness_t unit_liveness = {};
    loom_low_allocation_edge_alias_context_t context = {};
    context.placement = &placement;
    context.liveness = &liveness;
    context.unit_liveness = &unit_liveness;
    bool allows_overlap = false;
    IREE_CHECK_OK(loom_low_allocation_edge_alias_allows_counterpart_overlap(
        &context, &intervals[candidate_ordinal], &relation, &counterpart,
        /*destination_unit_offset=*/0, intervals[candidate_ordinal].unit_count,
        &allows_overlap));
    return allows_overlap;
  }

  // Arena blocks backing the module and its IR.
  iree_arena_block_pool_t pool_;
  // Registered low operations used by the API fixtures.
  loom_context_t context_;
  // Owned value and block identities.
  loom_module_t* module_ = nullptr;
  // Entry and earlier-use blocks precede the destination and source blocks.
  loom_region_t* region_ = nullptr;
};

TEST_F(LowAllocationEdgeAliasTest, UsesExactSourceDefinitionForDestination) {
  const auto source = AddArgument(kSourceBlock, 2);
  const auto destination = AddArgument(kDestinationBlock, 2);
  loom_op_t* branch = Branch(source);
  loom_liveness_interval_t intervals[] = {Interval(source, 4, 16, 10),
                                          Interval(destination, 8, 14, 8)};

  // The old destination survives the source definition at 10, although absent
  // at both its component acquisition at 2 and semantic live-through start
  // at 4.
  EXPECT_FALSE(AllowsOverlap(intervals, /*candidate_ordinal=*/1,
                             /*counterpart_ordinal=*/0,
                             /*counterpart_acquisition_start=*/2, branch));
  EXPECT_FALSE(AllowsOverlap(intervals, /*candidate_ordinal=*/1,
                             /*counterpart_ordinal=*/0,
                             /*counterpart_acquisition_start=*/4, branch));

  // Once the destination is dead at the actual definition, earlier physical
  // acquisition alone must not forbid the edge handoff.
  intervals[1].end_point = 9;
  EXPECT_TRUE(AllowsOverlap(intervals, /*candidate_ordinal=*/1,
                            /*counterpart_ordinal=*/0,
                            /*counterpart_acquisition_start=*/2, branch));
}

TEST_F(LowAllocationEdgeAliasTest, KeepsConcatPartDefinitionForSource) {
  const auto part = AddArgument(kSourceBlock, 1);
  const auto destination = AddArgument(kDestinationBlock, 2);
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena,
                          loom_region_block(region_, kSourceBlock), &builder);
  const loom_value_id_t parts[] = {part, part};
  loom_op_t* concat = nullptr;
  IREE_ASSERT_OK(loom_low_concat_build(&builder, parts, IREE_ARRAYSIZE(parts),
                                       loom_low_register_type(1, 0, 2),
                                       LOOM_LOCATION_UNKNOWN, &concat));
  const auto aggregate = loom_low_concat_result(concat);
  loom_op_t* branch = Branch(aggregate);
  const loom_liveness_interval_t intervals[] = {Interval(aggregate, 10, 16, 10),
                                                Interval(destination, 4, 8, 4),
                                                Interval(part, 2, 11, 5)};

  // Concat-source coalescing supplies the part interval with the aggregate's
  // downstream edge relation. The destination survives the part definition at
  // 5, although absent at its live-through start at 2 and aggregate definition
  // at 10.
  EXPECT_FALSE(AllowsOverlap(intervals, /*candidate_ordinal=*/2,
                             /*counterpart_ordinal=*/1,
                             /*counterpart_acquisition_start=*/4, branch));
}

}  // namespace
}  // namespace loom
