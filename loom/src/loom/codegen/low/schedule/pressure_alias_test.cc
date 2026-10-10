// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure_alias.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/pressure.h"

namespace loom {
namespace {

class PressureAliasTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void CheckTransfer(uint32_t alias_unit_count,
                     loom_low_schedule_value_flags_t source_flags) {
    // An active source owns two units borrowed by a surviving alias. The
    // candidate consumes that source and produces a new alias of the survivor.
    // Wider survivors already own their remaining units privately.
    loom_low_schedule_value_record_t values[3] = {};
    for (auto& value : values) {
      value.unit_count = alias_unit_count;
      value.flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      value.remaining_use_count = 2;
    }
    values[0].unit_count = 2;
    values[0].live_unit_count = 2;
    values[0].remaining_use_count = 1;
    values[0].flags |= source_flags;
    values[2].flags = 0;
    uint32_t value_producer_nodes[] = {
        LOOM_LOW_SCHEDULE_NODE_NONE,
        0,
        1,
    };
    loom_low_schedule_node_t nodes[2] = {};
    for (auto& node : nodes) {
      node.storage_relation_count = 1;
    }
    uint32_t relation_starts[] = {0, 1, 2};
    loom_segmented_storage_t relations;
    loom_segmented_storage_initialize(
        2 * sizeof(loom_low_schedule_storage_relation_t),
        alignof(loom_low_schedule_storage_relation_t), &relations);
    loom_low_schedule_storage_relation_t* rows = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(
        &relations, &arena_, reinterpret_cast<void**>(&rows)));
    for (uint32_t i = 0; i < 2; ++i) {
      rows[i] = {};
      rows[i].source_ordinal = i;
      rows[i].destination_ordinal = i + 1;
      rows[i].unit_count = values[i].unit_count;
      rows[i].kind = LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART;
    }
    loom_local_value_domain_t domain = {.value_count = IREE_ARRAYSIZE(values)};
    loom_low_schedule_build_state_t state = {
        .scratch_arena = &arena_,
        .value_domain = &domain,
        .values = values,
        .value_producer_nodes = value_producer_nodes,
        .nodes = nodes,
        .storage_relations = {relation_starts, &relations, 2, 2},
    };
    loom_low_schedule_pressure_state_t pressure = {};
    IREE_ASSERT_OK(loom_low_schedule_pressure_alias_initialize(
        &state, &pressure.storage_aliases));
    values[1].live_unit_count =
        alias_unit_count -
        loom_low_schedule_pressure_alias_append_scheduled_result(
            &state, &pressure, 0, 1);
    uint16_t operand_use_counts[] = {1, 1, 0};
    uint32_t scratch_counts[] = {0, 0, 0};
    loom_value_ordinal_t operand_ordinals[] = {0, 1, 0};
    pressure.candidate_operand_use_counts = operand_use_counts;
    pressure.candidate_scratch_counts = scratch_counts;
    pressure.candidate_operand_ordinals = operand_ordinals;
    pressure.candidate_operand_count = 2;

    const auto transfer =
        loom_low_schedule_pressure_alias_candidate_transfer_from_source(
            &state, &pressure, 0);
    EXPECT_EQ(transfer.live_units, 2u);
    const uint32_t predicted =
        loom_low_schedule_pressure_alias_candidate_result_units(
            &state, &pressure, 1, 2);
    const bool late = iree_any_bit_set(
        source_flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ);
    EXPECT_EQ(predicted, late ? 0u : alias_unit_count);
    if (!late) {
      EXPECT_EQ(loom_low_schedule_pressure_alias_transfer_from_source(
                    &state, &pressure, 0),
                2u);
      values[0].flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      values[0].live_unit_count = 0;
      values[0].remaining_use_count = 0;
    }
    --values[1].remaining_use_count;
    values[2].flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    EXPECT_EQ(loom_low_schedule_pressure_alias_append_scheduled_result(
                  &state, &pressure, 1, 2),
              predicted);
    loom_low_schedule_reset_candidate_operand_uses(&state, &pressure);
    EXPECT_EQ(values[1].candidate_transferred_units, 0u);
  }

  // Owns blocks for the relation index and mutable alias simulation.
  iree_arena_block_pool_t pool_;
  // Scratch state for each test's simulated schedule.
  iree_arena_allocator_t arena_;
};

TEST_F(PressureAliasTest, InheritsBeforeResultWrites) {
  CheckTransfer(2, 0);
  CheckTransfer(4, 0);
}

TEST_F(PressureAliasTest, LateInheritanceCannotCoalesceAnEarlierResult) {
  CheckTransfer(2, LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ);
  CheckTransfer(4, LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ);
}

}  // namespace
}  // namespace loom
