// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure_alias.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/completion_pressure.h"
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
                     loom_low_schedule_value_flags_t source_flags,
                     uint32_t required_mask = 0, bool alias_dies = false,
                     bool writable_result = false) {
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
    values[2].flags =
        writable_result ? LOOM_LOW_SCHEDULE_VALUE_FLAG_WRITTEN_IN_PLACE : 0;
    if (alias_dies) {
      values[1].remaining_use_count = 1;
      values[1].flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ;
    }
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
      if (required_mask & (1u << i)) {
        values[i].flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER;
      }
    }
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
      rows[i].flags = LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED;
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
    uint16_t completion_domains[] = {0};
    state.pressure_limits.constrained_completion_domain_ids_by_reg_class =
        completion_domains;
    loom_low_schedule_completion_domain_t completion = {
        .capacity = 8,
        .reg_class_id = 0,
    };
    state.pressure_limits.constrained_completion_domains = &completion;
    state.pressure_limits.constrained_completion_domain_count = 1;
    loom_low_schedule_completion_pressure_t required = {};
    pressure.completion_pressure = &required;
    IREE_ASSERT_OK(loom_low_schedule_pressure_alias_initialize(
        &state, &pressure.storage_aliases));
    values[1].live_unit_count =
        alias_unit_count -
        loom_low_schedule_pressure_alias_append_scheduled_result(
            &state, &pressure, 0, 1);
    const uint64_t source_required_units = required_mask & 1u ? 2 : 0;
    const uint64_t alias_required_units =
        required_mask & 2u ? alias_unit_count - 2 : 0;
    required.live_units = source_required_units + alias_required_units;
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
    const bool late = iree_any_bit_set(
        source_flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ);
    const int64_t transferred_required_units = required_mask & 2u ? 2 : 0;
    if (alias_dies) {
      // The alias inherits storage between the ordinary source read and its
      // own late read. It contributes to write-time pressure, not final demand.
      EXPECT_EQ(transfer.live_units, 0u);
      EXPECT_EQ(transfer.late_units, 2u);
      EXPECT_EQ(required.candidate_delta_units, 0);
      EXPECT_EQ(required.candidate_write_delta_units,
                late ? 0 : transferred_required_units);
      return;
    }
    EXPECT_EQ(transfer.live_units, 2u);
    const uint32_t predicted =
        loom_low_schedule_pressure_alias_candidate_result_units(
            &state, &pressure, 1, 2);
    EXPECT_EQ(required.candidate_delta_units, transferred_required_units);
    EXPECT_EQ(required.candidate_write_delta_units,
              late ? 0 : transferred_required_units);
    EXPECT_EQ(predicted, late || writable_result ? 0u : alias_unit_count);
    if (!late) {
      EXPECT_EQ(loom_low_schedule_pressure_alias_transfer_from_source(
                    &state, &pressure, 0),
                2u);
      EXPECT_EQ(required.live_units, source_required_units +
                                         alias_required_units +
                                         transferred_required_units);
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

  void CheckSourceBaseline(loom_low_storage_relation_kind_t kind,
                           uint16_t source_count, uint16_t relation_count,
                           uint32_t shared_units, bool writable_result = false,
                           bool required_relation = false) {
    for (bool source_first : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "kind=" << static_cast<uint32_t>(kind) << " sources="
                   << source_count << " relations=" << relation_count
                   << " source-first=" << source_first);
      loom_low_schedule_value_record_t values[3] = {};
      values[0].unit_count = kind == LOOM_LOW_STORAGE_RELATION_SUBRANGE ? 4 : 2;
      values[1].unit_count = values[0].unit_count;
      values[2].unit_count = 2u * relation_count;
      const loom_low_schedule_value_flags_t result_flags =
          writable_result ? LOOM_LOW_SCHEDULE_VALUE_FLAG_WRITTEN_IN_PLACE : 0;
      values[2].flags = result_flags;
      uint32_t producer_nodes[] = {LOOM_LOW_SCHEDULE_NODE_NONE,
                                   LOOM_LOW_SCHEDULE_NODE_NONE, 0};
      loom_low_schedule_node_t producer = {
          .operand_count = relation_count,
          .result_count = 1,
          .storage_relation_count = relation_count,
      };
      uint32_t relation_starts[] = {0, relation_count};
      loom_segmented_storage_t relations;
      loom_segmented_storage_initialize(
          relation_count * sizeof(loom_low_schedule_storage_relation_t),
          alignof(loom_low_schedule_storage_relation_t), &relations);
      loom_low_schedule_storage_relation_t* rows = nullptr;
      IREE_ASSERT_OK(loom_segmented_storage_append(
          &relations, &arena_, reinterpret_cast<void**>(&rows)));
      for (uint16_t i = 0; i < relation_count; ++i) {
        rows[i] = {};
        rows[i].source_ordinal = source_count == 1 ? 0 : i;
        rows[i].destination_ordinal = 2;
        rows[i].source_operand_index = i;
        rows[i].source_unit_offset =
            kind == LOOM_LOW_STORAGE_RELATION_SUBRANGE ? 1 : 0;
        rows[i].destination_unit_offset = 2u * i;
        rows[i].unit_count = 2;
        rows[i].kind = kind;
        rows[i].flags = required_relation
                            ? LOOM_LOW_STORAGE_RELATION_FLAG_HARD
                            : LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED;
        loom_low_schedule_node_operand_ordinals(&producer)[i] =
            rows[i].source_ordinal;
      }
      loom_low_schedule_node_result_ordinals(&producer)[0] = 2;
      loom_local_value_domain_t domain = {
          .value_count = IREE_ARRAYSIZE(values),
      };
      loom_low_schedule_build_state_t state = {
          .scratch_arena = &arena_,
          .value_domain = &domain,
          .values = values,
          .value_producer_nodes = producer_nodes,
          .nodes = &producer,
          .storage_relations = {relation_starts, &relations, 1, relation_count},
      };
      loom_low_schedule_pressure_state_t pressure = {};
      IREE_ASSERT_OK(loom_low_schedule_pressure_alias_initialize(
          &state, &pressure.storage_aliases));
      // A live-in result from another block cannot inherit a local source.
      EXPECT_EQ(loom_low_schedule_pressure_alias_append_source_baseline_result(
                    &state, &pressure, 1, 2),
                0u);
      EXPECT_EQ(pressure.storage_aliases.source_count, 0u);
      if (source_first) {
        for (uint16_t i = 0; i < source_count; ++i) {
          values[i].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
          values[i].live_unit_count = values[i].unit_count;
        }
      }
      const uint32_t owned =
          loom_low_schedule_pressure_alias_append_source_baseline_result(
              &state, &pressure, 0, 2);
      EXPECT_EQ(owned, source_first ? shared_units : 0u);
      values[2].flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      values[2].live_unit_count = values[2].unit_count - owned;
      if (!source_first) {
        uint32_t transferred = 0;
        for (uint16_t i = 0; i < source_count; ++i) {
          transferred += loom_low_schedule_pressure_alias_transfer_to_source(
              &state, &pressure, i);
          values[i].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
          values[i].live_unit_count = values[i].unit_count;
        }
        EXPECT_EQ(transferred, shared_units);
      }
      EXPECT_EQ(values[2].live_unit_count, values[2].unit_count - shared_units);
      uint32_t transferred = 0;
      for (uint16_t i = 0; i < source_count; ++i) {
        transferred += loom_low_schedule_pressure_alias_transfer_from_source(
            &state, &pressure, i);
        values[i].flags = 0;
        values[i].live_unit_count = 0;
      }
      EXPECT_EQ(transferred, shared_units);
      EXPECT_EQ(values[2].live_unit_count, values[2].unit_count);
      if (source_first) {
        loom_low_schedule_pressure_alias_deactivate_result(&state, &pressure,
                                                           2);
        for (uint16_t i = 0; i < source_count; ++i) {
          EXPECT_EQ(pressure.storage_aliases.source_unit_counts[i], 0u);
        }
        EXPECT_EQ(values[2].flags,
                  result_flags | LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE);
      }

      // Block boundaries discard claims even when their result is still live.
      // The pressure owner clears value flags separately from the alias index.
      loom_low_schedule_pressure_alias_reset(&pressure.storage_aliases);
      values[2].flags = result_flags | LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      EXPECT_EQ(pressure.storage_aliases.source_count, 0u);
      for (uint16_t i = 0; i < source_count; ++i) {
        EXPECT_EQ(pressure.storage_aliases.source_unit_counts[i], 0u);
        EXPECT_EQ(pressure.storage_aliases.source_heads[i],
                  LOOM_LOW_SCHEDULE_NODE_NONE);
      }
      EXPECT_EQ(loom_low_schedule_pressure_alias_append_source_baseline_result(
                    &state, &pressure, 0, 2),
                0u);
      transferred = 0;
      for (uint16_t i = 0; i < source_count; ++i) {
        transferred += loom_low_schedule_pressure_alias_transfer_to_source(
            &state, &pressure, i);
      }
      EXPECT_EQ(transferred, shared_units);
    }
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

TEST_F(PressureAliasTest, RequiredOwnershipFollowsTheTransferPhase) {
  for (uint32_t required_mask = 0; required_mask < 4; ++required_mask) {
    SCOPED_TRACE(required_mask);
    for (uint32_t alias_unit_count : {2u, 4u}) {
      CheckTransfer(alias_unit_count, 0, required_mask);
      CheckTransfer(alias_unit_count,
                    LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ,
                    required_mask);
    }
  }
}

TEST_F(PressureAliasTest, RequiredLateAliasRetainsTransientOwnership) {
  for (uint32_t required_mask = 0; required_mask < 4; ++required_mask) {
    SCOPED_TRACE(required_mask);
    for (uint32_t alias_unit_count : {2u, 4u}) {
      CheckTransfer(alias_unit_count, 0, required_mask, true);
      CheckTransfer(alias_unit_count,
                    LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ,
                    required_mask, true);
    }
  }
}

TEST_F(PressureAliasTest, SourceBaselineChargesSharedStorageOnce) {
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_SAME_STORAGE, 1, 1, 2);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_SUBRANGE, 1, 1, 2);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART, 2, 2, 4);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART, 1, 2, 2);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_DISJOINT_STORAGE, 1, 1, 0);
}

TEST_F(PressureAliasTest, WritableResultsOwnStorageAfterSourceTransfers) {
  for (uint32_t alias_unit_count : {2u, 4u}) {
    for (uint32_t required_mask = 0; required_mask < 8; ++required_mask) {
      CheckTransfer(alias_unit_count, 0, required_mask, false, true);
      CheckTransfer(alias_unit_count,
                    LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_LATE_READ,
                    required_mask, false, true);
    }
  }
}

TEST_F(PressureAliasTest, WritablePreferredResultsRetainIndependentPressure) {
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_SAME_STORAGE, 1, 1, 0, true);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_SUBRANGE, 1, 1, 0, true);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART, 2, 2, 0, true);
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART, 1, 2, 0, true);
}

TEST_F(PressureAliasTest, RequiredStorageStillSharesItsPressure) {
  CheckSourceBaseline(LOOM_LOW_STORAGE_RELATION_SAME_STORAGE, 1, 1, 2, true,
                      true);
}

}  // namespace
}  // namespace loom
