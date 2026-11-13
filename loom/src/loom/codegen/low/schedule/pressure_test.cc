// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/completion_demand.h"
#include "loom/codegen/low/schedule/completion_pressure.h"
#include "loom/codegen/low/schedule/dependencies.h"
#include "loom/codegen/low/schedule/dependency_incoming.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/target/residency.h"

namespace loom {
namespace {

class CandidatePressureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Owns blocks backing the native dependency and completion indexes.
  iree_arena_block_pool_t pool_;
  // Scratch released after each candidate-scoring test.
  iree_arena_allocator_t arena_;
};

TEST_F(CandidatePressureTest, RequiredOwnershipUsesRetainedInstructionPhases) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 4,
  };
  const loom_low_operand_t operands[] = {
      {.role = LOOM_LOW_OPERAND_ROLE_RESULT, .unit_count = 2},
      {.role = LOOM_LOW_OPERAND_ROLE_OPERAND, .unit_count = 2},
  };
  const uint16_t operand_indices[] = {0, 1, 1};
  const loom_low_descriptor_set_t descriptors = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
      .operand_indices = operand_indices,
      .operand_index_count = IREE_ARRAYSIZE(operand_indices),
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const loom_low_schedule_options_t options = {
      .strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  const loom_low_descriptor_t descriptor = {
      .operand_count = 3,
      .result_count = 1,
  };
  loom_low_schedule_node_t node = {
      .descriptor = &descriptor,
      .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
      .operand_count = 2,
      .result_count = 1,
      .value_ordinals = {.inline_value_ordinals = {0, 0, 1, 3}},
  };
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  uint32_t indegree = 0;
  loom_low_schedule_dependency_index_t dependencies;
  loom_low_schedule_dependency_detail_index_t details;
  IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
      &graph, 1, &arena_, &arena_, &indegree, &dependencies, &details));
  loom_low_schedule_dependency_incoming_index_t incoming;
  IREE_ASSERT_OK(loom_low_schedule_dependency_incoming_index_initialize(
      &dependencies, &node, &arena_, &incoming));
  loom_low_schedule_completion_demand_t completions;
  IREE_ASSERT_OK(loom_low_schedule_completion_demand_initialize(
      &dependencies, &incoming, 1, &arena_, &completions));

  // The candidate reads one two-unit owner twice and creates another. A
  // separate two-unit owner stays live. Inputs and output cross the block
  // boundary, so this one-node dependency index has no local edges.
  loom_low_schedule_value_record_t values[] = {
      {.live_unit_count = 2, .remaining_use_count = 2, .unit_count = 2},
      {.remaining_use_count = 1, .unit_count = 2},
      {.live_unit_count = 2, .remaining_use_count = 1, .unit_count = 2},
  };
  loom_value_id_t value_ids[] = {1, 2, 3};
  const loom_local_value_domain_t value_domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(values),
  };
  loom_low_schedule_completion_domain_t domain = {.capacity = 4};
  uint16_t domain_id = 0;
  uint32_t register_limit = 4;
  uint32_t activation_units = 0;
  loom_low_schedule_build_state_t state = {
      .options = &options,
      .scratch_arena = &arena_,
      .target = {.descriptor_set = &descriptors},
      .value_domain = &value_domain,
      .values = values,
      .nodes = &node,
      .dependency_index = dependencies,
      .node_constrained_activation_units = &activation_units,
      .pressure_limits =
          {
              .by_reg_class = &register_limit,
              .constrained_completion_domains = &domain,
              .constrained_completion_domain_ids_by_reg_class = &domain_id,
              .constrained_completion_domain_count = 1,
          },
  };
  loom_low_schedule_pressure_state_t pressure;
  IREE_ASSERT_OK(loom_low_schedule_pressure_initialize(&state, 1, &pressure));
  pressure.constrained_completion_demand = completions;
  pressure.current_live_units = 4;
  pressure.current_live_units_by_reg_class[0] = 4;
  auto& required = pressure.completion_pressure[0];
  IREE_ASSERT_OK(loom_low_schedule_pressure_initialize_unlock_summaries(
      &state, 1, &pressure));
  const loom_low_schedule_ready_policy_t ready = {};
  const struct {
    // Result storage overlaps inputs before their reads.
    bool early_write;
    // Source storage remains live through result writes.
    bool late_read;
  } phases[] = {{false, false}, {true, false}, {false, false},
                {false, true},  {true, true},  {false, false}};
  for (bool requires_register : {false, true}) {
    required.live_units = requires_register ? 4 : 0;
    for (const auto& phase : phases) {
      SCOPED_TRACE(::testing::Message() << "required=" << requires_register
                                        << " early=" << phase.early_write
                                        << " late=" << phase.late_read);
      node.flags = phase.late_read ? LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS : 0;
      if (phase.early_write) {
        node.flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER;
      }
      const loom_low_schedule_value_flags_t ownership =
          requires_register ? LOOM_LOW_SCHEDULE_VALUE_FLAG_CONSTRAINED_REGISTER
                            : 0;
      values[0].flags = ownership | LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      values[1].flags =
          ownership |
          (phase.early_write ? LOOM_LOW_SCHEDULE_VALUE_FLAG_EARLY_WRITE : 0);
      values[2].flags = ownership | LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      loom_low_schedule_register_debt_word_t debt = {};
      loom_low_schedule_candidate_score_t score = {};
      loom_low_schedule_pressure_score_candidate(&state, &pressure, &ready,
                                                 &indegree, 0, &debt, &score);
      EXPECT_EQ(score.projected_live_units, 4u);
      EXPECT_EQ(score.killed_live_units, 2u);
      EXPECT_EQ(score.produced_live_units, 2u);
      EXPECT_EQ(required.candidate_delta_units, 0);
      EXPECT_EQ(required.candidate_write_delta_units,
                requires_register && phase.late_read ? 2 : 0);
      EXPECT_EQ(required.candidate_early_added_units,
                requires_register && phase.early_write ? 2u : 0u);
      EXPECT_EQ(
          iree_any_bit_set(
              score.flags,
              LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_CONSTRAINED_CAPACITY),
          requires_register && (phase.early_write || phase.late_read));
      EXPECT_EQ(required.live_units, requires_register ? 4u : 0u);
      EXPECT_EQ(pressure.current_live_units, 4u);
      EXPECT_EQ(pressure.candidate_operand_count, 0u);
      EXPECT_EQ(values[0].remaining_use_count, 2u);
      EXPECT_EQ(values[0].flags, ownership | LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE);
      EXPECT_EQ(pressure.candidate_operand_use_counts[0], 0u);
    }
  }

  // Commit a spillable early-write candidate. Its six-unit temporary overlap
  // exceeds the four-unit limit more than its realized four-unit live set.
  // Only the realized penalty carries into the next scheduling decision.
  required.live_units = 0;
  node.flags = LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER;
  values[0].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_EARLY_WRITE;
  values[2].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  loom_low_schedule_register_debt_word_t debt = {};
  loom_low_schedule_candidate_score_t score = {};
  loom_low_schedule_pressure_score_candidate(&state, &pressure, &ready,
                                             &indegree, 0, &debt, &score);
  EXPECT_EQ(score.pressure_cliff_penalty, 3u);
  EXPECT_EQ(score.persistent_pressure_cliff_penalty, 1u);
  node.scheduled_ordinal = 0;
  loom_low_schedule_pressure_note_node_scheduled(&state, &pressure, 0, &score);
  EXPECT_EQ(pressure.current_persistent_pressure_penalty, 1u);
  EXPECT_EQ(pressure.current_live_units, 4u);
  EXPECT_EQ(values[0].remaining_use_count, 0u);
  EXPECT_EQ(values[0].live_unit_count, 0u);
  EXPECT_EQ(values[1].live_unit_count, 2u);
  EXPECT_EQ(values[2].live_unit_count, 2u);
}

TEST_F(CandidatePressureTest, RetainedPlacementAndReplayHaveSeparateLifetimes) {
  const loom_low_reg_class_t reg_class = {.alloc_unit_bits = 32};
  loom_low_operand_t operands[] = {
      {.role = LOOM_LOW_OPERAND_ROLE_RESULT, .unit_count = 1},
      {.role = LOOM_LOW_OPERAND_ROLE_OPERAND, .unit_count = 1},
  };
  const uint16_t operand_indices[] = {0, 1};
  const loom_low_descriptor_set_t descriptors = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
      .operand_indices = operand_indices,
      .operand_index_count = IREE_ARRAYSIZE(operand_indices),
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_descriptor_t descriptor = {.result_count = 1};
  loom_low_schedule_node_t node = {
      .descriptor = &descriptor,
      .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
      .result_count = 1,
  };
  // Both values cross the block boundary. Module IDs deliberately differ from
  // the local ordinals used by the candidate's operands and results.
  loom_low_schedule_value_record_t values[] = {
      {
          .live_unit_count = 1,
          .remaining_use_count = 1,
          .unit_count = 1,
          .flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE,
      },
      {.remaining_use_count = 1, .unit_count = 1},
  };
  loom_value_id_t value_ids[] = {7, 19};
  const loom_local_value_domain_t value_domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(values),
  };
  loom_low_schedule_options_t options = {
      .strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  loom_low_schedule_build_state_t state = {
      .options = &options,
      .scratch_arena = &arena_,
      .target = {.descriptor_set = &descriptors},
      .value_domain = &value_domain,
      .values = values,
      .nodes = &node,
  };
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  uint32_t indegree = 0;
  loom_low_schedule_dependency_detail_index_t details;
  IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
      &graph, 1, &arena_, &arena_, &indegree, &state.dependency_index,
      &details));
  loom_low_schedule_pressure_state_t pressure;
  IREE_ASSERT_OK(loom_low_schedule_pressure_initialize(&state, 1, &pressure));
  IREE_ASSERT_OK(loom_low_schedule_pressure_initialize_unlock_summaries(
      &state, 1, &pressure));
  pressure.current_live_units = 1;
  pressure.current_live_units_by_reg_class[0] = 1;
  const loom_low_schedule_ready_policy_t ready = {};

  for (bool captured : {false, true}) {
    node.operand_count = captured ? 1 : 0;
    descriptor.operand_count = node.operand_count + 1;
    node.value_ordinals.inline_value_ordinals[0] = captured ? 0 : 1;
    node.value_ordinals.inline_value_ordinals[1] = 1;
    for (bool replayable : {false, true}) {
      operands[0].flags =
          replayable ? LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE : 0;
      for (bool lowered_spills : {false, true}) {
        options.flags = lowered_spills
                            ? LOOM_LOW_SCHEDULE_FLAG_HAS_LOWERED_SPILL_TRAFFIC
                            : 0;
        for (loom_value_id_t placement : {0u, 7u, 19u}) {
          SCOPED_TRACE(::testing::Message() << "captured=" << captured
                                            << " replayable=" << replayable
                                            << " spills=" << lowered_spills
                                            << " placement=" << placement);
          uint64_t placement_word = 0;
          options.per_user_placement_values =
              placement == 0 ? iree_bitmap_t{}
                             : iree_bitmap_t{64, &placement_word};
          if (placement != 0) {
            iree_bitmap_set(options.per_user_placement_values, placement);
          }
          loom_low_schedule_register_debt_word_t debt = {};
          loom_low_schedule_candidate_score_t score = {};
          loom_low_schedule_pressure_score_candidate(
              &state, &pressure, &ready, &indegree, 0, &debt, &score);
          EXPECT_EQ(
              iree_any_bit_set(score.flags,
                               LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_STORAGE_SETUP),
              placement == value_ids[1]);
          EXPECT_EQ(
              iree_any_bit_set(
                  score.flags,
                  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_RETAINS_PRE_SPILL_PLACEMENT),
              placement == value_ids[1] && !lowered_spills);
          EXPECT_EQ(iree_any_bit_set(
                        score.flags,
                        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_REMATERIALIZABLE_LEAF),
                    replayable && !captured);
          EXPECT_EQ(score.killed_live_units, captured ? 1u : 0u);
          EXPECT_EQ(score.produced_live_units, 1u);
        }
      }
    }
  }
}

TEST(PressureTest, SourceCliffFloorRetainsReadWriteHighWater) {
  const loom_low_reg_class_t reg_class = {.alloc_unit_bits = 32};
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const iree_string_view_t resource_name = IREE_SV("registers");
  const loom_target_residency_cliff_t cliffs[] = {
      {0, 4, 4, 3}, {0, 8, 3, 2}, {0, 12, 2, 1}};
  const loom_target_residency_cliff_range_t range = {
      .start = 0,
      .count = IREE_ARRAYSIZE(cliffs),
  };
  const loom_target_residency_direct_resource_table_t resources = {
      .names = &resource_name,
      .cliffs = cliffs,
      .cliff_count = IREE_ARRAYSIZE(cliffs),
      .cliff_ranges = &range,
      .resource_count = 1,
  };
  const struct {
    // Units read by the node, shared by both operand occurrences.
    uint32_t input_units;
    // Units written by the node and live at its successor.
    uint32_t output_units;
    // Whether the inputs remain live through the result write.
    bool late_read;
    // First cliff strictly above the source-order high-water mark.
    uint32_t first_cliff;
  } cases[] = {
      {1, 3, false, 0}, {1, 3, true, 1},  {4, 4, false, 1},
      {4, 4, true, 2},  {8, 4, false, 2}, {8, 4, true, 3},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test_case.input_units << " -> " << test_case.output_units
                 << " late=" << test_case.late_read);
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = test_case.input_units, .register_class_id = 0},
        {.unit_count = test_case.output_units, .register_class_id = 0},
    };
    loom_local_value_domain_t domain = {.value_count = IREE_ARRAYSIZE(values)};
    const loom_liveness_block_relation_t block_liveness = {};
    const loom_liveness_dataflow_t liveness = {
        .blocks = &block_liveness,
        .block_count = 1,
    };
    loom_low_schedule_build_state_t state = {
        .pressure_cliffs = &resources,
        .target = {.descriptor_set = &descriptors},
        .value_domain = &domain,
        .liveness_dataflow = &liveness,
        .values = values,
    };
    uint64_t live_units[] = {0};
    uint32_t packing_reserve[] = {0};
    uint32_t first_cliffs[] = {0};
    uint16_t block_classes[1] = {};
    uint8_t block_class_touched[1] = {};
    loom_value_ordinal_t block_values[IREE_ARRAYSIZE(values)] = {};
    loom_low_schedule_pressure_state_t pressure = {
        .current_live_units_by_reg_class = live_units,
        .packing_reserve_units_by_reg_class = packing_reserve,
        .first_actionable_pressure_cliff_indices = first_cliffs,
        .block_reg_class_ids = block_classes,
        .block_reg_class_touched_flags = block_class_touched,
        .block_value_ordinals = block_values,
    };
    const loom_low_schedule_node_t consumer = {
        .operand_count = 1,
        .value_ordinals = {.inline_value_ordinals = {1}},
    };
    loom_low_schedule_node_t producer = {
        .operand_count = 2,
        .result_count = 1,
        .value_ordinals = {.inline_value_ordinals = {0, 0, 1, 3}},
    };
    if (test_case.late_read) {
      producer.flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS;
    }
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &consumer);
    EXPECT_EQ(pressure.current_live_units, test_case.output_units);
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &producer);
    EXPECT_EQ(pressure.current_live_units, test_case.input_units);
    EXPECT_EQ(live_units[0], test_case.input_units);
    EXPECT_EQ(values[0].live_unit_count, test_case.input_units);
    EXPECT_EQ(values[1].live_unit_count, 0u);
    EXPECT_EQ(pressure.block_value_count, 2u);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);

    // Starting another block clears live state, but cannot make a pressure
    // threshold already reached by the authored order actionable again.
    loom_low_schedule_initialize_source_pressure_block(&state, &pressure, 0);
    EXPECT_EQ(pressure.current_live_units, 0u);
    EXPECT_EQ(live_units[0], 0u);
    EXPECT_EQ(pressure.block_value_count, 0u);
    EXPECT_EQ(pressure.block_reg_class_count, 0u);
    EXPECT_EQ(block_class_touched[0], 0u);
    for (const auto& value : values) {
      EXPECT_EQ(value.flags, 0u);
      EXPECT_EQ(value.live_unit_count, 0u);
    }
    loom_low_schedule_reverse_source_pressure_node(&state, &pressure,
                                                   &consumer);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);
    EXPECT_EQ(pressure.current_live_units, test_case.output_units);
    loom_low_schedule_reset_source_pressure_sweep(&state, &pressure);
    EXPECT_EQ(pressure.current_live_units, 0u);
    EXPECT_EQ(first_cliffs[0], test_case.first_cliff);
  }
}

}  // namespace
}  // namespace loom
