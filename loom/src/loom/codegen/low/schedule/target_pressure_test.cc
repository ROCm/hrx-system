// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/target_pressure.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/completion_demand.h"
#include "loom/codegen/low/schedule/dependencies.h"
#include "loom/codegen/low/schedule/dependency_incoming.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/required_pressure.h"
#include "loom/target/residency.h"

namespace loom {
namespace {

TEST(TargetPressureTest, ResidencyHeadroomBelongsToGrowingResources) {
  const loom_low_reg_class_t classes[] = {
      {.alloc_unit_bits = 64},
      {.alloc_unit_bits = 32},
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  const iree_string_view_t resource_names[] = {IREE_SV("pointer"),
                                               IREE_SV("scalar")};
  const loom_target_residency_cliff_t cliffs[] = {{0, 8, 15, 14},
                                                  {0, 12, 14, 13}};
  const loom_target_residency_cliff_range_t ranges[] = {
      {.start = 0, .count = IREE_ARRAYSIZE(cliffs)},
      {.start = IREE_ARRAYSIZE(cliffs), .count = 0},
  };
  const loom_target_residency_model_t residency = {
      .best_tier = 15,
      .direct_resources =
          {
              .names = resource_names,
              .cliffs = cliffs,
              .cliff_count = IREE_ARRAYSIZE(cliffs),
              .cliff_ranges = ranges,
              .resource_count = IREE_ARRAYSIZE(ranges),
          },
  };
  const loom_low_schedule_options_t options = {
      .residency = loom_target_residency_view(&residency, UINT32_MAX),
  };
  const loom_low_schedule_build_state_t state = {
      .options = &options,
      .pressure_cliffs = &residency.direct_resources,
      .target = {.descriptor_set = &descriptors},
  };
  const uint32_t no_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
  const struct {
    // Human-readable pressure transition for assertion failures.
    const char* name;
    // Whether the candidate's pointer delta is current rather than stale.
    uint8_t pointer_touched;
    // Change to the five currently live pointer units.
    int64_t pointer_delta;
    // First cliff not already crossed by the source-order baseline.
    uint32_t first_cliff;
    // Residency tiers lost by the candidate.
    uint32_t penalty;
    // Additional growing-resource units before losing another tier.
    uint32_t units_until_cliff;
  } cases[] = {
      {"untouched pointer bank", 0, 99, 0, 0, no_cliff},
      {"unchanged pointer bank", 1, 0, 0, 0, no_cliff},
      {"shrinking pointer bank", 1, -1, 0, 0, no_cliff},
      {"growing below cliff", 1, 1, 0, 0, 2},
      {"one unit below cliff", 1, 2, 0, 0, 1},
      {"crossing cliff", 1, 3, 0, 1, no_cliff},
      {"first cliff already crossed by source", 1, 3, 1, 0, 4},
      {"next cliff remains actionable", 1, 7, 1, 1, no_cliff},
      {"both cliffs crossed", 1, 7, 0, 2, no_cliff},
      {"all cliffs already crossed by source", 1, 7, 2, 0, no_cliff},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    uint64_t live[] = {5, 0};
    int64_t delta[] = {test_case.pointer_delta, 1};
    uint8_t delta_touched[] = {test_case.pointer_touched, 1};
    uint16_t delta_classes[] = {1, 0};
    uint16_t block_classes[] = {0};
    uint8_t block_touched[] = {1, 0};
    uint32_t first_cliffs[] = {test_case.first_cliff, IREE_ARRAYSIZE(cliffs)};
    loom_low_schedule_pressure_state_t pressure = {
        .current_live_units_by_reg_class = live,
        .first_actionable_pressure_cliff_indices = first_cliffs,
        .block_reg_class_ids = block_classes,
        .block_reg_class_touched_flags = block_touched,
        .candidate_delta_units_by_reg_class = delta,
        .candidate_delta_touched_flags = delta_touched,
        .candidate_delta_touched_reg_class_ids = delta_classes,
        .candidate_delta_touched_count = 1u + test_case.pointer_touched,
        .block_reg_class_count = IREE_ARRAYSIZE(block_classes),
    };
    loom_low_schedule_candidate_score_t score = {
        .pressure_cliff_units = no_cliff,
        .units_until_pressure_cliff = no_cliff,
    };
    loom_low_schedule_target_pressure_score_candidate(
        &state, &pressure, LOOM_LOW_SCHEDULE_NODE_NONE, &score);
    EXPECT_EQ(score.pressure_cliff_penalty, test_case.penalty);
    EXPECT_EQ(score.persistent_pressure_cliff_penalty, test_case.penalty);
    EXPECT_EQ(score.units_until_pressure_cliff, test_case.units_until_cliff);
  }
}

class TargetPressureScoringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Owns completion indexes used by native candidate scoring.
  iree_arena_block_pool_t pool_;
  // Scratch for dependency and completion initialization.
  iree_arena_allocator_t arena_;
};

TEST_F(TargetPressureScoringTest, CompletionDomainsSharePressureSets) {
  const loom_low_reg_class_t classes[] = {
      {.alloc_unit_bits = 32, .allocatable_count = 3},
      {.alloc_unit_bits = 32, .allocatable_count = 4, .alias_set_id = 1},
      {.alloc_unit_bits = 64, .allocatable_count = 4, .alias_set_id = 1},
      {.alloc_unit_bits = 32},
      {.alloc_unit_bits = 32, .allocatable_count = 2},
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  loom_low_schedule_value_record_t values[IREE_ARRAYSIZE(classes)] = {};
  for (uint16_t reg_class_id = 0; reg_class_id < IREE_ARRAYSIZE(classes);
       ++reg_class_id) {
    values[reg_class_id].register_class_id = reg_class_id;
    if (reg_class_id != 4) {
      values[reg_class_id].flags |=
          LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
    }
  }
  loom_local_value_domain_t value_domain = {
      .value_count = IREE_ARRAYSIZE(values),
  };
  uint32_t capacities[] = {3, 4, 4, UINT32_MAX, 2};
  loom_low_schedule_pressure_set_limit_t pressure_sets[2] = {};
  pressure_sets[1] = {
      .live_unit_limit = 4,
      .unspillable_completion_domain_id = UINT16_MAX,
      .has_multiple_active_reg_classes = 1,
  };
  loom_low_schedule_build_state_t state = {
      .scratch_arena = &arena_,
      .target = {.descriptor_set = &descriptors},
      .value_domain = &value_domain,
      .values = values,
      .pressure_limits =
          {
              .by_reg_class = capacities,
              .pressure_sets = pressure_sets,
          },
  };

  IREE_ASSERT_OK(
      loom_low_schedule_target_pressure_initialize_completion_domains(&state));

  EXPECT_EQ(state.pressure_limits.unspillable_completion_domain_count, 2);
  const uint16_t* domain_ids =
      state.pressure_limits.unspillable_completion_domain_ids_by_reg_class;
  EXPECT_EQ(domain_ids[0], 0);
  EXPECT_EQ(domain_ids[1], 1);
  EXPECT_EQ(domain_ids[2], 1);
  EXPECT_EQ(domain_ids[3], UINT16_MAX);
  EXPECT_EQ(domain_ids[4], UINT16_MAX);
  EXPECT_EQ(pressure_sets[1].unspillable_completion_domain_id, 1);
  const loom_low_schedule_completion_domain_t* domains =
      state.pressure_limits.unspillable_completion_domains;
  EXPECT_EQ(domains[0].capacity, 3);
  EXPECT_EQ(domains[0].reg_class_id, 0);
  EXPECT_EQ(domains[1].capacity, 4);
  EXPECT_EQ(domains[1].reg_class_id, 2);
  for (iree_host_size_t i = 0; i < value_domain.value_count; ++i) {
    EXPECT_EQ(state.value_unspillable_completions[i].activation_units,
              UINT32_MAX);
    EXPECT_EQ(state.value_unspillable_completions[i].sink,
              LOOM_LOW_SCHEDULE_NODE_NONE);
  }
}

TEST_F(TargetPressureScoringTest, GrowthUsesCanonicalBankAndOwnership) {
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  loom_low_schedule_node_t node = {
      .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
  };
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
      &dependencies, &incoming, 2, &arena_, &completions));

  const uint32_t no_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
  const struct {
    // Live units in the candidate's bank before scheduling it.
    uint64_t current_units;
    // Net change after all input reads and result writes.
    int64_t delta_units;
    // Result units written before ordinary input reads.
    uint64_t early_units;
    // Dying input units retained through result writes.
    uint64_t late_units;
    // Whether the candidate increases demand beyond the bank's capacity.
    bool opens_debt;
    // Remaining capacity in a growing bank, or no cliff for other banks.
    uint32_t units_until_limit;
    // Capacity penalty at the peak read/write phase.
    uint32_t peak_penalty;
    // Capacity penalty after all reads and writes have completed.
    uint32_t persistent_penalty;
  } cases[] = {
      {3, 1, 0, 0, false, no_cliff, 1, 1}, {4, 1, 0, 0, true, no_cliff, 2, 2},
      {5, 0, 0, 0, false, no_cliff, 2, 2}, {5, -1, 0, 0, false, no_cliff, 1, 1},
      {4, 0, 1, 0, true, no_cliff, 2, 1},  {4, 0, 0, 1, true, no_cliff, 2, 1},
      {2, 0, 0, 0, false, no_cliff, 0, 0}, {2, -1, 0, 0, false, no_cliff, 0, 0},
      {2, 1, 0, 0, false, 1, 0, 0},        {2, 0, 1, 0, false, 1, 0, 0},
      {2, 0, 0, 1, false, 1, 0, 0},
  };
  for (uint16_t reg_class_id = 0; reg_class_id < 3; ++reg_class_id) {
    for (bool required : {false, true}) {
      for (bool unspillable : {false, true}) {
        for (const auto& test_case : cases) {
          SCOPED_TRACE(::testing::Message()
                       << "class=" << reg_class_id << " required=" << required
                       << " unspillable=" << unspillable
                       << " current=" << test_case.current_units
                       << " delta=" << test_case.delta_units
                       << " early=" << test_case.early_units
                       << " late=" << test_case.late_units);
          loom_low_reg_class_t classes[] = {
              {.alloc_unit_bits = 32, .allocatable_count = 4},
              {
                  .alloc_unit_bits = 32,
                  .allocatable_count = 4,
                  .alias_set_id = 1,
              },
              {
                  .alloc_unit_bits = 32,
                  .allocatable_count = 4,
                  .alias_set_id = 1,
              },
          };
          if (unspillable) {
            for (auto& reg_class : classes) {
              reg_class.flags |= LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE;
            }
          }
          const loom_low_descriptor_set_t descriptors = {
              .reg_classes = classes,
              .reg_class_count = IREE_ARRAYSIZE(classes),
          };
          uint32_t limits[] = {4, 4, 4};
          loom_low_schedule_pressure_set_limit_t sets[2] = {};
          sets[1] = {
              .live_unit_limit = 4,
              .representative_reg_class_id = 2,
              .unspillable_completion_domain_id = 1,
              .all_classes_unspillable = unspillable,
              .has_multiple_active_reg_classes = 1,
          };
          loom_low_schedule_completion_domain_t domains[] = {
              {.capacity = 4, .reg_class_id = 0},
              {.capacity = 4, .reg_class_id = 2},
          };
          uint16_t domain_ids[] = {0, 1, 1};
          loom_low_schedule_build_state_t state = {
              .target = {.descriptor_set = &descriptors},
              .nodes = &node,
              .pressure_limits =
                  {
                      .by_reg_class = limits,
                      .pressure_sets = sets,
                      .pressure_set_count = 1,
                      .unspillable_completion_domains = domains,
                      .unspillable_completion_domain_ids_by_reg_class =
                          domain_ids,
                      .unspillable_completion_domain_count = 2,
                  },
          };
          uint64_t live[3] = {};
          live[reg_class_id] = test_case.current_units;
          int64_t delta[3] = {};
          delta[reg_class_id] = test_case.delta_units;
          uint64_t early[3] = {};
          early[reg_class_id] = test_case.early_units;
          uint64_t late[3] = {};
          late[reg_class_id] = test_case.late_units;
          uint8_t touched[3] = {};
          touched[reg_class_id] = 1;
          uint32_t reserves[3] = {};
          uint32_t activation[2] = {};
          loom_low_schedule_required_pressure_t required_pressure[2] = {};
          if (required) {
            required_pressure[domain_ids[reg_class_id]] = {
                .candidate_delta_units = 1,
                .candidate_write_delta_units = 1,
                .candidate_early_added_units = test_case.early_units,
                .candidate_opens_lifetime = true,
            };
          }
          loom_low_schedule_pressure_set_record_t shared_pressure[2] = {};
          if (reg_class_id != 0) {
            shared_pressure[1] = {
                .current_live_units = test_case.current_units,
                .candidate_delta_units = test_case.delta_units,
                .candidate_lifetime =
                    {
                        .early_added_units = test_case.early_units,
                        .late_released_units = test_case.late_units,
                    },
                .flags = LOOM_LOW_SCHEDULE_PRESSURE_SET_FLAG_BLOCK_TOUCHED |
                         LOOM_LOW_SCHEDULE_PRESSURE_SET_FLAG_CANDIDATE_TOUCHED,
            };
          }
          uint16_t shared_ids[] = {1};
          loom_low_schedule_pressure_state_t pressure = {
              .current_live_units_by_reg_class = live,
              .required_pressure = required_pressure,
              .packing_reserve_units_by_reg_class = reserves,
              .pressure_sets =
                  {
                      .records = shared_pressure,
                      .block_ids = shared_ids,
                      .candidate_delta_touched_ids = shared_ids,
                      .block_count = reg_class_id != 0 ? 1u : 0u,
                      .candidate_delta_touched_count =
                          reg_class_id != 0 ? 1u : 0u,
                  },
              .block_reg_class_ids = &reg_class_id,
              .block_reg_class_touched_flags = touched,
              .candidate_delta_units_by_reg_class = delta,
              .candidate_lifetime =
                  {
                      .early_added_units = early,
                      .late_released_units = late,
                  },
              .candidate_unspillable_activation_units = activation,
              .unspillable_completion_demand = completions,
              .candidate_delta_touched_flags = touched,
              .candidate_delta_touched_reg_class_ids = &reg_class_id,
              .candidate_delta_touched_count = 1,
              .block_reg_class_count = 1,
          };
          loom_low_schedule_register_debt_word_t debt = {};
          loom_low_schedule_candidate_score_t score = {
              .register_debt = &debt,
              .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
              .units_until_pressure_cliff =
                  LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
              .active_unspillable_completion_capacity = UINT32_MAX,
              .active_unspillable_transaction_final_capacity = UINT32_MAX,
          };
          loom_low_schedule_target_pressure_score_candidate(
              &state, &pressure, LOOM_LOW_SCHEDULE_NODE_NONE, &score);
          const uint64_t expected_bank =
              test_case.opens_debt && !unspillable
                  ? UINT64_C(1) << (reg_class_id == 0 ? 0 : 2)
                  : 0;
          EXPECT_EQ(debt.banks, expected_bank);
          EXPECT_EQ(debt.required_banks, required ? expected_bank : 0);
          EXPECT_EQ(score.units_until_pressure_cliff,
                    test_case.units_until_limit);
          EXPECT_EQ(score.pressure_cliff_penalty, test_case.peak_penalty);
          EXPECT_EQ(score.persistent_pressure_cliff_penalty,
                    test_case.persistent_penalty);
          EXPECT_EQ(iree_any_bit_set(
                        score.flags,
                        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT),
                    expected_bank != 0);
          EXPECT_EQ(
              iree_any_bit_set(
                  score.flags,
                  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REQUIRED_REGISTER_DEBT),
              required && expected_bank != 0);
          EXPECT_EQ(
              iree_any_bit_set(
                  score.flags,
                  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY),
              unspillable && test_case.opens_debt);
        }
      }
    }
  }
}

TEST_F(TargetPressureScoringTest, ActiveCompletionsRetainDomainIdentity) {
  const loom_low_reg_class_t classes[] = {
      {
          .flags = LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE,
          .alloc_unit_bits = 32,
          .allocatable_count = 4,
      },
      {
          .flags = LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE,
          .alloc_unit_bits = 32,
          .allocatable_count = 4,
      },
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  loom_low_schedule_completion_domain_t domains[] = {
      {.capacity = 4, .reg_class_id = 0},
      {.capacity = 4, .reg_class_id = 1},
  };
  loom_low_schedule_node_t nodes[3] = {};
  for (auto& node : nodes) {
    node.scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  loom_low_schedule_dependency_graph_t graph;
  loom_low_schedule_dependency_graph_initialize(&graph);
  uint32_t indegrees[3] = {};
  loom_low_schedule_dependency_index_t dependencies;
  loom_low_schedule_dependency_detail_index_t details;
  IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
      &graph, IREE_ARRAYSIZE(nodes), &arena_, &arena_, indegrees, &dependencies,
      &details));
  loom_low_schedule_dependency_incoming_index_t incoming;
  IREE_ASSERT_OK(loom_low_schedule_dependency_incoming_index_initialize(
      &dependencies, nodes, &arena_, &incoming));
  loom_low_schedule_build_state_t state = {
      .target = {.descriptor_set = &descriptors},
      .nodes = nodes,
      .pressure_limits =
          {
              .unspillable_completion_domains = domains,
              .unspillable_completion_domain_count = IREE_ARRAYSIZE(domains),
          },
  };
  uint64_t live[2] = {};
  uint32_t reserves[2] = {};
  uint32_t activation[2] = {};
  uint8_t touched[2] = {};
  loom_low_schedule_required_pressure_t required[2] = {};
  loom_low_schedule_pressure_state_t pressure = {
      .current_live_units_by_reg_class = live,
      .required_pressure = required,
      .packing_reserve_units_by_reg_class = reserves,
      .candidate_unspillable_activation_units = activation,
      .candidate_delta_touched_flags = touched,
  };
  for (const auto kind :
       {LOOM_LOW_SCHEDULE_COMPLETION_NOMINATION_FINAL,
        LOOM_LOW_SCHEDULE_COMPLETION_NOMINATION_TRANSACTION}) {
    SCOPED_TRACE(kind);
    IREE_ASSERT_OK(loom_low_schedule_completion_demand_initialize(
        &dependencies, &incoming, IREE_ARRAYSIZE(domains), &arena_,
        &pressure.unspillable_completion_demand));
    // Each bank nominates its own consumer; the third node advances neither.
    for (uint16_t domain = 0; domain < IREE_ARRAYSIZE(domains); ++domain) {
      loom_low_schedule_completion_demand_nominate(
          &pressure.unspillable_completion_demand, kind, domain, domain);
    }
    for (uint64_t live_units : {3u, 4u}) {
      SCOPED_TRACE(live_units);
      for (uint16_t domain = 0; domain < IREE_ARRAYSIZE(domains); ++domain) {
        live[domain] = live_units;
        required[domain] = {
            .live_units = live_units,
            .non_replayable_live_units = live_units,
        };
      }
      for (uint32_t candidate = 0; candidate < IREE_ARRAYSIZE(nodes);
           ++candidate) {
        SCOPED_TRACE(candidate);
        loom_low_schedule_candidate_score_t score = {
            .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
            .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
            .active_unspillable_completion_capacity = UINT32_MAX,
            .active_unspillable_transaction_final_capacity = UINT32_MAX,
            .active_unspillable_completion_domain_id = UINT16_MAX,
        };
        loom_low_schedule_target_pressure_score_candidate(&state, &pressure,
                                                          candidate, &score);
        const bool advances_full_bank = live_units == 4 && candidate < 2;
        EXPECT_EQ(score.active_unspillable_completion_capacity,
                  advances_full_bank ? 4u : UINT32_MAX);
        EXPECT_EQ(score.active_unspillable_completion_domain_id,
                  advances_full_bank ? candidate : UINT16_MAX);
      }
    }
    // Filling one bank preserves its selected transaction's missing operand.
    // Only storage without an outside-domain replay recipe makes that
    // reservation stronger than the ordinary physical-pressure comparison.
    for (bool live_replayable : {false, true}) {
      for (bool candidate_replayable : {false, true}) {
        live[0] = 3;
        live[1] = 0;
        required[0] = {
            .live_units = 3,
            .non_replayable_live_units = live_replayable ? 2u : 3u,
            .candidate_delta_units = 1,
            .non_replayable_candidate_delta_units =
                candidate_replayable ? 0 : 1,
        };
        required[1] = {};
        for (uint32_t candidate = 0; candidate < IREE_ARRAYSIZE(nodes);
             ++candidate) {
          SCOPED_TRACE(::testing::Message()
                       << kind << "," << live_replayable << ","
                       << candidate_replayable << "," << candidate);
          loom_low_schedule_candidate_score_t score = {
              .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
              .units_until_pressure_cliff =
                  LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
              .active_unspillable_completion_capacity = UINT32_MAX,
              .active_unspillable_transaction_final_capacity = UINT32_MAX,
              .active_unspillable_completion_domain_id = UINT16_MAX,
          };
          loom_low_schedule_target_pressure_score_candidate(&state, &pressure,
                                                            candidate, &score);
          const bool consumes_headroom =
              kind == LOOM_LOW_SCHEDULE_COMPLETION_NOMINATION_TRANSACTION &&
              candidate != 0;
          EXPECT_EQ(
              iree_any_bit_set(
                  score.flags,
                  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_CONSUMES_COMPLETION_HEADROOM),
              consumes_headroom);
          EXPECT_EQ(
              score.consumes_non_replayable_headroom,
              consumes_headroom && !live_replayable && !candidate_replayable);
        }
      }
    }
  }
}

class RegisterPackingPressureTest : public ::testing::Test {
 protected:
  loom_low_schedule_candidate_score_t Score() {
    packed_units_[0] = 0;
    for (size_t i = 0; i < IREE_ARRAYSIZE(members_); ++i) {
      packed_units_[0] += loom_low_schedule_register_packing_contribution(
          live_[i], &members_[i]);
    }
    loom_low_schedule_candidate_score_t score = {
        .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
        .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
    };
    loom_low_schedule_score_candidate_register_packing_resources(
        &state_, &pressure_, LOOM_LOW_SCHEDULE_NODE_NONE, &score);
    return score;
  }

  // A general allocation domain and a singleton view share physical storage.
  loom_low_reg_class_t classes_[2] = {
      {.alloc_unit_bits = 64, .allocatable_count = 4, .alias_set_id = 1},
      {.alloc_unit_bits = 64, .allocatable_count = 1, .alias_set_id = 1},
  };
  // Two allocation domains draw from the same four-location physical bank.
  loom_low_register_packing_resource_member_t members_[2] = {
      {.reg_class_id = 0, .register_unit_count = 1, .resource_unit_count = 1},
      {.reg_class_id = 1, .register_unit_count = 1, .resource_unit_count = 1},
  };
  // Descriptor-owned shared capacity and member slice.
  loom_low_register_packing_resource_t resource_ = {
      .capacity = 4,
      .member_count = IREE_ARRAYSIZE(members_),
  };
  // The descriptor projection consumed by native resource scoring.
  loom_low_descriptor_set_t descriptors_ = {
      .reg_classes = classes_,
      .reg_class_count = IREE_ARRAYSIZE(classes_),
      .register_packing_resources = &resource_,
      .register_packing_resource_count = 1,
      .register_packing_resource_members = members_,
      .register_packing_resource_member_count = IREE_ARRAYSIZE(members_),
  };
  // Function-local capacity after reserved locations have been accounted for.
  uint32_t capacities_[1] = {4};
  // Native resource inputs; no IR or node completion query participates.
  loom_low_schedule_build_state_t state_ = {
      .target = {.descriptor_set = &descriptors_},
      .pressure_limits = {.register_packing_resource_capacities = capacities_},
  };
  // Current register units in the general and fixed-view domains.
  uint64_t live_[2] = {3, 1};
  // Result-minus-input changes for the candidate.
  int64_t delta_[2] = {};
  // Units allocated before ordinary reads complete.
  uint64_t early_[2] = {};
  // Dying units retained through result writes.
  uint64_t late_[2] = {};
  // Both class projections are present, including zero deltas.
  uint8_t touched_[2] = {1, 1};
  // Retained rounded sum of live member contributions.
  uint64_t packed_units_[1] = {};
  // No downstream activation is needed for these immediate phase decisions.
  uint32_t activation_[1] = {};
  // Mutable pressure projection over the native arrays above.
  loom_low_schedule_pressure_state_t pressure_ = {
      .current_live_units_by_reg_class = live_,
      .candidate_delta_units_by_reg_class = delta_,
      .candidate_lifetime =
          {
              .early_added_units = early_,
              .late_released_units = late_,
          },
      .candidate_register_packing_activation_units = activation_,
      .current_register_packing_units = packed_units_,
      .candidate_delta_touched_flags = touched_,
  };
};

TEST_F(RegisterPackingPressureTest, FixedViewConsumesGeneralBankCapacity) {
  // An independent load grows the general class from three to four. It fits
  // that class alone but overlaps a live fixed register in the same bank.
  delta_[0] = 1;
  const auto load = Score();
  EXPECT_EQ(load.pressure_cliff_penalty, 1u);
  EXPECT_EQ(load.persistent_pressure_cliff_penalty, 1u);
  EXPECT_EQ(load.pressure_cliff_source_kind,
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE);
  EXPECT_EQ(load.pressure_cliff_units, 5u);
  EXPECT_TRUE(iree_any_bit_set(
      load.flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_GROWS_PACKING_RESOURCE));

  // Completing the fixed-register transaction first leaves room for the load.
  delta_[0] = 0;
  delta_[1] = -1;
  const auto completion = Score();
  EXPECT_EQ(completion.pressure_cliff_penalty, 0u);
  EXPECT_EQ(completion.persistent_pressure_cliff_penalty, 0u);
  EXPECT_EQ(completion.flags, 0u);
}

TEST_F(RegisterPackingPressureTest, ReuseDoesNotSumSeparateLifetimePhases) {
  delta_[0] = 1;
  delta_[1] = -1;
  EXPECT_EQ(Score().pressure_cliff_penalty, 0u);

  // Early writes overlap the old fixed value; late reads overlap the new one.
  early_[0] = 1;
  const auto early_write = Score();
  EXPECT_EQ(early_write.pressure_cliff_penalty, 1u);
  EXPECT_EQ(early_write.persistent_pressure_cliff_penalty, 0u);
  early_[0] = 0;
  late_[1] = 1;
  const auto late_read = Score();
  EXPECT_EQ(late_read.pressure_cliff_penalty, 1u);
  EXPECT_EQ(late_read.persistent_pressure_cliff_penalty, 0u);
}

TEST_F(RegisterPackingPressureTest, RoundedMembersReleaseOnlyCompleteGroups) {
  members_[0].register_unit_count = 2;
  resource_.flags =
      LOOM_LOW_REGISTER_PACKING_RESOURCE_FLAG_HAS_AGGREGATE_MEMBER;
  capacities_[0] = 2;
  delta_[0] = -1;
  // Three narrow units round to two groups. Consuming one releases a group;
  // consuming one of four leaves both occupied beside the other member.
  for (uint64_t units : {3u, 4u}) {
    SCOPED_TRACE(units);
    live_[0] = units;
    const auto score = Score();
    // Read-time pressure still includes the consumed unit in either case.
    EXPECT_EQ(score.pressure_cliff_penalty, 1u);
    EXPECT_EQ(score.persistent_pressure_cliff_penalty, units == 3 ? 0u : 1u);
  }
}

TEST_F(RegisterPackingPressureTest, CompletionFollowsLiveNominations) {
  loom_low_schedule_node_t nodes[7] = {};
  for (auto& node : nodes) {
    node.scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  loom_low_schedule_value_record_t values[4] = {};
  for (auto& value : values) {
    value.register_class_id = 0;
    value.unit_count = 1;
  }
  uint32_t producers[] = {0, 1, 2, 3};
  uint32_t sinks[] = {4, 5, 4, 6, UINT32_MAX, UINT32_MAX, UINT32_MAX};
  loom_value_ordinal_t ordinals[] = {0, 1, 2, 3};
  uint32_t active_sink = LOOM_LOW_SCHEDULE_NODE_NONE;
  uint32_t active_count = 0;
  state_.nodes = nodes;
  state_.values = values;
  state_.value_producer_nodes = producers;
  state_.node_register_packing.completion_sinks = sinks;
  pressure_.block_value_ordinals = ordinals;
  pressure_.block_value_count = IREE_ARRAYSIZE(ordinals);
  pressure_.active_register_packing_completion_sinks = &active_sink;
  pressure_.active_register_packing_completion_value_counts = &active_count;
  auto add = [&](uint32_t ordinal) {
    values[ordinal].flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    loom_low_schedule_target_pressure_add_packing_completion_value(
        &state_, &pressure_, ordinal);
  };
  auto remove = [&](uint32_t ordinal) {
    loom_low_schedule_target_pressure_remove_packing_completion_value(
        &state_, &pressure_, ordinal);
    values[ordinal].flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  };
  auto advance = [&](uint32_t node) {
    loom_low_schedule_target_pressure_repair_packing_completions(
        &state_, &pressure_, node);
  };

  advance(0);
  EXPECT_EQ(active_sink, LOOM_LOW_SCHEDULE_NODE_NONE);
  add(1);
  EXPECT_EQ(active_sink, 5u);
  add(0);
  add(2);
  EXPECT_EQ(active_sink, 4u);
  EXPECT_EQ(active_count, 2u);
  remove(0);
  advance(0);
  EXPECT_EQ(active_sink, 4u);
  EXPECT_EQ(active_count, 1u);

  // Scheduling a completion retires its nomination even when its value has
  // further uses. The next live nomination remains selected independently.
  nodes[4].scheduled_ordinal = 0;
  advance(4);
  EXPECT_EQ(active_sink, 5u);
  remove(1);
  advance(1);
  EXPECT_EQ(active_sink, LOOM_LOW_SCHEDULE_NODE_NONE);
  advance(2);
  EXPECT_EQ(active_sink, LOOM_LOW_SCHEDULE_NODE_NONE);

  // An empty result is invalidated by a newly live producer, and block reset
  // discards the old nominations before another block initializes its values.
  add(3);
  EXPECT_EQ(active_sink, 6u);
  EXPECT_EQ(active_count, 1u);
  loom_low_schedule_target_pressure_reset_packing_completions(&state_,
                                                              &pressure_);
  EXPECT_EQ(active_sink, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(active_count, 0u);
}

}  // namespace
}  // namespace loom
