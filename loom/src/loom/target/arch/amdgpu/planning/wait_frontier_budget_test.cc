// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_frontier_budget.h"

#include <array>
#include <cstdint>
#include <limits>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static_assert(LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_SCORE_SCALED ==
                  UINT64_C(25000000000000) * 4 / 5 - UINT64_C(17313009454246),
              "raw score cap must apply the 1.25 safety factor exactly once");

loom_amdgpu_wait_frontier_precise_budget_input_t MakeBudgetInput(
    uint64_t precise_access_count) {
  loom_amdgpu_wait_frontier_precise_budget_input_t input = {};
  input.precise_access_count = precise_access_count;
  input.producer_counter_bit_count = precise_access_count;
  input.precise_read_count = precise_access_count / 2;
  input.precise_write_count = precise_access_count - input.precise_read_count;
  input.read_producer_counter_bit_count = input.precise_read_count;
  input.write_producer_counter_bit_count = input.precise_write_count;
  input.read_effective_term_sum = input.precise_read_count;
  input.write_effective_term_sum = input.precise_write_count;
  input.precise_word_count = (precise_access_count * 8 + 63) / 64;
  input.block_count = 2;
  input.reachable_block_count = 2;
  input.unfiltered_forward_edge_count = 1;
  input.memory_space_count =
      LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MEMORY_SPACE_COUNT;
  input.maximum_effective_term_count = precise_access_count == 0 ? 0 : 1;
  input.allocation_byte_count = 1;
  return input;
}

loom_amdgpu_wait_frontier_precise_budget_input_t MakeProductionBudgetInput() {
  auto input = MakeBudgetInput(128);
  input.producer_counter_bit_count = 128;
  input.read_producer_counter_bit_count = 112;
  input.write_producer_counter_bit_count = 16;
  input.storage_lease_count = 126;
  input.precise_read_count = 112;
  input.precise_write_count = 16;
  input.read_effective_term_sum = 96;
  input.write_effective_term_sum = 8;
  input.precise_word_count = 16;
  input.block_count = 76;
  input.reachable_block_count = 76;
  input.unfiltered_forward_edge_count = 80;
  input.filtered_forward_edge_count = 19;
  input.backedge_count = 2;
  input.local_reset_event_count = 24;
  input.local_reset_counter_bit_count = 41;
  input.maximum_effective_term_count = 1;
  input.effect_use_count = 543;
  input.cfg_edge_count = 101;
  input.scheduled_node_count = 2462;
  input.node_count = 2462;
  input.dependency_count = 488;
  input.dependency_read_query_count = 25;
  input.dependency_write_query_count = 51;
  input.barrier_query_count = 6;
  input.program_exit_query_count = 1;
  input.producer_completion_call_count = 101;
  input.producer_completion_guard_reject_count = 0;
  input.producer_completion_full_path_count = 101;
  input.producer_completion_full_path_read_space_count = 101;
  input.producer_completion_full_path_write_space_count = 0;
  // The fitted PF coefficient is exactly zero. Production conservatively
  // carries the checked F*P upper bound instead of allocating per-node scratch.
  input.producer_complete_precise_access_visit_count = 101 * 128;
  input.producer_complete_storage_lease_visit_count = 101 * 126;
  input.collector_heavy_path_possible = true;
  return input;
}

TEST(AmdgpuWaitFrontierBudgetTest, EveryCapIsInclusiveAndConjunctive) {
  using Usage = loom_amdgpu_wait_frontier_precise_budget_usage_t;
  struct CapCase {
    const char* name;
    uint64_t Usage::* field;
    uint64_t limit;
  };
  constexpr std::array<CapCase, 23> kCases = {{
      {"precise accesses", &Usage::precise_access_count,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ACCESS_COUNT},
      {"counter bits", &Usage::producer_counter_bit_count,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COUNTER_BITS},
      {"producer scans", &Usage::producer_scan_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRODUCER_SCANS},
      {"alias calls", &Usage::eligible_alias_calls,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_CALLS},
      {"alias bits", &Usage::eligible_alias_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_BIT_VISITS},
      {"term merges", &Usage::term_merge_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_TERM_MERGES},
      {"worklist pops", &Usage::worklist_pops,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_WORKLIST_POPS},
      {"edge evaluations", &Usage::edge_evaluations,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EDGE_EVALUATIONS},
      {"forward words", &Usage::forward_word_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FORWARD_WORD_VISITS},
      {"filtered accesses", &Usage::filtered_access_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_ACCESS_VISITS},
      {"filtered bits", &Usage::filtered_producer_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_BIT_VISITS},
      {"collapse accesses", &Usage::collapse_access_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_ACCESS_VISITS},
      {"collapse bits", &Usage::collapse_producer_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_BIT_VISITS},
      {"local drain accesses", &Usage::local_drain_access_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_ACCESS_VISITS},
      {"local drain bits", &Usage::local_drain_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_BIT_VISITS},
      {"begin block words", &Usage::begin_block_word_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_WORD_VISITS},
      {"begin block collapses", &Usage::begin_block_collapse_access_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_VISITS},
      {"begin block collapse bits",
       &Usage::begin_block_collapse_producer_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_BIT_VISITS},
      {"dynamic drain bits", &Usage::dynamic_incoming_drain_bit_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_DYNAMIC_DRAIN_BIT_VISITS},
      {"effective terms", &Usage::maximum_effective_term_count,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EFFECTIVE_TERMS},
      {"allocation bytes", &Usage::allocation_byte_count,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALLOCATION_BYTES},
      {"pre-admission visits", &Usage::pre_admission_visits,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRE_ADMISSION_VISITS},
      {"additive structural score", &Usage::structural_score_scaled,
       LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_SCORE_SCALED},
  }};

  for (const CapCase& cap : kCases) {
    SCOPED_TRACE(cap.name);
    Usage usage = {};
    usage.*(cap.field) = cap.limit;
    EXPECT_TRUE(
        loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
    usage.*(cap.field) = cap.limit + 1;
    EXPECT_FALSE(
        loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
  }
}

TEST(AmdgpuWaitFrontierBudgetTest, CheckedArithmeticRejectsOverflow) {
  auto input = MakeBudgetInput(2);
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;

  input.precise_access_count = std::numeric_limits<uint64_t>::max();
  input.precise_read_count = input.precise_access_count;
  input.precise_write_count = 0;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.storage_lease_count = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.read_effective_term_sum = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.unfiltered_forward_edge_count = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.effect_use_count = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.producer_completion_call_count = std::numeric_limits<uint64_t>::max();
  input.producer_completion_full_path_count =
      std::numeric_limits<uint64_t>::max();
  input.producer_complete_precise_access_visit_count =
      std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, RejectsInconsistentTupleDimensions) {
  auto input = MakeBudgetInput(2);
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;

  ++input.precise_read_count;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  ++input.precise_word_count;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.producer_counter_bit_count = 17;
  input.read_producer_counter_bit_count = 17;
  input.write_producer_counter_bit_count = 0;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  ++input.read_producer_counter_bit_count;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.reachable_block_count = 3;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.local_reset_event_count = 2;
  input.local_reset_counter_bit_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  --input.memory_space_count;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.xcnt_group_count =
      LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_XCNT_GROUP_COUNT + 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.producer_completion_call_count = 2;
  input.producer_completion_guard_reject_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input.producer_completion_full_path_count = 1;
  input.producer_complete_precise_access_visit_count = 2;
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input.producer_completion_full_path_read_space_count = 2;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input.producer_completion_full_path_read_space_count = 0;
  input.producer_completion_full_path_write_space_count = 2;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));

  input = MakeBudgetInput(2);
  input.direct_memory_query_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.unclassified_memory_query_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.direct_producer_completion_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  input = MakeBudgetInput(2);
  input.unclassified_producer_completion_count = 1;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, PreAdmissionVisitCapIsInclusive) {
  uint64_t visits = 0;
  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
      /*effect_use_count=*/0, /*block_count=*/0, /*cfg_edge_count=*/0,
      /*scheduled_node_count=*/0, /*node_count=*/131072,
      /*dependency_count=*/0, &visits));
  EXPECT_EQ(visits, 131072u);
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage = {};
  usage.pre_admission_visits = visits;
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));

  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
      /*effect_use_count=*/0, /*block_count=*/0, /*cfg_edge_count=*/0,
      /*scheduled_node_count=*/0, /*node_count=*/131073,
      /*dependency_count=*/0, &visits));
  EXPECT_EQ(visits, 131073u);
  usage.pre_admission_visits = visits;
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, PreAdmissionVisitArithmeticIsChecked) {
  uint64_t visits = 7;
  EXPECT_FALSE(loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
      std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0, 0, &visits));
}

TEST(AmdgpuWaitFrontierBudgetTest, EffectiveTermBoundaryIs64) {
  auto input = MakeBudgetInput(2);
  input.maximum_effective_term_count = 64;
  EXPECT_TRUE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));
  input.maximum_effective_term_count = 65;
  EXPECT_FALSE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));
}

TEST(AmdgpuWaitFrontierBudgetTest, ProducerBitDensityBoundaryIsOnePerAccess) {
  auto input = MakeBudgetInput(2);
  EXPECT_TRUE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));

  input.producer_counter_bit_count = 3;
  input.write_producer_counter_bit_count = 2;
  EXPECT_FALSE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));
}

TEST(AmdgpuWaitFrontierBudgetTest, PreciseAccessCapAndProducerDensityAreExact) {
  auto input = MakeBudgetInput(344);
  EXPECT_TRUE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));

  input = MakeBudgetInput(345);
  EXPECT_FALSE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));

  input = MakeBudgetInput(344);
  input.producer_counter_bit_count = 344;
  input.read_producer_counter_bit_count = input.precise_read_count;
  input.write_producer_counter_bit_count = input.precise_write_count;
  EXPECT_TRUE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));
  ++input.producer_counter_bit_count;
  ++input.write_producer_counter_bit_count;
  EXPECT_FALSE(loom_amdgpu_wait_frontier_precise_budget_is_admitted(&input));
}

TEST(AmdgpuWaitFrontierBudgetTest, DirectionalProducerBitsBoundAliasWork) {
  auto input = MakeBudgetInput(5);
  input.precise_read_count = 3;
  input.precise_write_count = 2;
  input.producer_counter_bit_count = 5;
  input.read_producer_counter_bit_count = 1;
  input.write_producer_counter_bit_count = 4;
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.eligible_alias_calls, 12u);
  EXPECT_EQ(usage.eligible_alias_bit_visits, 14u);
}

TEST(AmdgpuWaitFrontierBudgetTest, AdmitsP256WithHeadroom) {
  const auto input = MakeBudgetInput(256);
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.producer_scan_visits, 65536u);
  EXPECT_LT(usage.forward_word_visits,
            LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FORWARD_WORD_VISITS);
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, P344IsAdmittedAndP512ExceedsScanBudget) {
  auto input = MakeBudgetInput(344);
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.producer_scan_visits, 118336u);
  EXPECT_LE(usage.producer_scan_visits,
            LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRODUCER_SCANS);
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));

  input = MakeBudgetInput(512);
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.producer_scan_visits, 262144u);
  EXPECT_GT(usage.producer_scan_visits,
            LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRODUCER_SCANS);
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
}

TEST(AmdgpuWaitFrontierBudgetTest,
     DerivedUsageKeepsParallelFilteredAndBackedgeCostsDistinct) {
  auto input = MakeBudgetInput(2);
  input.unfiltered_forward_edge_count = 2;
  input.filtered_forward_edge_count = 3;
  input.backedge_count = 4;
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  // H = Q + L + 16*M + X + 1 = 2 + 0 + 96 + 0 + 1 = 99.
  EXPECT_EQ(usage.worklist_pops, 198u);
  EXPECT_EQ(usage.edge_evaluations, 9u * 99u);
  EXPECT_EQ(usage.forward_word_visits, 2u * 99u);
  EXPECT_EQ(usage.filtered_access_visits, 2u * 3u * 99u);
  EXPECT_EQ(usage.collapse_access_visits, 2u * 4u * 99u);
  EXPECT_EQ(usage.begin_block_word_visits, 5u);
  EXPECT_EQ(usage.begin_block_collapse_access_visits, 8u);
}

TEST(AmdgpuWaitFrontierBudgetTest, ProductionTupleIsAdmitted) {
  auto input = MakeProductionBudgetInput();
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.producer_scan_visits, 16384u);
  EXPECT_EQ(usage.eligible_alias_calls, 3584u);
  EXPECT_EQ(usage.eligible_alias_bit_visits, 3584u);
  EXPECT_EQ(usage.term_merge_visits, 4864u);
  EXPECT_EQ(usage.worklist_pops, 26676u);
  EXPECT_EQ(usage.edge_evaluations, 35451u);
  EXPECT_EQ(usage.forward_word_visits, 449280u);
  EXPECT_EQ(usage.filtered_access_visits, 853632u);
  EXPECT_EQ(usage.filtered_producer_bit_visits, 853632u);
  EXPECT_EQ(usage.collapse_access_visits, 89856u);
  EXPECT_EQ(usage.collapse_producer_bit_visits, 89856u);
  EXPECT_EQ(usage.local_drain_access_visits, 3072u);
  EXPECT_EQ(usage.local_drain_bit_visits, 5248u);
  EXPECT_EQ(usage.begin_block_word_visits, 1584u);
  EXPECT_EQ(usage.begin_block_collapse_access_visits, 256u);
  EXPECT_EQ(usage.begin_block_collapse_producer_bit_visits, 256u);
  EXPECT_EQ(usage.dynamic_incoming_drain_bit_visits, 77824u);
  EXPECT_EQ(usage.pre_admission_visits, 12965u);
  EXPECT_EQ(usage.memory_query_calls, 367u);
  EXPECT_EQ(usage.memory_query_precise_access_visits, 34048u);
  EXPECT_EQ(usage.memory_query_precise_bit_visits, 28704u);
  EXPECT_EQ(usage.producer_complete_precise_access_visits, 12928u);
  EXPECT_EQ(usage.producer_complete_storage_lease_visits, 12726u);
  EXPECT_EQ(usage.structural_score_scaled, UINT64_C(2525524395062));
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, FirstProductionShapedScoreAboveBudget) {
  auto input = MakeProductionBudgetInput();
  input.program_exit_query_count = 182;
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.structural_score_scaled, UINT64_C(2686106200819));
  EXPECT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));

  ++input.program_exit_query_count;
  ASSERT_TRUE(
      loom_amdgpu_wait_frontier_precise_budget_calculate(&input, &usage));
  EXPECT_EQ(usage.structural_score_scaled, UINT64_C(2686993393116));
  EXPECT_FALSE(
      loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage));
}

TEST(AmdgpuWaitFrontierBudgetTest, StorageLeaseDimensionRaisesFixedPointWork) {
  auto without_lease = MakeBudgetInput(1);
  loom_amdgpu_wait_frontier_precise_budget_usage_t baseline;
  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_calculate(&without_lease,
                                                                 &baseline));

  auto with_lease = without_lease;
  with_lease.storage_lease_count = 1;
  loom_amdgpu_wait_frontier_precise_budget_usage_t lease_usage;
  ASSERT_TRUE(loom_amdgpu_wait_frontier_precise_budget_calculate(&with_lease,
                                                                 &lease_usage));
  EXPECT_EQ(lease_usage.worklist_pops, baseline.worklist_pops + 2);
  EXPECT_EQ(lease_usage.edge_evaluations, baseline.edge_evaluations + 1);
  EXPECT_EQ(lease_usage.forward_word_visits, baseline.forward_word_visits + 1);
}

}  // namespace
}  // namespace loom
