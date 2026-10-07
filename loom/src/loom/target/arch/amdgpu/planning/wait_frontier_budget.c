// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_frontier_budget.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"

#define LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_BITS_PER_WORD UINT64_C(64)

#define LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_FLOOR_SCALED UINT64_C(491585961000)
#define LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_SLOPE_SCALED UINT64_C(26474491)
#define LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_HEAVY_PREMIUM_SCALED \
  UINT64_C(1539208063000)
#define LOOM_AMDGPU_WAIT_FRONTIER_QUERY_CALL_COST_SCALED UINT64_C(809881225)
#define LOOM_AMDGPU_WAIT_FRONTIER_QUERY_BIT_COST_SCALED UINT64_C(4831942)
#define LOOM_AMDGPU_WAIT_FRONTIER_PRODUCER_CALL_COST_SCALED UINT64_C(582256419)

static bool loom_amdgpu_wait_frontier_budget_mul3(uint64_t a, uint64_t b,
                                                  uint64_t c,
                                                  uint64_t* out_value) {
  uint64_t product = 0;
  return iree_checked_mul_u64(a, b, &product) &&
         iree_checked_mul_u64(product, c, out_value);
}

static bool loom_amdgpu_wait_frontier_budget_add_product(
    uint64_t coefficient, uint64_t value, uint64_t* inout_total) {
  uint64_t product = 0;
  return iree_checked_mul_u64(coefficient, value, &product) &&
         iree_checked_add_u64(*inout_total, product, inout_total);
}

bool loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
    uint64_t effect_use_count, uint64_t block_count, uint64_t cfg_edge_count,
    uint64_t scheduled_node_count, uint64_t node_count,
    uint64_t dependency_count, uint64_t* out_visits) {
  IREE_ASSERT_ARGUMENT(out_visits);
  *out_visits = 0;
  return loom_amdgpu_wait_frontier_budget_add_product(2, effect_use_count,
                                                      out_visits) &&
         loom_amdgpu_wait_frontier_budget_add_product(11, block_count,
                                                      out_visits) &&
         loom_amdgpu_wait_frontier_budget_add_product(7, cfg_edge_count,
                                                      out_visits) &&
         loom_amdgpu_wait_frontier_budget_add_product(3, scheduled_node_count,
                                                      out_visits) &&
         iree_checked_add_u64(*out_visits, node_count, out_visits) &&
         iree_checked_add_u64(*out_visits, dependency_count, out_visits);
}

bool loom_amdgpu_wait_frontier_precise_budget_calculate(
    const loom_amdgpu_wait_frontier_precise_budget_input_t* input,
    loom_amdgpu_wait_frontier_precise_budget_usage_t* out_usage) {
  IREE_ASSERT_ARGUMENT(input);
  IREE_ASSERT_ARGUMENT(out_usage);
  *out_usage = (loom_amdgpu_wait_frontier_precise_budget_usage_t){0};

  uint64_t classified_access_count = 0;
  uint64_t classified_producer_bit_count = 0;
  uint64_t classified_producer_completion_count = 0;
  uint64_t producer_full_path_space_count = 0;
  uint64_t maximum_producer_precise_access_visits = 0;
  uint64_t expected_producer_storage_lease_visits = 0;
  uint64_t precise_bit_capacity = 0;
  uint64_t rounded_precise_bit_capacity = 0;
  if (!iree_checked_add_u64(input->precise_read_count,
                            input->precise_write_count,
                            &classified_access_count) ||
      classified_access_count != input->precise_access_count ||
      !iree_checked_add_u64(input->read_producer_counter_bit_count,
                            input->write_producer_counter_bit_count,
                            &classified_producer_bit_count) ||
      classified_producer_bit_count != input->producer_counter_bit_count ||
      !iree_checked_mul_u64(input->precise_access_count,
                            LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT,
                            &precise_bit_capacity) ||
      input->producer_counter_bit_count > precise_bit_capacity ||
      input->producer_counter_bit_count > input->precise_access_count ||
      !iree_checked_add_u64(precise_bit_capacity,
                            LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_BITS_PER_WORD - 1,
                            &rounded_precise_bit_capacity) ||
      input->precise_word_count !=
          rounded_precise_bit_capacity /
              LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_BITS_PER_WORD ||
      input->reachable_block_count > input->block_count ||
      input->local_reset_counter_bit_count < input->local_reset_event_count ||
      input->memory_space_count !=
          LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MEMORY_SPACE_COUNT ||
      input->xcnt_group_count >
          LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_XCNT_GROUP_COUNT ||
      input->scheduled_node_count > input->node_count ||
      input->dependency_read_query_count > input->scheduled_node_count ||
      input->dependency_write_query_count > input->scheduled_node_count ||
      input->barrier_query_count > input->scheduled_node_count ||
      input->program_exit_query_count > input->scheduled_node_count ||
      !iree_checked_add_u64(input->producer_completion_guard_reject_count,
                            input->producer_completion_full_path_count,
                            &classified_producer_completion_count) ||
      classified_producer_completion_count !=
          input->producer_completion_call_count ||
      input->producer_completion_full_path_count >
          input->producer_completion_call_count ||
      input->producer_completion_full_path_read_space_count >
          input->producer_completion_full_path_count ||
      input->producer_completion_full_path_write_space_count >
          input->producer_completion_full_path_count ||
      input->direct_memory_query_count != 0 ||
      input->unclassified_memory_query_count != 0 ||
      input->direct_producer_completion_count != 0 ||
      input->unclassified_producer_completion_count != 0 ||
      !iree_checked_add_u64(
          input->producer_completion_full_path_read_space_count,
          input->producer_completion_full_path_write_space_count,
          &producer_full_path_space_count) ||
      !iree_checked_mul_u64(input->producer_completion_full_path_count,
                            input->precise_access_count,
                            &maximum_producer_precise_access_visits) ||
      input->producer_complete_precise_access_visit_count >
          maximum_producer_precise_access_visits ||
      !iree_checked_mul_u64(input->producer_completion_full_path_count,
                            input->storage_lease_count,
                            &expected_producer_storage_lease_visits) ||
      input->producer_complete_storage_lease_visit_count !=
          expected_producer_storage_lease_visits) {
    return false;
  }

  uint64_t coarse_memory_bit_count = 0;
  uint64_t lattice_dimension_count = 0;
  uint64_t lattice_height = 0;
  uint64_t unfiltered_edge_evaluations = 0;
  uint64_t filtered_edge_evaluations = 0;
  uint64_t backedge_evaluations = 0;
  uint64_t term_merge_left = 0;
  uint64_t term_merge_right = 0;
  uint64_t term_merge_sum = 0;
  uint64_t forward_edge_count = 0;
  uint64_t query_base_calls = 0;
  uint64_t twice_barrier_query_count = 0;
  uint64_t twice_producer_completion_full_path_count = 0;
  uint64_t query_bit_visits = 0;
  uint64_t query_bit_component = 0;
  uint64_t collector_score = 0;
  uint64_t query_score = 0;
  uint64_t producer_score = 0;
  bool valid =
      iree_checked_mul_u64(input->memory_space_count, 16,
                           &coarse_memory_bit_count) &&
      iree_checked_add_u64(input->producer_counter_bit_count,
                           input->storage_lease_count,
                           &lattice_dimension_count) &&
      iree_checked_add_u64(lattice_dimension_count, coarse_memory_bit_count,
                           &lattice_dimension_count) &&
      iree_checked_add_u64(lattice_dimension_count, input->xcnt_group_count,
                           &lattice_dimension_count) &&
      iree_checked_add_u64(lattice_dimension_count, 1, &lattice_height) &&
      iree_checked_mul_u64(input->precise_access_count,
                           input->precise_access_count,
                           &out_usage->producer_scan_visits) &&
      loom_amdgpu_wait_frontier_budget_mul3(2, input->precise_read_count,
                                            input->precise_write_count,
                                            &out_usage->eligible_alias_calls) &&
      iree_checked_mul_u64(input->precise_read_count,
                           input->write_producer_counter_bit_count,
                           &term_merge_left) &&
      iree_checked_mul_u64(input->precise_write_count,
                           input->read_producer_counter_bit_count,
                           &term_merge_right) &&
      iree_checked_add_u64(term_merge_left, term_merge_right,
                           &out_usage->eligible_alias_bit_visits) &&
      iree_checked_mul_u64(input->precise_write_count,
                           input->read_effective_term_sum, &term_merge_left) &&
      iree_checked_mul_u64(input->precise_read_count,
                           input->write_effective_term_sum,
                           &term_merge_right) &&
      iree_checked_add_u64(term_merge_left, term_merge_right,
                           &term_merge_sum) &&
      iree_checked_mul_u64(2, term_merge_sum, &out_usage->term_merge_visits) &&
      iree_checked_mul_u64(input->reachable_block_count, lattice_height,
                           &out_usage->worklist_pops) &&
      iree_checked_mul_u64(input->unfiltered_forward_edge_count, lattice_height,
                           &unfiltered_edge_evaluations) &&
      iree_checked_mul_u64(input->filtered_forward_edge_count, lattice_height,
                           &filtered_edge_evaluations) &&
      iree_checked_mul_u64(input->backedge_count, lattice_height,
                           &backedge_evaluations) &&
      iree_checked_add_u64(unfiltered_edge_evaluations,
                           filtered_edge_evaluations,
                           &out_usage->edge_evaluations) &&
      iree_checked_add_u64(out_usage->edge_evaluations, backedge_evaluations,
                           &out_usage->edge_evaluations) &&
      loom_amdgpu_wait_frontier_budget_mul3(
          input->precise_word_count, input->unfiltered_forward_edge_count,
          lattice_height, &out_usage->forward_word_visits) &&
      loom_amdgpu_wait_frontier_budget_mul3(
          input->precise_access_count, input->filtered_forward_edge_count,
          lattice_height, &out_usage->filtered_access_visits) &&
      iree_checked_mul_u64(filtered_edge_evaluations,
                           input->producer_counter_bit_count,
                           &out_usage->filtered_producer_bit_visits) &&
      loom_amdgpu_wait_frontier_budget_mul3(
          input->precise_access_count, input->backedge_count, lattice_height,
          &out_usage->collapse_access_visits) &&
      iree_checked_mul_u64(backedge_evaluations,
                           input->producer_counter_bit_count,
                           &out_usage->collapse_producer_bit_visits) &&
      iree_checked_mul_u64(input->precise_access_count,
                           input->local_reset_event_count,
                           &out_usage->local_drain_access_visits) &&
      iree_checked_mul_u64(input->precise_access_count,
                           input->local_reset_counter_bit_count,
                           &out_usage->local_drain_bit_visits) &&
      iree_checked_add_u64(input->unfiltered_forward_edge_count,
                           input->filtered_forward_edge_count,
                           &forward_edge_count) &&
      iree_checked_mul_u64(forward_edge_count, input->precise_word_count,
                           &out_usage->begin_block_word_visits) &&
      iree_checked_mul_u64(input->backedge_count, input->precise_access_count,
                           &out_usage->begin_block_collapse_access_visits) &&
      iree_checked_mul_u64(
          input->backedge_count, input->producer_counter_bit_count,
          &out_usage->begin_block_collapse_producer_bit_visits) &&
      loom_amdgpu_wait_frontier_budget_mul3(
          input->precise_access_count, LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT,
          input->block_count, &out_usage->dynamic_incoming_drain_bit_visits) &&
      loom_amdgpu_wait_frontier_precise_budget_pre_admission_visits(
          input->effect_use_count, input->block_count, input->cfg_edge_count,
          input->scheduled_node_count, input->node_count,
          input->dependency_count, &out_usage->pre_admission_visits) &&
      iree_checked_mul_u64(2, input->barrier_query_count,
                           &twice_barrier_query_count) &&
      iree_checked_add_u64(input->block_count,
                           input->dependency_read_query_count,
                           &query_base_calls) &&
      iree_checked_add_u64(query_base_calls,
                           input->dependency_write_query_count,
                           &query_base_calls) &&
      iree_checked_add_u64(query_base_calls, twice_barrier_query_count,
                           &query_base_calls) &&
      iree_checked_add_u64(query_base_calls, input->program_exit_query_count,
                           &query_base_calls) &&
      iree_checked_mul_u64(2, input->producer_completion_full_path_count,
                           &twice_producer_completion_full_path_count) &&
      iree_checked_add_u64(query_base_calls,
                           twice_producer_completion_full_path_count,
                           &out_usage->memory_query_calls) &&
      iree_checked_mul_u64(input->precise_access_count, query_base_calls,
                           &out_usage->memory_query_precise_access_visits) &&
      iree_checked_mul_u64(input->precise_access_count,
                           producer_full_path_space_count, &term_merge_left) &&
      iree_checked_add_u64(out_usage->memory_query_precise_access_visits,
                           term_merge_left,
                           &out_usage->memory_query_precise_access_visits) &&
      iree_checked_mul_u64(input->producer_counter_bit_count,
                           input->block_count, &query_bit_visits) &&
      iree_checked_mul_u64(input->write_producer_counter_bit_count,
                           input->dependency_read_query_count,
                           &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &query_bit_visits) &&
      iree_checked_mul_u64(input->read_producer_counter_bit_count,
                           input->dependency_write_query_count,
                           &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &query_bit_visits) &&
      iree_checked_mul_u64(input->producer_counter_bit_count,
                           twice_barrier_query_count, &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &query_bit_visits) &&
      iree_checked_mul_u64(input->write_producer_counter_bit_count,
                           input->program_exit_query_count,
                           &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &query_bit_visits) &&
      iree_checked_mul_u64(
          input->read_producer_counter_bit_count,
          input->producer_completion_full_path_read_space_count,
          &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &query_bit_visits) &&
      iree_checked_mul_u64(
          input->write_producer_counter_bit_count,
          input->producer_completion_full_path_write_space_count,
          &query_bit_component) &&
      iree_checked_add_u64(query_bit_visits, query_bit_component,
                           &out_usage->memory_query_precise_bit_visits) &&
      iree_checked_mul_u64(LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_SLOPE_SCALED,
                           out_usage->pre_admission_visits, &collector_score);
  if (!valid) {
    *out_usage = (loom_amdgpu_wait_frontier_precise_budget_usage_t){0};
    return false;
  }

  out_usage->precise_access_count = input->precise_access_count;
  out_usage->producer_counter_bit_count = input->producer_counter_bit_count;
  out_usage->maximum_effective_term_count = input->maximum_effective_term_count;
  out_usage->allocation_byte_count = input->allocation_byte_count;
  out_usage->producer_complete_precise_access_visits =
      input->producer_complete_precise_access_visit_count;
  out_usage->producer_complete_storage_lease_visits =
      input->producer_complete_storage_lease_visit_count;
  collector_score = iree_max(collector_score,
                             LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_FLOOR_SCALED);
  if (input->collector_heavy_path_possible &&
      !iree_checked_add_u64(
          collector_score,
          LOOM_AMDGPU_WAIT_FRONTIER_COLLECTOR_HEAVY_PREMIUM_SCALED,
          &collector_score)) {
    *out_usage = (loom_amdgpu_wait_frontier_precise_budget_usage_t){0};
    return false;
  }
  if (!loom_amdgpu_wait_frontier_budget_add_product(
          LOOM_AMDGPU_WAIT_FRONTIER_QUERY_CALL_COST_SCALED,
          out_usage->memory_query_calls, &query_score) ||
      !loom_amdgpu_wait_frontier_budget_add_product(
          LOOM_AMDGPU_WAIT_FRONTIER_QUERY_BIT_COST_SCALED,
          out_usage->memory_query_precise_bit_visits, &query_score) ||
      !loom_amdgpu_wait_frontier_budget_add_product(
          LOOM_AMDGPU_WAIT_FRONTIER_PRODUCER_CALL_COST_SCALED,
          input->producer_completion_full_path_count, &producer_score) ||
      !iree_checked_add_u64(collector_score, query_score,
                            &out_usage->structural_score_scaled) ||
      !iree_checked_add_u64(out_usage->structural_score_scaled, producer_score,
                            &out_usage->structural_score_scaled)) {
    *out_usage = (loom_amdgpu_wait_frontier_precise_budget_usage_t){0};
    return false;
  }
  return true;
}

bool loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(
    const loom_amdgpu_wait_frontier_precise_budget_usage_t* usage) {
  IREE_ASSERT_ARGUMENT(usage);
  return usage->precise_access_count <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ACCESS_COUNT &&
         usage->pre_admission_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRE_ADMISSION_VISITS &&
         usage->producer_counter_bit_count <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COUNTER_BITS &&
         usage->producer_scan_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_PRODUCER_SCANS &&
         usage->eligible_alias_calls <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_CALLS &&
         usage->eligible_alias_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALIAS_BIT_VISITS &&
         usage->term_merge_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_TERM_MERGES &&
         usage->worklist_pops <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_WORKLIST_POPS &&
         usage->edge_evaluations <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EDGE_EVALUATIONS &&
         usage->forward_word_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FORWARD_WORD_VISITS &&
         usage->filtered_access_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_ACCESS_VISITS &&
         usage->filtered_producer_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_FILTERED_BIT_VISITS &&
         usage->collapse_access_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_ACCESS_VISITS &&
         usage->collapse_producer_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_COLLAPSE_BIT_VISITS &&
         usage->local_drain_access_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_ACCESS_VISITS &&
         usage->local_drain_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_LOCAL_DRAIN_BIT_VISITS &&
         usage->begin_block_word_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_WORD_VISITS &&
         usage->begin_block_collapse_access_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_VISITS &&
         usage->begin_block_collapse_producer_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_BEGIN_BLOCK_COLLAPSE_BIT_VISITS &&
         usage->dynamic_incoming_drain_bit_visits <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_DYNAMIC_DRAIN_BIT_VISITS &&
         usage->maximum_effective_term_count <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_EFFECTIVE_TERMS &&
         usage->allocation_byte_count <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_ALLOCATION_BYTES &&
         usage->structural_score_scaled <=
             LOOM_AMDGPU_WAIT_FRONTIER_PRECISE_MAX_SCORE_SCALED;
}

bool loom_amdgpu_wait_frontier_precise_budget_is_admitted(
    const loom_amdgpu_wait_frontier_precise_budget_input_t* input) {
  loom_amdgpu_wait_frontier_precise_budget_usage_t usage;
  return loom_amdgpu_wait_frontier_precise_budget_calculate(input, &usage) &&
         loom_amdgpu_wait_frontier_precise_budget_usage_is_admitted(&usage);
}
