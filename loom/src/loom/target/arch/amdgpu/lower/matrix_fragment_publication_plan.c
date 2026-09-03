// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_publication_plan.h"

#include <stdint.h>

#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/matrix.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/target/arch/amdgpu/matrix/types.h"
#include "loom/util/fact_table.h"

typedef bool (*loom_amdgpu_fragment_publication_visit_fn_t)(const loom_op_t* op,
                                                            void* user_data);

static bool loom_amdgpu_fragment_publication_visit_op(
    const loom_op_t* op, loom_amdgpu_fragment_publication_visit_fn_t visit,
    void* user_data) {
  if (!visit(op, user_data)) {
    return false;
  }
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = loom_op_regions(op)[region_index];
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      for (const loom_op_t* child_op = block->first_op; child_op != NULL;
           child_op = child_op->next_op) {
        if (!loom_amdgpu_fragment_publication_visit_op(child_op, visit,
                                                       user_data)) {
          return false;
        }
      }
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_publication_visit_function(
    loom_func_like_t source_function,
    loom_amdgpu_fragment_publication_visit_fn_t visit, void* user_data) {
  const loom_region_t* body = loom_func_like_body(source_function);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
      if (!loom_amdgpu_fragment_publication_visit_op(op, visit, user_data)) {
        return false;
      }
    }
  }
  return true;
}

typedef struct loom_amdgpu_fragment_publication_cost_t {
  // Relative wavefront memory-request cost after lane coalescing.
  uint64_t memory_request_cost;
  // Relative target instruction issue cost for publication preparation.
  uint64_t instruction_issue_cost;
} loom_amdgpu_fragment_publication_cost_t;

static bool loom_amdgpu_fragment_publication_accumulate_cost(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_publication_cost_t* cost) {
  // Source-origin terms are common to every exact layout candidate. Compare
  // the candidate-owned lane/register mapping; runtime view strides still
  // need to be subgroup-uniform so those relative intervals are static.
  const bool uses_crosslane =
      loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
          plan->epilogue_strategy);
  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
    if (!loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
            plan, packet->register_index, /*element_index=*/0)) {
      return false;
    }
    const uint32_t per_lane_packet_byte_count =
        uses_crosslane ? LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                             plan->element_byte_count
                       : (uint32_t)packet->result_register_count *
                             plan->element_byte_count;
    loom_low_lower_memory_subgroup_access_report_t geometry = {0};
    loom_amdgpu_memory_calculate_subgroup_geometry(
        &plan->address_layout, (uint8_t)layout->wave_size,
        per_lane_packet_byte_count, &geometry);
    cost->memory_request_cost += geometry.contiguous_region_count;
  }
  cost->instruction_issue_cost +=
      loom_amdgpu_fragment_memory_publication_issue_count(descriptor_set, plan);
  return true;
}

static bool loom_amdgpu_fragment_publication_payload_has_native_storage(
    const loom_value_fact_table_t* fact_table, loom_value_id_t payload) {
  loom_vector_fragment_fact_t fragment;
  return loom_vector_fragment_fact_query_value_facts(
             &fact_table->context,
             loom_value_fact_table_lookup(fact_table, payload), &fragment) &&
         iree_all_bits_set(fragment.flags,
                           LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE);
}

typedef struct loom_amdgpu_fragment_publication_evaluation_t {
  const loom_module_t* module;
  const loom_value_fact_table_t* fact_table;
  const loom_view_region_table_t* view_regions;
  const loom_target_bundle_t* bundle;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_amdgpu_target_facts_t* target_facts;
  loom_func_like_t source_function;
  const loom_amdgpu_matrix_fragment_layout_t* canonical_layout;
  const loom_amdgpu_matrix_fragment_layout_t* candidate_layout;
  loom_amdgpu_fragment_publication_cost_t cost;
  uint32_t relevant_store_count;
  bool valid;
} loom_amdgpu_fragment_publication_evaluation_t;

static bool loom_amdgpu_fragment_publication_evaluate_op(const loom_op_t* op,
                                                         void* user_data) {
  loom_amdgpu_fragment_publication_evaluation_t* evaluation = user_data;
  if (!loom_vector_fragment_store_isa(op)) {
    return true;
  }
  loom_amdgpu_fragment_memory_plan_t canonical_plan = {0};
  if (!loom_amdgpu_analyze_vector_fragment_memory_plan_for_layout(
          evaluation->module, evaluation->fact_table, evaluation->view_regions,
          evaluation->bundle, evaluation->descriptor_set,
          evaluation->target_facts, evaluation->source_function, op,
          evaluation->canonical_layout, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          &canonical_plan)) {
    return true;
  }

  ++evaluation->relevant_store_count;
  if (evaluation->candidate_layout != evaluation->canonical_layout &&
      !loom_amdgpu_fragment_publication_payload_has_native_storage(
          evaluation->fact_table, loom_vector_fragment_store_value(op))) {
    evaluation->valid = false;
    return false;
  }
  loom_amdgpu_fragment_memory_plan_t candidate_plan = {0};
  if (!loom_amdgpu_analyze_vector_fragment_memory_plan_for_layout(
          evaluation->module, evaluation->fact_table, evaluation->view_regions,
          evaluation->bundle, evaluation->descriptor_set,
          evaluation->target_facts, evaluation->source_function, op,
          evaluation->candidate_layout, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          &candidate_plan) ||
      !loom_amdgpu_fragment_publication_accumulate_cost(
          evaluation->descriptor_set, evaluation->candidate_layout,
          &candidate_plan, &evaluation->cost)) {
    evaluation->valid = false;
    return false;
  }
  return true;
}

static loom_amdgpu_fragment_publication_evaluation_t
loom_amdgpu_fragment_publication_evaluate_layout(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function,
    const loom_amdgpu_matrix_fragment_layout_t* canonical_layout,
    const loom_amdgpu_matrix_fragment_layout_t* candidate_layout) {
  loom_amdgpu_fragment_publication_evaluation_t evaluation = {
      .module = module,
      .fact_table = fact_table,
      .view_regions = view_regions,
      .bundle = bundle,
      .descriptor_set = descriptor_set,
      .target_facts = target_facts,
      .source_function = source_function,
      .canonical_layout = canonical_layout,
      .candidate_layout = candidate_layout,
      .valid = true,
  };
  loom_amdgpu_fragment_publication_visit_function(
      source_function, loom_amdgpu_fragment_publication_evaluate_op,
      &evaluation);
  return evaluation;
}

typedef struct loom_amdgpu_fragment_publication_overlap_t {
  const loom_module_t* module;
  const loom_value_fact_table_t* fact_table;
  const loom_view_region_table_t* view_regions;
  const loom_target_bundle_t* bundle;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_amdgpu_target_facts_t* target_facts;
  loom_func_like_t source_function;
  const loom_amdgpu_matrix_fragment_layout_t* lhs_layout;
  const loom_amdgpu_matrix_fragment_layout_t* rhs_layout;
  bool found;
} loom_amdgpu_fragment_publication_overlap_t;

static bool loom_amdgpu_fragment_publication_find_overlap(const loom_op_t* op,
                                                          void* user_data) {
  loom_amdgpu_fragment_publication_overlap_t* overlap = user_data;
  if (!loom_vector_fragment_store_isa(op)) {
    return true;
  }
  loom_amdgpu_fragment_memory_plan_t plan = {0};
  if (!loom_amdgpu_analyze_vector_fragment_memory_plan_for_layout(
          overlap->module, overlap->fact_table, overlap->view_regions,
          overlap->bundle, overlap->descriptor_set, overlap->target_facts,
          overlap->source_function, op, overlap->lhs_layout,
          LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, &plan) ||
      !loom_amdgpu_analyze_vector_fragment_memory_plan_for_layout(
          overlap->module, overlap->fact_table, overlap->view_regions,
          overlap->bundle, overlap->descriptor_set, overlap->target_facts,
          overlap->source_function, op, overlap->rhs_layout,
          LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, &plan)) {
    return true;
  }
  overlap->found = true;
  return false;
}

static bool loom_amdgpu_fragment_publication_layouts_overlap(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function,
    const loom_amdgpu_matrix_fragment_layout_t* lhs_layout,
    const loom_amdgpu_matrix_fragment_layout_t* rhs_layout) {
  loom_amdgpu_fragment_publication_overlap_t overlap = {
      .module = module,
      .fact_table = fact_table,
      .view_regions = view_regions,
      .bundle = bundle,
      .descriptor_set = descriptor_set,
      .target_facts = target_facts,
      .source_function = source_function,
      .lhs_layout = lhs_layout,
      .rhs_layout = rhs_layout,
  };
  loom_amdgpu_fragment_publication_visit_function(
      source_function, loom_amdgpu_fragment_publication_find_overlap, &overlap);
  return overlap.found;
}

typedef struct loom_amdgpu_fragment_source_layouts_t {
  const loom_module_t* module;
  const loom_value_fact_table_t* fact_table;
  const loom_target_bundle_t* bundle;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_amdgpu_target_facts_t* target_facts;
  bool kinds[LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT];
} loom_amdgpu_fragment_source_layouts_t;

static bool loom_amdgpu_fragment_publication_collect_source_layout(
    const loom_op_t* op, void* user_data) {
  loom_amdgpu_fragment_source_layouts_t* layouts = user_data;
  if (!loom_vector_mma_isa(op)) {
    return true;
  }
  const loom_matrix_fragment_layout_t* selected_layout =
      loom_amdgpu_matrix_select_source_fragment_layout(
          layouts->module, layouts->fact_table, layouts->bundle,
          layouts->descriptor_set, layouts->target_facts, op);
  if (selected_layout != NULL && selected_layout->canonical_kind <
                                     LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT) {
    layouts->kinds[selected_layout->canonical_kind] = true;
  }
  return true;
}

static const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_fragment_publication_source_layout(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function,
    const loom_amdgpu_matrix_fragment_layout_t* requested_layout,
    bool* out_has_ambiguous_source_layout) {
  *out_has_ambiguous_source_layout = false;
  loom_amdgpu_fragment_source_layouts_t source_layouts = {
      .module = module,
      .fact_table = fact_table,
      .bundle = bundle,
      .descriptor_set = descriptor_set,
      .target_facts = target_facts,
  };
  loom_amdgpu_fragment_publication_visit_function(
      source_function, loom_amdgpu_fragment_publication_collect_source_layout,
      &source_layouts);

  const loom_amdgpu_matrix_fragment_layout_t* source_layout = NULL;
  if (requested_layout->kind < LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT &&
      source_layouts.kinds[requested_layout->kind]) {
    source_layout = requested_layout;
  } else {
    for (uint32_t kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN + 1u;
         kind < LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT; ++kind) {
      if (!source_layouts.kinds[kind]) {
        continue;
      }
      const loom_amdgpu_matrix_fragment_layout_t* candidate_layout =
          loom_amdgpu_matrix_fragment_layout_for_kind(
              (loom_amdgpu_matrix_fragment_layout_kind_t)kind);
      if (candidate_layout == NULL ||
          !loom_amdgpu_fragment_publication_layouts_overlap(
              module, fact_table, view_regions, bundle, descriptor_set,
              target_facts, source_function, requested_layout,
              candidate_layout)) {
        continue;
      }
      if (source_layout != NULL) {
        *out_has_ambiguous_source_layout = true;
        return requested_layout;
      }
      source_layout = candidate_layout;
    }
  }
  if (source_layout == NULL) {
    return requested_layout;
  }

  for (uint32_t kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN + 1u;
       kind < LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT; ++kind) {
    if (!source_layouts.kinds[kind] || kind == source_layout->kind) {
      continue;
    }
    const loom_amdgpu_matrix_fragment_layout_t* candidate_layout =
        loom_amdgpu_matrix_fragment_layout_for_kind(
            (loom_amdgpu_matrix_fragment_layout_kind_t)kind);
    if (candidate_layout != NULL &&
        loom_amdgpu_fragment_publication_layouts_overlap(
            module, fact_table, view_regions, bundle, descriptor_set,
            target_facts, source_function, source_layout, candidate_layout)) {
      *out_has_ambiguous_source_layout = true;
      return source_layout;
    }
  }
  return source_layout;
}

static bool loom_amdgpu_fragment_publication_cost_is_less(
    loom_amdgpu_fragment_publication_cost_t lhs,
    loom_amdgpu_fragment_publication_cost_t rhs) {
  return lhs.memory_request_cost < rhs.memory_request_cost ||
         (lhs.memory_request_cost == rhs.memory_request_cost &&
          lhs.instruction_issue_cost < rhs.instruction_issue_cost);
}

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_select_matrix_fragment_publication_layout(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function,
    const loom_amdgpu_matrix_fragment_layout_t* contract_layout) {
  if (module == NULL || fact_table == NULL || view_regions == NULL ||
      bundle == NULL || descriptor_set == NULL || target_facts == NULL ||
      source_function.op == NULL || source_function.vtable == NULL ||
      contract_layout == NULL ||
      contract_layout->canonical_kind != contract_layout->kind) {
    return contract_layout;
  }

  bool has_ambiguous_source_layout = false;
  contract_layout = loom_amdgpu_fragment_publication_source_layout(
      module, fact_table, view_regions, bundle, descriptor_set, target_facts,
      source_function, contract_layout, &has_ambiguous_source_layout);
  if (has_ambiguous_source_layout) {
    return contract_layout;
  }

  const loom_amdgpu_matrix_fragment_layout_t* selected_layout = contract_layout;
  loom_amdgpu_fragment_publication_evaluation_t selected_evaluation =
      loom_amdgpu_fragment_publication_evaluate_layout(
          module, fact_table, view_regions, bundle, descriptor_set,
          target_facts, source_function, contract_layout, contract_layout);
  if (!selected_evaluation.valid ||
      selected_evaluation.relevant_store_count == 0) {
    return contract_layout;
  }
  for (uint32_t kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN + 1u;
       kind < LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT; ++kind) {
    const loom_amdgpu_matrix_fragment_layout_t* candidate_layout =
        loom_amdgpu_matrix_fragment_layout_for_kind(
            (loom_amdgpu_matrix_fragment_layout_kind_t)kind);
    if (candidate_layout == NULL || candidate_layout == contract_layout ||
        candidate_layout->canonical_kind != contract_layout->kind) {
      continue;
    }
    const loom_amdgpu_fragment_publication_evaluation_t candidate_evaluation =
        loom_amdgpu_fragment_publication_evaluate_layout(
            module, fact_table, view_regions, bundle, descriptor_set,
            target_facts, source_function, contract_layout, candidate_layout);
    if (!candidate_evaluation.valid ||
        candidate_evaluation.relevant_store_count !=
            selected_evaluation.relevant_store_count ||
        !loom_amdgpu_fragment_publication_cost_is_less(
            candidate_evaluation.cost, selected_evaluation.cost)) {
      continue;
    }
    selected_layout = candidate_layout;
    selected_evaluation = candidate_evaluation;
  }
  return selected_layout;
}
