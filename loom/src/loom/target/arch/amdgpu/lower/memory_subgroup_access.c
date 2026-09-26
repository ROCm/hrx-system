// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/codegen/low/lower/participation.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/topology.h"

#define LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE 64

typedef struct loom_amdgpu_memory_byte_interval_t {
  // Inclusive relative byte offset of the interval begin.
  uint64_t begin;
  // Exclusive relative byte offset of the interval end.
  uint64_t end;
} loom_amdgpu_memory_byte_interval_t;

static bool loom_amdgpu_memory_coordinate_value(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    const uint32_t coordinates[LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_],
    int64_t* out_value) {
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  if (loom_value_facts_is_exact(value_facts)) {
    *out_value = value_facts.range_lo;
    return true;
  }
  const loom_value_fact_topology_domain_t* domain =
      loom_value_facts_topology_domain(value_facts);
  if (!domain ||
      (domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID &&
       domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID)) {
    return false;
  }
  *out_value = coordinates[domain->axis];
  return true;
}

static bool loom_amdgpu_memory_projection_value(
    const loom_value_fact_table_t* facts,
    const loom_symbolic_projection_t* projection,
    const uint32_t coordinates[LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_],
    int64_t* out_value) {
  int64_t value = 0;
  if (!loom_amdgpu_memory_coordinate_value(facts, projection->value_id,
                                           coordinates, &value) ||
      !iree_checked_mul_i64(value, projection->scale, &value) ||
      !iree_checked_add_i64(value, projection->offset, &value)) {
    return false;
  }
  value /= projection->divisor;
  if (projection->modulus) {
    value %= projection->modulus;
  }
  *out_value = value;
  return true;
}

// Interpret only the shared owner's completed numeric certificate. No source
// operations are decoded and no producer expansion occurs during lane queries.
static bool loom_amdgpu_memory_expression_value(
    const loom_symbolic_expr_context_t* expressions,
    const loom_low_lower_participation_operand_t* summary,
    const uint32_t coordinates[LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_],
    int64_t* out_value) {
  if (summary->projection) {
    return loom_amdgpu_memory_projection_value(
        expressions->fact_table, summary->projection, coordinates, out_value);
  }
  const loom_symbolic_expr_t* expression = &summary->expression;
  if (!loom_symbolic_expr_is_linear(expression)) {
    return false;
  }
  int64_t value = expression->constant;
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t* term = &expression->terms[i];
    int64_t coordinate = 0;
    if (!loom_amdgpu_memory_coordinate_value(expressions->fact_table,
                                             term->value_id, coordinates,
                                             &coordinate)) {
      loom_symbolic_expr_summary_t term_summary;
      if (!loom_symbolic_expr_context_try_lookup_summary(
              expressions, term->value_id, &term_summary) ||
          !term_summary.projection ||
          !loom_amdgpu_memory_projection_value(expressions->fact_table,
                                               term_summary.projection,
                                               coordinates, &coordinate)) {
        return false;
      }
    }
    if (!iree_checked_mul_i64(coordinate, term->coefficient, &coordinate) ||
        !iree_checked_add_i64(value, coordinate, &value)) {
      return false;
    }
  }
  *out_value = value;
  return true;
}

static bool loom_amdgpu_memory_comparison_mask(
    const loom_low_lower_participation_t* participation,
    const loom_symbolic_expr_context_t* expressions,
    const loom_target_workgroup_size_t* workgroup_size, uint8_t subgroup_size,
    uint64_t* out_mask) {
  const loom_low_lower_participation_condition_t* condition =
      participation->condition;
  const uint32_t plane_size = workgroup_size->x * workgroup_size->y;
  const uint32_t flat_size = plane_size * workgroup_size->z;
  for (uint32_t wave_begin = 0; wave_begin < flat_size;
       wave_begin += subgroup_size) {
    uint64_t mask = 0;
    for (uint8_t lane = 0; lane < subgroup_size; ++lane) {
      const uint32_t linear_id = wave_begin + lane;
      const uint32_t coordinates[] = {
          linear_id % workgroup_size->x,
          (linear_id / workgroup_size->x) % workgroup_size->y,
          linear_id / plane_size, lane};
      int64_t lhs_value = 0, rhs_value = 0;
      if (!loom_amdgpu_memory_expression_value(expressions, &condition->lhs,
                                               coordinates, &lhs_value) ||
          !loom_amdgpu_memory_expression_value(expressions, &condition->rhs,
                                               coordinates, &rhs_value)) {
        return false;
      }
      const loom_value_facts_t lhs_facts =
          loom_value_facts_exact_i64(lhs_value);
      const loom_value_facts_t rhs_facts =
          loom_value_facts_exact_i64(rhs_value);
      bool result = false;
      if (!loom_condition_integer_comparison_evaluate(
              &condition->comparison, &expressions->fact_table->context,
              &lhs_facts, &rhs_facts, &result)) {
        return false;
      }
      if (result == condition->assumed_truth) {
        mask |= UINT64_C(1) << lane;
      }
    }
    if (mask == 0 || (wave_begin != 0 && mask != *out_mask)) {
      return false;
    }
    *out_mask = mask;
  }
  return true;
}

iree_status_t loom_amdgpu_memory_prove_subgroup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint8_t subgroup_size, loom_amdgpu_memory_subgroup_proof_t* out_proof) {
  IREE_ASSERT_GT(subgroup_size, 0u);
  *out_proof = (loom_amdgpu_memory_subgroup_proof_t){0};
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_func_like_t function =
      loom_low_lower_context_source_function(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_amdgpu_required_workgroup_size_from_facts(
          module, function, bundle, fact_table, &workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.y == 0 || workgroup_size.z == 0) {
    out_proof->unknown_reason = IREE_SV("active-lane-workgroup-size-unknown");
    return iree_ok_status();
  }
  const uint64_t flat_workgroup_size =
      (uint64_t)workgroup_size.x * workgroup_size.y * workgroup_size.z;
  if (flat_workgroup_size % subgroup_size != 0) {
    out_proof->unknown_reason = IREE_SV("active-lane-partial-subgroup");
    return iree_ok_status();
  }

  loom_low_lower_participation_t participation;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_subgroup_participation(
      context, source_op, &participation));
  if (participation.kind == LOOM_LOW_LOWER_PARTICIPATION_FULL) {
    out_proof->active_lane_mask =
        subgroup_size == 64 ? UINT64_MAX : (UINT64_C(1) << subgroup_size) - 1u;
    out_proof->proof = IREE_SV("subgroup-uniform-control-full-wave");
  } else if (participation.kind == LOOM_LOW_LOWER_PARTICIPATION_COMPARISON) {
    if (!loom_amdgpu_memory_comparison_mask(
            &participation,
            loom_low_lower_context_symbolic_expr_context(context),
            &workgroup_size, subgroup_size, &out_proof->active_lane_mask)) {
      out_proof->unknown_reason = IREE_SV("active-lane-predicate-unproven");
      return iree_ok_status();
    }
    out_proof->proof = IREE_SV("single-entry-comparison-all-waves");
  } else {
    out_proof->unknown_reason = IREE_SV("active-lane-control-not-uniform");
    return iree_ok_status();
  }
  out_proof->is_proven = true;
  out_proof->workgroup_size = workgroup_size;
  return iree_ok_status();
}

static void loom_amdgpu_memory_sort_byte_intervals(
    uint8_t interval_count, loom_amdgpu_memory_byte_interval_t* intervals) {
  for (uint8_t i = 1; i < interval_count; ++i) {
    const loom_amdgpu_memory_byte_interval_t value = intervals[i];
    uint8_t insert_index = i;
    while (insert_index > 0) {
      const loom_amdgpu_memory_byte_interval_t previous =
          intervals[insert_index - 1];
      if (previous.begin < value.begin ||
          (previous.begin == value.begin && previous.end <= value.end)) {
        break;
      }
      intervals[insert_index] = previous;
      --insert_index;
    }
    intervals[insert_index] = value;
  }
}

void loom_amdgpu_memory_calculate_subgroup_geometry(
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    uint8_t subgroup_size, uint64_t active_lane_mask,
    uint32_t per_lane_packet_byte_count,
    loom_low_lower_memory_subgroup_access_report_t* out_report) {
  IREE_ASSERT_GT(subgroup_size, 0u);
  IREE_ASSERT_LE(subgroup_size, LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE);
  const uint64_t subgroup_lane_mask =
      subgroup_size == 64 ? UINT64_MAX : (UINT64_C(1) << subgroup_size) - 1u;
  IREE_ASSERT_NE(active_lane_mask, 0u);
  IREE_ASSERT_EQ(active_lane_mask & ~subgroup_lane_mask, 0u);
  IREE_ASSERT_GT(per_lane_packet_byte_count, 0u);
  loom_amdgpu_memory_byte_interval_t
      intervals[LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE] = {0};
  uint8_t active_lane_count = 0;
  bool has_previous_lane = false;
  uint64_t previous_lane_offset = 0;
  for (uint8_t lane = 0; lane < subgroup_size; ++lane) {
    if ((active_lane_mask & (UINT64_C(1) << lane)) == 0) {
      continue;
    }
    const uint64_t byte_offset =
        loom_amdgpu_fragment_memory_relative_lane_byte_offset(address_layout,
                                                              lane);
    intervals[active_lane_count++] = (loom_amdgpu_memory_byte_interval_t){
        .begin = byte_offset,
        .end = byte_offset + per_lane_packet_byte_count,
    };
    if (has_previous_lane) {
      const uint64_t delta = byte_offset >= previous_lane_offset
                                 ? byte_offset - previous_lane_offset
                                 : previous_lane_offset - byte_offset;
      out_report->maximum_adjacent_lane_delta_bytes =
          iree_max(out_report->maximum_adjacent_lane_delta_bytes, delta);
    }
    previous_lane_offset = byte_offset;
    has_previous_lane = true;
  }

  loom_amdgpu_memory_sort_byte_intervals(active_lane_count, intervals);
  uint64_t region_begin = intervals[0].begin;
  uint64_t region_end = intervals[0].end;
  uint64_t previous_start = intervals[0].begin;
  out_report->distinct_lane_address_count = 1;
  out_report->contiguous_region_count = 1;
  for (uint8_t i = 1; i < active_lane_count; ++i) {
    const loom_amdgpu_memory_byte_interval_t interval = intervals[i];
    if (interval.begin != previous_start) {
      ++out_report->distinct_lane_address_count;
      previous_start = interval.begin;
    }
    if (interval.begin <= region_end) {
      region_end = iree_max(region_end, interval.end);
      continue;
    }
    out_report->subgroup_unique_byte_count += region_end - region_begin;
    out_report->maximum_uncovered_byte_gap_bytes =
        iree_max(out_report->maximum_uncovered_byte_gap_bytes,
                 interval.begin - region_end);
    ++out_report->contiguous_region_count;
    region_begin = interval.begin;
    region_end = interval.end;
  }
  out_report->subgroup_unique_byte_count += region_end - region_begin;
  out_report->subgroup_span_byte_count = region_end - intervals[0].begin;
  out_report->subgroup_requested_byte_count =
      (uint64_t)active_lane_count * per_lane_packet_byte_count;
  out_report->interval_coverage = out_report->contiguous_region_count == 1
                                      ? IREE_SV("dense")
                                      : IREE_SV("gapped");
}

iree_status_t loom_amdgpu_fragment_memory_report_subgroup_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_offset_t* runtime_offset,
    const loom_low_descriptor_memory_effect_summary_t* issued,
    loom_low_lower_memory_subgroup_access_report_t* out_report) {
  static_assert(
      LOOM_MATRIX_FRAGMENT_AXIS_COUNT ==
          LOOM_LOW_LOWER_MEMORY_SUBGROUP_ACCESS_TERM_CAPACITY,
      "fragment lane terms must fit the source-low report representation");
  IREE_ASSERT_GT(layout->wave_size, 0u);
  IREE_ASSERT_LE(layout->wave_size, LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE);
  *out_report = (loom_low_lower_memory_subgroup_access_report_t){
      .proof = IREE_SVL("unknown"),
      .lane_address_proof = IREE_SVL("unproven"),
      .active_lane_proof = IREE_SVL("unproven"),
      .subgroup_size = layout->wave_size,
      .lane_term_count = plan->address_layout.lane_term_count,
      .linear_lane_byte_stride = plan->address_layout.linear_lane_byte_stride,
  };
  if (plan->address_layout.lane_term_count == 0) {
    out_report->lane_mapping = IREE_SV("uniform");
  } else if (plan->address_layout.linear_lane_byte_stride != 0) {
    out_report->lane_mapping = IREE_SV("linear");
  } else {
    out_report->lane_mapping = IREE_SV("digit-terms");
  }
  for (uint8_t i = 0; i < plan->address_layout.lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* source_term =
        &plan->address_layout.lane_terms[i];
    out_report->lane_terms[i] = (loom_low_lower_memory_subgroup_access_term_t){
        .divisor = source_term->divisor,
        .modulus = source_term->modulus,
        .byte_stride = source_term->byte_stride,
    };
  }

  uint32_t per_lane_packet_byte_count = 0;
  uint16_t unknown_width_count = 0;
  if (plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    per_lane_packet_byte_count = issued->read_byte_count;
    unknown_width_count = issued->read_unknown_width_count;
  } else {
    IREE_ASSERT_EQ(plan->operation_kind,
                   LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE);
    per_lane_packet_byte_count = issued->write_byte_count;
    unknown_width_count = issued->write_unknown_width_count;
  }
  if (unknown_width_count != 0) {
    out_report->unknown_reason = IREE_SV("packet-effect-width-unknown");
    return iree_ok_status();
  }
  IREE_ASSERT_GT(per_lane_packet_byte_count, 0u);
  out_report->per_lane_packet_byte_count = per_lane_packet_byte_count;

  loom_amdgpu_memory_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_subgroup(
      context, source_op, layout->wave_size, &active_lane_proof));
  if (!active_lane_proof.is_proven) {
    out_report->unknown_reason = active_lane_proof.unknown_reason;
    return iree_ok_status();
  }
  out_report->active_lane_proof = active_lane_proof.proof;
  out_report->active_lane_count =
      (uint8_t)iree_math_count_ones_u64(active_lane_proof.active_lane_mask);

  if (!runtime_offset->is_subgroup_uniform) {
    out_report->lane_mapping = IREE_SV("runtime-axis-terms");
    out_report->unknown_reason = IREE_SV("address-runtime-fragment-stride");
    return iree_ok_status();
  }

  if (!plan->dynamic_base_is_subgroup_uniform) {
    out_report->unknown_reason =
        IREE_SV("address-dynamic-base-not-subgroup-uniform");
    return iree_ok_status();
  }
  out_report->lane_address_proof =
      IREE_SV("compiled-fragment-lane-register-layout");

  loom_amdgpu_memory_calculate_subgroup_geometry(
      &plan->address_layout, layout->wave_size,
      active_lane_proof.active_lane_mask, per_lane_packet_byte_count,
      out_report);
  out_report->proof = IREE_SV("exact");
  return iree_ok_status();
}
