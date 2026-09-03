// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"

#include "loom/analysis/control_uniformity.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/topology.h"

#define LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE 64

typedef struct loom_amdgpu_memory_subgroup_access_state_t {
  // Reusable execution-uniformity analysis for the source function.
  loom_control_uniformity_info_t control_uniformity;
} loom_amdgpu_memory_subgroup_access_state_t;

typedef struct loom_amdgpu_memory_byte_interval_t {
  // Inclusive relative byte offset of the interval begin.
  uint64_t begin;
  // Exclusive relative byte offset of the interval end.
  uint64_t end;
} loom_amdgpu_memory_byte_interval_t;

static int loom_amdgpu_memory_subgroup_access_state_key;

static iree_status_t loom_amdgpu_memory_control_uniformity(
    loom_low_lower_context_t* context,
    loom_control_uniformity_info_t** out_control_uniformity) {
  loom_amdgpu_memory_subgroup_access_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_memory_subgroup_access_state_key, sizeof(*state),
      (void**)&state));
  if (state->control_uniformity.module == NULL) {
    loom_control_uniformity_info_initialize(
        loom_low_lower_context_module(context),
        loom_low_lower_context_fact_table(context),
        loom_low_lower_context_function_arena(context),
        &state->control_uniformity);
  }
  *out_control_uniformity = &state->control_uniformity;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_memory_prove_full_subgroup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint8_t subgroup_size, loom_amdgpu_memory_lane_source_t lane_source,
    loom_amdgpu_memory_full_subgroup_proof_t* out_proof) {
  IREE_ASSERT_GT(subgroup_size, 0u);
  *out_proof = (loom_amdgpu_memory_full_subgroup_proof_t){0};
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_func_like_t function =
      loom_low_lower_context_source_function(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (lane_source == LOOM_AMDGPU_MEMORY_LANE_SOURCE_WORKITEM_X) {
    loom_target_workgroup_size_t workgroup_size = {0};
    if (!loom_amdgpu_required_workgroup_size_from_facts(
            module, function, bundle, fact_table, &workgroup_size) ||
        workgroup_size.x == 0) {
      out_proof->unknown_reason = IREE_SV("active-lane-workgroup-size-unknown");
      return iree_ok_status();
    }
    if (workgroup_size.x % subgroup_size != 0) {
      out_proof->unknown_reason = IREE_SV("active-lane-workitem-x-wrap");
      return iree_ok_status();
    }
  } else {
    uint32_t flat_workgroup_size = 0;
    if (!loom_amdgpu_required_flat_workgroup_size_from_facts(
            module, function, bundle, fact_table, &flat_workgroup_size) ||
        flat_workgroup_size == 0) {
      out_proof->unknown_reason = IREE_SV("active-lane-workgroup-size-unknown");
      return iree_ok_status();
    }
    if (flat_workgroup_size % subgroup_size != 0) {
      out_proof->unknown_reason = IREE_SV("active-lane-partial-subgroup");
      return iree_ok_status();
    }
  }

  loom_control_uniformity_info_t* control_uniformity = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_memory_control_uniformity(context, &control_uniformity));
  loom_control_uniformity_failure_t failure = {0};
  bool control_is_uniform = false;
  IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_execution(
      control_uniformity, source_op, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP,
      &failure, &control_is_uniform));
  if (!control_is_uniform) {
    out_proof->unknown_reason = IREE_SV("active-lane-control-not-uniform");
    return iree_ok_status();
  }
  out_proof->is_full_subgroup = true;
  out_proof->proof = IREE_SV("subgroup-uniform-control-full-wave");
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
    uint8_t subgroup_size, uint32_t per_lane_packet_byte_count,
    loom_low_lower_memory_subgroup_access_report_t* out_report) {
  IREE_ASSERT_GT(subgroup_size, 0u);
  IREE_ASSERT_LE(subgroup_size, LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE);
  IREE_ASSERT_GT(per_lane_packet_byte_count, 0u);
  uint64_t lane_offsets[LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE] = {0};
  loom_amdgpu_memory_byte_interval_t
      intervals[LOOM_AMDGPU_MEMORY_MAX_SUBGROUP_SIZE] = {0};
  for (uint8_t lane = 0; lane < subgroup_size; ++lane) {
    const uint64_t byte_offset =
        loom_amdgpu_fragment_memory_relative_lane_byte_offset(address_layout,
                                                              lane);
    lane_offsets[lane] = byte_offset;
    intervals[lane] = (loom_amdgpu_memory_byte_interval_t){
        .begin = byte_offset,
        .end = byte_offset + per_lane_packet_byte_count,
    };
    if (lane != 0) {
      const uint64_t previous_offset = lane_offsets[lane - 1];
      const uint64_t delta = byte_offset >= previous_offset
                                 ? byte_offset - previous_offset
                                 : previous_offset - byte_offset;
      out_report->maximum_adjacent_lane_delta_bytes =
          iree_max(out_report->maximum_adjacent_lane_delta_bytes, delta);
    }
  }

  loom_amdgpu_memory_sort_byte_intervals(subgroup_size, intervals);
  uint64_t region_begin = intervals[0].begin;
  uint64_t region_end = intervals[0].end;
  uint64_t previous_start = intervals[0].begin;
  out_report->distinct_lane_address_count = 1;
  out_report->contiguous_region_count = 1;
  for (uint8_t i = 1; i < subgroup_size; ++i) {
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
      (uint64_t)subgroup_size * per_lane_packet_byte_count;
  out_report->interval_coverage = out_report->contiguous_region_count == 1
                                      ? IREE_SV("dense")
                                      : IREE_SV("gapped");
}

iree_status_t loom_amdgpu_fragment_memory_report_subgroup_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index,
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

  loom_amdgpu_memory_full_subgroup_proof_t active_lane_proof = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_prove_full_subgroup(
      context, source_op, layout->wave_size,
      LOOM_AMDGPU_MEMORY_LANE_SOURCE_SUBGROUP_LANE, &active_lane_proof));
  if (!active_lane_proof.is_full_subgroup) {
    out_report->unknown_reason = active_lane_proof.unknown_reason;
    return iree_ok_status();
  }
  out_report->active_lane_proof = active_lane_proof.proof;

  if (!loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
          plan, packet->register_index, element_index)) {
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
      &plan->address_layout, layout->wave_size, per_lane_packet_byte_count,
      out_report);
  out_report->proof = IREE_SV("exact");
  return iree_ok_status();
}
