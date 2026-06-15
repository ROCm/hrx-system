// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/diagnostics.h"

#include <inttypes.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/op_defs.h"

#define LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS 16

static iree_status_t loom_low_schedule_emit(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

iree_status_t loom_low_schedule_emit_missing_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    iree_string_view_t opcode) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(opcode),
      loom_param_string(state->target.descriptor_set_key),
  };
  return loom_low_schedule_emit(state, op, LOOM_ERR_TARGET_045, params,
                                IREE_ARRAYSIZE(params));
}

static bool loom_low_schedule_interval_contains_point(
    const loom_liveness_interval_t* interval, uint32_t point) {
  return interval->start_point <= point && point < interval->end_point;
}

static bool loom_low_schedule_first_pressure_cliff_for_reg_class(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id,
    const loom_low_schedule_pressure_cliff_t** out_cliff) {
  *out_cliff = NULL;
  if (state->pressure_cliff_ranges == NULL ||
      reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= state->target.descriptor_set->reg_class_count) {
    return false;
  }
  const loom_low_schedule_pressure_cliff_range_t range =
      state->pressure_cliff_ranges[reg_class_id];
  if (range.count == 0) {
    return false;
  }
  *out_cliff = &state->options->pressure_cliffs.values[range.start];
  return true;
}

static iree_status_t loom_low_schedule_pressure_budget_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_liveness_value_class_t value_class, uint32_t* out_budget,
    bool* out_has_budget) {
  *out_budget = 0;
  *out_has_budget = false;
  if (value_class.type_kind != LOOM_TYPE_REGISTER ||
      !state->target.descriptor_set ||
      value_class.register_descriptor_set_stable_id !=
          state->target.descriptor_set->stable_id ||
      value_class.register_class_id >=
          state->target.descriptor_set->reg_class_count) {
    return iree_ok_status();
  }
  const uint16_t reg_class_id = value_class.register_class_id;
  const loom_low_reg_class_t* reg_class =
      &state->target.descriptor_set->reg_classes[reg_class_id];
  const loom_low_schedule_pressure_cliff_t* first_cliff = NULL;
  if (loom_low_schedule_first_pressure_cliff_for_reg_class(state, reg_class_id,
                                                           &first_cliff)) {
    *out_budget = first_cliff->cliff_units;
    *out_has_budget = true;
    return iree_ok_status();
  }
  if (reg_class->allocatable_count == 0) {
    return iree_ok_status();
  }
  *out_budget = reg_class->allocatable_count;
  *out_has_budget = true;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_collect_pressure_contributors(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary,
    const iree_string_view_t** out_contributors,
    iree_host_size_t* out_contributor_count) {
  iree_string_view_t* contributors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS,
      sizeof(*contributors), (void**)&contributors));
  iree_host_size_t contributor_count = 0;
  bool overflowed = false;
  for (uint32_t point_attempt = 0; point_attempt < 2; ++point_attempt) {
    if (point_attempt != 0 && contributor_count != 0) {
      break;
    }
    if (point_attempt != 0 && summary->peak_point == UINT32_MAX) {
      break;
    }
    uint32_t point = summary->peak_point + point_attempt;
    for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
      const loom_liveness_interval_t* interval = &liveness->intervals[i];
      if (!loom_liveness_value_class_equal(interval->value_class,
                                           summary->value_class) ||
          !loom_low_schedule_interval_contains_point(interval, point)) {
        continue;
      }
      if (contributor_count < LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS) {
        contributors[contributor_count++] =
            loom_low_diagnostic_value_name(state->module, interval->value_id);
      } else {
        overflowed = true;
      }
    }
  }
  if (overflowed &&
      contributor_count == LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS) {
    contributors[contributor_count - 1] = IREE_SV("...");
  }
  if (contributor_count == 0) {
    contributors[contributor_count++] = IREE_SV("<none>");
  }
  *out_contributors = contributors;
  *out_contributor_count = contributor_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_pressure_summary(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary) {
  uint32_t budget = 0;
  bool has_budget = false;
  IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_budget_for_class(
      state, summary->value_class, &budget, &has_budget));
  if (!has_budget) {
    return iree_ok_status();
  }

  const iree_string_view_t* contributors = NULL;
  iree_host_size_t contributor_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_schedule_collect_pressure_contributors(
      state, liveness, summary, &contributors, &contributor_count));

  const loom_op_t* origin_op =
      summary->peak_op ? summary->peak_op : state->function_op;
  iree_string_view_t operation_name =
      summary->peak_op ? loom_op_name(state->module, summary->peak_op)
                       : IREE_SV("<block-boundary>");
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          state->target.descriptor_set, summary->value_class)),
      loom_param_u32(budget),
      loom_param_u32(summary->peak_live_units),
      loom_param_string(
          loom_low_diagnostic_block_name(state->module, summary->peak_block)),
      loom_param_string(operation_name),
      loom_param_string_list(contributors, contributor_count),
  };
  return loom_low_schedule_emit(state, origin_op, LOOM_ERR_BACKEND_003, params,
                                IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_schedule_emit_pressure_diagnostics(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_pressure_summary(
        state, liveness, &liveness->pressure_summaries[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_node_diagnostic_label(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, iree_string_view_t* out_label) {
  *out_label = IREE_SV("<unknown>");
  if (node == NULL) {
    return iree_ok_status();
  }
  if (node->descriptor != NULL) {
    *out_label = loom_low_descriptor_set_string(
        state->target.descriptor_set, node->descriptor->key_string_offset);
    return iree_ok_status();
  }
  *out_label = loom_op_name(state->module, node->op);
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_candidate_decision(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_candidate_decision_t* decision) {
  if (decision->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* chosen_node = NULL;
  if (decision->chosen_node < state->scheduled_node_count) {
    chosen_node = &state->nodes[decision->chosen_node];
  }
  const loom_low_schedule_node_t* rejected_node = NULL;
  if (decision->rejected_node < state->scheduled_node_count) {
    rejected_node = &state->nodes[decision->rejected_node];
  }
  const loom_block_t* block = NULL;
  if (decision->block_index < state->body->block_count) {
    block = state->blocks[decision->block_index].block;
  }
  iree_string_view_t chosen_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, chosen_node, &chosen_label));
  iree_string_view_t rejected_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, rejected_node, &rejected_label));
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_block_name(state->module, block)),
      loom_param_u32(decision->scheduled_ordinal),
      loom_param_u32(decision->ready_candidate_count),
      loom_param_string(chosen_label),
      loom_param_string(rejected_label),
      loom_param_u32(decision->chosen_dependency_latency_cycles),
      loom_param_u32(decision->chosen_latency_cycles),
      loom_param_u32(decision->chosen_pair_affinity_score),
      loom_param_u64(decision->chosen_projected_live_units),
      loom_param_u64(decision->chosen_killed_live_units),
      loom_param_u64(decision->chosen_produced_live_units),
      loom_param_u32(decision->chosen_data_ready_stall_cycles),
      loom_param_u32(decision->chosen_resource_stall_cycles),
      loom_param_u32(decision->chosen_hazard_stall_cycles),
      loom_param_u32(decision->chosen_effective_stall_cycles),
      loom_param_u32(decision->chosen_pressure_cliff_reg_class_id),
      loom_param_u32(decision->chosen_pressure_cliff_units),
      loom_param_u32(decision->chosen_pressure_cliff_penalty),
      loom_param_u32(decision->chosen_units_until_pressure_cliff),
      loom_param_u32(decision->rejected_dependency_latency_cycles),
      loom_param_u32(decision->rejected_latency_cycles),
      loom_param_u32(decision->rejected_pair_affinity_score),
      loom_param_u64(decision->rejected_projected_live_units),
      loom_param_u64(decision->rejected_killed_live_units),
      loom_param_u64(decision->rejected_produced_live_units),
      loom_param_u32(decision->rejected_data_ready_stall_cycles),
      loom_param_u32(decision->rejected_resource_stall_cycles),
      loom_param_u32(decision->rejected_hazard_stall_cycles),
      loom_param_u32(decision->rejected_effective_stall_cycles),
      loom_param_u32(decision->rejected_pressure_cliff_reg_class_id),
      loom_param_u32(decision->rejected_pressure_cliff_units),
      loom_param_u32(decision->rejected_pressure_cliff_penalty),
      loom_param_u32(decision->rejected_units_until_pressure_cliff),
  };
  const loom_op_t* origin_op =
      chosen_node && chosen_node->op ? chosen_node->op : state->function_op;
  return loom_low_schedule_emit(state, origin_op, LOOM_ERR_BACKEND_015, params,
                                IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_schedule_emit_candidate_decision_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->candidate_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_candidate_decision(
        state, &state->candidate_decisions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_model_summary(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_model_summary_t* summary) {
  if (summary->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* first_node = NULL;
  if (summary->first_node < state->scheduled_node_count) {
    first_node = &state->nodes[summary->first_node];
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(summary->schedule_class_name),
      loom_param_string(loom_low_model_quality_name(summary->model_quality)),
      loom_param_string(loom_low_latency_kind_name(summary->latency_kind)),
      loom_param_u32(summary->latency_cycles),
      loom_param_u32(summary->issue_use_count),
      loom_param_u32(summary->hazard_count),
      loom_param_u32(summary->use_count),
  };
  const loom_op_t* origin_op =
      first_node && first_node->op ? first_node->op : state->function_op;
  return loom_low_schedule_emit(state, origin_op, LOOM_ERR_BACKEND_016, params,
                                IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_schedule_emit_model_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->model_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_model_summary(
        state, &state->model_summaries[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_resource_bottleneck(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_resource_summary_t* summary) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(summary->resource_name),
      loom_param_u32(summary->capacity_per_cycle),
      loom_param_u32(summary->contention_group_id),
      loom_param_u32(summary->use_count),
      loom_param_u64(summary->total_unit_cycles),
      loom_param_u64(summary->estimated_min_cycles),
      loom_param_u32(summary->peak_units_per_cycle),
  };
  return loom_low_schedule_emit(state, state->function_op, LOOM_ERR_BACKEND_013,
                                params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_schedule_emit_resource_diagnostics(
    loom_low_schedule_build_state_t* state) {
  uint64_t maximum_estimated_min_cycles = 0;
  for (iree_host_size_t i = 0; i < state->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &state->resource_summaries[i];
    if (summary->estimated_min_cycles > maximum_estimated_min_cycles) {
      maximum_estimated_min_cycles = summary->estimated_min_cycles;
    }
  }
  if (maximum_estimated_min_cycles == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &state->resource_summaries[i];
    if (summary->estimated_min_cycles != maximum_estimated_min_cycles) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_resource_bottleneck(state, summary));
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_hazard_gap_packet_index(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (hazard_gap->block_index >= state->body->block_count) {
    return UINT32_MAX;
  }
  const loom_low_schedule_block_t* block =
      &state->blocks[hazard_gap->block_index];
  const uint64_t packet_index =
      (uint64_t)block->scheduled_node_start + scheduled_ordinal;
  return packet_index <= UINT32_MAX ? (uint32_t)packet_index : UINT32_MAX;
}

static iree_string_view_t loom_low_schedule_hazard_reference_name(
    const loom_low_schedule_hazard_gap_t* hazard_gap, char* buffer,
    iree_host_size_t buffer_capacity) {
  if (!iree_string_view_is_empty(hazard_gap->resource_name)) {
    return hazard_gap->resource_name;
  }
  if (buffer_capacity == 0) {
    return IREE_SV("<unknown>");
  }
  iree_string_view_t reference_kind =
      loom_low_hazard_reference_kind_name(hazard_gap->reference_kind);
  int length = iree_snprintf(buffer, buffer_capacity, "%.*s:%" PRIu16,
                             (int)reference_kind.size, reference_kind.data,
                             hazard_gap->reference_id);
  if (length < 0) {
    return IREE_SV("<unknown>");
  }
  iree_host_size_t size = (iree_host_size_t)length;
  if (size >= buffer_capacity) {
    size = buffer_capacity - 1;
  }
  return iree_make_string_view(buffer, size);
}

static iree_status_t loom_low_schedule_emit_hazard_gap(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_hazard_gap_t* hazard_gap) {
  const loom_low_schedule_node_t* consumer_node = NULL;
  if (hazard_gap->consumer_node < state->scheduled_node_count) {
    consumer_node = &state->nodes[hazard_gap->consumer_node];
  }
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_schedule_node_diagnostic_label(
      state, consumer_node, &descriptor_key));
  char reference_buffer[32];
  iree_string_view_t reference_name = loom_low_schedule_hazard_reference_name(
      hazard_gap, reference_buffer, sizeof(reference_buffer));
  uint32_t producer_packet = loom_low_schedule_hazard_gap_packet_index(
      state, hazard_gap, hazard_gap->producer_scheduled_ordinal);
  uint32_t consumer_packet = loom_low_schedule_hazard_gap_packet_index(
      state, hazard_gap, hazard_gap->consumer_scheduled_ordinal);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(loom_low_diagnostic_string_or_placeholder(
          descriptor_key, IREE_SV("<unknown>"))),
      loom_param_string(loom_low_hazard_kind_name(hazard_gap->kind)),
      loom_param_string(
          loom_low_hazard_reference_kind_name(hazard_gap->reference_kind)),
      loom_param_string(reference_name),
      loom_param_u32(hazard_gap->required_distance),
      loom_param_u32(hazard_gap->actual_distance),
      loom_param_u32(hazard_gap->required_delay),
      loom_param_u32(producer_packet),
      loom_param_u32(consumer_packet),
  };
  const loom_op_t* origin_op =
      consumer_node ? consumer_node->op : state->function_op;
  return loom_low_schedule_emit(state, origin_op, LOOM_ERR_BACKEND_014, params,
                                IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_schedule_emit_hazard_gap_diagnostics(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->hazard_gap_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_hazard_gap(state, &state->hazard_gaps[i]));
  }
  return iree_ok_status();
}
