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
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .module = table->module,
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_schedule_emit_with_related(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count) {
  loom_diagnostic_emission_t emission = {
      .module = table->module,
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = related_op_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_string_view_t loom_low_schedule_dependency_kind_name(
    loom_low_schedule_dependency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_DEPENDENCY_SSA:
      return IREE_SV("ssa");
    case LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT:
      return IREE_SV("effect");
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STATE:
      return IREE_SV("state");
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE:
      return IREE_SV("storage");
    default:
      return IREE_SV("unknown");
  }
}

static const loom_low_schedule_node_t* loom_low_schedule_failure_node(
    const loom_low_schedule_table_t* table, uint32_t node_index) {
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return NULL;
  }
  return &table->nodes[node_index];
}

static iree_string_view_t loom_low_schedule_node_diagnostic_label(
    const loom_low_schedule_table_t* table,
    const loom_low_schedule_node_t* node) {
  if (node == NULL) return IREE_SV("<unknown>");
  return node->descriptor != NULL ? loom_low_descriptor_set_string(
                                        table->target.descriptor_set,
                                        node->descriptor->key_string_offset)
                                  : loom_op_name(table->module, node->op);
}

static iree_string_view_t loom_low_schedule_operand_index_name(
    uint32_t operand_index, char* buffer, iree_host_size_t buffer_capacity) {
  if (operand_index == UINT32_MAX) {
    return IREE_SV("none");
  }
  if (buffer_capacity == 0) {
    return IREE_SV("<unknown>");
  }
  int length =
      iree_snprintf(buffer, buffer_capacity, "%" PRIu32, operand_index);
  if (length < 0) {
    return IREE_SV("<unknown>");
  }
  iree_host_size_t size = (iree_host_size_t)length;
  if (size >= buffer_capacity) {
    size = buffer_capacity - 1;
  }
  return iree_make_string_view(buffer, size);
}

static iree_status_t loom_low_schedule_emit_dependency_cycle(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_low_schedule_failure_t* failure) {
  if (!loom_low_schedule_failure_is_present(failure)) {
    return iree_ok_status();
  }

  const loom_low_schedule_node_t* producer_node =
      loom_low_schedule_failure_node(table, failure->producer_node);
  const loom_low_schedule_node_t* consumer_node =
      loom_low_schedule_failure_node(table, failure->consumer_node);
  const iree_string_view_t producer_label =
      loom_low_schedule_node_diagnostic_label(table, producer_node);
  const iree_string_view_t consumer_label =
      loom_low_schedule_node_diagnostic_label(table, consumer_node);

  iree_string_view_t
      cycle_labels[LOOM_LOW_SCHEDULE_FAILURE_CYCLE_NODE_CAPACITY];
  iree_host_size_t cycle_label_count = failure->cycle_node_count;
  if (cycle_label_count > IREE_ARRAYSIZE(cycle_labels)) {
    cycle_label_count = IREE_ARRAYSIZE(cycle_labels);
  }
  for (iree_host_size_t i = 0; i < cycle_label_count; ++i) {
    const loom_low_schedule_node_t* cycle_node =
        loom_low_schedule_failure_node(table, failure->cycle_nodes[i]);
    cycle_labels[i] =
        loom_low_schedule_node_diagnostic_label(table, cycle_node);
  }
  if (cycle_label_count == 0) {
    cycle_labels[cycle_label_count++] = IREE_SV("<unavailable>");
  }

  const loom_block_t* block = NULL;
  if (failure->block_index < table->block_count) {
    block = table->blocks[failure->block_index].block;
  }
  char operand_buffer[32];
  iree_string_view_t operand_index = loom_low_schedule_operand_index_name(
      failure->operand_index, operand_buffer, sizeof(operand_buffer));
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_block_name(table->module, block)),
      loom_param_u32(failure->block_index),
      loom_param_u32(failure->scheduled_node_count),
      loom_param_u32(failure->block_node_count),
      loom_param_u32(failure->unscheduled_node_count),
      loom_param_u32(failure->producer_node),
      loom_param_string(producer_label),
      loom_param_u32(failure->consumer_node),
      loom_param_string(consumer_label),
      loom_param_string(
          loom_low_schedule_dependency_kind_name(failure->dependency_kind)),
      loom_param_string(operand_index),
      loom_param_string_list(cycle_labels, cycle_label_count),
      loom_param_bool(iree_all_bits_set(
          failure->flags, LOOM_LOW_SCHEDULE_FAILURE_FLAG_CYCLE_PATH_TRUNCATED)),
      loom_param_bool(iree_all_bits_set(
          failure->flags, LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY)),
  };
  loom_diagnostic_related_op_t related_ops[2];
  iree_host_size_t related_op_count = 0;
  if (producer_node != NULL) {
    related_ops[related_op_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("cycle producer"),
        .op = producer_node->op,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
  }
  if (consumer_node != NULL) {
    loom_diagnostic_field_ref_t field_ref = loom_diagnostic_field_ref_none();
    if (failure->operand_index <= UINT16_MAX) {
      field_ref = loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                            (uint16_t)failure->operand_index);
    }
    related_ops[related_op_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("cycle consumer"),
        .op = consumer_node->op,
        .field_ref = field_ref,
    };
  }
  const loom_op_t* origin_op =
      consumer_node != NULL ? consumer_node->op : table->function_op;
  return loom_low_schedule_emit_with_related(
      table, emitter, origin_op, LOOM_ERR_BACKEND_044, params,
      IREE_ARRAYSIZE(params), related_ops, related_op_count);
}

static bool loom_low_schedule_interval_contains_point(
    const loom_liveness_interval_t* interval, uint32_t point) {
  return interval->start_point <= point && point < interval->end_point;
}

static iree_host_size_t loom_low_schedule_collect_pressure_contributors(
    const loom_low_schedule_table_t* table,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary,
    iree_string_view_t* contributors) {
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
            loom_low_diagnostic_value_name(table->module, interval->value_id);
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
  return contributor_count;
}

static iree_status_t loom_low_schedule_emit_pressure_summary(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_pressure_summary_t* summary, uint32_t budget) {
  if (budget == UINT32_MAX) {
    return iree_ok_status();
  }

  iree_string_view_t contributors[LOOM_LOW_SCHEDULE_MAX_PRESSURE_CONTRIBUTORS];
  const iree_host_size_t contributor_count =
      loom_low_schedule_collect_pressure_contributors(table, liveness, summary,
                                                      contributors);

  const loom_op_t* origin_op =
      summary->peak_op ? summary->peak_op : table->function_op;
  iree_string_view_t operation_name =
      summary->peak_op ? loom_op_name(table->module, summary->peak_op)
                       : IREE_SV("<block-boundary>");
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, summary->value_class)),
      loom_param_u32(budget),
      loom_param_u32(summary->peak_live_units),
      loom_param_string(
          loom_low_diagnostic_block_name(table->module, summary->peak_block)),
      loom_param_string(operation_name),
      loom_param_string_list(contributors, contributor_count),
  };
  return loom_low_schedule_emit(table, emitter, origin_op, LOOM_ERR_BACKEND_003,
                                params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_pressure_diagnostics(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_pressure_summary(
        table, emitter, liveness, &liveness->pressure_summaries[i],
        table->pressure_summary_budgets[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_candidate_decision(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_low_schedule_candidate_decision_t* decision) {
  if (decision->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* chosen_node = NULL;
  if (decision->chosen_node < table->scheduled_node_count) {
    chosen_node = &table->nodes[decision->chosen_node];
  }
  const loom_low_schedule_node_t* rejected_node = NULL;
  if (decision->rejected_node < table->scheduled_node_count) {
    rejected_node = &table->nodes[decision->rejected_node];
  }
  const loom_block_t* block = NULL;
  if (decision->block_index < table->block_count) {
    block = table->blocks[decision->block_index].block;
  }
  const iree_string_view_t chosen_label =
      loom_low_schedule_node_diagnostic_label(table, chosen_node);
  const iree_string_view_t rejected_label =
      loom_low_schedule_node_diagnostic_label(table, rejected_node);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_block_name(table->module, block)),
      loom_param_u32(decision->scheduled_ordinal),
      loom_param_u32(decision->ready_candidate_count),
      loom_param_u32(decision->scored_candidate_count),
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
      loom_param_u32(decision->chosen_completion_wait_cycles),
      loom_param_u32(decision->chosen_effective_stall_cycles),
      loom_param_string(decision->chosen_pressure_cliff_source),
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
      loom_param_u32(decision->rejected_completion_wait_cycles),
      loom_param_u32(decision->rejected_effective_stall_cycles),
      loom_param_string(decision->rejected_pressure_cliff_source),
      loom_param_u32(decision->rejected_pressure_cliff_units),
      loom_param_u32(decision->rejected_pressure_cliff_penalty),
      loom_param_u32(decision->rejected_units_until_pressure_cliff),
  };
  const loom_op_t* origin_op =
      chosen_node && chosen_node->op ? chosen_node->op : table->function_op;
  return loom_low_schedule_emit(table, emitter, origin_op, LOOM_ERR_BACKEND_015,
                                params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_candidate_decision_diagnostics(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->candidate_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_candidate_decision(
        table, emitter, &table->candidate_decisions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_model_summary(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_low_schedule_model_summary_t* summary) {
  if (summary->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* first_node = NULL;
  if (summary->first_node < table->scheduled_node_count) {
    first_node = &table->nodes[summary->first_node];
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(summary->schedule_class_name),
      loom_param_string(loom_low_model_quality_name(summary->model_quality)),
      loom_param_string(loom_low_latency_kind_name(summary->latency_kind)),
      loom_param_u32(summary->latency_cycles),
      loom_param_u32(summary->issue_use_count),
      loom_param_u32(summary->hazard_count),
      loom_param_u32(summary->use_count),
  };
  const loom_op_t* origin_op =
      first_node && first_node->op ? first_node->op : table->function_op;
  return loom_low_schedule_emit(table, emitter, origin_op, LOOM_ERR_BACKEND_016,
                                params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_model_diagnostics(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->model_summary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_model_summary(
        table, emitter, &table->model_summaries[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_emit_resource_bottleneck(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_low_schedule_resource_summary_t* summary) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(summary->resource_name),
      loom_param_u32(summary->capacity_per_cycle),
      loom_param_u32(summary->contention_group_id),
      loom_param_u32(summary->use_count),
      loom_param_u64(summary->total_unit_cycles),
      loom_param_u64(summary->estimated_min_cycles),
      loom_param_u32(summary->peak_units_per_cycle),
  };
  return loom_low_schedule_emit(table, emitter, table->function_op,
                                LOOM_ERR_BACKEND_013, params,
                                IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_resource_diagnostics(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter) {
  uint64_t maximum_estimated_min_cycles = 0;
  for (iree_host_size_t i = 0; i < table->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &table->resource_summaries[i];
    if (summary->estimated_min_cycles > maximum_estimated_min_cycles) {
      maximum_estimated_min_cycles = summary->estimated_min_cycles;
    }
  }
  if (maximum_estimated_min_cycles == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < table->resource_summary_count; ++i) {
    const loom_low_schedule_resource_summary_t* summary =
        &table->resource_summaries[i];
    if (summary->estimated_min_cycles != maximum_estimated_min_cycles) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_resource_bottleneck(table, emitter, summary));
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_hazard_gap_packet_index(
    const loom_low_schedule_table_t* table,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (hazard_gap->block_index >= table->block_count) {
    return UINT32_MAX;
  }
  const loom_low_schedule_block_t* block =
      &table->blocks[hazard_gap->block_index];
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
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter,
    const loom_low_schedule_hazard_gap_t* hazard_gap) {
  const loom_low_schedule_node_t* consumer_node = NULL;
  if (hazard_gap->consumer_node < table->scheduled_node_count) {
    consumer_node = &table->nodes[hazard_gap->consumer_node];
  }
  const iree_string_view_t descriptor_key =
      loom_low_schedule_node_diagnostic_label(table, consumer_node);
  char reference_buffer[32];
  iree_string_view_t reference_name = loom_low_schedule_hazard_reference_name(
      hazard_gap, reference_buffer, sizeof(reference_buffer));
  uint32_t producer_packet = loom_low_schedule_hazard_gap_packet_index(
      table, hazard_gap, hazard_gap->producer_scheduled_ordinal);
  uint32_t consumer_packet = loom_low_schedule_hazard_gap_packet_index(
      table, hazard_gap, hazard_gap->consumer_scheduled_ordinal);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
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
      consumer_node ? consumer_node->op : table->function_op;
  return loom_low_schedule_emit(table, emitter, origin_op, LOOM_ERR_BACKEND_014,
                                params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_schedule_emit_hazard_gap_diagnostics(
    const loom_low_schedule_table_t* table, iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->hazard_gap_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_hazard_gap(
        table, emitter, &table->hazard_gaps[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_diagnostics_emit(
    const loom_low_schedule_table_t* table,
    loom_low_schedule_diagnostic_flags_t flags,
    iree_diagnostic_emitter_t emitter) {
  if (emitter.fn == NULL) return iree_ok_status();
  if (table->error_count != 0) {
    return loom_low_schedule_emit_dependency_cycle(table, emitter,
                                                   &table->failure);
  }
  if (iree_any_bit_set(flags, LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_emit_pressure_diagnostics(
        table, emitter, &table->liveness));
  }
  if (iree_any_bit_set(flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_candidate_decision_diagnostics(table, emitter));
  }
  if (iree_any_bit_set(flags, LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_model_diagnostics(table, emitter));
  }
  if (iree_any_bit_set(flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_resource_diagnostics(table, emitter));
  }
  if (iree_any_bit_set(flags, LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_hazard_gap_diagnostics(table, emitter));
  }
  return iree_ok_status();
}
