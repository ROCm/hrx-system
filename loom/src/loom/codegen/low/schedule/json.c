// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/json.h"

#include <inttypes.h>

#include "loom/analysis/liveness_json.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_low_schedule_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_schedule_json_function_name(
    const loom_low_schedule_table_t* table) {
  if (loom_low_function_def_isa(table->function_op)) {
    return loom_low_schedule_json_symbol_name(
        table->module, loom_low_function_callee(table->function_op));
  }
  return IREE_SV("<unnamed>");
}

static const char* loom_low_schedule_json_node_kind(
    loom_low_schedule_node_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_NODE_STRUCTURAL:
      return "structural";
    case LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR:
      return "descriptor";
    case LOOM_LOW_SCHEDULE_NODE_TERMINATOR:
      return "terminator";
    default:
      return "unknown";
  }
}

static const char* loom_low_schedule_json_dependency_kind(
    loom_low_schedule_dependency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_DEPENDENCY_SSA:
      return "ssa";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT:
      return "effect";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL:
      return "control";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR:
      return "anchor";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STATE:
      return "state";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE:
      return "storage";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_schedule_json_write_nullable_string(
    loom_output_stream_t* stream, iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_low_schedule_json_write_u16_or_null(
    loom_output_stream_t* stream, uint16_t value, uint16_t null_value) {
  if (value == null_value) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu16, value);
}

static iree_status_t loom_low_schedule_json_write_u32_or_null(
    loom_output_stream_t* stream, uint32_t value, uint32_t null_value) {
  if (value == null_value) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_low_schedule_json_write_hazard_reference(
    loom_output_stream_t* stream, loom_low_hazard_reference_kind_t kind,
    uint16_t reference_id, iree_string_view_t resource_name) {
  if (kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE) {
    return loom_low_schedule_json_write_nullable_string(stream, resource_name);
  }
  return loom_output_stream_write_format(stream, "%" PRIu16, reference_id);
}

static iree_status_t loom_low_schedule_json_descriptor_key(
    const loom_low_schedule_table_t* table,
    const loom_low_schedule_node_t* node, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  if (node->descriptor == NULL) {
    return iree_ok_status();
  }
  *out_key = loom_low_descriptor_set_string(
      table->target.descriptor_set, node->descriptor->key_string_offset);
  return iree_ok_status();
}

iree_status_t loom_low_schedule_format_json(
    const loom_low_schedule_table_t* table, iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.low.schedule.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_schedule_json_function_name(table)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, table->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, table->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"block_count\":%zu,\"node_count\":%zu,\"dependency_count\":%zu"
      ",\"candidate_decision_count\":%zu"
      ",\"resource_use_count\":%zu,\"effect_use_count\":%zu"
      ",\"hazard_use_count\":%zu"
      ",\"hazard_gap_count\":%zu"
      ",\"model_summary_count\":%zu,\"resource_summary_count\":%zu",
      table->block_count, table->node_count, table->dependency_count,
      table->candidate_decision_count, table->resource_use_count,
      table->effect_use_count, table->hazard_use_count, table->hazard_gap_count,
      table->model_summary_count, table->resource_summary_count));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"blocks\":["));
  for (iree_host_size_t i = 0; i < table->block_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_block_t* block = &table->blocks[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        "{\"index\":%zu,\"node_start\":%" PRIu32 ",\"node_count\":%" PRIu32
        ",\"scheduled_node_start\":%" PRIu32
        ",\"scheduled_node_count\":%" PRIu32 "}",
        i, block->node_start, block->node_count, block->scheduled_node_start,
        block->scheduled_node_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"nodes\":["));
  for (iree_host_size_t i = 0; i < table->node_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_node_t* node = &table->nodes[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        "{\"index\":%zu,\"block\":%" PRIu32 ",\"source_ordinal\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"kind\":",
        i, node->block_index, node->source_ordinal, node->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        &stream, loom_low_schedule_json_node_kind(node->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"op\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, loom_op_name(table->module, node->op)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"descriptor\":"));
    iree_string_view_t descriptor_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_json_descriptor_key(table, node, &descriptor_key));
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_json_write_nullable_string(&stream, descriptor_key));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"schedule_class\":"));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string(
        &stream, node->schedule_class_name));
    iree_string_view_t latency_kind_name =
        loom_low_latency_kind_name(node->latency_kind);
    iree_string_view_t model_quality_name =
        loom_low_model_quality_name(node->model_quality);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"latency_cycles\":%" PRIu16
        ",\"latency_kind\":%u,\"latency_kind_name\":\"%.*s\""
        ",\"model_quality\":%u,\"model_quality_name\":\"%.*s\""
        ",\"issue_use_count\":%" PRIu16 ",\"hazard_count\":%" PRIu16
        ",\"effect_count\":%" PRIu16 "}",
        node->latency_cycles, (unsigned)node->latency_kind,
        (int)latency_kind_name.size, latency_kind_name.data,
        (unsigned)node->model_quality, (int)model_quality_name.size,
        model_quality_name.data, node->issue_use_count, node->hazard_count,
        node->effect_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, ",\"scheduled_node_indices\":["));
  for (iree_host_size_t i = 0; i < table->scheduled_node_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, "%" PRIu32, table->scheduled_node_indices[i]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"dependencies\":["));
  for (iree_host_size_t i = 0; i < table->dependency_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_low_schedule_dependency_t* dependency = &table->dependencies[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, "{\"from\":%" PRIu32 ",\"to\":%" PRIu32 ",\"kind\":",
        dependency->producer_node, dependency->consumer_node));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        &stream, loom_low_schedule_json_dependency_kind(dependency->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"operand\":"));
    if (dependency->operand_index == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "%" PRIu32, dependency->operand_index));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  if (table->pressure_step_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_pressure_steps\":["));
    for (iree_host_size_t i = 0; i < table->pressure_step_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_pressure_step_t* step = &table->pressure_steps[i];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"node\":%" PRIu32 ",\"block\":%" PRIu32
          ",\"scheduled_ordinal\":%" PRIu32 ",\"live_units_before\":%" PRIu64
          ",\"killed_live_units\":%" PRIu64 ",\"produced_live_units\":%" PRIu64
          ",\"live_units_after\":%" PRIu64 "}",
          step->node_index, step->block_index, step->scheduled_ordinal,
          step->live_units_before, step->killed_live_units,
          step->produced_live_units, step->live_units_after));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->candidate_decision_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"candidate_decisions\":["));
    for (iree_host_size_t i = 0; i < table->candidate_decision_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_candidate_decision_t* decision =
          &table->candidate_decisions[i];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"block\":%" PRIu32 ",\"scheduled_ordinal\":%" PRIu32
          ",\"candidate_count\":%" PRIu32 ",\"chosen_node\":%" PRIu32
          ",\"chosen_dependency_latency_cycles\":%" PRIu16
          ",\"chosen_latency_cycles\":%" PRIu16
          ",\"chosen_pair_affinity_score\":%" PRIu16
          ",\"chosen_projected_live_units\":%" PRIu64
          ",\"chosen_killed_live_units\":%" PRIu64
          ",\"chosen_produced_live_units\":%" PRIu64
          ",\"chosen_data_ready_stall_cycles\":%" PRIu32
          ",\"chosen_resource_stall_cycles\":%" PRIu32
          ",\"chosen_hazard_stall_cycles\":%" PRIu32
          ",\"chosen_effective_stall_cycles\":%" PRIu32
          ",\"chosen_bottleneck_resource_id\":",
          decision->block_index, decision->scheduled_ordinal,
          decision->ready_candidate_count, decision->chosen_node,
          decision->chosen_dependency_latency_cycles,
          decision->chosen_latency_cycles, decision->chosen_pair_affinity_score,
          decision->chosen_projected_live_units,
          decision->chosen_killed_live_units,
          decision->chosen_produced_live_units,
          decision->chosen_data_ready_stall_cycles,
          decision->chosen_resource_stall_cycles,
          decision->chosen_hazard_stall_cycles,
          decision->chosen_effective_stall_cycles));
      if (decision->chosen_bottleneck_resource_id == LOOM_LOW_RESOURCE_NONE) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
      } else {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            &stream, "%" PRIu16, decision->chosen_bottleneck_resource_id));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"chosen_pressure_cliff_penalty\":%" PRIu32,
          decision->chosen_pressure_cliff_penalty));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          &stream, ",\"chosen_pressure_cliff_reg_class_id\":"));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u16_or_null(
          &stream, decision->chosen_pressure_cliff_reg_class_id,
          LOOM_LOW_REG_CLASS_NONE));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          &stream, ",\"chosen_pressure_cliff_units\":"));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
          &stream, decision->chosen_pressure_cliff_units,
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          &stream, ",\"chosen_units_until_pressure_cliff\":"));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
          &stream, decision->chosen_units_until_pressure_cliff,
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(&stream, ",\"rejected_node\":"));
      if (decision->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream,
            "null,\"rejected_dependency_latency_cycles\":null"
            ",\"rejected_latency_cycles\":null"
            ",\"rejected_pair_affinity_score\":null"
            ",\"rejected_projected_live_units\":null"
            ",\"rejected_killed_live_units\":null"
            ",\"rejected_produced_live_units\":null"
            ",\"rejected_data_ready_stall_cycles\":null"
            ",\"rejected_resource_stall_cycles\":null"
            ",\"rejected_hazard_stall_cycles\":null"
            ",\"rejected_effective_stall_cycles\":null"
            ",\"rejected_bottleneck_resource_id\":null"
            ",\"rejected_pressure_cliff_penalty\":null"
            ",\"rejected_pressure_cliff_reg_class_id\":null"
            ",\"rejected_pressure_cliff_units\":null"
            ",\"rejected_units_until_pressure_cliff\":null}"));
      } else {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            &stream,
            "%" PRIu32 ",\"rejected_dependency_latency_cycles\":%" PRIu16
            ",\"rejected_latency_cycles\":%" PRIu16
            ",\"rejected_pair_affinity_score\":%" PRIu16
            ",\"rejected_projected_live_units\":%" PRIu64
            ",\"rejected_killed_live_units\":%" PRIu64
            ",\"rejected_produced_live_units\":%" PRIu64
            ",\"rejected_data_ready_stall_cycles\":%" PRIu32
            ",\"rejected_resource_stall_cycles\":%" PRIu32
            ",\"rejected_hazard_stall_cycles\":%" PRIu32
            ",\"rejected_effective_stall_cycles\":%" PRIu32
            ",\"rejected_bottleneck_resource_id\":",
            decision->rejected_node,
            decision->rejected_dependency_latency_cycles,
            decision->rejected_latency_cycles,
            decision->rejected_pair_affinity_score,
            decision->rejected_projected_live_units,
            decision->rejected_killed_live_units,
            decision->rejected_produced_live_units,
            decision->rejected_data_ready_stall_cycles,
            decision->rejected_resource_stall_cycles,
            decision->rejected_hazard_stall_cycles,
            decision->rejected_effective_stall_cycles));
        if (decision->rejected_bottleneck_resource_id ==
            LOOM_LOW_RESOURCE_NONE) {
          IREE_RETURN_IF_ERROR(
              loom_output_stream_write_cstring(&stream, "null"));
        } else {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              &stream, "%" PRIu16, decision->rejected_bottleneck_resource_id));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            &stream, ",\"rejected_pressure_cliff_penalty\":%" PRIu32,
            decision->rejected_pressure_cliff_penalty));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"rejected_pressure_cliff_reg_class_id\":"));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u16_or_null(
            &stream, decision->rejected_pressure_cliff_reg_class_id,
            LOOM_LOW_REG_CLASS_NONE));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"rejected_pressure_cliff_units\":"));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
            &stream, decision->rejected_pressure_cliff_units,
            LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"rejected_units_until_pressure_cliff\":"));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
            &stream, decision->rejected_units_until_pressure_cliff,
            LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->resource_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_resource_uses\":["));
    for (iree_host_size_t i = 0; i < table->resource_use_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_resource_use_t* resource_use =
          &table->resource_uses[i];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"node\":%" PRIu32 ",\"block\":%" PRIu32
          ",\"scheduled_ordinal\":%" PRIu32 ",\"issue_use_ordinal\":%" PRIu16
          ",\"resource\":",
          resource_use->node_index, resource_use->block_index,
          resource_use->scheduled_ordinal, resource_use->issue_use_ordinal));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(&stream, resource_use->resource_name));
      iree_string_view_t resource_kind_name =
          loom_low_resource_kind_name(resource_use->resource_kind);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          ",\"resource_kind\":%u,\"resource_kind_name\":"
          "\"%.*s\",\"resource_flags\":%" PRIu16
          ",\"capacity_per_cycle\":%" PRIu16 ",\"contention_group\":%" PRIu16
          ",\"stage\":%" PRIu16 ",\"cycles\":%" PRIu16 ",\"units\":%" PRIu16
          "}",
          (unsigned)resource_use->resource_kind, (int)resource_kind_name.size,
          resource_kind_name.data, resource_use->resource_flags,
          resource_use->capacity_per_cycle, resource_use->contention_group_id,
          resource_use->stage, resource_use->cycles, resource_use->units));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->effect_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_effect_uses\":["));
    for (iree_host_size_t i = 0; i < table->effect_use_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_effect_use_t* effect_use = &table->effect_uses[i];
      iree_string_view_t effect_kind_name =
          loom_low_effect_kind_name(effect_use->kind);
      iree_string_view_t memory_space_name =
          loom_low_memory_space_name(effect_use->memory_space);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"node\":%" PRIu32 ",\"block\":%" PRIu32
          ",\"scheduled_ordinal\":%" PRIu32 ",\"effect_ordinal\":%" PRIu16
          ",\"kind\":%u,\"kind_name\":\"%.*s\",\"memory_space\":%u"
          ",\"memory_space_name\":\"%.*s\",\"scope\":%" PRIu16
          ",\"effect_flags\":%" PRIu16 ",\"counter\":%" PRIu16
          ",\"width_bits\":%" PRIu16 "}",
          effect_use->node_index, effect_use->block_index,
          effect_use->scheduled_ordinal, effect_use->effect_ordinal,
          (unsigned)effect_use->kind, (int)effect_kind_name.size,
          effect_kind_name.data, (unsigned)effect_use->memory_space,
          (int)memory_space_name.size, memory_space_name.data,
          effect_use->scope_id, effect_use->effect_flags,
          effect_use->counter_id, effect_use->width_bits));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->hazard_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_hazard_uses\":["));
    for (iree_host_size_t i = 0; i < table->hazard_use_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_hazard_use_t* hazard_use = &table->hazard_uses[i];
      iree_string_view_t hazard_kind_name =
          loom_low_hazard_kind_name(hazard_use->kind);
      iree_string_view_t reference_kind_name =
          loom_low_hazard_reference_kind_name(hazard_use->reference_kind);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"node\":%" PRIu32 ",\"block\":%" PRIu32
          ",\"scheduled_ordinal\":%" PRIu32 ",\"hazard_ordinal\":%" PRIu16
          ",\"kind\":%u,\"kind_name\":\"%.*s\",\"reference_kind\":%u"
          ",\"reference_kind_name\":\"%.*s\",\"reference\":",
          hazard_use->node_index, hazard_use->block_index,
          hazard_use->scheduled_ordinal, hazard_use->hazard_ordinal,
          (unsigned)hazard_use->kind, (int)hazard_kind_name.size,
          hazard_kind_name.data, (unsigned)hazard_use->reference_kind,
          (int)reference_kind_name.size, reference_kind_name.data));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_hazard_reference(
          &stream, hazard_use->reference_kind, hazard_use->reference_id,
          hazard_use->resource_name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          ",\"producer_stage\":%" PRIu16 ",\"consumer_stage\":%" PRIu16
          ",\"distance\":%" PRIu16 ",\"hazard_flags\":%" PRIu16 "}",
          hazard_use->producer_stage, hazard_use->consumer_stage,
          hazard_use->distance, hazard_use->hazard_flags));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->hazard_gap_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"scheduled_hazard_gaps\":["));
    for (iree_host_size_t i = 0; i < table->hazard_gap_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_hazard_gap_t* hazard_gap = &table->hazard_gaps[i];
      iree_string_view_t hazard_kind_name =
          loom_low_hazard_kind_name(hazard_gap->kind);
      iree_string_view_t reference_kind_name =
          loom_low_hazard_reference_kind_name(hazard_gap->reference_kind);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"producer_node\":%" PRIu32 ",\"consumer_node\":%" PRIu32
          ",\"block\":%" PRIu32 ",\"producer_scheduled_ordinal\":%" PRIu32
          ",\"consumer_scheduled_ordinal\":%" PRIu32
          ",\"producer_hazard_ordinal\":%" PRIu16
          ",\"consumer_hazard_ordinal\":%" PRIu16
          ",\"kind\":%u,\"kind_name\":\"%.*s\",\"reference_kind\":%u"
          ",\"reference_kind_name\":\"%.*s\",\"reference\":",
          hazard_gap->producer_node, hazard_gap->consumer_node,
          hazard_gap->block_index, hazard_gap->producer_scheduled_ordinal,
          hazard_gap->consumer_scheduled_ordinal,
          hazard_gap->producer_hazard_ordinal,
          hazard_gap->consumer_hazard_ordinal, (unsigned)hazard_gap->kind,
          (int)hazard_kind_name.size, hazard_kind_name.data,
          (unsigned)hazard_gap->reference_kind, (int)reference_kind_name.size,
          reference_kind_name.data));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_hazard_reference(
          &stream, hazard_gap->reference_kind, hazard_gap->reference_id,
          hazard_gap->resource_name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          ",\"producer_stage\":%" PRIu16 ",\"consumer_stage\":%" PRIu16
          ",\"required_distance\":%" PRIu16 ",\"actual_distance\":%" PRIu32
          ",\"required_delay\":%" PRIu16 ",\"hazard_flags\":%" PRIu16 "}",
          hazard_gap->producer_stage, hazard_gap->consumer_stage,
          hazard_gap->required_distance, hazard_gap->actual_distance,
          hazard_gap->required_delay, hazard_gap->hazard_flags));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->model_summary_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"model_summaries\":["));
    for (iree_host_size_t i = 0; i < table->model_summary_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_model_summary_t* summary =
          &table->model_summaries[i];
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(&stream, "{\"schedule_class\":"));
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
          &stream, summary->schedule_class_name));
      iree_string_view_t latency_kind_name =
          loom_low_latency_kind_name(summary->latency_kind);
      iree_string_view_t model_quality_name =
          loom_low_model_quality_name(summary->model_quality);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          ",\"first_node\":%" PRIu32 ",\"use_count\":%" PRIu32
          ",\"latency_cycles\":%" PRIu16
          ",\"latency_kind\":%u,\"latency_kind_name\":\"%.*s\""
          ",\"model_quality\":%u,\"model_quality_name\":\"%.*s\""
          ",\"issue_use_count\":%" PRIu16 ",\"hazard_count\":%" PRIu16 "}",
          summary->first_node, summary->use_count, summary->latency_cycles,
          (unsigned)summary->latency_kind, (int)latency_kind_name.size,
          latency_kind_name.data, (unsigned)summary->model_quality,
          (int)model_quality_name.size, model_quality_name.data,
          summary->issue_use_count, summary->hazard_count));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  if (table->resource_summary_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"resource_summaries\":["));
    for (iree_host_size_t i = 0; i < table->resource_summary_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const loom_low_schedule_resource_summary_t* summary =
          &table->resource_summaries[i];
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(&stream, "{\"resource\":"));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(&stream, summary->resource_name));
      iree_string_view_t resource_kind_name =
          loom_low_resource_kind_name(summary->resource_kind);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          ",\"resource_kind\":%u,\"resource_kind_name\":"
          "\"%.*s\",\"resource_flags\":%" PRIu16
          ",\"capacity_per_cycle\":%" PRIu16 ",\"contention_group\":%" PRIu16
          ",\"use_count\":%" PRIu32 ",\"total_occupied_cycles\":%" PRIu64
          ",\"total_unit_cycles\":%" PRIu64 ",\"estimated_min_cycles\":%" PRIu64
          ",\"peak_units_per_cycle\":%" PRIu16 "}",
          (unsigned)summary->resource_kind, (int)resource_kind_name.size,
          resource_kind_name.data, summary->resource_flags,
          summary->capacity_per_cycle, summary->contention_group_id,
          summary->use_count, summary->total_occupied_cycles,
          summary->total_unit_cycles, summary->estimated_min_cycles,
          summary->peak_units_per_cycle));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"liveness\":"));
  loom_low_descriptor_text_print_context_t type_print_context;
  loom_low_descriptor_text_print_context_initialize_for_set(
      table->target.descriptor_set, &type_print_context);
  IREE_RETURN_IF_ERROR(loom_liveness_format_json(
      &table->liveness, &type_print_context.options, builder));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  return iree_ok_status();
}
