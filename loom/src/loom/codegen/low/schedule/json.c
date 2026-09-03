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
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STATE:
      return "state";
    case LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE:
      return "storage";
    default:
      return "unknown";
  }
}

static const char* loom_low_schedule_json_failure_kind(
    loom_low_schedule_failure_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_SCHEDULE_FAILURE_NONE:
      return "none";
    case LOOM_LOW_SCHEDULE_FAILURE_DEPENDENCY_CYCLE:
      return "dependency_cycle";
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

static iree_status_t loom_low_schedule_json_write_nullable_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_string_field(object, name, value);
}

static iree_status_t loom_low_schedule_json_write_nullable_u16_field(
    loom_json_object_writer_t* object, iree_string_view_t name, uint16_t value,
    uint16_t null_value) {
  if (value == null_value) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_low_schedule_json_write_nullable_u32_field(
    loom_json_object_writer_t* object, iree_string_view_t name, uint32_t value,
    uint32_t null_value) {
  if (value == null_value) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
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

iree_status_t loom_low_schedule_hazard_gap_write_json_fields(
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    loom_json_object_writer_t* object) {
  const iree_string_view_t hazard_kind_name =
      loom_low_hazard_kind_name(hazard_gap->kind);
  const iree_string_view_t reference_kind_name =
      loom_low_hazard_reference_kind_name(hazard_gap->reference_kind);
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("producer_node"), hazard_gap->producer_node));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("consumer_node"), hazard_gap->consumer_node));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("block"), hazard_gap->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("producer_scheduled_ordinal"),
      hazard_gap->producer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("consumer_scheduled_ordinal"),
      hazard_gap->consumer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("producer_hazard_ordinal"),
      hazard_gap->producer_hazard_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("consumer_hazard_ordinal"),
      hazard_gap->consumer_hazard_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("kind"), (uint32_t)hazard_gap->kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("kind_name"), hazard_kind_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("reference_kind"), (uint32_t)hazard_gap->reference_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("reference_kind_name"), reference_kind_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("reference")));
  IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_hazard_reference(
      object->stream, hazard_gap->reference_kind, hazard_gap->reference_id,
      hazard_gap->resource_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("producer_stage"), hazard_gap->producer_stage));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("consumer_stage"), hazard_gap->consumer_stage));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("required_distance"), hazard_gap->required_distance));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("actual_distance"), hazard_gap->actual_distance));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      object, IREE_SV("required_delay"), hazard_gap->required_delay));
  return loom_json_object_write_uint32_field(object, IREE_SV("hazard_flags"),
                                             hazard_gap->hazard_flags);
}

static const iree_string_view_t kLoomLowScheduleRejectedMetricNames[] = {
    IREE_SVL("rejected_node"),
    IREE_SVL("rejected_dependency_latency_cycles"),
    IREE_SVL("rejected_latency_cycles"),
    IREE_SVL("rejected_pair_affinity_score"),
    IREE_SVL("rejected_projected_live_units"),
    IREE_SVL("rejected_killed_live_units"),
    IREE_SVL("rejected_produced_live_units"),
    IREE_SVL("rejected_data_ready_stall_cycles"),
    IREE_SVL("rejected_resource_stall_cycles"),
    IREE_SVL("rejected_hazard_stall_cycles"),
    IREE_SVL("rejected_completion_wait_cycles"),
    IREE_SVL("rejected_effective_stall_cycles"),
    IREE_SVL("rejected_bottleneck_resource_id"),
    IREE_SVL("rejected_pressure_cliff_penalty"),
    IREE_SVL("rejected_pressure_cliff_source"),
    IREE_SVL("rejected_pressure_cliff_units"),
    IREE_SVL("rejected_units_until_pressure_cliff"),
};

iree_status_t loom_low_schedule_format_json(
    const loom_low_schedule_table_t* table, iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.low.schedule.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_schedule_json_function_name(table)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), table->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"), table->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("block_count"), table->block_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("node_count"), table->node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("dependency_count"), table->dependencies.count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("dependency_group_count"),
      table->dependency_group_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unlock_summary_publication_count"),
      table->unlock_summary_publication_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("candidate_decision_count"),
      table->candidate_decision_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("resource_use_count"), table->resource_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("effect_use_count"), table->effect_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("hazard_use_count"), table->hazard_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("hazard_gap_count"), table->hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("model_summary_count"), table->model_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("resource_summary_count"),
      table->resource_summary_count));
  if (table->error_count != 0) {
    const loom_low_schedule_failure_t* failure = &table->failure;
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("error_count"), table->error_count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("failure")));
    loom_json_object_writer_t failure_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &failure_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &failure_object, IREE_SV("kind"),
        iree_make_cstring_view(
            loom_low_schedule_json_failure_kind(failure->kind))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &failure_object, IREE_SV("flags"), failure->flags));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &failure_object, IREE_SV("cycle_path_truncated"),
        iree_all_bits_set(
            failure->flags,
            LOOM_LOW_SCHEDULE_FAILURE_FLAG_CYCLE_PATH_TRUNCATED)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &failure_object, IREE_SV("witness_edge_only"),
        iree_all_bits_set(failure->flags,
                          LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &failure_object, IREE_SV("block"), failure->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &failure_object, IREE_SV("block_node_count"),
        failure->block_node_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &failure_object, IREE_SV("scheduled_node_count"),
        failure->scheduled_node_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &failure_object, IREE_SV("unscheduled_node_count"),
        failure->unscheduled_node_count));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &failure_object, IREE_SV("producer_node")));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
        &stream, failure->producer_node, LOOM_LOW_SCHEDULE_NODE_NONE));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &failure_object, IREE_SV("consumer_node")));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
        &stream, failure->consumer_node, LOOM_LOW_SCHEDULE_NODE_NONE));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &failure_object, IREE_SV("dependency_kind"),
        iree_make_cstring_view(
            loom_low_schedule_json_dependency_kind(failure->dependency_kind))));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&failure_object, IREE_SV("operand")));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_u32_or_null(
        &stream, failure->operand_index, UINT32_MAX));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&failure_object, IREE_SV("cycle_nodes")));
    loom_json_array_writer_t cycle_nodes;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &cycle_nodes));
    for (uint32_t i = 0; i < failure->cycle_node_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_write_uint32_element(
          &cycle_nodes, failure->cycle_nodes[i]));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&cycle_nodes));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&failure_object));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("blocks")));
  loom_json_array_writer_t blocks;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &blocks));
  for (iree_host_size_t i = 0; i < table->block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&blocks));
    const loom_low_schedule_block_t* block = &table->blocks[i];
    loom_json_object_writer_t block_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &block_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &block_object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("node_start"), block->node_start));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("node_count"), block->node_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("scheduled_node_start"),
        block->scheduled_node_start));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("scheduled_node_count"),
        block->scheduled_node_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&block_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&blocks));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
      &object, IREE_SV("source_order_boundaries")));
  loom_json_array_writer_t source_order_boundaries;
  IREE_RETURN_IF_ERROR(
      loom_json_array_begin(&stream, &source_order_boundaries));
  for (iree_host_size_t i = 0; i < table->node_count; ++i) {
    if (!iree_any_bit_set(table->nodes[i].flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_host_size_element(&source_order_boundaries, i));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&source_order_boundaries));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("nodes")));
  loom_json_array_writer_t nodes;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &nodes));
  for (iree_host_size_t i = 0; i < table->node_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&nodes));
    const loom_low_schedule_node_t* node = &table->nodes[i];
    loom_json_object_writer_t node_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &node_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &node_object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("block"), node->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("source_ordinal"), node->source_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("scheduled_ordinal"), node->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &node_object, IREE_SV("kind"),
        iree_make_cstring_view(loom_low_schedule_json_node_kind(node->kind))));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &node_object, IREE_SV("op"), loom_op_name(table->module, node->op)));
    iree_string_view_t descriptor_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_json_descriptor_key(table, node, &descriptor_key));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&node_object, IREE_SV("descriptor")));
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_json_write_nullable_string(&stream, descriptor_key));
    const loom_low_schedule_class_t* schedule_class = node->schedule_class;
    iree_string_view_t schedule_class_name = iree_string_view_empty();
    if (schedule_class != NULL) {
      schedule_class_name = loom_low_descriptor_set_string(
          table->target.descriptor_set, schedule_class->name_string_offset);
    }
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&node_object, IREE_SV("schedule_class")));
    IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string(
        &stream, schedule_class_name));
    const uint16_t latency_cycles =
        schedule_class ? schedule_class->latency_cycles : 0;
    const loom_low_latency_kind_t latency_kind =
        schedule_class ? schedule_class->latency_kind
                       : LOOM_LOW_LATENCY_KIND_UNKNOWN;
    const loom_low_model_quality_t model_quality =
        schedule_class ? schedule_class->model_quality
                       : LOOM_LOW_MODEL_QUALITY_UNKNOWN;
    const uint16_t issue_use_count =
        schedule_class ? schedule_class->issue_use_count : 0;
    const uint16_t hazard_count =
        schedule_class ? schedule_class->hazard_count : 0;
    const uint16_t effect_count =
        node->descriptor ? node->descriptor->effect_count : 0;
    iree_string_view_t latency_kind_name =
        loom_low_latency_kind_name(latency_kind);
    iree_string_view_t model_quality_name =
        loom_low_model_quality_name(model_quality);
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("latency_cycles"), latency_cycles));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("latency_kind"), (uint32_t)latency_kind));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &node_object, IREE_SV("latency_kind_name"), latency_kind_name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("model_quality"), (uint32_t)model_quality));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &node_object, IREE_SV("model_quality_name"), model_quality_name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("issue_use_count"), issue_use_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("hazard_count"), hazard_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &node_object, IREE_SV("effect_count"), effect_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&node_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&nodes));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("scheduled_node_indices")));
  loom_json_array_writer_t scheduled_node_indices;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &scheduled_node_indices));
  for (iree_host_size_t i = 0; i < table->scheduled_node_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_uint32_element(
        &scheduled_node_indices, table->scheduled_node_indices[i]));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&scheduled_node_indices));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("dependencies")));
  loom_json_array_writer_t dependencies;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &dependencies));
  for (uint32_t i = 0; i < table->dependencies.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&dependencies));
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&table->dependencies, i);
    loom_json_object_writer_t dependency_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &dependency_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &dependency_object, IREE_SV("from"), dependency->producer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &dependency_object, IREE_SV("to"), dependency->consumer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &dependency_object, IREE_SV("kind"),
        iree_make_cstring_view(
            loom_low_schedule_json_dependency_kind(dependency->kind))));
    if (dependency->operand_index == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
          &dependency_object, IREE_SV("operand")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &dependency_object, IREE_SV("operand"), dependency->operand_index));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&dependency_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&dependencies));

  if (table->pressure_step_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("scheduled_pressure_steps")));
    loom_json_array_writer_t pressure_steps;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &pressure_steps));
    for (iree_host_size_t i = 0; i < table->pressure_step_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&pressure_steps));
      const loom_low_schedule_pressure_step_t* step = &table->pressure_steps[i];
      loom_json_object_writer_t step_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &step_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &step_object, IREE_SV("node"), step->node_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &step_object, IREE_SV("block"), step->block_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &step_object, IREE_SV("scheduled_ordinal"), step->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &step_object, IREE_SV("live_units_before"), step->live_units_before));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &step_object, IREE_SV("killed_live_units"), step->killed_live_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &step_object, IREE_SV("produced_live_units"),
          step->produced_live_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &step_object, IREE_SV("live_units_after"), step->live_units_after));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&step_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&pressure_steps));
  }

  if (table->candidate_decision_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("candidate_decisions")));
    loom_json_array_writer_t candidate_decisions;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &candidate_decisions));
    for (iree_host_size_t i = 0; i < table->candidate_decision_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&candidate_decisions));
      const loom_low_schedule_candidate_decision_t* decision =
          &table->candidate_decisions[i];
      loom_json_object_writer_t decision_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &decision_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("block"), decision->block_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("scheduled_ordinal"),
          decision->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("ready_candidate_count"),
          decision->ready_candidate_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("scored_candidate_count"),
          decision->scored_candidate_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_node"), decision->chosen_node));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_dependency_latency_cycles"),
          decision->chosen_dependency_latency_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_latency_cycles"),
          decision->chosen_latency_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_pair_affinity_score"),
          decision->chosen_pair_affinity_score));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &decision_object, IREE_SV("chosen_projected_live_units"),
          decision->chosen_projected_live_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &decision_object, IREE_SV("chosen_killed_live_units"),
          decision->chosen_killed_live_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &decision_object, IREE_SV("chosen_produced_live_units"),
          decision->chosen_produced_live_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_data_ready_stall_cycles"),
          decision->chosen_data_ready_stall_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_resource_stall_cycles"),
          decision->chosen_resource_stall_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_hazard_stall_cycles"),
          decision->chosen_hazard_stall_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_completion_wait_cycles"),
          decision->chosen_completion_wait_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_effective_stall_cycles"),
          decision->chosen_effective_stall_cycles));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u16_field(
          &decision_object, IREE_SV("chosen_bottleneck_resource_id"),
          decision->chosen_bottleneck_resource_id, LOOM_LOW_RESOURCE_NONE));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &decision_object, IREE_SV("chosen_pressure_cliff_penalty"),
          decision->chosen_pressure_cliff_penalty));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string_field(
          &decision_object, IREE_SV("chosen_pressure_cliff_source"),
          decision->chosen_pressure_cliff_source));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u32_field(
          &decision_object, IREE_SV("chosen_pressure_cliff_units"),
          decision->chosen_pressure_cliff_units,
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u32_field(
          &decision_object, IREE_SV("chosen_units_until_pressure_cliff"),
          decision->chosen_units_until_pressure_cliff,
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
      if (decision->rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
        for (iree_host_size_t j = 0;
             j < IREE_ARRAYSIZE(kLoomLowScheduleRejectedMetricNames); ++j) {
          IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
              &decision_object, kLoomLowScheduleRejectedMetricNames[j]));
        }
      } else {
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_node"),
            decision->rejected_node));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_dependency_latency_cycles"),
            decision->rejected_dependency_latency_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_latency_cycles"),
            decision->rejected_latency_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_pair_affinity_score"),
            decision->rejected_pair_affinity_score));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &decision_object, IREE_SV("rejected_projected_live_units"),
            decision->rejected_projected_live_units));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &decision_object, IREE_SV("rejected_killed_live_units"),
            decision->rejected_killed_live_units));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &decision_object, IREE_SV("rejected_produced_live_units"),
            decision->rejected_produced_live_units));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_data_ready_stall_cycles"),
            decision->rejected_data_ready_stall_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_resource_stall_cycles"),
            decision->rejected_resource_stall_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_hazard_stall_cycles"),
            decision->rejected_hazard_stall_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_completion_wait_cycles"),
            decision->rejected_completion_wait_cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_effective_stall_cycles"),
            decision->rejected_effective_stall_cycles));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u16_field(
            &decision_object, IREE_SV("rejected_bottleneck_resource_id"),
            decision->rejected_bottleneck_resource_id, LOOM_LOW_RESOURCE_NONE));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &decision_object, IREE_SV("rejected_pressure_cliff_penalty"),
            decision->rejected_pressure_cliff_penalty));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_string_field(
            &decision_object, IREE_SV("rejected_pressure_cliff_source"),
            decision->rejected_pressure_cliff_source));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u32_field(
            &decision_object, IREE_SV("rejected_pressure_cliff_units"),
            decision->rejected_pressure_cliff_units,
            LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
        IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_nullable_u32_field(
            &decision_object, IREE_SV("rejected_units_until_pressure_cliff"),
            decision->rejected_units_until_pressure_cliff,
            LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE));
      }
      IREE_RETURN_IF_ERROR(loom_json_object_end(&decision_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&candidate_decisions));
  }

  if (table->resource_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("scheduled_resource_uses")));
    loom_json_array_writer_t resource_uses;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &resource_uses));
    const loom_low_descriptor_set_t* descriptor_set =
        table->target.descriptor_set;
    for (iree_host_size_t scheduled_index = 0;
         scheduled_index < table->scheduled_node_count; ++scheduled_index) {
      const uint32_t node_index =
          table->scheduled_node_indices[scheduled_index];
      const loom_low_schedule_node_t* node = &table->nodes[node_index];
      const loom_low_schedule_class_t* schedule_class = node->schedule_class;
      if (schedule_class == NULL) continue;
      for (uint16_t issue_use_ordinal = 0;
           issue_use_ordinal < schedule_class->issue_use_count;
           ++issue_use_ordinal) {
        const loom_low_issue_use_t* issue_use =
            &descriptor_set->issue_uses[schedule_class->issue_use_start +
                                        issue_use_ordinal];
        IREE_ASSERT_LT(issue_use->resource_id, descriptor_set->resource_count);
        const loom_low_resource_t* resource =
            &descriptor_set->resources[issue_use->resource_id];
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&resource_uses));
        loom_json_object_writer_t use_object;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &use_object));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("node"), node_index));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("block"), node->block_index));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("scheduled_ordinal"),
            node->scheduled_ordinal));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("issue_use_ordinal"), issue_use_ordinal));
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &use_object, IREE_SV("resource"),
            loom_low_descriptor_set_string(descriptor_set,
                                           resource->name_string_offset)));
        const iree_string_view_t resource_kind_name =
            loom_low_resource_kind_name(resource->kind);
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("resource_kind"), (uint32_t)resource->kind));
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &use_object, IREE_SV("resource_kind_name"), resource_kind_name));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("resource_flags"), resource->flags));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("capacity_per_cycle"),
            resource->capacity_per_cycle));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("contention_group"),
            resource->contention_group_id));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("stage"), issue_use->stage));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("cycles"), issue_use->cycles));
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &use_object, IREE_SV("units"), issue_use->units));
        IREE_RETURN_IF_ERROR(loom_json_object_end(&use_object));
      }
    }
    IREE_ASSERT_EQ(resource_uses.element_count, table->resource_use_count);
    IREE_RETURN_IF_ERROR(loom_json_array_end(&resource_uses));
  }

  if (table->effect_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("scheduled_effect_uses")));
    loom_json_array_writer_t effect_uses;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &effect_uses));
    for (iree_host_size_t i = 0; i < table->effect_use_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&effect_uses));
      const loom_low_schedule_effect_use_t* effect_use = &table->effect_uses[i];
      iree_string_view_t effect_kind_name =
          loom_low_effect_kind_name(effect_use->kind);
      iree_string_view_t memory_space_name =
          loom_low_memory_space_name(effect_use->memory_space);
      loom_json_object_writer_t effect_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &effect_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("node"), effect_use->node_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("block"), effect_use->block_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("scheduled_ordinal"),
          effect_use->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("effect_ordinal"),
          effect_use->effect_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("kind"), (uint32_t)effect_use->kind));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &effect_object, IREE_SV("kind_name"), effect_kind_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("memory_space"),
          (uint32_t)effect_use->memory_space));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &effect_object, IREE_SV("memory_space_name"), memory_space_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("scope"), effect_use->scope_id));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("effect_flags"), effect_use->effect_flags));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("counter"), effect_use->counter_id));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &effect_object, IREE_SV("width_bits"), effect_use->width_bits));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&effect_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&effect_uses));
  }

  if (table->hazard_use_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("scheduled_hazard_uses")));
    loom_json_array_writer_t hazard_uses;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &hazard_uses));
    for (iree_host_size_t i = 0; i < table->hazard_use_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&hazard_uses));
      const loom_low_schedule_hazard_use_t* hazard_use = &table->hazard_uses[i];
      iree_string_view_t hazard_kind_name =
          loom_low_hazard_kind_name(hazard_use->kind);
      iree_string_view_t reference_kind_name =
          loom_low_hazard_reference_kind_name(hazard_use->reference_kind);
      loom_json_object_writer_t hazard_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &hazard_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("node"), hazard_use->node_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("block"), hazard_use->block_index));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("scheduled_ordinal"),
          hazard_use->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("hazard_ordinal"),
          hazard_use->hazard_ordinal));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("kind"), (uint32_t)hazard_use->kind));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &hazard_object, IREE_SV("kind_name"), hazard_kind_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("reference_kind"),
          (uint32_t)hazard_use->reference_kind));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &hazard_object, IREE_SV("reference_kind_name"), reference_kind_name));
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&hazard_object, IREE_SV("reference")));
      IREE_RETURN_IF_ERROR(loom_low_schedule_json_write_hazard_reference(
          &stream, hazard_use->reference_kind, hazard_use->reference_id,
          hazard_use->resource_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("producer_stage"),
          hazard_use->producer_stage));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("consumer_stage"),
          hazard_use->consumer_stage));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("distance"), hazard_use->distance));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &hazard_object, IREE_SV("hazard_flags"), hazard_use->hazard_flags));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&hazard_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&hazard_uses));
  }

  if (table->hazard_gap_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("scheduled_hazard_gaps")));
    loom_json_array_writer_t hazard_gaps;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &hazard_gaps));
    for (iree_host_size_t i = 0; i < table->hazard_gap_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&hazard_gaps));
      const loom_low_schedule_hazard_gap_t* hazard_gap = &table->hazard_gaps[i];
      loom_json_object_writer_t gap_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &gap_object));
      IREE_RETURN_IF_ERROR(loom_low_schedule_hazard_gap_write_json_fields(
          hazard_gap, &gap_object));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&gap_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&hazard_gaps));
  }

  if (table->model_summary_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("model_summaries")));
    loom_json_array_writer_t model_summaries;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &model_summaries));
    for (iree_host_size_t i = 0; i < table->model_summary_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&model_summaries));
      const loom_low_schedule_model_summary_t* summary =
          &table->model_summaries[i];
      loom_json_object_writer_t summary_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &summary_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &summary_object, IREE_SV("schedule_class"),
          summary->schedule_class_name));
      iree_string_view_t latency_kind_name =
          loom_low_latency_kind_name(summary->latency_kind);
      iree_string_view_t model_quality_name =
          loom_low_model_quality_name(summary->model_quality);
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("first_node"), summary->first_node));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("use_count"), summary->use_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("latency_cycles"), summary->latency_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("latency_kind"),
          (uint32_t)summary->latency_kind));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &summary_object, IREE_SV("latency_kind_name"), latency_kind_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("model_quality"),
          (uint32_t)summary->model_quality));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &summary_object, IREE_SV("model_quality_name"), model_quality_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("issue_use_count"),
          summary->issue_use_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("hazard_count"), summary->hazard_count));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&summary_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&model_summaries));
  }

  if (table->resource_summary_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("resource_summaries")));
    loom_json_array_writer_t resource_summaries;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &resource_summaries));
    for (iree_host_size_t i = 0; i < table->resource_summary_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&resource_summaries));
      const loom_low_schedule_resource_summary_t* summary =
          &table->resource_summaries[i];
      loom_json_object_writer_t summary_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &summary_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &summary_object, IREE_SV("resource"), summary->resource_name));
      iree_string_view_t resource_kind_name =
          loom_low_resource_kind_name(summary->resource_kind);
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("resource_kind"),
          (uint32_t)summary->resource_kind));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &summary_object, IREE_SV("resource_kind_name"), resource_kind_name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("resource_flags"), summary->resource_flags));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("capacity_per_cycle"),
          summary->capacity_per_cycle));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("contention_group"),
          summary->contention_group_id));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("use_count"), summary->use_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &summary_object, IREE_SV("total_occupied_cycles"),
          summary->total_occupied_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &summary_object, IREE_SV("total_unit_cycles"),
          summary->total_unit_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &summary_object, IREE_SV("estimated_min_cycles"),
          summary->estimated_min_cycles));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &summary_object, IREE_SV("peak_units_per_cycle"),
          summary->peak_units_per_cycle));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&summary_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&resource_summaries));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("liveness")));
  if (table->liveness.region != NULL) {
    loom_low_descriptor_text_print_context_t type_print_context;
    loom_low_descriptor_text_print_context_initialize_for_set(
        table->target.descriptor_set, &type_print_context);
    IREE_RETURN_IF_ERROR(loom_liveness_format_json(
        &table->liveness, &type_print_context.options, builder));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
  }
  return loom_json_object_end(&object);
}
