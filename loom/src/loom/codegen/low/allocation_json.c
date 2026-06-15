// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_json.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_low_allocation_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) return IREE_SV("<unnamed>");
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_allocation_json_function_name(
    const loom_low_allocation_table_t* table) {
  if (loom_low_function_def_isa(table->function_op)) {
    return loom_low_allocation_json_symbol_name(
        table->module, loom_low_function_callee(table->function_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_allocation_json_type_kind_name(
    loom_type_kind_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_STORAGE:
      return IREE_SV("storage");
    case LOOM_TYPE_COUNT_:
      break;
  }
  return IREE_SV("unknown");
}

static const char* loom_low_allocation_json_mode_name(uint8_t mode) {
  switch (mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return "virtual";
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return "assigned";
    case LOOM_LOW_ALLOCATION_FIXED:
      return "fixed";
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_remark_kind_name(
    loom_low_allocation_remark_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_REMARK_SPILL:
      return "spill";
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_copy_kind_name(
    loom_low_allocation_copy_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
      return "coalesced";
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return "materialized";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_allocation_json_write_string_or_null(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream,
                                        module->strings.entries[string_id]);
}

static iree_status_t loom_low_allocation_json_write_scalar_name_or_null(
    loom_scalar_type_t scalar_type, loom_output_stream_t* stream) {
  const char* name = loom_scalar_type_name(scalar_type);
  if (!name) return loom_output_stream_write_cstring(stream, "null");
  return loom_json_write_escaped_cstring(stream, name);
}

static iree_status_t loom_low_allocation_json_write_type(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options, loom_type_t type,
    loom_output_stream_t* stream) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_text_print_type_with_options(
      type, module, &escape_stream, type_print_options));
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_low_allocation_json_write_value(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options,
    loom_value_id_t value_id, loom_output_stream_t* stream) {
  const loom_module_t* module = table->module;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"id\":%u,\"name\":", (unsigned)value_id));
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_or_null(
        module, value->name_id, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"type\":"));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_type(
        module, type_print_options, value->type, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "null,\"type\":null"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_value_class(
    const loom_low_allocation_table_t* table,
    loom_liveness_value_class_t value_class, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"type_kind\":%u,\"type_kind_name\":",
      (unsigned)value_class.type_kind));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_allocation_json_type_kind_name(value_class.type_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"element_type\":%u,\"element_type_name\":",
      (unsigned)value_class.element_type));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_scalar_name_or_null(
      value_class.element_type, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"register_class\":"));
  if (value_class.type_kind == LOOM_TYPE_REGISTER &&
      value_class.register_descriptor_set_stable_id ==
          table->target.descriptor_set->stable_id &&
      value_class.register_class_id <
          table->target.descriptor_set->reg_class_count) {
    const loom_low_reg_class_t* reg_class =
        &table->target.descriptor_set
             ->reg_classes[value_class.register_class_id];
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_low_descriptor_set_string(table->target.descriptor_set,
                                               reg_class->name_string_offset)));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_location(
    const loom_low_allocation_assignment_t* assignment,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream,
      loom_low_allocation_location_kind_name(assignment->location_kind)));
  const char* base_name =
      assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT
          ? "slot"
          : "base";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"%s\":%" PRIu32 ",\"count\":%" PRIu32 "}", base_name,
      assignment->location_base, assignment->location_count));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_assignment(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"value\":", index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, assignment->value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"class\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value_class(
      table, assignment->value_class, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"start_point\":%" PRIu32 ",\"end_point\":%" PRIu32
      ",\"unit_count\":%" PRIu32 ",\"location\":",
      assignment->start_point, assignment->end_point, assignment->unit_count));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_json_write_location(assignment, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_spill_plan(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_spill_plan_t* spill_plan =
      &table->spill_plans[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "{\"index\":%zu,\"assignment\":%" PRIu32 ",\"slot\":%" PRIu32
      ",\"space\":",
      index, spill_plan->assignment_index, spill_plan->slot_index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_spill_slot_space_name(spill_plan->slot_space)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, spill_plan->value_id, stream));
  return loom_output_stream_write_format(
      stream,
      ",\"byte_size\":%" PRIu32 ",\"byte_alignment\":%" PRIu32
      ",\"store_count\":%" PRIu32 ",\"reload_count\":%" PRIu32 "}",
      spill_plan->byte_size, spill_plan->byte_alignment,
      spill_plan->store_count, spill_plan->reload_count);
}

static iree_status_t loom_low_allocation_json_write_remark(
    const loom_low_allocation_table_t* table, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_remark_t* remark = &table->remarks[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_remark_kind_name(remark->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"assignment\":%" PRIu32 ",\"budget_units\":",
      remark->assignment_index));
  if (remark->budget_units == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, "%" PRIu32,
                                                         remark->budget_units));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"required_units\":%" PRIu32 "}", remark->required_units));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_copy_decision(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_copy_decision_t* copy_decision =
      &table->copy_decisions[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"kind\":", index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_copy_kind_name(copy_decision->kind)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"source_value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, copy_decision->source_value_id, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"result_value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, copy_decision->result_value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"source_assignment\":%" PRIu32 ",\"result_assignment\":%" PRIu32 "}",
      copy_decision->source_assignment_index,
      copy_decision->result_assignment_index));
  return iree_ok_status();
}

iree_status_t loom_low_allocation_format_json(
    const loom_low_allocation_table_t* table, iree_string_builder_t* builder) {
  loom_low_descriptor_text_print_context_t type_print_context;
  loom_low_descriptor_text_print_context_initialize_for_set(
      table->target.descriptor_set, &type_print_context);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.low.allocation.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_allocation_json_function_name(table)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, table->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, table->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"allocation_mode\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, loom_low_allocation_json_mode_name(table->allocation_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"assignment_count\":%zu,\"remark_count\":%zu"
      ",\"copy_decision_count\":%zu,\"spill_count\":%zu"
      ",\"spill_plan_count\":%zu"
      ",\"coalesced_copy_count\":%zu,\"materialized_copy_count\":%zu",
      table->assignment_count, table->remark_count, table->copy_decision_count,
      table->spill_count, table->spill_plan_count, table->coalesced_copy_count,
      table->materialized_copy_count));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"assignments\":["));
  for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_assignment(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"spill_plans\":["));
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_spill_plan(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"copy_decisions\":["));
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_copy_decision(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"remarks\":["));
  for (iree_host_size_t i = 0; i < table->remark_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_remark(table, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  return iree_ok_status();
}
