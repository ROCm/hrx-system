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

static const char* loom_low_allocation_json_failure_blocking_kind_name(
    loom_low_allocation_failure_blocking_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET:
      return "interval-exceeds-budget";
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT:
      return "active-assignment";
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT:
      return "location-constraint";
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION:
      return "no-assignable-location";
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN:
    default:
      return "unknown";
  }
}

static bool loom_low_allocation_json_has_storage_lease_details(
    const loom_low_allocation_table_t* table) {
  return table->storage_leases.records != NULL ||
         table->storage_leases.record_count != 0 ||
         table->storage_lease_instance_count != 0 ||
         table->storage_release_action_count != 0;
}

static iree_status_t loom_low_allocation_json_write_u32_or_null(
    uint32_t value, loom_output_stream_t* stream) {
  if (value == UINT32_MAX) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
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

static iree_status_t loom_low_allocation_json_write_string_view_or_null(
    iree_string_view_t value, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_low_allocation_json_write_scalar_name_or_null(
    loom_scalar_type_t scalar_type, loom_output_stream_t* stream) {
  const char* name = loom_scalar_type_name(scalar_type);
  if (!name) return loom_output_stream_write_cstring(stream, "null");
  return loom_json_write_escaped_cstring(stream, name);
}

static iree_status_t loom_low_allocation_json_write_host_size_or_null(
    iree_host_size_t value, iree_host_size_t null_value,
    loom_output_stream_t* stream) {
  if (value == null_value) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%zu", value);
}

static const char* loom_low_allocation_json_storage_lease_kind_name(
    loom_low_storage_lease_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_STORAGE_LEASE_SOURCE_READ:
      return "source_read";
    case LOOM_LOW_STORAGE_LEASE_RESULT_WRITE:
      return "result_write";
    case LOOM_LOW_STORAGE_LEASE_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_storage_lease_attachment_name(
    loom_low_storage_lease_attachment_t attachment) {
  switch (attachment) {
    case LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND:
      return "operand";
    case LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT:
      return "result";
    case LOOM_LOW_STORAGE_LEASE_ATTACHMENT_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* loom_low_allocation_json_storage_lease_release_scope_name(
    loom_low_storage_lease_release_scope_t release_scope) {
  switch (release_scope) {
    case LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS:
      return "progress_class";
    case LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_UNKNOWN:
    default:
      return "unknown";
  }
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

static iree_status_t loom_low_allocation_json_write_location_parts(
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_allocation_location_kind_name(location_kind)));
  const char* base_name =
      location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT ? "slot"
                                                               : "base";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"%s\":%" PRIu32 ",\"count\":%" PRIu32 "}", base_name,
      location_base, location_count));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_location(
    const loom_low_allocation_assignment_t* assignment,
    loom_output_stream_t* stream) {
  return loom_low_allocation_json_write_location_parts(
      assignment->location_kind, assignment->location_base,
      assignment->location_count, stream);
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

static iree_status_t loom_low_allocation_json_write_register_class_or_null(
    const loom_low_allocation_table_t* table, uint16_t descriptor_reg_class_id,
    loom_output_stream_t* stream) {
  if (descriptor_reg_class_id >=
      table->target.descriptor_set->reg_class_count) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  const loom_low_reg_class_t* reg_class =
      &table->target.descriptor_set->reg_classes[descriptor_reg_class_id];
  return loom_json_write_escaped_string(
      stream, loom_low_descriptor_set_string(table->target.descriptor_set,
                                             reg_class->name_string_offset));
}

static iree_status_t loom_low_allocation_json_write_storage_lease_record(
    const loom_low_allocation_table_t* table, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_storage_lease_record_t* record =
      &table->storage_leases.records[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"packet\":", index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_host_size_or_null(
      record->packet_index, LOOM_LOW_STORAGE_LEASE_PACKET_NONE, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"node\":%" PRIu32 ",\"block\":%" PRIu32
      ",\"scheduled_ordinal\":%" PRIu32 ",\"kind\":%u,\"kind_name\":",
      record->node_index, record->block_index, record->scheduled_ordinal,
      (unsigned)record->kind));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_storage_lease_kind_name(record->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"attachment\":%u,\"attachment_name\":", (unsigned)record->attachment));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_storage_lease_attachment_name(
                  record->attachment)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"attachment_index\":%" PRIu16 ",\"unit_offset\":%" PRIu32
      ",\"unit_count\":%" PRIu32
      ",\"release_scope\":%u"
      ",\"release_scope_name\":",
      record->attachment_index, record->unit_offset, record->unit_count,
      (unsigned)record->release_scope));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_storage_lease_release_scope_name(
                  record->release_scope)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"release_class_id\":%" PRIu16 ",\"release_class_name\":",
      record->release_class_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      record->release_class_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"release_action_id\":%" PRIu16 ",\"release_action_name\":",
      record->release_action_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      record->release_action_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"release_reason_id\":%" PRIu16 ",\"release_reason_name\":",
      record->release_reason_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      record->release_reason_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"flags\":%" PRIu16
      ",\"starts_at_issue\":%s"
      ",\"release_before_boundary\":%s,\"may_carry_across_boundary\":%s"
      ",\"release_for_pressure\":%s}",
      record->flags,
      iree_all_bits_set(record->flags,
                        LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE)
          ? "true"
          : "false",
      iree_all_bits_set(record->flags,
                        LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY)
          ? "true"
          : "false",
      iree_all_bits_set(record->flags,
                        LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY)
          ? "true"
          : "false",
      iree_all_bits_set(record->flags,
                        LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE)
          ? "true"
          : "false"));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_json_write_storage_lease_instance(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_allocation_storage_lease_t* lease =
      &table->storage_lease_instances[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "{\"index\":%zu,\"lease_record\":%" PRIu32 ",\"assignment\":%" PRIu32
      ",\"value\":",
      index, lease->lease_record_index, lease->assignment_index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, lease->value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"start_point\":%" PRIu32 ",\"end_point\":%" PRIu32
      ",\"release_action\":",
      lease->start_point, lease->end_point));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_u32_or_null(
      lease->release_action_index, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"reg_class_id\":%" PRIu16 ",\"reg_class_name\":",
      lease->descriptor_reg_class_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_register_class_or_null(
      table, lease->descriptor_reg_class_id, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"location\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_location_parts(
      lease->location_kind, lease->location_base, lease->location_count,
      stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_allocation_json_write_storage_release_action(
    const loom_low_allocation_table_t* table, iree_host_size_t index,
    loom_output_stream_t* stream) {
  const loom_low_storage_release_action_t* action =
      &table->storage_release_actions[index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"insertion_packet\":", index));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_host_size_or_null(
      action->insertion_packet_index, LOOM_LOW_STORAGE_LEASE_PACKET_NONE,
      stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"insertion_node\":%" PRIu32 ",\"block\":%" PRIu32
      ",\"scheduled_ordinal\":%" PRIu32 ",\"release_class_id\":%" PRIu16
      ",\"release_class_name\":",
      action->insertion_node_index, action->block_index,
      action->scheduled_ordinal, action->release_class_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      action->release_class_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"release_action_id\":%" PRIu16 ",\"release_action_name\":",
      action->release_action_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      action->release_action_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"release_reason_id\":%" PRIu16 ",\"release_reason_name\":",
      action->release_reason_id));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_string_view_or_null(
      action->release_reason_name, stream));
  return loom_output_stream_write_format(
      stream,
      ",\"required_progress\":%" PRIu32 ",\"lease_record\":%" PRIu32 "}",
      action->required_progress, action->lease_record_index);
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

static iree_status_t loom_low_allocation_json_write_failure_location(
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_allocation_location_kind_name(location_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"base\":"));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_json_write_u32_or_null(location_base, stream));
  return loom_output_stream_write_format(stream, ",\"count\":%" PRIu32 "}",
                                         location_count);
}

static iree_status_t loom_low_allocation_json_write_failure(
    const loom_low_allocation_table_t* table,
    const loom_text_print_options_t* type_print_options,
    loom_json_object_writer_t* object) {
  if (!loom_low_allocation_failure_is_present(&table->failure)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("failure")));
  loom_output_stream_t* stream = object->stream;
  const loom_low_allocation_failure_t* failure = &table->failure;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"code\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, failure->failure_code));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"blocking_kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_allocation_json_failure_blocking_kind_name(
                  failure->blocking_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"value\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
      table, type_print_options, failure->value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"class\":"));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value_class(
      table, failure->value_class, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"start_point\":%" PRIu32 ",\"end_point\":%" PRIu32
      ",\"required_units\":%" PRIu32 ",\"budget_units\":",
      failure->start_point, failure->end_point, failure->required_unit_count));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_u32_or_null(
      failure->budget_units, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"peak_live_units\":%" PRIu32 ",\"location\":",
      failure->peak_live_units));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_failure_location(
      failure->location_kind, failure->location_base, failure->location_count,
      stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"conflict\":"));
  if (failure->conflict_value_id == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"assignment\":%" PRIu32 ",\"value\":",
        failure->conflict_assignment_index));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_value(
        table, type_print_options, failure->conflict_value_id, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"start_point\":%" PRIu32 ",\"end_point\":%" PRIu32 ",\"location\":",
        failure->conflict_start_point, failure->conflict_end_point));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_failure_location(
        failure->conflict_location_kind, failure->conflict_location_base,
        failure->conflict_location_count, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t loom_low_allocation_format_json(
    const loom_low_allocation_table_t* table, iree_string_builder_t* builder) {
  loom_low_descriptor_text_print_context_t type_print_context;
  loom_low_descriptor_text_print_context_initialize_for_set(
      table->target.descriptor_set, &type_print_context);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.low.allocation.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_allocation_json_function_name(table)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), table->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"), table->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("allocation_mode"),
      iree_make_cstring_view(
          loom_low_allocation_json_mode_name(table->allocation_mode))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("assignment_count"), table->assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("remark_count"), table->remark_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("copy_decision_count"), table->copy_decision_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("spill_count"), table->spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("spill_plan_count"), table->spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("coalesced_copy_count"), table->coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("materialized_copy_count"),
      table->materialized_copy_count));

  const bool has_storage_lease_details =
      loom_low_allocation_json_has_storage_lease_details(table);
  if (has_storage_lease_details) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("storage_lease_count"),
        table->storage_leases.record_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("storage_lease_instance_count"),
        table->storage_lease_instance_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("storage_release_action_count"),
        table->storage_release_action_count));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("assignments")));
  loom_json_array_writer_t assignments;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &assignments));
  for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&assignments));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_assignment(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&assignments));

  if (has_storage_lease_details) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("storage_leases")));
    loom_json_array_writer_t storage_leases;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &storage_leases));
    for (iree_host_size_t i = 0; i < table->storage_leases.record_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&storage_leases));
      IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_storage_lease_record(
          table, i, &stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&storage_leases));

    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("storage_lease_instances")));
    loom_json_array_writer_t storage_lease_instances;
    IREE_RETURN_IF_ERROR(
        loom_json_array_begin(&stream, &storage_lease_instances));
    for (iree_host_size_t i = 0; i < table->storage_lease_instance_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_json_array_begin_element(&storage_lease_instances));
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_json_write_storage_lease_instance(
              table, &type_print_context.options, i, &stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&storage_lease_instances));

    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("storage_release_actions")));
    loom_json_array_writer_t storage_release_actions;
    IREE_RETURN_IF_ERROR(
        loom_json_array_begin(&stream, &storage_release_actions));
    for (iree_host_size_t i = 0; i < table->storage_release_action_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_json_array_begin_element(&storage_release_actions));
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_json_write_storage_release_action(table, i,
                                                                &stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&storage_release_actions));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("spill_plans")));
  loom_json_array_writer_t spill_plans;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &spill_plans));
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&spill_plans));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_spill_plan(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&spill_plans));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("copy_decisions")));
  loom_json_array_writer_t copy_decisions;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &copy_decisions));
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&copy_decisions));
    IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_copy_decision(
        table, &type_print_context.options, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&copy_decisions));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("remarks")));
  loom_json_array_writer_t remarks;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &remarks));
  for (iree_host_size_t i = 0; i < table->remark_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&remarks));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_json_write_remark(table, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&remarks));
  IREE_RETURN_IF_ERROR(loom_low_allocation_json_write_failure(
      table, &type_print_context.options, &object));
  return loom_json_object_end(&object);
}
