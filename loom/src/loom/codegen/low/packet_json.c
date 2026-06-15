// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_json.h"

#include <inttypes.h>
#include <math.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_low_packet_json_symbol_name(
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

static iree_string_view_t loom_low_packet_json_function_name(
    const loom_low_schedule_table_t* schedule) {
  if (loom_low_function_def_isa(schedule->function_op)) {
    return loom_low_packet_json_symbol_name(
        schedule->module, loom_low_function_callee(schedule->function_op));
  }
  return IREE_SV("<unnamed>");
}

static const char* loom_low_packet_json_allocation_mode_name(uint8_t mode) {
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

static const char* loom_low_packet_json_node_kind_name(
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

static iree_status_t loom_low_packet_json_write_string_id_or_null(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream,
                                        module->strings.entries[string_id]);
}

static iree_status_t loom_low_packet_json_write_string_id_or_fallback(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id != LOOM_STRING_ID_INVALID &&
      string_id < module->strings.count) {
    return loom_json_write_escaped_string(stream,
                                          module->strings.entries[string_id]);
  }
  char buffer[32];
  int length = iree_snprintf(buffer, sizeof(buffer), "<name:%" PRIu32 ">",
                             (uint32_t)string_id);
  return loom_json_write_escaped_string(stream,
                                        iree_make_string_view(buffer, length));
}

static iree_status_t loom_low_packet_json_write_string_view_or_null(
    iree_string_view_t value, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_low_packet_json_write_hazard_reference(
    loom_output_stream_t* stream, loom_low_hazard_reference_kind_t kind,
    uint16_t reference_id, iree_string_view_t resource_name) {
  if (kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE) {
    return loom_low_packet_json_write_string_view_or_null(resource_name,
                                                          stream);
  }
  return loom_output_stream_write_format(stream, "%" PRIu16, reference_id);
}

static iree_status_t loom_low_packet_json_write_nullable_u16(
    uint16_t value, uint16_t absent_value, loom_output_stream_t* stream) {
  if (value == absent_value) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu16, value);
}

static iree_status_t loom_low_packet_json_write_nullable_u32(
    uint32_t value, uint32_t absent_value, loom_output_stream_t* stream) {
  if (value == absent_value) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_low_packet_json_write_type(
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

static iree_status_t loom_low_packet_json_write_location(
    const loom_low_allocation_assignment_t* assignment,
    iree_host_size_t assignment_index, loom_output_stream_t* stream) {
  if (!assignment) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"assignment\":%zu,\"kind\":", assignment_index));
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

static iree_status_t loom_low_packet_json_write_value(
    const loom_low_allocation_table_t* allocation,
    const loom_text_print_options_t* type_print_options,
    loom_value_id_t value_id, loom_output_stream_t* stream) {
  const loom_module_t* module = allocation->module;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"id\":%" PRIu32 ",\"name\":", value_id));
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        module, value->name_id, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"type\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_type(
        module, type_print_options, value->type, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "null,\"type\":null"));
  }
  uint32_t assignment_index = UINT32_MAX;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                      &assignment_index);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"location\":"));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_location(
      assignment, assignment_index, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_low_packet_json_write_value_array(
    const loom_low_allocation_table_t* allocation,
    const loom_text_print_options_t* type_print_options,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value(
        allocation, type_print_options, values[i], stream));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_low_packet_json_write_hazard_gaps(
    const loom_low_schedule_table_t* schedule, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < schedule->hazard_gap_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_low_schedule_hazard_gap_t* hazard_gap =
        &schedule->hazard_gaps[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"index\":%zu,\"producer_packet\":", i));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_nullable_u32(
        loom_low_packet_hazard_gap_packet_index(
            schedule, hazard_gap, hazard_gap->producer_scheduled_ordinal),
        LOOM_LOW_PACKET_INDEX_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"consumer_packet\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_nullable_u32(
        loom_low_packet_hazard_gap_packet_index(
            schedule, hazard_gap, hazard_gap->consumer_scheduled_ordinal),
        LOOM_LOW_PACKET_INDEX_NONE, stream));
    iree_string_view_t hazard_kind_name =
        loom_low_hazard_kind_name(hazard_gap->kind);
    iree_string_view_t reference_kind_name =
        loom_low_hazard_reference_kind_name(hazard_gap->reference_kind);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"producer_node\":%" PRIu32 ",\"consumer_node\":%" PRIu32
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
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_hazard_reference(
        stream, hazard_gap->reference_kind, hazard_gap->reference_id,
        hazard_gap->resource_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"producer_stage\":%" PRIu16 ",\"consumer_stage\":%" PRIu16
        ",\"required_distance\":%" PRIu16 ",\"actual_distance\":%" PRIu32
        ",\"required_delay\":%" PRIu16 ",\"hazard_flags\":%" PRIu16 "}",
        hazard_gap->producer_stage, hazard_gap->consumer_stage,
        hazard_gap->required_distance, hazard_gap->actual_distance,
        hazard_gap->required_delay, hazard_gap->hazard_flags));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_low_packet_json_write_f64(
    double value, loom_output_stream_t* stream) {
  if (!isfinite(value)) {
    return loom_json_write_escaped_cstring(
        stream, isnan(value) ? "nan" : (value < 0.0 ? "-inf" : "inf"));
  }
  char buffer[32];
  int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", value);
  return loom_output_stream_write(stream,
                                  iree_make_string_view(buffer, length));
}

static iree_status_t loom_low_packet_json_write_symbol_attr(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"module\":%" PRIu16 ",\"symbol\":%" PRIu16 ",\"name\":",
      symbol_ref.module_id, symbol_ref.symbol_id));
  if (loom_symbol_ref_is_valid(symbol_ref) && symbol_ref.module_id == 0 &&
      symbol_ref.symbol_id < module->symbols.count) {
    const loom_symbol_t* symbol =
        &module->symbols.entries[symbol_ref.symbol_id];
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        module, symbol->name_id, stream));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_low_packet_json_write_type_attr(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options, loom_type_id_t type_id,
    loom_output_stream_t* stream) {
  if (type_id < module->types.count) {
    return loom_low_packet_json_write_type(
        module, type_print_options, module->types.entries[type_id], stream);
  }
  char buffer[32];
  int length =
      iree_snprintf(buffer, sizeof(buffer), "type<%" PRIu32 ">", type_id);
  return loom_json_write_escaped_string(stream,
                                        iree_make_string_view(buffer, length));
}

static iree_status_t loom_low_packet_json_write_predicate_arg(
    const loom_predicate_t* predicate, uint8_t arg_index,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  switch (predicate->arg_tags[arg_index]) {
    case LOOM_PRED_ARG_NONE: {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, "none"));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ",\"value\":null}"));
      return iree_ok_status();
    }
    case LOOM_PRED_ARG_VALUE: {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, "value"));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"value\":%" PRId64 "}", predicate->args[arg_index]));
      return iree_ok_status();
    }
    case LOOM_PRED_ARG_CONST: {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, "const"));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"value\":%" PRId64 "}", predicate->args[arg_index]));
      return iree_ok_status();
    }
    default: {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, "unknown"));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"value\":%" PRId64 "}", predicate->args[arg_index]));
      return iree_ok_status();
    }
  }
}

static iree_status_t loom_low_packet_json_write_predicate_list_attr(
    const loom_attribute_t* attr, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (uint16_t i = 0; i < attr->count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_predicate_t* predicate = &attr->predicate_list[i];
    const char* kind_name = loom_predicate_kind_name(predicate->kind);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\":"));
    if (kind_name) {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, kind_name));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, "unknown"));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"args\":["));
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      if (j > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
      }
      IREE_RETURN_IF_ERROR(
          loom_low_packet_json_write_predicate_arg(predicate, j, stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_low_packet_json_write_attr(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options,
    const loom_attribute_t* attr, loom_output_stream_t* stream, uint8_t depth) {
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute nesting exceeds %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  switch (attr->kind) {
    case LOOM_ATTR_ABSENT:
      return loom_output_stream_write_cstring(stream, "null");
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, attr->i64);
    case LOOM_ATTR_F64:
      return loom_low_packet_json_write_f64(attr->f64, stream);
    case LOOM_ATTR_STRING:
      return loom_low_packet_json_write_string_id_or_null(
          module, attr->string_id, stream);
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(stream,
                                              attr->raw ? "true" : "false");
    case LOOM_ATTR_ENUM:
      return loom_output_stream_write_format(stream, "%" PRIu64, attr->raw);
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "%" PRId64, attr->i64_array[i]));
      }
      return loom_output_stream_write_char(stream, ']');
    }
    case LOOM_ATTR_SYMBOL:
      return loom_low_packet_json_write_symbol_attr(module, attr->symbol,
                                                    stream);
    case LOOM_ATTR_TYPE:
      return loom_low_packet_json_write_type_attr(module, type_print_options,
                                                  attr->type_id, stream);
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_low_packet_json_write_predicate_list_attr(attr, stream);
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
        }
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_fallback(
            module, entry->name_id, stream));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ':'));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
            module, type_print_options, &entry->value, stream,
            (uint8_t)(depth + 1)));
      }
      return loom_output_stream_write_char(stream, '}');
    }
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(module, (uint16_t)attr->encoding_id);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "{\"id\":%" PRIu32 ",\"name\":", attr->encoding_id));
      if (encoding) {
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
            module, encoding->name_id, stream));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, ",\"alias\":"));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
            module, encoding->alias_id, stream));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, ",\"attributes\":{"));
        for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
          if (i > 0) {
            IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
          }
          const loom_named_attr_t* entry = &encoding->attributes[i];
          IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_fallback(
              module, entry->name_id, stream));
          IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ':'));
          IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
              module, type_print_options, &entry->value, stream,
              (uint8_t)(depth + 1)));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream,
                                             "null,\"alias\":null,"
                                             "\"attributes\":{}"));
      }
      return loom_output_stream_write_char(stream, '}');
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %u",
                              (unsigned)attr->kind);
  }
}

static const loom_named_attr_t* loom_low_packet_json_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id < module->strings.count &&
        iree_string_view_equal(module->strings.entries[attr->name_id], name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_low_packet_json_write_named_attrs(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options,
    loom_named_attr_slice_t attrs, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_named_attr_t* entry = &attrs.entries[i];
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_fallback(
        module, entry->name_id, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ':'));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
        module, type_print_options, &entry->value, stream, 0));
  }
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_low_packet_json_write_generic_attrs(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options, const loom_op_t* op,
    loom_output_stream_t* stream) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '{'));
  bool did_emit_attr = false;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (attrs[i].kind == LOOM_ATTR_ABSENT) {
      continue;
    }
    if (did_emit_attr) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    if (vtable && vtable->attr_descriptors) {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
          stream, loom_attr_descriptor_name(&vtable->attr_descriptors[i])));
    } else {
      char buffer[16];
      int length =
          iree_snprintf(buffer, sizeof(buffer), "attr%" PRIu8, (uint8_t)i);
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
          stream, iree_make_string_view(buffer, length)));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ':'));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
        module, type_print_options, &attrs[i], stream, 0));
    did_emit_attr = true;
  }
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_low_packet_json_write_descriptor_string_or_null(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, loom_output_stream_t* stream) {
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  return loom_low_packet_json_write_string_view_or_null(value, stream);
}

static iree_status_t loom_low_packet_json_write_block_ref(
    const loom_low_schedule_table_t* schedule, const loom_block_t* block,
    loom_output_stream_t* stream) {
  uint32_t block_index = loom_low_packet_block_index(schedule, block);
  if (block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, block_index);
}

static iree_status_t loom_low_packet_json_write_successors(
    const loom_low_schedule_table_t* schedule, const loom_op_t* op,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"index\":%" PRIu8 ",\"block\":", i));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_json_write_block_ref(schedule, successors[i], stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_low_packet_json_write_low_packet_attrs(
    const loom_low_schedule_table_t* schedule,
    const loom_text_print_options_t* type_print_options,
    const loom_low_schedule_node_t* node, loom_output_stream_t* stream) {
  const loom_module_t* module = schedule->module;
  loom_named_attr_slice_t attrs = {0};
  if (loom_low_op_isa(node->op)) {
    attrs = loom_low_op_attrs(node->op);
  } else if (loom_low_const_isa(node->op)) {
    attrs = loom_low_const_attrs(node->op);
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"attributes\":"));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_named_attrs(
      module, type_print_options, attrs, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"immediates\":"));

  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor =
      descriptor_set ? node->descriptor : NULL;
  if (!descriptor) {
    return loom_output_stream_write_cstring(stream, "[]");
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const uint32_t immediate_index = descriptor->immediate_start + i;
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    iree_string_view_t name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"index\":%" PRIu16 ",\"name\":", i));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_json_write_string_view_or_null(name, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_low_immediate_kind_name(immediate->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"bit_width\":%" PRIu16 ",\"flags\":%" PRIu16
        ",\"encoding_id\":%" PRIu16 ",\"enum_domain_id\":",
        immediate->bit_width, immediate->flags, immediate->encoding_id));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_nullable_u16(
        immediate->enum_domain_id, LOOM_LOW_ENUM_DOMAIN_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"value\":"));
    const loom_named_attr_t* attr =
        loom_low_packet_json_find_named_attr(module, attrs, name);
    if (attr) {
      IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
          module, type_print_options, &attr->value, stream, 0));
    } else if (iree_any_bit_set(immediate->flags,
                                LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "%" PRId64 ",\"defaulted\":true", immediate->default_value));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_low_packet_json_write_packet(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_text_print_options_t* type_print_options,
    const loom_low_packet_view_t* packet, loom_output_stream_t* stream) {
  const loom_low_schedule_node_t* node = packet->node;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "{\"index\":%zu,\"node\":%" PRIu32 ",\"block\":%" PRIu32
      ",\"source_ordinal\":%" PRIu32 ",\"scheduled_ordinal\":%" PRIu32
      ",\"kind\":",
      packet->packet_index, packet->node_index, node->block_index,
      node->source_ordinal, node->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_low_packet_json_node_kind_name(node->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"op\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_op_name(schedule->module, node->op)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"descriptor\":"));
  if (descriptor) {
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_descriptor_string_or_null(
        descriptor_set, descriptor->key_string_offset, stream));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"mnemonic\":"));
  if (descriptor) {
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_descriptor_string_or_null(
        descriptor_set, descriptor->mnemonic_string_offset, stream));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"encoding_id\":"));
  if (descriptor) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%" PRIu16, descriptor->encoding_id));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"descriptor_flags\":"));
  if (descriptor) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "%" PRIu16, descriptor->flags));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"schedule_class\":"));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_view_or_null(
      node->schedule_class_name, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"latency_cycles\":%" PRIu16 ",\"latency_kind\":",
      node->latency_cycles));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_latency_kind_name(node->latency_kind)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"model_quality\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_low_model_quality_name(node->model_quality)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"issue_use_count\":%" PRIu16 ",\"hazard_count\":%" PRIu16
      ",\"effect_count\":%" PRIu16 ",\"results\":",
      node->issue_use_count, node->hazard_count, node->effect_count));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
      allocation, type_print_options, loom_op_const_results(node->op),
      node->op->result_count, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"operands\":"));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
      allocation, type_print_options, loom_op_const_operands(node->op),
      node->op->operand_count, stream));
  if (loom_low_op_isa(node->op) || loom_low_const_isa(node->op)) {
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_low_packet_attrs(
        schedule, type_print_options, node, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"attributes\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_generic_attrs(
        schedule->module, type_print_options, node->op, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"immediates\":[]"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"successors\":"));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_json_write_successors(schedule, node->op, stream));
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_low_packet_json_write(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder) {
  loom_low_descriptor_text_print_context_t type_print_context;
  loom_low_descriptor_text_print_context_initialize_for_set(
      allocation->target.descriptor_set, &type_print_context);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.low.packet.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_packet_json_function_name(schedule)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"allocation_mode\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream,
      loom_low_packet_json_allocation_mode_name(allocation->allocation_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"block_count\":%zu,\"packet_count\":%zu,\"assignment_count\":%zu"
      ",\"spill_count\":%zu,\"hazard_gap_count\":%zu",
      schedule->block_count, schedule->scheduled_node_count,
      allocation->assignment_count, allocation->spill_count,
      schedule->hazard_gap_count));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"blocks\":["));
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    const loom_low_schedule_block_t* block_record = &schedule->blocks[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, "{\"index\":%zu,\"label\":", i));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        schedule->module, block_record->block->label_id, &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"packet_start\":%" PRIu32 ",\"packet_count\":%" PRIu32 ",\"args\":",
        block_record->scheduled_node_start,
        block_record->scheduled_node_count));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
        allocation, &type_print_context.options, block_record->block->arg_ids,
        block_record->block->arg_count, &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '}'));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ']'));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"packets\":["));
  for (iree_host_size_t i = 0; i < loom_low_packet_count(schedule); ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    loom_low_packet_view_t packet;
    IREE_RETURN_IF_ERROR(
        loom_low_packet_view_at(schedule, allocation, i, &packet));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_packet(
        schedule, allocation, &type_print_context.options, &packet, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ']'));

  if (schedule->hazard_gap_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hazard_gaps\":"));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_json_write_hazard_gaps(schedule, &stream));
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '}'));
  return iree_ok_status();
}

iree_status_t loom_low_packet_format_json(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));

  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch));
  iree_status_t status =
      loom_low_packet_json_write(schedule, allocation, builder);
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
