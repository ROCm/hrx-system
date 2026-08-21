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
#include "loom/codegen/low/schedule/json.h"
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

static iree_string_view_t loom_low_packet_json_string_id_or_fallback(
    const loom_module_t* module, loom_string_id_t string_id, char* buffer,
    iree_host_size_t buffer_capacity) {
  if (string_id != LOOM_STRING_ID_INVALID &&
      string_id < module->strings.count) {
    return module->strings.entries[string_id];
  }
  int length = iree_snprintf(buffer, buffer_capacity, "<name:%" PRIu32 ">",
                             (uint32_t)string_id);
  return iree_make_string_view(buffer, length);
}

static iree_status_t loom_low_packet_json_write_string_view_or_null(
    iree_string_view_t value, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("assignment"), assignment_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_low_allocation_location_kind_name(assignment->location_kind)));
  const char* base_name =
      assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT
          ? "slot"
          : "base";
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, iree_make_cstring_view(base_name), assignment->location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("count"), assignment->location_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_low_packet_json_write_value(
    const loom_low_allocation_table_t* allocation,
    const loom_text_print_options_t* type_print_options,
    loom_value_id_t value_id, loom_output_stream_t* stream) {
  const loom_module_t* module = allocation->module;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("id"), value_id));
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("name")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        module, value->name_id, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("type")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_type(
        module, type_print_options, value->type, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("name")));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("type")));
  }
  uint32_t assignment_index = UINT32_MAX;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_try_map_active_value_assignment(allocation, value_id,
                                                          &assignment_index);
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("location")));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_location(
      assignment, assignment_index, stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_low_packet_json_write_value_array(
    const loom_low_allocation_table_t* allocation,
    const loom_text_print_options_t* type_print_options,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value(
        allocation, type_print_options, values[i], stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_low_packet_json_write_hazard_gaps(
    const loom_low_schedule_table_t* schedule, loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < schedule->hazard_gap_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    const loom_low_schedule_hazard_gap_t* hazard_gap =
        &schedule->hazard_gaps[i];
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_host_size_field(&object, IREE_SV("index"), i));
    const uint32_t producer_packet = loom_low_packet_hazard_gap_packet_index(
        schedule, hazard_gap, hazard_gap->producer_scheduled_ordinal);
    if (producer_packet == LOOM_LOW_PACKET_INDEX_NONE) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
          &object, IREE_SV("producer_packet")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("producer_packet"), producer_packet));
    }
    const uint32_t consumer_packet = loom_low_packet_hazard_gap_packet_index(
        schedule, hazard_gap, hazard_gap->consumer_scheduled_ordinal);
    if (consumer_packet == LOOM_LOW_PACKET_INDEX_NONE) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
          &object, IREE_SV("consumer_packet")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("consumer_packet"), consumer_packet));
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_hazard_gap_write_json_fields(hazard_gap, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("module"), symbol_ref.module_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("symbol"), symbol_ref.symbol_id));
  if (loom_symbol_ref_is_valid(symbol_ref) && symbol_ref.module_id == 0 &&
      symbol_ref.symbol_id < module->symbols.count) {
    const loom_symbol_t* symbol =
        &module->symbols.entries[symbol_ref.symbol_id];
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("name")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        module, symbol->name_id, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("name")));
  }
  return loom_json_object_end(&object);
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
  iree_string_view_t kind_name = IREE_SV("unknown");
  switch (predicate->arg_tags[arg_index]) {
    case LOOM_PRED_ARG_NONE:
      kind_name = IREE_SV("none");
      break;
    case LOOM_PRED_ARG_VALUE:
      kind_name = IREE_SV("value");
      break;
    case LOOM_PRED_ARG_CONST:
      kind_name = IREE_SV("const");
      break;
    default:
      break;
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("kind"), kind_name));
  if (predicate->arg_tags[arg_index] == LOOM_PRED_ARG_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("value")));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
        &object, IREE_SV("value"), predicate->args[arg_index]));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_low_packet_json_write_predicate_list_attr(
    const loom_attribute_t* attr, loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (uint16_t i = 0; i < attr->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    const loom_predicate_t* predicate = &attr->predicate_list[i];
    const char* kind_name = loom_predicate_kind_name(predicate->kind);
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("kind"),
        iree_make_cstring_view(kind_name ? kind_name : "unknown")));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("args")));
    loom_json_array_writer_t args;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &args));
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&args));
      IREE_RETURN_IF_ERROR(
          loom_low_packet_json_write_predicate_arg(predicate, j, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&args));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_low_packet_json_write_signed_enum_set_attr(
    const loom_attribute_t* attr, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  static const iree_string_view_t field_names[2] = {
      IREE_SVL("positive"),
      IREE_SVL("negative"),
  };
  for (iree_host_size_t polarity = 0; polarity < IREE_ARRAYSIZE(field_names);
       ++polarity) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, field_names[polarity]));
    loom_json_array_writer_t values;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &values));
    const uint64_t* words =
        attr->count > 0 ? attr->signed_enum_set_words + polarity * attr->count
                        : NULL;
    for (uint16_t word_index = 0; word_index < attr->count; ++word_index) {
      for (uint8_t bit_index = 0; bit_index < 64; ++bit_index) {
        if (!iree_any_bit_set(words[word_index], UINT64_C(1) << bit_index)) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_json_array_write_uint64_element(
            &values, (uint64_t)word_index * 64 + bit_index));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&values));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_low_packet_json_write_attr(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options,
    const loom_attribute_t* attr, loom_output_stream_t* stream, uint8_t depth) {
  if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute nesting exceeds %u",
                            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
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
    case LOOM_ATTR_ENUM_ARRAY: {
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_json_array_write_uint64_element(&array, attr->enum_array[i]));
      }
      return loom_json_array_end(&array);
    }
    case LOOM_ATTR_SIGNED_ENUM_SET:
      return loom_low_packet_json_write_signed_enum_set_attr(attr, stream);
    case LOOM_ATTR_I64_ARRAY: {
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_json_array_write_int64_element(&array, attr->i64_array[i]));
      }
      return loom_json_array_end(&array);
    }
    case LOOM_ATTR_SYMBOL:
      return loom_low_packet_json_write_symbol_attr(module, attr->symbol,
                                                    stream);
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_symbol_attr(
            module, attr->symbol_refs[i], stream));
      }
      return loom_json_array_end(&array);
    }
    case LOOM_ATTR_TYPE:
      return loom_low_packet_json_write_type_attr(module, type_print_options,
                                                  attr->type_id, stream);
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_low_packet_json_write_predicate_list_attr(attr, stream);
    case LOOM_ATTR_DICT: {
      loom_json_object_writer_t object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
      for (uint16_t i = 0; i < attr->count; ++i) {
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        char name_buffer[32];
        const iree_string_view_t name =
            loom_low_packet_json_string_id_or_fallback(
                module, entry->name_id, name_buffer, sizeof(name_buffer));
        IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, name));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
            module, type_print_options, &entry->value, stream,
            (uint8_t)(depth + 1)));
      }
      return loom_json_object_end(&object);
    }
    case LOOM_ATTR_PARAMETERIZED: {
      const loom_parameterized_attr_descriptor_t* descriptor =
          loom_context_resolve_parameterized_attr(
              module->context,
              (loom_parameterized_attr_kind_t)attr->reserved_1);
      loom_json_object_writer_t object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("family")));
      if (descriptor) {
        IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
            stream, loom_bstring_view(descriptor->name)));
      } else {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
      }
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("parameters")));
      loom_json_object_writer_t parameters;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &parameters));
      for (uint16_t i = 0; i < attr->count; ++i) {
        iree_string_view_t name = IREE_SV("<unknown>");
        if (descriptor && i < descriptor->parameter_count) {
          name =
              loom_attr_descriptor_name(&descriptor->parameter_descriptors[i]);
        }
        IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&parameters, name));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
            module, type_print_options, &attr->parameterized_slots[i], stream,
            (uint8_t)(depth + 1)));
      }
      IREE_RETURN_IF_ERROR(loom_json_object_end(&parameters));
      return loom_json_object_end(&object);
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
            module, type_print_options, &attr->parameterized_array[i], stream,
            (uint8_t)(depth + 1)));
      }
      return loom_json_array_end(&array);
    }
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(module, (uint16_t)attr->encoding_id);
      loom_json_object_writer_t object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("id"), attr->encoding_id));
      if (encoding) {
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&object, IREE_SV("name")));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
            module, encoding->name_id, stream));
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&object, IREE_SV("alias")));
        IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
            module, encoding->alias_id, stream));
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&object, IREE_SV("attributes")));
        loom_json_object_writer_t attributes;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &attributes));
        for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
          const loom_named_attr_t* entry = &encoding->attributes[i];
          char name_buffer[32];
          const iree_string_view_t name =
              loom_low_packet_json_string_id_or_fallback(
                  module, entry->name_id, name_buffer, sizeof(name_buffer));
          IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&attributes, name));
          IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
              module, type_print_options, &entry->value, stream,
              (uint8_t)(depth + 1)));
        }
        IREE_RETURN_IF_ERROR(loom_json_object_end(&attributes));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_json_object_write_null_field(&object, IREE_SV("name")));
        IREE_RETURN_IF_ERROR(
            loom_json_object_write_null_field(&object, IREE_SV("alias")));
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&object, IREE_SV("attributes")));
        loom_json_object_writer_t attributes;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &attributes));
        IREE_RETURN_IF_ERROR(loom_json_object_end(&attributes));
      }
      return loom_json_object_end(&object);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    char name_buffer[32];
    const iree_string_view_t name = loom_low_packet_json_string_id_or_fallback(
        module, entry->name_id, name_buffer, sizeof(name_buffer));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, name));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
        module, type_print_options, &entry->value, stream, 0));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_low_packet_json_write_generic_attrs(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options, const loom_op_t* op,
    loom_output_stream_t* stream) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (attrs[i].kind == LOOM_ATTR_ABSENT) {
      continue;
    }
    iree_string_view_t name;
    char name_buffer[16];
    if (vtable && vtable->attr_descriptors) {
      name = loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
    } else {
      int length =
          iree_snprintf(name_buffer, sizeof(name_buffer), "attr%" PRIu8, i);
      name = iree_make_string_view(name_buffer, length);
    }
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, name));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
        module, type_print_options, &attrs[i], stream, 0));
  }
  return loom_json_object_end(&object);
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
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint32_field(&object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("block")));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_json_write_block_ref(schedule, successors[i], stream));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_low_packet_json_write_low_packet_attrs(
    const loom_low_schedule_table_t* schedule,
    const loom_text_print_options_t* type_print_options,
    const loom_low_schedule_node_t* node,
    loom_json_object_writer_t* packet_object) {
  loom_output_stream_t* stream = packet_object->stream;
  const loom_module_t* module = schedule->module;
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  (void)loom_low_packet_try_op_attrs(node->op, &attrs, NULL);
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(packet_object, IREE_SV("attributes")));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_named_attrs(
      module, type_print_options, attrs, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(packet_object, IREE_SV("immediates")));
  loom_json_array_writer_t immediates;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &immediates));

  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor =
      descriptor_set ? node->descriptor : NULL;
  if (!descriptor) {
    return loom_json_array_end(&immediates);
  }

  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&immediates));
    const uint32_t immediate_index = descriptor->immediate_start + i;
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    iree_string_view_t name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    loom_json_object_writer_t immediate_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &immediate_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &immediate_object, IREE_SV("index"), i));
    if (iree_string_view_is_empty(name)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(&immediate_object,
                                                             IREE_SV("name")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &immediate_object, IREE_SV("name"), name));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &immediate_object, IREE_SV("kind"),
        loom_low_immediate_kind_name(immediate->kind)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &immediate_object, IREE_SV("bit_width"), immediate->bit_width));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &immediate_object, IREE_SV("flags"), immediate->flags));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &immediate_object, IREE_SV("encoding_id"), immediate->encoding_id));
    if (immediate->enum_domain_id == LOOM_LOW_ENUM_DOMAIN_NONE) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
          &immediate_object, IREE_SV("enum_domain_id")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &immediate_object, IREE_SV("enum_domain_id"),
          immediate->enum_domain_id));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&immediate_object, IREE_SV("value")));
    const loom_named_attr_t* attr =
        loom_low_packet_json_find_named_attr(module, attrs, name);
    if (attr) {
      IREE_RETURN_IF_ERROR(loom_low_packet_json_write_attr(
          module, type_print_options, &attr->value, stream, 0));
    } else if (iree_any_bit_set(immediate->flags,
                                LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "%" PRId64, immediate->default_value));
      IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
          &immediate_object, IREE_SV("defaulted"), true));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&immediate_object));
  }
  return loom_json_array_end(&immediates);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), packet->packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node"), packet->node_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block"), node->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_ordinal"), node->source_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("scheduled_ordinal"), node->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      iree_make_cstring_view(loom_low_packet_json_node_kind_name(node->kind))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("op"), loom_op_name(schedule->module, node->op)));
  if (descriptor) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("descriptor")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_descriptor_string_or_null(
        descriptor_set, descriptor->key_string_offset, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("descriptor")));
  }
  if (descriptor) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("mnemonic")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_descriptor_string_or_null(
        descriptor_set, descriptor->mnemonic_string_offset, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("mnemonic")));
  }
  if (descriptor) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("encoding_id"), descriptor->encoding_id));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("encoding_id")));
  }
  if (descriptor) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("descriptor_flags"), descriptor->flags));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
        &object, IREE_SV("descriptor_flags")));
  }
  const loom_low_schedule_class_t* schedule_class = node->schedule_class;
  iree_string_view_t schedule_class_name = iree_string_view_empty();
  if (schedule_class != NULL) {
    schedule_class_name = loom_low_descriptor_set_string(
        descriptor_set, schedule_class->name_string_offset);
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("schedule_class")));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_view_or_null(
      schedule_class_name, stream));
  const uint16_t latency_cycles =
      schedule_class ? schedule_class->latency_cycles : 0;
  const loom_low_latency_kind_t latency_kind =
      schedule_class ? schedule_class->latency_kind
                     : LOOM_LOW_LATENCY_KIND_UNKNOWN;
  const loom_low_model_quality_t model_quality =
      schedule_class ? schedule_class->model_quality
                     : LOOM_LOW_MODEL_QUALITY_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("latency_cycles"), latency_cycles));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("latency_kind"),
      loom_low_latency_kind_name(latency_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("model_quality"),
      loom_low_model_quality_name(model_quality)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issue_use_count"),
      schedule_class ? schedule_class->issue_use_count : 0));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("hazard_count"),
      schedule_class ? schedule_class->hazard_count : 0));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("effect_count"),
      descriptor ? descriptor->effect_count : 0));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("results")));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
      allocation, type_print_options, loom_op_const_results(node->op),
      node->op->result_count, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("operands")));
  IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
      allocation, type_print_options, loom_op_const_operands(node->op),
      node->op->operand_count, stream));
  if (loom_low_packet_try_op_attrs(node->op, NULL, NULL)) {
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_low_packet_attrs(
        schedule, type_print_options, node, &object));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("attributes")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_generic_attrs(
        schedule->module, type_print_options, node->op, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("immediates")));
    loom_json_array_writer_t immediates;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &immediates));
    IREE_RETURN_IF_ERROR(loom_json_array_end(&immediates));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("successors")));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_json_write_successors(schedule, node->op, stream));
  return loom_json_object_end(&object);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.low.packet.v0")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_packet_json_function_name(schedule)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), schedule->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"), schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("allocation_mode"),
      iree_make_cstring_view(loom_low_packet_json_allocation_mode_name(
          allocation->allocation_mode))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("block_count"), schedule->block_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("packet_count"), schedule->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("assignment_count"), allocation->assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("spill_count"), allocation->spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("hazard_gap_count"), schedule->hazard_gap_count));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("blocks")));
  loom_json_array_writer_t blocks;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &blocks));
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&blocks));
    const loom_low_schedule_block_t* block_record = &schedule->blocks[i];
    loom_json_object_writer_t block_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &block_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &block_object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&block_object, IREE_SV("label")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_string_id_or_null(
        schedule->module, block_record->block->label_id, &stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("packet_start"),
        block_record->scheduled_node_start));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &block_object, IREE_SV("packet_count"),
        block_record->scheduled_node_count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&block_object, IREE_SV("args")));
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_value_array(
        allocation, &type_print_context.options, block_record->block->arg_ids,
        block_record->block->arg_count, &stream));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&block_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&blocks));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("packets")));
  loom_json_array_writer_t packets;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &packets));
  for (iree_host_size_t i = 0; i < loom_low_packet_count(schedule); ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&packets));
    const loom_low_packet_view_t packet = loom_low_packet_at(schedule, i);
    IREE_RETURN_IF_ERROR(loom_low_packet_json_write_packet(
        schedule, allocation, &type_print_context.options, &packet, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&packets));

  if (schedule->hazard_gap_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("hazard_gaps")));
    IREE_RETURN_IF_ERROR(
        loom_low_packet_json_write_hazard_gaps(schedule, &stream));
  }

  return loom_json_object_end(&object);
}

iree_status_t loom_low_packet_format_json(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder) {
  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch));
  iree_status_t status =
      loom_low_packet_json_write(schedule, allocation, builder);
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
