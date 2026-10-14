// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/config/schema.h"

#include "loom/config/application.h"
#include "loom/format/text/printer.h"
#include "loom/ops/config/ops.h"
#include "loom/util/json.h"

static iree_status_t loom_config_write_printed_type_string(
    const loom_module_t* module, loom_type_t type,
    loom_output_stream_t* stream) {
  loom_json_string_writer_t string;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &string));
  IREE_RETURN_IF_ERROR(loom_text_print_type(type, module, &string.stream));
  return loom_json_string_end(&string);
}

static iree_status_t loom_config_write_printed_attribute_string(
    const loom_module_t* module, loom_attribute_t attribute,
    loom_output_stream_t* stream) {
  loom_json_string_writer_t string;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &string));
  IREE_RETURN_IF_ERROR(
      loom_text_print_attribute(&attribute, module, &string.stream));
  return loom_json_string_end(&string);
}

static iree_string_view_t loom_config_predicate_arg_kind_name(uint8_t tag) {
  switch ((loom_predicate_arg_tag_t)tag) {
    case LOOM_PRED_ARG_NONE:
      return IREE_SV("none");
    case LOOM_PRED_ARG_VALUE:
      return IREE_SV("value");
    case LOOM_PRED_ARG_CONST:
      return IREE_SV("const");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_config_format_predicate_arg_json(
    const loom_predicate_t* predicate, uint8_t arg_index,
    loom_output_stream_t* stream) {
  uint8_t tag = LOOM_PRED_ARG_NONE;
  int64_t value = 0;
  if (arg_index < predicate->arg_count) {
    tag = predicate->arg_tags[arg_index];
    value = predicate->args[arg_index];
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), loom_config_predicate_arg_kind_name(tag)));
  switch ((loom_predicate_arg_tag_t)tag) {
    case LOOM_PRED_ARG_VALUE: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
          &object, IREE_SV("value_id"), value));
      break;
    }
    case LOOM_PRED_ARG_CONST: {
      IREE_RETURN_IF_ERROR(
          loom_json_object_write_int64_field(&object, IREE_SV("value"), value));
      break;
    }
    default:
      break;
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_config_format_predicate_json(
    const loom_predicate_t* predicate, loom_output_stream_t* stream) {
  const char* kind_name = loom_predicate_kind_name(predicate->kind);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      kind_name ? iree_make_cstring_view(kind_name) : IREE_SV("unknown")));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("args")));
  loom_json_array_writer_t arguments;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &arguments));
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&arguments));
    IREE_RETURN_IF_ERROR(
        loom_config_format_predicate_arg_json(predicate, i, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&arguments));
  return loom_json_object_end(&object);
}

static iree_status_t loom_config_format_predicates_json(
    loom_attribute_t predicates, loom_output_stream_t* stream) {
  if (predicates.kind != LOOM_ATTR_ABSENT &&
      predicates.kind != LOOM_ATTR_PREDICATE_LIST) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config constraints must be a predicate list");
  }
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST && predicates.count > 0 &&
      !predicates.predicate_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config constraints predicate list is missing");
  }
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST) {
    for (uint16_t i = 0; i < predicates.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(loom_config_format_predicate_json(
          &predicates.predicate_list[i], stream));
    }
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_config_format_schema_entry_json(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_output_stream_t* stream) {
  loom_op_t* op = symbol->defining_op;
  const bool is_decl = loom_config_decl_isa(op);
  const bool is_def = loom_config_def_isa(op);
  if (!is_decl && !is_def) {
    const iree_string_view_t symbol_name =
        loom_config_symbol_name(module, symbol);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config symbol '@%.*s' is not a config.decl/def",
                            (int)symbol_name.size, symbol_name.data);
  }

  const loom_value_id_t config_value = loom_config_symbol_result_value(op);
  if (config_value == LOOM_VALUE_ID_INVALID ||
      config_value >= module->values.count) {
    const iree_string_view_t symbol_name =
        loom_config_symbol_name(module, symbol);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config symbol '@%.*s' has no result value",
                            (int)symbol_name.size, symbol_name.data);
  }
  const loom_type_t config_type = loom_module_value_type(module, config_value);

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("name"), loom_config_symbol_name(module, symbol)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"), is_decl ? IREE_SV("decl") : IREE_SV("def")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_bool_field(&object, IREE_SV("required"), is_decl));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("type")));
  IREE_RETURN_IF_ERROR(
      loom_config_write_printed_type_string(module, config_type, stream));
  if (is_def) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("default")));
    IREE_RETURN_IF_ERROR(loom_config_write_printed_attribute_string(
        module, loom_config_def_value(op), stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("constraints")));
  const loom_attribute_t predicates = is_decl ? loom_config_decl_predicates(op)
                                              : loom_config_def_predicates(op);
  IREE_RETURN_IF_ERROR(loom_config_format_predicates_json(predicates, stream));
  return loom_json_object_end(&object);
}

iree_status_t loom_config_format_schema_json(const loom_module_t* module,
                                             loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("configs")));
  loom_json_array_writer_t configs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &configs));
  iree_host_size_t config_count = 0;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_config_symbol_is_config(symbol)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&configs));
    IREE_RETURN_IF_ERROR(
        loom_config_format_schema_entry_json(module, symbol, stream));
    ++config_count;
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&configs));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), config_count));
  return loom_json_object_end(&object);
}
