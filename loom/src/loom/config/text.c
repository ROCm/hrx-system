// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/config/text.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/api.h"
#include "loom/config/application.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/attribute.h"
#include "loom/ir/encoding.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/encoding/roles.h"
#include "loom/util/stream.h"

void loom_config_text_materialize_options_initialize(
    loom_config_text_materialize_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

static iree_status_t loom_config_text_parse_encoding_value(
    loom_module_t* module, loom_type_t type, iree_string_view_t value_text,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    loom_attribute_t* out_attr) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  iree_status_t status = iree_string_builder_append_cstring(
      &builder, "config.def @__config_value = ");
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, value_text);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, " : ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_print_type(type, module, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }

  loom_module_t* parsed_module = NULL;
  if (iree_status_is_ok(status)) {
    loom_text_parse_options_t parse_options = {
        .max_errors = 20,
    };
    status = loom_text_parse(iree_string_builder_view(&builder),
                             IREE_SV("<config>"), module->context, block_pool,
                             &parse_options, &parsed_module);
  }
  if (iree_status_is_ok(status) && !parsed_module) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to parse config encoding value '%.*s'",
                              (int)value_text.size, value_text.data);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* op = loom_module_block(parsed_module)->first_op;
    if (!op || !loom_config_def_isa(op)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "failed to parse config encoding value");
    } else {
      loom_type_t remapped_type = {0};
      status = loom_config_remap_type_and_value(
          parsed_module, module,
          loom_module_value_type(parsed_module, loom_config_def_type(op)),
          loom_config_def_value(op), block_pool, &remapped_type, out_attr);
      if (iree_status_is_ok(status) && !loom_type_equal(remapped_type, type)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parsed config encoding value has an incompatible type");
      }
    }
  }

  if (parsed_module) {
    loom_module_free(parsed_module);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_config_text_check_encoding_role(
    const loom_module_t* module, iree_string_view_t key, loom_type_t type,
    loom_attribute_t value) {
  loom_encoding_role_t expected_role = loom_type_encoding_role(type);
  if (expected_role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return iree_ok_status();
  }
  const loom_encoding_t* encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(value));
  loom_encoding_role_t actual_role =
      loom_encoding_static_role(module, encoding);
  if (actual_role == expected_role) {
    return iree_ok_status();
  }
  iree_string_view_t expected = loom_encoding_role_description(expected_role);
  iree_string_view_t actual = loom_encoding_role_description(actual_role);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' expects %.*s, got %.*s", (int)key.size,
                          key.data, (int)expected.size, expected.data,
                          (int)actual.size, actual.data);
}

static iree_status_t loom_config_text_parse_scalar_value(
    iree_string_view_t key, loom_type_t type, iree_string_view_t value_text,
    loom_attribute_t* out_attr) {
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    if (iree_string_view_equal_case(value_text, IREE_SV("true"))) {
      *out_attr = loom_attr_bool(true);
      return iree_ok_status();
    }
    if (iree_string_view_equal_case(value_text, IREE_SV("false"))) {
      *out_attr = loom_attr_bool(false);
      return iree_ok_status();
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(value_text, &value) ||
        (value != 0 && value != 1)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config '%.*s' expects i1 value true/false/0/1, got '%.*s'",
          (int)key.size, key.data, (int)value_text.size, value_text.data);
    }
    *out_attr = loom_attr_bool(value != 0);
    return iree_ok_status();
  }

  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
      loom_scalar_type_is_integer(scalar_type)) {
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(value_text, &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config '%.*s' expects integer value, got '%.*s'",
                              (int)key.size, key.data, (int)value_text.size,
                              value_text.data);
    }
    int64_t domain_lo = INT64_MIN;
    int64_t domain_hi = INT64_MAX;
    if (loom_scalar_type_integer_domain(scalar_type, &domain_lo, &domain_hi) &&
        (value < domain_lo || value > domain_hi)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "config '%.*s' value %" PRId64 " is outside the %s domain [%" PRId64
          ", %" PRId64 "]",
          (int)key.size, key.data, value, loom_scalar_type_name(scalar_type),
          domain_lo, domain_hi);
    }
    *out_attr = loom_attr_i64(value);
    return iree_ok_status();
  }

  if (loom_scalar_type_is_float(scalar_type)) {
    double value = 0.0;
    if (!iree_string_view_atod(value_text, &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config '%.*s' expects floating-point value, got '%.*s'",
          (int)key.size, key.data, (int)value_text.size, value_text.data);
    }
    *out_attr = loom_attr_f64(value);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' has unsupported scalar type %s",
                          (int)key.size, key.data,
                          loom_scalar_type_name(scalar_type));
}

static iree_status_t loom_config_text_parse_value(
    loom_module_t* module, iree_string_view_t key, loom_type_t type,
    iree_string_view_t value_text, iree_arena_block_pool_t* block_pool,
    loom_attribute_t* out_attr) {
  if (loom_type_is_scalar(type)) {
    return loom_config_text_parse_scalar_value(key, type, value_text, out_attr);
  }
  if (loom_type_is_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_config_text_parse_encoding_value(
        module, type, value_text, block_pool, iree_allocator_system(),
        out_attr));
    return loom_config_text_check_encoding_role(module, key, type, *out_attr);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' has unsupported type", (int)key.size,
                          key.data);
}

iree_status_t loom_config_text_materialize_module(
    loom_module_t* module,
    const loom_config_text_materialize_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_config_application_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(block_pool);
  const loom_config_text_binding_set_t* binding_set = options->binding_set;
  const iree_host_size_t binding_count =
      binding_set ? binding_set->binding_count : 0;

  loom_config_application_result_t result = {0};
  if (binding_count == 0) {
    if (out_result) {
      *out_result = result;
    }
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const loom_config_text_binding_t* binding = &binding_set->bindings[i];
    iree_string_view_t key = binding->key;
    iree_string_view_t value_text = binding->value;
    const uint16_t symbol_id = loom_config_find_symbol(module, key);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      ++result.ignored_count;
      continue;
    }

    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (!loom_config_symbol_is_config(symbol)) {
      ++result.ignored_count;
      continue;
    }
    loom_op_t* old_op = symbol->defining_op;
    loom_value_id_t config_value = loom_config_symbol_result_value(old_op);
    loom_type_t config_type = loom_module_value_type(module, config_value);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(loom_config_text_parse_value(
        module, key, config_type, value_text, block_pool, &value));
    IREE_RETURN_IF_ERROR(loom_config_apply_exact_value(
        module, loom_config_symbol_name(module, symbol), old_op, config_type,
        value));
    IREE_RETURN_IF_ERROR(loom_config_applied_value_sink_emit(
        options->applied_value_sink, module, key, value));
    ++result.materialized_count;
  }

  if (out_result) {
    *out_result = result;
  }
  return iree_ok_status();
}
