// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/config/text.h"

#include <string.h>

#include "iree/base/api.h"
#include "loom/config/application.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/format/text/parser.h"
#include "loom/ir/attribute.h"
#include "loom/ir/encoding.h"
#include "loom/ops/encoding/roles.h"

void loom_config_text_materialize_options_initialize(
    loom_config_text_materialize_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

// Per-binding diagnostic ownership, borrowed only while the value is applied.
typedef struct loom_config_text_diagnostic_context_t {
  // Destination module owning parsed attributes and declaration identities.
  loom_module_t* module;
  // Declaration whose value is being parsed and checked.
  const loom_op_t* op;
  // Caller-owned source resolver and diagnostic sink.
  const loom_config_text_materialize_options_t* options;
} loom_config_text_diagnostic_context_t;

static iree_status_t loom_config_text_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  const loom_config_text_diagnostic_context_t* context = user_data;
  if (!context->options->diagnostic_sink.fn) {
    return iree_ok_status();
  }
  loom_diagnostic_t diagnostic;
  loom_diagnostic_related_location_t
      related[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  loom_source_resolve_diagnostic(
      context->module, context->options->source_resolver, LOOM_EMITTER_PASS,
      emission, related, &diagnostic);
  return loom_diagnostic_emit(&context->options->diagnostic_sink, &diagnostic);
}

static iree_status_t loom_config_text_capture_parse_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  const loom_config_text_diagnostic_context_t* context = user_data;
  if (!context->options->diagnostic_sink.fn) {
    return iree_ok_status();
  }
  loom_diagnostic_t annotated = *diagnostic;
  loom_diagnostic_related_location_t
      related[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  const iree_host_size_t count =
      iree_min(diagnostic->related_location_count, IREE_ARRAYSIZE(related));
  for (iree_host_size_t i = 0; i < count; ++i) {
    related[i] = diagnostic->related_locations[i];
  }
  annotated.related_locations = related;
  annotated.related_location_count = count;
  annotated.related_location_omitted_count +=
      diagnostic->related_location_count - count;
  loom_source_range_t declaration;
  if (loom_source_resolve(context->options->source_resolver, context->module,
                          context->op->location, &declaration)) {
    if (count < IREE_ARRAYSIZE(related)) {
      related[annotated.related_location_count++] =
          (loom_diagnostic_related_location_t){
              .label = IREE_SV("configuration declared here"),
              .source_location = declaration,
          };
    } else {
      ++annotated.related_location_omitted_count;
    }
  }
  return loom_diagnostic_emit(&context->options->diagnostic_sink, &annotated);
}

static bool loom_config_text_parse_scalar_value(loom_type_t type,
                                                iree_string_view_t text,
                                                loom_attribute_t* out_attr) {
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    if (iree_string_view_equal_case(text, IREE_SV("true"))) {
      *out_attr = loom_attr_bool(true);
      return true;
    }
    if (iree_string_view_equal_case(text, IREE_SV("false"))) {
      *out_attr = loom_attr_bool(false);
      return true;
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(text, &value) ||
        (value != 0 && value != 1)) {
      return false;
    }
    *out_attr = loom_attr_bool(value != 0);
    return true;
  }
  if (loom_scalar_type_is_float(scalar_type)) {
    double value = 0.0;
    if (!iree_string_view_atod(text, &value)) {
      return false;
    }
    *out_attr = loom_attr_f64(value);
    return true;
  }
  int64_t value = 0;
  if (!iree_string_view_atoi_int64(text, &value)) {
    return false;
  }
  int64_t domain_lo = INT64_MIN;
  int64_t domain_hi = INT64_MAX;
  if (loom_scalar_type_integer_domain(scalar_type, &domain_lo, &domain_hi) &&
      (value < domain_lo || value > domain_hi)) {
    return false;
  }
  *out_attr = loom_attr_i64(value);
  return true;
}

static iree_status_t loom_config_text_parse_value(
    loom_config_text_diagnostic_context_t* context, iree_string_view_t key,
    loom_type_t type, iree_string_view_t text, loom_attribute_t* out_attr,
    bool* out_accepted) {
  *out_accepted = false;
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_config_text_emit,
      .user_data = context,
  };
  if (loom_type_is_scalar(type)) {
    *out_accepted = loom_config_text_parse_scalar_value(type, text, out_attr);
    if (*out_accepted) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(key),
        loom_param_string(text),
        loom_param_type(type),
    };
    const loom_diagnostic_emission_t emission = {
        .op = context->op,
        .error = LOOM_ERR_SYMBOL_011,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }

  const loom_attr_descriptor_t descriptor = {.attr_kind = LOOM_ATTR_ENCODING};
  const loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_config_text_capture_parse_diagnostic,
                          .user_data = context},
  };
  IREE_RETURN_IF_ERROR(loom_text_parse_attribute(
      text, IREE_SV("<configuration>"), context->module, &descriptor,
      &parse_options, out_attr, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  const loom_encoding_role_t expected_role = loom_type_encoding_role(type);
  if (expected_role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return iree_ok_status();
  }
  const loom_encoding_t* encoding = loom_module_encoding(
      context->module, loom_attr_as_encoding_id(*out_attr));
  const loom_encoding_role_t actual_role =
      loom_encoding_static_role(context->module, encoding);
  if (actual_role == expected_role) {
    return iree_ok_status();
  }
  *out_accepted = false;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(key),
      loom_param_string(loom_encoding_role_description(expected_role)),
      loom_param_string(loom_encoding_role_description(actual_role)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = context->op,
      .error = LOOM_ERR_SYMBOL_012,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_config_text_materialize_module(
    loom_module_t* module,
    const loom_config_text_materialize_options_t* options,
    loom_config_application_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  const loom_config_text_binding_set_t* binding_set = options->binding_set;
  const iree_host_size_t binding_count =
      binding_set ? binding_set->binding_count : 0;

  *out_result = (loom_config_application_result_t){0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status) &&
                               !out_result->error_count;
       ++i) {
    const loom_config_text_binding_t* binding = &binding_set->bindings[i];
    iree_string_view_t key = binding->key;
    iree_string_view_t value_text = binding->value;
    const uint16_t symbol_id = loom_config_find_symbol(module, key);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      ++out_result->ignored_count;
      continue;
    }

    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (!loom_config_symbol_is_config(symbol)) {
      ++out_result->ignored_count;
      continue;
    }
    loom_op_t* old_op = symbol->defining_op;
    loom_value_id_t config_value = loom_config_symbol_result_value(old_op);
    loom_type_t config_type = loom_module_value_type(module, config_value);

    loom_config_text_diagnostic_context_t context = {
        .module = module,
        .op = old_op,
        .options = options,
    };
    const iree_diagnostic_emitter_t emitter = {
        .fn = loom_config_text_emit,
        .user_data = &context,
    };
    loom_attribute_t value = loom_attr_absent();
    bool accepted = false;
    status = loom_config_text_parse_value(&context, key, config_type,
                                          value_text, &value, &accepted);
    if (iree_status_is_ok(status) && accepted) {
      status = loom_config_apply_exact_value(module, key, old_op, config_type,
                                             value, emitter, &accepted);
    }
    if (iree_status_is_ok(status) && !accepted) {
      ++out_result->error_count;
    }
    if (!iree_status_is_ok(status) || !accepted) {
      continue;
    }
    ++out_result->materialized_count;
    status = loom_config_applied_value_sink_emit(options->applied_value_sink,
                                                 module, key, value);
  }
  return status;
}
