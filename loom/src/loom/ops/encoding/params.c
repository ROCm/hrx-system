// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/params.h"

static loom_encoding_define_param_resolution_t
loom_encoding_define_param_resolution_ok(void) {
  return (loom_encoding_define_param_resolution_t){
      .issue = LOOM_ENCODING_DEFINE_PARAM_ISSUE_NONE,
      .name_id = LOOM_STRING_ID_INVALID,
      .value_id = LOOM_VALUE_ID_INVALID,
      .expected_type = LOOM_TYPE_CONSTRAINT_ANY,
  };
}

static loom_encoding_define_param_resolution_t
loom_encoding_define_param_resolution_issue(
    loom_encoding_define_param_issue_t issue, loom_string_id_t name_id,
    loom_value_id_t value_id, loom_type_constraint_t expected_type) {
  return (loom_encoding_define_param_resolution_t){
      .issue = issue,
      .name_id = name_id,
      .value_id = value_id,
      .expected_type = expected_type,
  };
}

loom_encoding_define_param_resolution_t loom_encoding_define_resolve_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_value_id_t* dynamic_value_slots,
    loom_encoding_define_resolved_params_t* out_params) {
  const uint8_t descriptor_count = descriptor->dynamic_parameter_count;
  for (uint8_t i = 0; i < descriptor_count; ++i) {
    dynamic_value_slots[i] = LOOM_VALUE_ID_INVALID;
  }
  *out_params = (loom_encoding_define_resolved_params_t){
      .spec = params->spec,
      .descriptor = descriptor,
      .static_attrs = params->static_attrs,
      .dynamic_values =
          {
              .values = dynamic_value_slots,
              .count = descriptor_count,
          },
  };

  iree_host_size_t static_index = 0;
  uint8_t descriptor_index = 0;
  for (iree_host_size_t dynamic_index = 0;
       dynamic_index < params->dynamic_names.count; ++dynamic_index) {
    const loom_named_attr_t* dynamic_entry =
        &params->dynamic_names.entries[dynamic_index];
    const iree_string_view_t dynamic_name =
        module->strings.entries[dynamic_entry->name_id];

    while (static_index < params->static_attrs.count) {
      const loom_named_attr_t* static_entry =
          &params->static_attrs.entries[static_index];
      const iree_string_view_t static_name =
          module->strings.entries[static_entry->name_id];
      const int comparison =
          iree_string_view_compare(static_name, dynamic_name);
      if (comparison >= 0) {
        if (comparison == 0) {
          return loom_encoding_define_param_resolution_issue(
              LOOM_ENCODING_DEFINE_PARAM_ISSUE_DUPLICATE_STATIC_DYNAMIC,
              dynamic_entry->name_id, LOOM_VALUE_ID_INVALID,
              LOOM_TYPE_CONSTRAINT_ANY);
        }
        break;
      }
      ++static_index;
    }

    while (descriptor_index < descriptor_count) {
      const loom_encoding_dynamic_parameter_descriptor_t* dynamic_descriptor =
          &descriptor->dynamic_parameter_descriptors[descriptor_index];
      const int comparison = iree_string_view_compare(
          loom_bstring_view(dynamic_descriptor->name), dynamic_name);
      if (comparison >= 0) break;
      ++descriptor_index;
    }
    if (descriptor_index == descriptor_count ||
        !loom_bstring_equal(
            descriptor->dynamic_parameter_descriptors[descriptor_index].name,
            dynamic_name)) {
      return loom_encoding_define_param_resolution_issue(
          LOOM_ENCODING_DEFINE_PARAM_ISSUE_UNKNOWN_DYNAMIC,
          dynamic_entry->name_id, LOOM_VALUE_ID_INVALID,
          LOOM_TYPE_CONSTRAINT_ANY);
    }

    IREE_ASSERT(dynamic_entry->value.kind == LOOM_ATTR_I64);
    IREE_ASSERT(dynamic_entry->value.i64 >= 0);
    IREE_ASSERT(dynamic_entry->value.i64 < params->dynamic_values.count);
    const loom_value_id_t value_id =
        params->dynamic_values.values[dynamic_entry->value.i64];
    const loom_type_constraint_t expected_type =
        descriptor->dynamic_parameter_descriptors[descriptor_index]
            .type_constraint;
    if (!loom_type_satisfies_constraint(
            loom_module_value_type(module, value_id), expected_type)) {
      return loom_encoding_define_param_resolution_issue(
          LOOM_ENCODING_DEFINE_PARAM_ISSUE_DYNAMIC_TYPE_MISMATCH,
          dynamic_entry->name_id, value_id, expected_type);
    }
    dynamic_value_slots[descriptor_index] = value_id;
    ++descriptor_index;
  }

  return loom_encoding_define_param_resolution_ok();
}
