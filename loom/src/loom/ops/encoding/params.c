// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/params.h"

static void loom_encoding_define_initialize_resolved_params(
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params) {
  const uint8_t descriptor_count = descriptor->dynamic_parameter_count;
  for (uint8_t i = 0; i < descriptor_count; ++i) {
    dynamic_binding_slots[i] = (loom_encoding_define_dynamic_binding_t){
        .value_id = LOOM_VALUE_ID_INVALID,
        .operand_ordinal = UINT16_MAX,
    };
  }
  *out_params = (loom_encoding_define_resolved_params_t){
      .spec = params->spec,
      .descriptor = descriptor,
      .static_attrs = params->static_attrs,
      .dynamic_bindings = dynamic_binding_slots,
      .dynamic_binding_count = descriptor_count,
  };
}

void loom_encoding_resolve_static_params(
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_t* spec,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params) {
  const loom_encoding_define_param_view_t params = {
      .spec = spec,
      .static_attrs = loom_encoding_attrs(spec),
  };
  loom_encoding_define_initialize_resolved_params(
      descriptor, &params, dynamic_binding_slots, out_params);
}

bool loom_encoding_define_try_resolve_unverified_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params) {
  loom_encoding_define_initialize_resolved_params(
      descriptor, params, dynamic_binding_slots, out_params);
  if (!params->spec ||
      !loom_encoding_static_parameters_are_valid(params->spec) ||
      loom_module_encoding_family_descriptor(module, params->spec) !=
          descriptor ||
      params->dynamic_names.count != params->dynamic_values.count ||
      params->dynamic_names.count > descriptor->dynamic_parameter_count ||
      (params->dynamic_names.count > 0 &&
       (!params->dynamic_names.entries || !params->dynamic_values.values))) {
    return false;
  }

  uint64_t seen_operand_ordinals[(UINT8_MAX + 63u) / 64u];
  const uint8_t operand_word_count =
      (uint8_t)((params->dynamic_values.count + 63u) / 64u);
  for (uint8_t i = 0; i < operand_word_count; ++i) {
    seen_operand_ordinals[i] = 0;
  }
  iree_host_size_t static_index = 0;
  uint8_t descriptor_index = 0;
  for (iree_host_size_t dynamic_index = 0;
       dynamic_index < params->dynamic_names.count; ++dynamic_index) {
    const loom_named_attr_t* dynamic_entry =
        &params->dynamic_names.entries[dynamic_index];
    if (dynamic_entry->name_id == LOOM_STRING_ID_INVALID ||
        dynamic_entry->name_id >= module->strings.count) {
      return false;
    }
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
        if (comparison == 0) return false;
        break;
      }
      ++static_index;
    }

    int descriptor_comparison = 1;
    while (descriptor_index < descriptor->dynamic_parameter_count) {
      const loom_encoding_dynamic_parameter_descriptor_t* dynamic_descriptor =
          &descriptor->dynamic_parameter_descriptors[descriptor_index];
      descriptor_comparison = iree_string_view_compare(
          loom_bstring_view(dynamic_descriptor->name), dynamic_name);
      if (descriptor_comparison >= 0) break;
      ++descriptor_index;
    }
    if (descriptor_index == descriptor->dynamic_parameter_count ||
        descriptor_comparison != 0 ||
        dynamic_entry->value.kind != LOOM_ATTR_I64 ||
        dynamic_entry->value.i64 < 0 ||
        dynamic_entry->value.i64 >= params->dynamic_values.count) {
      return false;
    }

    const uint16_t operand_ordinal = (uint16_t)dynamic_entry->value.i64;
    const uint64_t operand_bit = UINT64_C(1) << (operand_ordinal & 63);
    uint64_t* operand_word = &seen_operand_ordinals[operand_ordinal >> 6];
    if ((*operand_word & operand_bit) != 0) return false;
    *operand_word |= operand_bit;
    const loom_value_id_t value_id =
        params->dynamic_values.values[operand_ordinal];
    if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count ||
        !loom_type_satisfies_constraint(
            loom_module_value_type(module, value_id),
            descriptor->dynamic_parameter_descriptors[descriptor_index]
                .type_constraint)) {
      return false;
    }
    dynamic_binding_slots[descriptor_index] =
        (loom_encoding_define_dynamic_binding_t){
            .value_id = value_id,
            .operand_ordinal = operand_ordinal,
        };
    ++descriptor_index;
  }

  return true;
}

void loom_encoding_define_resolve_verified_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params) {
  const uint8_t descriptor_count = descriptor->dynamic_parameter_count;
  if (params->dynamic_names.count == descriptor_count) {
    *out_params = (loom_encoding_define_resolved_params_t){
        .spec = params->spec,
        .descriptor = descriptor,
        .static_attrs = params->static_attrs,
        .dynamic_bindings = dynamic_binding_slots,
        .dynamic_binding_count = descriptor_count,
    };
    for (uint8_t i = 0; i < descriptor_count; ++i) {
      const loom_named_attr_t* dynamic_entry =
          &params->dynamic_names.entries[i];
      const uint16_t operand_ordinal = (uint16_t)dynamic_entry->value.i64;
      dynamic_binding_slots[i] = (loom_encoding_define_dynamic_binding_t){
          .value_id = params->dynamic_values.values[operand_ordinal],
          .operand_ordinal = operand_ordinal,
      };
    }
    return;
  }

  loom_encoding_define_initialize_resolved_params(
      descriptor, params, dynamic_binding_slots, out_params);

  uint8_t descriptor_index = 0;
  for (iree_host_size_t dynamic_index = 0;
       dynamic_index < params->dynamic_names.count; ++dynamic_index) {
    const loom_named_attr_t* dynamic_entry =
        &params->dynamic_names.entries[dynamic_index];
    const iree_string_view_t dynamic_name =
        module->strings.entries[dynamic_entry->name_id];
    while (!loom_bstring_equal(
        descriptor->dynamic_parameter_descriptors[descriptor_index].name,
        dynamic_name)) {
      ++descriptor_index;
      IREE_ASSERT(descriptor_index < descriptor->dynamic_parameter_count);
    }

    IREE_ASSERT(dynamic_entry->value.kind == LOOM_ATTR_I64);
    const uint16_t operand_ordinal = (uint16_t)dynamic_entry->value.i64;
    dynamic_binding_slots[descriptor_index] =
        (loom_encoding_define_dynamic_binding_t){
            .value_id = params->dynamic_values.values[operand_ordinal],
            .operand_ordinal = operand_ordinal,
        };
    ++descriptor_index;
  }
}
