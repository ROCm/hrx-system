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

void loom_encoding_define_resolve_verified_params(
    const loom_module_t* module,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params) {
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
