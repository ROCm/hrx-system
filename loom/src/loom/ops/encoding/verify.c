// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"

static iree_status_t loom_encoding_emit(iree_diagnostic_emitter_t emitter,
                                        const loom_op_t* op,
                                        const loom_error_def_t* error,
                                        const loom_diagnostic_param_t* params,
                                        iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_encoding_define_emit_duplicate_static_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t name_id) {
  iree_string_view_t param_name = module->strings.entries[name_id];
  loom_diagnostic_param_t params[] = {
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_006, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_define_emit_unknown_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t encoding_name,
    loom_string_id_t name_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(module->strings.entries[name_id]),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_008, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_define_emit_dynamic_type_mismatch(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t encoding_name,
    loom_string_id_t name_id, loom_value_id_t value_id,
    loom_type_constraint_t expected_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(module->strings.entries[name_id]),
      loom_param_type(loom_module_value_type(module, value_id)),
      loom_param_string(
          iree_make_cstring_view(loom_type_constraint_name(expected_type))),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_009, params,
                            IREE_ARRAYSIZE(params));
}

// Resolves sorted authored parameters and diagnoses family-contract
// violations at the public verifier boundary.
static iree_status_t loom_encoding_define_resolve_params(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t encoding_name,
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_define_param_view_t* params,
    loom_encoding_define_dynamic_binding_t* dynamic_binding_slots,
    loom_encoding_define_resolved_params_t* out_params, bool* out_valid) {
  *out_valid = false;
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
          return loom_encoding_define_emit_duplicate_static_dynamic_param(
              module, op, emitter, dynamic_entry->name_id);
        }
        break;
      }
      ++static_index;
    }

    int descriptor_comparison = 1;
    while (descriptor_index < descriptor_count) {
      const loom_encoding_dynamic_parameter_descriptor_t* dynamic_descriptor =
          &descriptor->dynamic_parameter_descriptors[descriptor_index];
      descriptor_comparison = iree_string_view_compare(
          loom_bstring_view(dynamic_descriptor->name), dynamic_name);
      if (descriptor_comparison >= 0) break;
      ++descriptor_index;
    }
    if (descriptor_index == descriptor_count || descriptor_comparison != 0) {
      return loom_encoding_define_emit_unknown_dynamic_param(
          module, op, emitter, encoding_name, dynamic_entry->name_id);
    }

    IREE_ASSERT(dynamic_entry->value.kind == LOOM_ATTR_I64);
    IREE_ASSERT(dynamic_entry->value.i64 >= 0);
    IREE_ASSERT(dynamic_entry->value.i64 < params->dynamic_values.count);
    const uint16_t operand_ordinal = (uint16_t)dynamic_entry->value.i64;
    const loom_value_id_t value_id =
        params->dynamic_values.values[operand_ordinal];
    const loom_type_constraint_t expected_type =
        descriptor->dynamic_parameter_descriptors[descriptor_index]
            .type_constraint;
    if (!loom_type_satisfies_constraint(
            loom_module_value_type(module, value_id), expected_type)) {
      return loom_encoding_define_emit_dynamic_type_mismatch(
          module, op, emitter, encoding_name, dynamic_entry->name_id, value_id,
          expected_type);
    }
    dynamic_binding_slots[descriptor_index] =
        (loom_encoding_define_dynamic_binding_t){
            .value_id = value_id,
            .operand_ordinal = operand_ordinal,
        };
    ++descriptor_index;
  }

  *out_valid = true;
  return iree_ok_status();
}

// Returns a name present in both lexically ordered parameter namespaces.
static loom_string_id_t loom_encoding_define_find_duplicate_param(
    const loom_module_t* module,
    const loom_encoding_define_param_view_t* params) {
  iree_host_size_t static_index = 0;
  iree_host_size_t dynamic_index = 0;
  while (static_index < params->static_attrs.count &&
         dynamic_index < params->dynamic_names.count) {
    const loom_string_id_t static_name_id =
        params->static_attrs.entries[static_index].name_id;
    const loom_string_id_t dynamic_name_id =
        params->dynamic_names.entries[dynamic_index].name_id;
    const int comparison =
        iree_string_view_compare(module->strings.entries[static_name_id],
                                 module->strings.entries[dynamic_name_id]);
    if (comparison == 0) return dynamic_name_id;
    if (comparison < 0) {
      ++static_index;
    } else {
      ++dynamic_index;
    }
  }
  return LOOM_STRING_ID_INVALID;
}

static iree_status_t loom_encoding_define_emit_result_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, loom_type_t actual_type,
    loom_type_t expected_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_type(actual_type),
      loom_param_type(expected_type),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_012, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                            IREE_ARRAYSIZE(params));
}

static uint16_t loom_encoding_dynamic_sentinel_count(loom_attribute_t values) {
  if (values.kind != LOOM_ATTR_I64_ARRAY) return 0;
  uint16_t dynamic_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.i64_array[i] == INT64_MIN) ++dynamic_count;
  }
  return dynamic_count;
}

static iree_status_t loom_encoding_verify_dynamic_index_count(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_attribute_t static_values,
    uint16_t dynamic_count) {
  uint16_t expected_dynamic_count =
      loom_encoding_dynamic_sentinel_count(static_values);
  if (dynamic_count == expected_dynamic_count) return iree_ok_status();

  iree_string_view_t op_name = loom_op_name(module, op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_u32(dynamic_count),
      loom_param_u32(expected_dynamic_count),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_STRUCTURE_001, params,
                            IREE_ARRAYSIZE(params));
}

iree_status_t loom_encoding_layout_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_attribute_t static_strides =
      loom_encoding_layout_strided_static_strides(op);
  if (static_strides.kind == LOOM_ATTR_I64_ARRAY) {
    for (uint16_t i = 0; i < static_strides.count; ++i) {
      int64_t static_stride = static_strides.i64_array[i];
      if (static_stride < 0 && static_stride != INT64_MIN) {
        return loom_encoding_emit_attribute_value_constraint(
            emitter, op, IREE_SV("static_strides"), static_stride,
            IREE_SV("stride >= 0 or dynamic sentinel"));
      }
    }
  }
  return loom_encoding_verify_dynamic_index_count(
      module, op, emitter, loom_encoding_layout_strided_static_strides(op),
      loom_encoding_layout_strided_strides(op).count);
}

iree_status_t loom_encoding_layout_assume_strided_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  int64_t rank = loom_encoding_layout_assume_strided_rank(op);
  if (rank >= 0 && rank <= UINT8_MAX) return iree_ok_status();
  return loom_encoding_emit_attribute_value_constraint(
      emitter, op, IREE_SV("rank"), rank, IREE_SV("rank in [0, 255]"));
}

iree_status_t loom_encoding_define_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, op);
  if (!params.spec) return iree_ok_status();

  iree_string_view_t encoding_name =
      module->strings.entries[params.spec->name_id];

  const loom_encoding_vtable_t* vtable =
      loom_module_encoding_vtable(module, params.spec);
  const uint8_t dynamic_parameter_count =
      vtable ? vtable->descriptor->dynamic_parameter_count : 0;
  loom_encoding_define_dynamic_binding_t* dynamic_binding_slots =
      dynamic_parameter_count ? iree_alloca(dynamic_parameter_count *
                                            sizeof(*dynamic_binding_slots))
                              : NULL;
  loom_encoding_define_resolved_params_t resolved_params;
  if (vtable) {
    bool params_valid = false;
    IREE_RETURN_IF_ERROR(loom_encoding_define_resolve_params(
        module, op, emitter, encoding_name, vtable->descriptor, &params,
        dynamic_binding_slots, &resolved_params, &params_valid));
    if (!params_valid) return iree_ok_status();
  } else {
    const loom_string_id_t duplicate_name_id =
        loom_encoding_define_find_duplicate_param(module, &params);
    if (duplicate_name_id != LOOM_STRING_ID_INVALID) {
      return loom_encoding_define_emit_duplicate_static_dynamic_param(
          module, op, emitter, duplicate_name_id);
    }
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_encoding_define_result(op));
  if (loom_type_is_encoding(result_type)) {
    loom_encoding_role_t result_role = loom_type_encoding_role(result_type);
    loom_encoding_role_t expected_role =
        loom_encoding_static_role(module, params.spec);
    if (result_role != expected_role) {
      return loom_encoding_define_emit_result_role_error(
          emitter, op, encoding_name, result_type,
          loom_type_encoding_with_role(expected_role));
    }
  }

  if (vtable && vtable->verify_define) {
    IREE_RETURN_IF_ERROR(
        vtable->verify_define(module, op, &resolved_params, emitter));
  }

  return iree_ok_status();
}

iree_status_t loom_encoding_assume_spec_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_assume_spec_spec(op));
  if (!spec) return iree_ok_status();

  loom_type_t result_type =
      loom_module_value_type(module, loom_encoding_assume_spec_result(op));
  if (!loom_type_is_encoding(result_type)) return iree_ok_status();

  loom_encoding_role_t result_role = loom_type_encoding_role(result_type);
  loom_encoding_role_t expected_role = loom_encoding_static_role(module, spec);
  if (result_role == expected_role) return iree_ok_status();

  iree_string_view_t encoding_name = module->strings.entries[spec->name_id];
  return loom_encoding_define_emit_result_role_error(
      emitter, op, encoding_name, result_type,
      loom_type_encoding_with_role(expected_role));
}

iree_status_t loom_encoding_isa_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_isa_spec(op));
  if (!spec) return iree_ok_status();

  const loom_type_t operand_type =
      loom_module_value_type(module, loom_encoding_isa_enc(op));
  if (!loom_type_is_encoding(operand_type)) return iree_ok_status();

  const loom_encoding_role_t operand_role =
      loom_type_encoding_role(operand_type);
  const loom_encoding_role_t spec_role =
      loom_encoding_static_role(module, spec);
  if (operand_role == LOOM_ENCODING_ROLE_UNKNOWN || operand_role == spec_role) {
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_string(module->strings.entries[spec->name_id]),
      loom_param_type(loom_type_encoding_with_role(spec_role)),
      loom_param_type(operand_type),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_021, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_verify_match_requirements(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_attribute_t requirements) {
  if (loom_encoding_match_attr_has_element_format(requirements) ||
      loom_encoding_match_attr_has_payload_packing(requirements) ||
      loom_encoding_match_attr_has_affine(requirements)) {
    return iree_ok_status();
  }
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_022,
                            /*params=*/NULL, /*param_count=*/0);
}

iree_status_t loom_encoding_matches_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  return loom_encoding_verify_match_requirements(
      op, emitter, loom_encoding_matches_requirements(op));
}

iree_status_t loom_encoding_assume_match_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_encoding_verify_match_requirements(
      op, emitter, loom_encoding_assume_match_requirements(op));
}
