// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/families.h"

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/encoding/matrix_operand.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

iree_status_t loom_encoding_register_physical_storage_family(
    loom_context_t* context);

static iree_string_view_t loom_encoding_turboquant_kv_name(void) {
  return IREE_SV("turboquant_kv");
}

static iree_string_view_t loom_encoding_matrix_operand_name(void) {
  return IREE_SV("matrix_operand");
}

static bool loom_encoding_string_id_equal(const loom_module_t* module,
                                          loom_string_id_t string_id,
                                          iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_encoding_find_param(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_encoding_string_attr_value(const loom_module_t* module,
                                            loom_attribute_t attr,
                                            iree_string_view_t* out_value) {
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return false;
  }
  *out_value = module->strings.entries[attr.string_id];
  return true;
}

static bool loom_encoding_matrix_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("element_format")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("payload_packing")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("payload_elements")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("payload_registers")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_topology")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_format")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("secondary_scale_format")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_group_elements")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_operands")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("affine")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("rounding")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("codebook")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("sparsity")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("zero_scale_fallback"));
}

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

static iree_status_t loom_encoding_emit_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, const loom_error_def_t* error,
    iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_kind_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    loom_attr_kind_t actual_kind, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_010, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_value_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t actual_value, iree_string_view_t expected_values) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(actual_value),
      loom_param_string(expected_values),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_013, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_attribute_constraint_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(param_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t encoding_name,
    iree_string_view_t param_name, loom_value_id_t value_id,
    iree_string_view_t expected_type) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_009, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_011, params,
                            IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_mutually_exclusive_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_a,
    iree_string_view_t param_b) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_a),
      loom_param_string(param_b),
  };
  return loom_encoding_emit(emitter, op, LOOM_ERR_ENCODING_014, params,
                            IREE_ARRAYSIZE(params));
}

typedef enum loom_encoding_matrix_param_requirement_e {
  LOOM_ENCODING_MATRIX_PARAM_OPTIONAL = 0,
  LOOM_ENCODING_MATRIX_PARAM_REQUIRED = 1,
} loom_encoding_matrix_param_requirement_t;

static iree_status_t loom_encoding_matrix_static_i64(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    loom_encoding_matrix_param_requirement_t requirement, int64_t default_value,
    int64_t* out_value, bool* out_ok) {
  *out_value = default_value;
  *out_ok = false;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    if (requirement == LOOM_ENCODING_MATRIX_PARAM_OPTIONAL) {
      *out_ok = true;
      return iree_ok_status();
    }
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_matrix_operand_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  if (entry->value.kind != LOOM_ATTR_I64) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
  }
  *out_value = loom_attr_as_i64(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_matrix_static_bool(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool default_value, bool* out_value, bool* out_ok) {
  *out_value = default_value;
  *out_ok = false;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (entry->value.kind != LOOM_ATTR_BOOL) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("bool"));
  }
  *out_value = loom_attr_as_bool(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_matrix_static_symbol(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    loom_encoding_matrix_operand_symbol_set_t symbol_set,
    loom_encoding_matrix_param_requirement_t requirement,
    uint64_t default_value, uint64_t* out_value, bool* out_ok) {
  *out_value = default_value;
  *out_ok = false;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    if (requirement == LOOM_ENCODING_MATRIX_PARAM_OPTIONAL) {
      *out_ok = true;
      return iree_ok_status();
    }
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_matrix_operand_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  if (entry->value.kind != LOOM_ATTR_STRING) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("symbol"));
  }
  iree_string_view_t symbol = iree_string_view_empty();
  if (!loom_encoding_string_attr_value(module, entry->value, &symbol)) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("symbol"));
  }
  if (!loom_encoding_matrix_operand_lookup_symbol(symbol_set, symbol,
                                                  out_value)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_matrix_operand_name(), param_name, symbol,
        loom_encoding_matrix_operand_expected_symbols(symbol_set));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_matrix_verify_param_names(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    if (!loom_encoding_matrix_static_param_name_is_supported(module,
                                                             entry->name_id)) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(emitter, op,
                                            loom_encoding_matrix_operand_name(),
                                            LOOM_ERR_ENCODING_008, param_name);
    }
  }
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_matrix_operand_name(),
                                          LOOM_ERR_ENCODING_008, param_name);
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_matrix_verify_i64_range(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, int64_t value,
    iree_string_view_t expected_constraint) {
  return loom_encoding_emit_attribute_constraint_error(
      emitter, op, param_name, value, expected_constraint);
}

static iree_status_t loom_encoding_matrix_verify_optional_static_i64(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t default_value, int64_t* out_value, bool* out_ok) {
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_i64(
      module, op, params, emitter, param_name,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, default_value, out_value, out_ok));
  if (!*out_ok) return iree_ok_status();
  if (*out_value < 0) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, param_name, *out_value, IREE_SV("non-negative i64"));
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_matrix_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_verify_param_names(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t element_format = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("element_format"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
      LOOM_ENCODING_MATRIX_PARAM_REQUIRED,
      LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN, &element_format, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("element_format"), 0,
        IREE_SV("non-none numeric format symbol"));
  }

  uint64_t payload_packing = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("payload_packing"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL,
      LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT, &payload_packing,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t scale_format = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("scale_format"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE,
      &scale_format, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t secondary_scale_format = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("secondary_scale_format"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE,
      &secondary_scale_format, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t scale_topology = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("scale_topology"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE,
      &scale_topology, &param_ok));
  if (!param_ok) return iree_ok_status();

  int64_t payload_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_i64(
      module, op, params, emitter, IREE_SV("payload_elements"),
      LOOM_ENCODING_MATRIX_PARAM_REQUIRED, /*default_value=*/0,
      &payload_elements, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (payload_elements <= 0 || payload_elements > UINT16_MAX) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("payload_elements"), payload_elements,
        IREE_SV("positive and <= 65535"));
  }

  int64_t payload_registers = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_i64(
      module, op, params, emitter, IREE_SV("payload_registers"),
      LOOM_ENCODING_MATRIX_PARAM_REQUIRED, /*default_value=*/0,
      &payload_registers, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (payload_registers < 0 || payload_registers > UINT16_MAX) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("payload_registers"), payload_registers,
        IREE_SV("non-negative and <= 65535"));
  }
  if (iree_any_bit_set((uint32_t)payload_packing,
                       LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) &&
      payload_registers == 0) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("payload_registers"), payload_registers,
        IREE_SV("positive for target-fragment payloads"));
  }

  int64_t scale_group_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_verify_optional_static_i64(
      module, op, params, emitter, IREE_SV("scale_group_elements"),
      /*default_value=*/0, &scale_group_elements, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (scale_group_elements > UINT16_MAX) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("scale_group_elements"), scale_group_elements,
        IREE_SV("non-negative and <= 65535"));
  }

  int64_t scale_operands = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_verify_optional_static_i64(
      module, op, params, emitter, IREE_SV("scale_operands"),
      /*default_value=*/0, &scale_operands, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (scale_operands > UINT16_MAX) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("scale_operands"), scale_operands,
        IREE_SV("non-negative and <= 65535"));
  }

  uint64_t affine = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("affine"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_AFFINE_POLICY_NONE,
      &affine, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t rounding = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("rounding"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_ROUNDING_POLICY_NONE,
      &rounding, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t codebook = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("codebook"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE,
      &codebook, &param_ok));
  if (!param_ok) return iree_ok_status();

  uint64_t sparsity = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_symbol(
      module, op, params, emitter, IREE_SV("sparsity"),
      LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY,
      LOOM_ENCODING_MATRIX_PARAM_OPTIONAL, LOOM_VALUE_FACT_SPARSITY_POLICY_NONE,
      &sparsity, &param_ok));
  if (!param_ok) return iree_ok_status();

  bool zero_scale_fallback = false;
  IREE_RETURN_IF_ERROR(loom_encoding_matrix_static_bool(
      module, op, params, emitter, IREE_SV("zero_scale_fallback"),
      /*default_value=*/false, &zero_scale_fallback, &param_ok));
  if (!param_ok) return iree_ok_status();

  loom_value_fact_encoded_operand_schema_t encoded_operand = {
      .scale_format = (uint64_t)scale_format,
      .secondary_scale_format = (uint64_t)secondary_scale_format,
      .scale_topology = (uint32_t)scale_topology,
      .scale_group_element_count = (uint16_t)scale_group_elements,
      .scale_operand_count = (uint16_t)scale_operands,
      .flags = zero_scale_fallback
                   ? LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK
                   : 0,
  };
  if (!loom_value_fact_encoded_operand_schema_scale_is_complete(
          encoded_operand)) {
    return loom_encoding_matrix_verify_i64_range(
        emitter, op, IREE_SV("scale"), 1,
        IREE_SV("all-zero for none or topology/group/operand count for "
                "scaled"));
  }
  return iree_ok_status();
}

static bool loom_encoding_turboquant_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("first_stage_bits")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("logical_element")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("logical_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("pack_order")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("qjl_rows")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("record_bytes")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("residual_bits")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scalar_quantizer")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("transform_family"));
}

static bool loom_encoding_turboquant_dynamic_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id, IREE_SV("centroids")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("qjl_transform")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("thresholds")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("transform"));
}

static iree_status_t loom_encoding_turboquant_require_static_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    const loom_named_attr_t** out_param) {
  *out_param = NULL;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_turboquant_kv_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  *out_param = entry;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    const loom_named_attr_t** out_param) {
  *out_param = NULL;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->dynamic_names, param_name);
  if (!entry) {
    return loom_encoding_emit_param_error(emitter, op,
                                          loom_encoding_turboquant_kv_name(),
                                          LOOM_ERR_ENCODING_007, param_name);
  }
  *out_param = entry;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_i64(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t* out_value, bool* out_ok) {
  *out_value = 0;
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (entry->value.kind != LOOM_ATTR_I64) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
  }
  *out_value = loom_attr_as_i64(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_string(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    iree_string_view_t* out_value, bool* out_ok) {
  *out_value = iree_string_view_empty();
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (!loom_encoding_string_attr_value(module, entry->value, out_value)) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("string"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_turboquant_transform_family_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("hadamard")) ||
         iree_string_view_equal(value, IREE_SV("hadamard_sign")) ||
         iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"));
}

static bool loom_encoding_numeric_transform_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id, IREE_SV("family")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("normalization")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems"));
}

static bool loom_encoding_numeric_transform_dynamic_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("matrix")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("permutation")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("seed")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("signs"));
}

static bool loom_encoding_numeric_transform_family_supported(
    iree_string_view_t value) {
  return loom_encoding_numeric_transform_family_from_name(value) !=
         LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN;
}

static iree_string_view_t loom_encoding_numeric_transform_family_quoted(
    iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("hadamard"))) {
    return IREE_SV("'hadamard'");
  }
  if (iree_string_view_equal(value, IREE_SV("hadamard_sign"))) {
    return IREE_SV("'hadamard_sign'");
  }
  if (iree_string_view_equal(value, IREE_SV("jl_dense"))) {
    return IREE_SV("'jl_dense'");
  }
  if (iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"))) {
    return IREE_SV("'sign_permute_hadamard'");
  }
  return value;
}

static bool loom_encoding_numeric_transform_normalization_supported(
    iree_string_view_t value) {
  loom_encoding_numeric_transform_normalization_t normalization =
      LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE;
  return loom_encoding_numeric_transform_normalization_from_name(
      value, &normalization);
}

static bool loom_encoding_numeric_transform_param_is_extent(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems"));
}

static iree_status_t loom_encoding_numeric_transform_verify_static_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_numeric_transform_static_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_numeric_transform_name(),
          LOOM_ERR_ENCODING_008, param_name);
    }

    if (loom_encoding_numeric_transform_param_is_extent(module,
                                                        entry->name_id)) {
      if (entry->value.kind != LOOM_ATTR_I64) {
        return loom_encoding_emit_static_kind_error(
            emitter, op, loom_encoding_numeric_transform_name(), param_name,
            (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
      }
      int64_t extent = loom_attr_as_i64(entry->value);
      if (extent <= 0) {
        return loom_encoding_emit_attribute_constraint_error(
            emitter, op, param_name, extent, IREE_SV("positive extent"));
      }
      continue;
    }

    iree_string_view_t string_value = iree_string_view_empty();
    if (!loom_encoding_string_attr_value(module, entry->value, &string_value)) {
      return loom_encoding_emit_static_kind_error(
          emitter, op, loom_encoding_numeric_transform_name(), param_name,
          (loom_attr_kind_t)entry->value.kind, IREE_SV("string"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("family"))) {
      if (loom_encoding_numeric_transform_family_supported(string_value)) {
        continue;
      }
      return loom_encoding_emit_static_value_error(
          emitter, op, loom_encoding_numeric_transform_name(), param_name,
          string_value,
          IREE_SV("'hadamard', 'hadamard_sign', 'jl_dense', or "
                  "'sign_permute_hadamard'"));
    }

    if (loom_encoding_numeric_transform_normalization_supported(string_value)) {
      continue;
    }
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_numeric_transform_name(), param_name,
        string_value, IREE_SV("'none' or 'orthonormal'"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_dynamic_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_numeric_transform_dynamic_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_numeric_transform_name(),
          LOOM_ERR_ENCODING_008, param_name);
    }

    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    if (!loom_encoding_define_dynamic_param_value(params, entry, &value_id)) {
      return iree_ok_status();
    }

    loom_type_t actual_type = loom_module_value_type(module, value_id);
    if (loom_encoding_numeric_transform_param_is_extent(module,
                                                        entry->name_id) ||
        loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("seed"))) {
      if (loom_type_equal(actual_type,
                          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id, IREE_SV("index"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("signs"))) {
      if (loom_type_is_vector(actual_type) &&
          loom_type_element_type(actual_type) == LOOM_SCALAR_TYPE_I1) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id, IREE_SV("i1 vector"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("permutation"))) {
      if (loom_type_is_vector(actual_type) &&
          loom_type_satisfies_constraint(
              actual_type,
              LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT)) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id,
          IREE_SV("vector with index or non-i1 integer elements"));
    }

    if (loom_type_is_vector(actual_type) &&
        loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
      continue;
    }
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_numeric_transform_name(), param_name,
        value_id, IREE_SV("floating-point vector"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_require_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool family_must_be_static, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* static_param =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (static_param) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (!family_must_be_static &&
      loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(emitter, op,
                                        loom_encoding_numeric_transform_name(),
                                        LOOM_ERR_ENCODING_007, param_name);
}

static iree_status_t loom_encoding_numeric_transform_require_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  if (loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(emitter, op,
                                        loom_encoding_numeric_transform_name(),
                                        LOOM_ERR_ENCODING_007, param_name);
}

static bool loom_encoding_numeric_transform_has_dynamic_param(
    const loom_module_t* module,
    const loom_encoding_define_param_view_t* params,
    iree_string_view_t param_name) {
  return loom_encoding_find_param(module, params->dynamic_names, param_name) !=
         NULL;
}

static iree_status_t loom_encoding_numeric_transform_reject_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  if (!loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(emitter, op,
                                        loom_encoding_numeric_transform_name(),
                                        LOOM_ERR_ENCODING_008, param_name);
}

static iree_status_t loom_encoding_numeric_transform_verify_no_dynamic_payload(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("signs"), out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("permutation"), out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("matrix"), out_ok));
  if (!*out_ok) return iree_ok_status();
  return loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("seed"), out_ok);
}

static iree_status_t loom_encoding_numeric_transform_verify_no_matrix(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  return loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("matrix"), out_ok);
}

static iree_status_t
loom_encoding_numeric_transform_verify_hadamard_sign_payload(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  bool has_signs = loom_encoding_numeric_transform_has_dynamic_param(
      module, params, IREE_SV("signs"));
  bool has_seed = loom_encoding_numeric_transform_has_dynamic_param(
      module, params, IREE_SV("seed"));
  if (!has_signs && !has_seed) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_numeric_transform_name(),
        LOOM_ERR_ENCODING_007, IREE_SV("signs or seed"));
  }
  if (has_signs && has_seed) {
    return loom_encoding_emit_mutually_exclusive_param_error(
        emitter, op, loom_encoding_numeric_transform_name(), IREE_SV("signs"),
        IREE_SV("seed"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_jl_normalization(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, params->static_attrs, IREE_SV("normalization"));
  if (!entry) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t value = iree_string_view_empty();
  if (!loom_encoding_string_attr_value(module, entry->value, &value)) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      IREE_SV("normalization"), value, IREE_SV("'none'"));
}

static iree_status_t loom_encoding_numeric_transform_verify_family_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* family_entry =
      loom_encoding_find_param(module, params->static_attrs, IREE_SV("family"));
  if (!family_entry) return iree_ok_status();

  iree_string_view_t family_name = iree_string_view_empty();
  if (!loom_encoding_string_attr_value(module, family_entry->value,
                                       &family_name)) {
    return iree_ok_status();
  }

  switch (loom_encoding_numeric_transform_family_from_name(family_name)) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD:
      return loom_encoding_numeric_transform_verify_no_dynamic_payload(
          module, op, params, emitter, out_ok);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_verify_hadamard_sign_payload(
              module, op, params, emitter, out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(
          module, op, params, emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("signs"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("seed"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(
          module, op, params, emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("matrix"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("signs"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("seed"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_jl_normalization(
          module, op, params, emitter, out_ok);
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_encoding_numeric_transform_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_static_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_dynamic_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("family"),
      /*family_must_be_static=*/true, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("input_elems"),
      /*family_must_be_static=*/false, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("output_elems"),
      /*family_must_be_static=*/false, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_family_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_static_param_kinds(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_turboquant_static_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(emitter, op,
                                            loom_encoding_turboquant_kv_name(),
                                            LOOM_ERR_ENCODING_008, param_name);
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_param_names(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_turboquant_dynamic_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(emitter, op,
                                            loom_encoding_turboquant_kv_name(),
                                            LOOM_ERR_ENCODING_008, param_name);
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_ceil_bits_to_bytes(
    int64_t bit_count, int64_t* out_byte_count) {
  *out_byte_count = 0;
  int64_t padded_bits = 0;
  if (!iree_checked_add_i64(bit_count, 7, &padded_bits)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bit count overflows int64");
  }
  *out_byte_count = padded_bits / 8;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_min_record_bytes(
    int64_t logical_elems, int64_t first_stage_bits, int64_t qjl_rows,
    int64_t residual_bits, int64_t* out_record_bytes) {
  *out_record_bytes = 0;
  int64_t code_bits = 0;
  int64_t code_bytes = 0;
  int64_t residual_total_bits = 0;
  int64_t residual_bytes = 0;
  if (!iree_checked_mul_i64(logical_elems, first_stage_bits, &code_bits) ||
      !iree_checked_mul_i64(qjl_rows, residual_bits, &residual_total_bits)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "turboquant record bit count overflows int64");
  }
  IREE_RETURN_IF_ERROR(
      loom_encoding_turboquant_ceil_bits_to_bytes(code_bits, &code_bytes));
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_ceil_bits_to_bytes(
      residual_total_bits, &residual_bytes));
  if (!iree_checked_add_i64(code_bytes, residual_bytes, out_record_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "turboquant record byte count overflows int64");
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_view(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t expected_static_length, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, entry, &value_id)) {
    return iree_ok_status();
  }

  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_view(actual_type) || loom_type_rank(actual_type) != 1 ||
      !loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        value_id, IREE_SV("rank-1 floating-point view"));
  }

  if (!loom_type_dim_is_dynamic_at(actual_type, 0)) {
    int64_t actual_length = loom_type_dim_static_size_at(actual_type, 0);
    if (actual_length != expected_static_length) {
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, param_name, actual_length,
          IREE_SV("static view length matching schema"));
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_transform(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, entry, &value_id)) {
    return iree_ok_status();
  }

  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_encoding(actual_type)) {
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        value_id, IREE_SV("encoding<transform>"));
  }
  if (loom_type_encoding_role(actual_type) !=
      LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM) {
    return loom_encoding_emit_role_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM));
  }
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_try_get_numeric_transform_family(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t* out_family) {
  *out_family = iree_string_view_empty();
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) return false;

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec ||
      !loom_encoding_string_id_equal(module, params.spec->name_id,
                                     loom_encoding_numeric_transform_name())) {
    return false;
  }
  const loom_named_attr_t* family_entry =
      loom_encoding_find_param(module, params.static_attrs, IREE_SV("family"));
  if (!family_entry) return false;
  return loom_encoding_string_attr_value(module, family_entry->value,
                                         out_family);
}

static iree_status_t loom_encoding_turboquant_verify_transform_family(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    iree_string_view_t expected_family, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->dynamic_names, param_name);
  if (!entry) {
    *out_ok = true;
    return iree_ok_status();
  }

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_define_dynamic_param_value(params, entry, &value_id)) {
    return iree_ok_status();
  }

  iree_string_view_t actual_family = iree_string_view_empty();
  if (!loom_encoding_try_get_numeric_transform_family(module, value_id,
                                                      &actual_family)) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(actual_family, expected_family)) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t expected =
      loom_encoding_numeric_transform_family_quoted(expected_family);
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_turboquant_kv_name(), param_name,
      actual_family, expected);
}

static iree_status_t loom_encoding_turboquant_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_static_param_kinds(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_param_names(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  int64_t first_stage_bits = 0;
  int64_t logical_elems = 0;
  int64_t qjl_rows = 0;
  int64_t record_bytes = 0;
  int64_t residual_bits = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("first_stage_bits"),
      &first_stage_bits, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("logical_elems"), &logical_elems,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("qjl_rows"), &qjl_rows, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("record_bytes"), &record_bytes,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("residual_bits"), &residual_bits,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  if (first_stage_bits < 1 || first_stage_bits > 8) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("first_stage_bits"), first_stage_bits,
        IREE_SV("between 1 and 8"));
  }
  if (logical_elems <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("logical_elems"), logical_elems,
        IREE_SV("positive"));
  }
  if (qjl_rows <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("qjl_rows"), qjl_rows, IREE_SV("positive"));
  }
  if (record_bytes <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("positive"));
  }
  if (residual_bits != 1) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("residual_bits"), residual_bits,
        IREE_SV("equal to 1"));
  }

  iree_string_view_t logical_element = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("logical_element"), &logical_element,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  loom_scalar_type_t logical_scalar = LOOM_SCALAR_TYPE_COUNT_;
  if (!loom_scalar_type_parse(logical_element, &logical_scalar) ||
      !loom_scalar_type_is_float(logical_scalar)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("logical_element"), logical_element,
        IREE_SV("floating-point scalar type"));
  }

  iree_string_view_t pack_order = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("pack_order"), &pack_order,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(pack_order, IREE_SV("lsb0"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(), IREE_SV("pack_order"),
        pack_order, IREE_SV("'lsb0'"));
  }

  iree_string_view_t scalar_quantizer = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("scalar_quantizer"),
      &scalar_quantizer, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(scalar_quantizer, IREE_SV("lloyd_max"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("scalar_quantizer"), scalar_quantizer, IREE_SV("'lloyd_max'"));
  }

  iree_string_view_t transform_family = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("transform_family"),
      &transform_family, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!loom_encoding_turboquant_transform_family_supported(transform_family)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("transform_family"), transform_family,
        IREE_SV("'hadamard', 'hadamard_sign', or 'sign_permute_hadamard'"));
  }

  int64_t minimum_record_bytes = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_min_record_bytes(
      logical_elems, first_stage_bits, qjl_rows, residual_bits,
      &minimum_record_bytes));
  if (record_bytes < minimum_record_bytes) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("large enough for first-stage codes and QJL residual bits"));
  }

  int64_t centroid_count = 1LL << first_stage_bits;
  int64_t threshold_count = centroid_count - 1;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter, IREE_SV("centroids"), centroid_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter, IREE_SV("thresholds"), threshold_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_transform(
      module, op, params, emitter, IREE_SV("qjl_transform"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter, IREE_SV("qjl_transform"),
      IREE_SV("jl_dense"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_transform(
      module, op, params, emitter, IREE_SV("transform"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter, IREE_SV("transform"), transform_family,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_dense_vtable = {
    .name = IREE_SVL("dense"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

static const loom_encoding_vtable_t loom_encoding_strided_vtable = {
    .name = IREE_SVL("strided"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

static const loom_encoding_vtable_t loom_encoding_q8_0_vtable = {
    .name = IREE_SVL("q8_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q4_0_vtable = {
    .name = IREE_SVL("ggml_q4_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q8_0_vtable = {
    .name = IREE_SVL("ggml_q8_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_q6_k_vtable = {
    .name = IREE_SVL("q6_k"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q6_k_vtable = {
    .name = IREE_SVL("ggml_q6_k"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_iq_grid_vtable = {
    .name = IREE_SVL("ggml_iq_grid"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_loom_fp4_table_vtable = {
    .name = IREE_SVL("loom_fp4_table"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ieee_fp8_e4m3_vtable = {
    .name = IREE_SVL("ieee_fp8_e4m3"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_matrix_operand_vtable = {
    .name = IREE_SVL("matrix_operand"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    .verify_define = loom_encoding_matrix_verify_define,
};

static const loom_encoding_vtable_t loom_encoding_numeric_transform_vtable = {
    .name = IREE_SVL("numeric_transform"),
    .role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM,
    .verify_define = loom_encoding_numeric_transform_verify_define,
};

static const loom_encoding_vtable_t loom_encoding_orthogonal_transform_vtable =
    {
        .name = IREE_SVL("orthogonal_transform"),
        .role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM,
};

static const loom_encoding_vtable_t loom_encoding_turboquant_kv_vtable = {
    .name = IREE_SVL("turboquant_kv"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    .verify_define = loom_encoding_turboquant_verify_define,
};

static const loom_encoding_vtable_t* const loom_encoding_builtin_vtables[] = {
    &loom_encoding_dense_vtable,
    &loom_encoding_strided_vtable,
    &loom_encoding_q8_0_vtable,
    &loom_encoding_ggml_q4_0_vtable,
    &loom_encoding_ggml_q8_0_vtable,
    &loom_encoding_q6_k_vtable,
    &loom_encoding_ggml_q6_k_vtable,
    &loom_encoding_ggml_iq_grid_vtable,
    &loom_encoding_loom_fp4_table_vtable,
    &loom_encoding_ieee_fp8_e4m3_vtable,
    &loom_encoding_matrix_operand_vtable,
    &loom_encoding_numeric_transform_vtable,
    &loom_encoding_orthogonal_transform_vtable,
    &loom_encoding_turboquant_kv_vtable,
};

iree_status_t loom_context_register_builtin_encoding_vtables(
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_encoding_register_physical_storage_family(context));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_encoding_builtin_vtables); ++i) {
    IREE_RETURN_IF_ERROR(loom_context_register_encoding_vtable(
        context, loom_encoding_builtin_vtables[i]));
  }
  return iree_ok_status();
}
