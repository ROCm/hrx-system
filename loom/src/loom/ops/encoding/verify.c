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

  for (iree_host_size_t static_index = 0;
       static_index < params.static_attrs.count; ++static_index) {
    loom_string_id_t static_name_id =
        params.static_attrs.entries[static_index].name_id;
    for (iree_host_size_t dynamic_index = 0;
         dynamic_index < params.dynamic_names.count; ++dynamic_index) {
      if (params.dynamic_names.entries[dynamic_index].name_id ==
          static_name_id) {
        return loom_encoding_define_emit_duplicate_static_dynamic_param(
            module, op, emitter, static_name_id);
      }
    }
  }

  iree_string_view_t encoding_name =
      module->strings.entries[params.spec->name_id];

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

  const loom_encoding_vtable_t* vtable =
      loom_context_lookup_encoding_vtable(module->context, encoding_name);
  if (vtable && vtable->verify_define) {
    IREE_RETURN_IF_ERROR(vtable->verify_define(module, op, &params, emitter));
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
