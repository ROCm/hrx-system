// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"

static loom_type_t loom_global_symbol_type(const loom_module_t* module,
                                           const loom_symbol_t* symbol) {
  loom_type_t none = {0};
  if (!symbol || !symbol->defining_op) return none;
  if (loom_global_constant_isa(symbol->defining_op)) {
    return loom_module_value_type(
        module, loom_global_constant_type(symbol->defining_op));
  }
  if (loom_global_variable_isa(symbol->defining_op)) {
    return loom_module_value_type(
        module, loom_global_variable_type(symbol->defining_op));
  }
  return none;
}

static bool loom_global_types_match(loom_type_t global_type,
                                    loom_type_t use_type) {
  loom_type_kind_t kind = loom_type_kind(global_type);
  if (kind != loom_type_kind(use_type)) return false;
  if (!loom_type_is_shaped(global_type)) {
    return loom_type_equal(global_type, use_type);
  }

  if (!loom_type_element_type_equals(global_type, use_type)) return false;
  if (!loom_type_rank_equals(global_type, use_type)) return false;

  uint8_t rank = loom_type_rank(global_type);
  for (uint8_t dimension = 0; dimension < rank; ++dimension) {
    uint64_t global_dim = loom_type_dim(global_type, dimension);
    uint64_t use_dim = loom_type_dim(use_type, dimension);
    if (!loom_dim_is_dynamic(global_dim)) {
      if (loom_dim_is_dynamic(use_dim)) return false;
      if (loom_dim_static_size(global_dim) != loom_dim_static_size(use_dim)) {
        return false;
      }
      continue;
    }

    if (!loom_dim_is_dynamic(use_dim)) {
      return false;
    }

    loom_value_id_t global_value_id = loom_dim_value_id(global_dim);
    loom_value_id_t use_value_id = loom_dim_value_id(use_dim);
    for (uint8_t prior_dimension = 0; prior_dimension < dimension;
         ++prior_dimension) {
      uint64_t prior_global_dim = loom_type_dim(global_type, prior_dimension);
      if (!loom_dim_is_dynamic(prior_global_dim) ||
          loom_dim_value_id(prior_global_dim) != global_value_id) {
        continue;
      }

      uint64_t prior_use_dim = loom_type_dim(use_type, prior_dimension);
      if (!loom_dim_is_dynamic(prior_use_dim) ||
          loom_dim_value_id(prior_use_dim) != use_value_id) {
        return false;
      }
      break;
    }
  }

  if (!loom_type_has_encoding(global_type)) {
    return !loom_type_has_encoding(use_type);
  }
  if (loom_type_has_static_encoding(global_type)) {
    return loom_type_has_static_encoding(use_type) &&
           loom_type_encoding_equals(global_type, use_type);
  }
  return loom_type_has_ssa_encoding(global_type) &&
         loom_type_has_ssa_encoding(use_type);
}

static bool loom_global_type_references_dim_result(loom_type_t value_type,
                                                   loom_value_id_t value_id) {
  uint8_t rank = loom_type_rank(value_type);
  for (uint8_t dimension = 0; dimension < rank; ++dimension) {
    uint64_t dim = loom_type_dim(value_type, dimension);
    if (!loom_dim_is_dynamic(dim)) continue;
    if (loom_dim_value_id(dim) == value_id) return true;
  }
  return false;
}

static bool loom_global_type_references_encoding_result(
    loom_type_t value_type, loom_value_id_t value_id) {
  return loom_type_has_ssa_encoding(value_type) &&
         loom_type_encoding_value_id(value_type) == value_id;
}

static iree_status_t loom_global_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_global_verify_result_type(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_slice_t results,
    uint16_t result_index, loom_type_constraint_t constraint) {
  if (result_index == UINT16_MAX || result_index >= results.count) {
    return iree_ok_status();
  }
  loom_type_t type =
      loom_module_value_type(module, results.values[result_index]);
  if (loom_type_satisfies_constraint(type, constraint)) {
    return iree_ok_status();
  }

  char result_name_buffer[32];
  iree_snprintf(result_name_buffer, sizeof(result_name_buffer), "result %u",
                result_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(result_name_buffer)),
      loom_param_type(type),
      loom_param_string(
          iree_make_cstring_view(loom_type_constraint_name(constraint))),
  };
  return loom_global_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_global_verify_load_results(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t results = loom_global_load_result(op);
  if (results.count == 0) {
    iree_string_view_t op_name = loom_op_name(module, op);
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(0),
        loom_param_u32(1),
    };
    return loom_global_emit(emitter, op, LOOM_ERR_STRUCTURE_002, params,
                            IREE_ARRAYSIZE(params));
  }

  loom_type_t value_type = loom_module_value_type(module, results.values[0]);
  uint16_t referenced_count = 0;

  for (uint16_t result_index = 0; result_index < results.count;
       ++result_index) {
    loom_value_id_t result_id = results.values[result_index];
    bool referenced_as_dim =
        loom_global_type_references_dim_result(value_type, result_id);
    bool referenced_as_encoding =
        loom_global_type_references_encoding_result(value_type, result_id);
    if (!referenced_as_dim && !referenced_as_encoding) continue;

    if (referenced_as_dim) {
      IREE_RETURN_IF_ERROR(loom_global_verify_result_type(
          module, op, emitter, results, result_index,
          LOOM_TYPE_CONSTRAINT_INDEX));
    }
    if (referenced_as_encoding) {
      IREE_RETURN_IF_ERROR(loom_global_verify_result_type(
          module, op, emitter, results, result_index,
          LOOM_TYPE_CONSTRAINT_ANY_ENCODING));
    }
    if (result_index > 0) ++referenced_count;
  }

  uint16_t expected_result_count = (uint16_t)(1 + referenced_count);
  if (results.count != expected_result_count) {
    iree_string_view_t op_name = loom_op_name(module, op);
    loom_diagnostic_param_t params[] = {
        loom_param_string(op_name),
        loom_param_u32(results.count),
        loom_param_u32(expected_result_count),
    };
    return loom_global_emit(emitter, op, LOOM_ERR_STRUCTURE_002, params,
                            IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_global_emit_type_mismatch(
    const loom_op_t* op, const loom_op_t* definition_op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_name,
    loom_type_t use_type, loom_type_t global_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_type(use_type),
      loom_param_string(IREE_SV("referenced global")),
      loom_param_type(global_type),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_global_emit_initializer_kind_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("initializer")),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_global_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_global_emit_initializer_type_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t global_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("global type")),
      loom_param_type(global_type),
      loom_param_string(IREE_SV("scalar")),
  };
  return loom_global_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_global_verify_initializer(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t global_value_id,
    loom_attribute_t initializer) {
  if (loom_attr_is_absent(initializer)) {
    return iree_ok_status();
  }

  loom_type_t global_type = loom_module_value_type(module, global_value_id);
  if (!loom_type_is_scalar(global_type)) {
    return loom_global_emit_initializer_type_mismatch(op, emitter, global_type);
  }

  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (!loom_attr_matches_scalar_type(
          initializer, loom_type_element_type(global_type), &expected_kind)) {
    return loom_global_emit_initializer_kind_mismatch(
        op, emitter, (loom_attr_kind_t)initializer.kind, expected_kind);
  }
  return iree_ok_status();
}

iree_status_t loom_global_constant_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_global_verify_initializer(module, op, emitter,
                                        loom_global_constant_type(op),
                                        loom_global_constant_initializer(op));
}

iree_status_t loom_global_variable_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_global_verify_initializer(module, op, emitter,
                                        loom_global_variable_type(op),
                                        loom_global_variable_initializer(op));
}

iree_status_t loom_global_load_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t ref = loom_global_load_global(op);
  if (!loom_symbol_ref_is_valid(ref) ||
      ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }

  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_global_verify_load_results(module, op, emitter));

  loom_value_slice_t results = loom_global_load_result(op);
  if (results.count == 0) return iree_ok_status();
  loom_type_t global_type = loom_global_symbol_type(module, symbol);
  loom_type_t loaded_type = loom_module_value_type(module, results.values[0]);
  if (!loom_global_types_match(global_type, loaded_type)) {
    IREE_RETURN_IF_ERROR(loom_global_emit_type_mismatch(
        op, symbol->defining_op, emitter, IREE_SV("loaded value"), loaded_type,
        global_type));
  }
  return iree_ok_status();
}

iree_status_t loom_global_store_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t ref = loom_global_store_global(op);
  if (!loom_symbol_ref_is_valid(ref) ||
      ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }

  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return iree_ok_status();
  }

  loom_type_t global_type = loom_global_symbol_type(module, symbol);
  loom_type_t stored_type =
      loom_module_value_type(module, loom_global_store_value(op));
  if (!loom_global_types_match(global_type, stored_type)) {
    IREE_RETURN_IF_ERROR(loom_global_emit_type_mismatch(
        op, symbol->defining_op, emitter, IREE_SV("stored value"), stored_type,
        global_type));
  }
  return iree_ok_status();
}
