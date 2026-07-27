// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"
#include "loom/ops/view/access_verifier.h"
#include "loom/ops/view/ops.h"

static iree_status_t loom_view_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_view_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_view_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_view_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_verify_type_has_encoding(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field_name, loom_type_t type) {
  if (!loom_type_is_view(type) || loom_type_has_encoding(type)) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(IREE_SV("view type")),
  };
  return loom_view_emit(emitter, op, LOOM_ERR_ENCODING_001, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_verify_optional_cache_policy(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t cache_scope_attr_index, uint16_t cache_temporal_attr_index,
    loom_cache_policy_access_t access) {
  loom_attribute_t cache_scope_attr = loom_op_attrs(op)[cache_scope_attr_index];
  loom_attribute_t cache_temporal_attr =
      loom_op_attrs(op)[cache_temporal_attr_index];
  bool has_cache_scope = !loom_attr_is_absent(cache_scope_attr);
  bool has_cache_temporal = !loom_attr_is_absent(cache_temporal_attr);
  if (!has_cache_scope && !has_cache_temporal) return iree_ok_status();
  if (!has_cache_scope) {
    return loom_view_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_scope"), 0,
        IREE_SV("present when cache_temporal is present"));
  }
  if (!has_cache_temporal) {
    return loom_view_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_temporal"), 0,
        IREE_SV("present when cache_scope is present"));
  }
  if (cache_scope_attr.kind != LOOM_ATTR_ENUM ||
      cache_temporal_attr.kind != LOOM_ATTR_ENUM) {
    return iree_ok_status();
  }

  uint8_t cache_scope = loom_attr_as_enum(cache_scope_attr);
  uint8_t cache_temporal = loom_attr_as_enum(cache_temporal_attr);
  loom_cache_policy_error_t error =
      loom_cache_policy_validate(cache_scope, cache_temporal, access);
  if (error == LOOM_CACHE_POLICY_ERROR_NONE) return iree_ok_status();
  iree_string_view_t attr_name = loom_cache_policy_error_attr_name(error);
  int64_t actual_value =
      iree_string_view_equal(attr_name, IREE_SV("cache_scope"))
          ? cache_scope
          : cache_temporal;
  return loom_view_emit_attribute_value_constraint(
      emitter, op, attr_name, actual_value,
      loom_cache_policy_error_expected_constraint(error));
}

static iree_status_t loom_view_verify_atomic_kind(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type, uint8_t kind,
    bool allow_exchange) {
  if (!allow_exchange && loom_atomic_kind_is_exchange(kind)) {
    return loom_view_emit_attribute_value_constraint(
        emitter, op, IREE_SV("kind"), kind,
        IREE_SV("non-exchange atomic reduce kind"));
  }
  if (!loom_atomic_kind_is_valid(kind)) return iree_ok_status();
  if (!loom_type_is_scalar(value_type)) return iree_ok_status();

  loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_atomic_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_atomic_kind_accepts_float(kind)) {
    return iree_ok_status();
  }

  iree_string_view_t expected_constraint =
      loom_atomic_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for atomic kind")
          : IREE_SV("floating-point element type for atomic kind");
  return loom_view_emit_operand_constraint(emitter, op, value_name, value_type,
                                           expected_constraint);
}

static iree_status_t loom_view_verify_atomic_cmpxchg_ordering(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint8_t success_ordering, uint8_t failure_ordering) {
  loom_atomic_cmpxchg_ordering_error_t error =
      loom_atomic_cmpxchg_ordering_validate(success_ordering, failure_ordering);
  if (error == LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE) {
    return iree_ok_status();
  }
  iree_string_view_t attr_name =
      loom_atomic_cmpxchg_ordering_error_attr_name(error);
  return loom_view_emit_attribute_value_constraint(
      emitter, op, attr_name, failure_ordering,
      loom_atomic_cmpxchg_ordering_error_expected_constraint(error));
}

static iree_status_t loom_view_refine_verify_static_dimensions(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t source_type, loom_type_t result_type) {
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis) ||
        loom_type_dim_is_dynamic_at(result_type, axis)) {
      continue;
    }
    int64_t source_size = loom_type_dim_static_size_at(source_type, axis);
    int64_t result_size = loom_type_dim_static_size_at(result_type, axis);
    if (source_size == result_size) continue;

    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("source static dimension")),
        loom_param_i64(source_size),
        loom_param_string(IREE_SV("result static dimension")),
        loom_param_i64(result_size),
    };
    return loom_view_emit(emitter, op, LOOM_ERR_SHAPE_001, params,
                          IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_view_subview_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_attribute_t static_offsets = loom_view_subview_static_offsets(op);
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_subview_source(op));
  return loom_view_verify_index_list_rank(
      module, op, emitter, IREE_SV("source"), source_type, static_offsets,
      loom_view_subview_offsets(op).count);
}

iree_status_t loom_view_refine_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_refine_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_refine_result(op));

  IREE_RETURN_IF_ERROR(loom_view_verify_type_has_encoding(
      op, emitter, IREE_SV("source type layout"), source_type));
  IREE_RETURN_IF_ERROR(loom_view_verify_type_has_encoding(
      op, emitter, IREE_SV("result type layout"), result_type));

  if (!loom_type_is_view(source_type) || !loom_type_is_view(result_type) ||
      !loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_view_refine_verify_static_dimensions(op, emitter, source_type,
                                                   result_type);
}

iree_status_t loom_view_load_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_load_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_load_static_indices(op), loom_view_load_indices(op).count));
  return loom_view_verify_optional_cache_policy(
      emitter, op, loom_view_load_cache_scope_ATTR_INDEX,
      loom_view_load_cache_temporal_ATTR_INDEX, LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_view_store_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_store_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_store_static_indices(op), loom_view_store_indices(op).count));
  return loom_view_verify_optional_cache_policy(
      emitter, op, loom_view_store_cache_scope_ATTR_INDEX,
      loom_view_store_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_view_atomic_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_atomic_reduce_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_atomic_reduce_static_indices(op),
      loom_view_atomic_reduce_indices(op).count));

  loom_type_t value_type =
      loom_module_value_type(module, loom_view_atomic_reduce_value(op));
  IREE_RETURN_IF_ERROR(
      loom_view_verify_atomic_kind(emitter, op, IREE_SV("value"), value_type,
                                   loom_view_atomic_reduce_kind(op), false));
  return loom_view_verify_optional_cache_policy(
      emitter, op, loom_view_atomic_reduce_cache_scope_ATTR_INDEX,
      loom_view_atomic_reduce_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_view_atomic_rmw_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_atomic_rmw_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_atomic_rmw_static_indices(op),
      loom_view_atomic_rmw_indices(op).count));

  loom_type_t value_type =
      loom_module_value_type(module, loom_view_atomic_rmw_value(op));
  IREE_RETURN_IF_ERROR(
      loom_view_verify_atomic_kind(emitter, op, IREE_SV("value"), value_type,
                                   loom_view_atomic_rmw_kind(op), true));
  return loom_view_verify_optional_cache_policy(
      emitter, op, loom_view_atomic_rmw_cache_scope_ATTR_INDEX,
      loom_view_atomic_rmw_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_view_atomic_cmpxchg_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_atomic_cmpxchg_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_atomic_cmpxchg_static_indices(op),
      loom_view_atomic_cmpxchg_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_view_verify_atomic_cmpxchg_ordering(
      emitter, op, loom_view_atomic_cmpxchg_success_ordering(op),
      loom_view_atomic_cmpxchg_failure_ordering(op)));
  return loom_view_verify_optional_cache_policy(
      emitter, op, loom_view_atomic_cmpxchg_cache_scope_ATTR_INDEX,
      loom_view_atomic_cmpxchg_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_view_prefetch_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_attribute_t static_indices = loom_view_prefetch_static_indices(op);
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_prefetch_view(op));
  return loom_view_verify_index_list_rank(module, op, emitter, IREE_SV("view"),
                                          view_type, static_indices,
                                          loom_view_prefetch_indices(op).count);
}
