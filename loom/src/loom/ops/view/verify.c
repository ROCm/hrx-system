// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
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

static iree_status_t loom_view_verify_atomic_kind(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type, uint8_t kind,
    bool allow_exchange) {
  if (iree_any_bit_set(op->instance_flags, LOOM_MEMORY_ACCESS_FLAG_NOFTZ) &&
      kind != LOOM_ATOMIC_KIND_ADDF) {
    return loom_view_emit_attribute_value_constraint(
        emitter, op, IREE_SV("memory_flags"), op->instance_flags,
        IREE_SV("noftz requires floating-point atomic addition"));
  }
  if (!allow_exchange && loom_atomic_kind_is_exchange(kind)) {
    return loom_view_emit_attribute_value_constraint(
        emitter, op, IREE_SV("kind"), kind,
        IREE_SV("non-exchange atomic reduce kind"));
  }
  if (!loom_atomic_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  if (!loom_type_is_scalar(value_type)) {
    return iree_ok_status();
  }

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
    if (source_size == result_size) {
      continue;
    }

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
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_subview_result(op));
  if (loom_type_is_view(source_type) && loom_type_is_view(result_type) &&
      loom_type_view_alignment(source_type) !=
          loom_type_view_alignment(result_type)) {
    return loom_view_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("the same element-access alignment as the result view"));
  }
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

  if (!loom_type_is_view(source_type) || !loom_type_is_view(result_type) ||
      !loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_view_refine_verify_static_dimensions(op, emitter, source_type,
                                                   result_type);
}

iree_status_t loom_view_bitcast_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_type_t types[] = {
      loom_module_value_type(module, loom_view_bitcast_source(op)),
      loom_module_value_type(module, loom_view_bitcast_result(op)),
  };
  const iree_string_view_t names[] = {IREE_SV("source"), IREE_SV("result")};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(types); ++i) {
    if (!loom_type_is_view(types[i])) {
      continue;
    }
    int32_t bit_count =
        loom_scalar_type_bitwidth(loom_type_element_type(types[i]));
    if (loom_type_has_encoding(types[i]) || bit_count <= 0 ||
        bit_count % 8 != 0) {
      loom_diagnostic_param_t params[] = {
          loom_param_string(names[i]),
          loom_param_type(types[i]),
          loom_param_string(IREE_SV(
              "dense unencoded view with byte-addressable scalar elements")),
      };
      return loom_view_emit(emitter, op,
                            i == 0 ? LOOM_ERR_TYPE_003 : LOOM_ERR_TYPE_004,
                            params, IREE_ARRAYSIZE(params));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_view_load_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_load_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_load_static_indices(op), loom_view_load_indices(op).count));
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_LOAD,
                                  emitter);
}

iree_status_t loom_view_store_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_view_store_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_view_store_static_indices(op), loom_view_store_indices(op).count));
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_STORE,
                                  emitter);
}

loom_trait_flags_t loom_view_atomic_load_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  (void)module;
  const loom_trait_flags_t traits = op->traits & ~LOOM_TRAIT_MEMORY_FENCE;
  return loom_view_atomic_load_ordering(op) == LOOM_ATOMIC_ORDERING_RELAXED
             ? traits
             : traits | LOOM_TRAIT_MEMORY_FENCE;
}

loom_trait_flags_t loom_view_atomic_store_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  (void)module;
  const loom_trait_flags_t traits = op->traits & ~LOOM_TRAIT_MEMORY_FENCE;
  return loom_view_atomic_store_ordering(op) == LOOM_ATOMIC_ORDERING_RELAXED
             ? traits
             : traits | LOOM_TRAIT_MEMORY_FENCE;
}

iree_status_t loom_view_atomic_load_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"),
      loom_module_value_type(module, loom_view_atomic_load_view(op)),
      loom_view_atomic_load_static_indices(op),
      loom_view_atomic_load_indices(op).count));
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_LOAD,
                                  emitter);
}

iree_status_t loom_view_atomic_store_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"),
      loom_module_value_type(module, loom_view_atomic_store_view(op)),
      loom_view_atomic_store_static_indices(op),
      loom_view_atomic_store_indices(op).count));
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_STORE,
                                  emitter);
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
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_ATOMIC,
                                  emitter);
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
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_ATOMIC,
                                  emitter);
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
  return loom_cache_policy_verify(module, op, LOOM_CACHE_POLICY_ACCESS_ATOMIC,
                                  emitter);
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
