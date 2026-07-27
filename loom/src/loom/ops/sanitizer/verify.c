// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/view/access_verifier.h"

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_sanitizer_emit(iree_diagnostic_emitter_t emitter,
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

static bool loom_sanitizer_optional_attr_is_present(const loom_op_t* op,
                                                    uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static iree_status_t loom_sanitizer_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_static_access_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t offset, int64_t size, int64_t bound) {
  int64_t total = INT64_MAX;
  if (offset <= INT64_MAX - size) {
    total = offset + size;
  }
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),  loom_param_i64(offset), loom_param_i64(size),
      loom_param_i64(total), loom_param_i64(bound),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_SUBRANGE_004, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_static_extent_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint64_t actual_count,
    uint64_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32((uint32_t)iree_min(actual_count, UINT32_MAX)),
      loom_param_i64((int64_t)iree_min(expected_count, (uint64_t)INT64_MAX)),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_SUBRANGE_003, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_static_extent_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, int64_t extent) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("static_extents")),
      loom_param_i64(extent),
      loom_param_string(IREE_SV("positive extent")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_assert_access_verify_static_extents(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t view_type, loom_attribute_t static_indices,
    loom_attribute_t static_extents) {
  if (loom_attr_is_absent(static_extents) ||
      static_extents.kind != LOOM_ATTR_I64_ARRAY ||
      !loom_type_is_view(view_type)) {
    return iree_ok_status();
  }

  const uint8_t view_rank = loom_type_rank(view_type);
  if (static_extents.count != view_rank) {
    return loom_sanitizer_emit_static_extent_count_mismatch(
        emitter, op, IREE_SV("static_extents"), static_extents.count,
        view_rank);
  }

  for (uint16_t axis = 0; axis < static_extents.count; ++axis) {
    const int64_t extent = static_extents.i64_array[axis];
    if (extent <= 0) {
      return loom_sanitizer_emit_static_extent_constraint(emitter, op, extent);
    }
    if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
        axis >= static_indices.count) {
      continue;
    }
    const int64_t origin = static_indices.i64_array[axis];
    if (origin == INT64_MIN || loom_type_dim_is_dynamic_at(view_type, axis)) {
      continue;
    }
    const int64_t bound = loom_type_dim_static_size_at(view_type, axis);
    if (origin < 0 || origin > bound || extent > bound - origin) {
      return loom_sanitizer_emit_static_access_out_of_bounds(
          emitter, op, axis, origin, extent, bound);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_emit_static_count_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    int64_t static_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("static_count")),
      loom_param_i64(static_count),
      loom_param_string(IREE_SV("positive count")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_static_stride_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, int64_t stride) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("static_strides")),
      loom_param_i64(stride),
      loom_param_string(IREE_SV("non-negative stride")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_static_stride_progress_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("static_strides")),
      loom_param_i64(0),
      loom_param_string(IREE_SV("non-zero stride for repeated accesses")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_assert_accesses_verify_static_shape(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t view_type, loom_attribute_t static_indices,
    loom_attribute_t static_extents, loom_attribute_t static_strides,
    int64_t static_count) {
  if (!loom_type_is_view(view_type)) return iree_ok_status();
  if (static_count <= 0) {
    return loom_sanitizer_emit_static_count_constraint(emitter, op,
                                                       static_count);
  }

  const uint8_t view_rank = loom_type_rank(view_type);
  if (static_extents.kind != LOOM_ATTR_I64_ARRAY ||
      static_extents.count != view_rank) {
    return loom_sanitizer_emit_static_extent_count_mismatch(
        emitter, op, IREE_SV("static_extents"), static_extents.count,
        view_rank);
  }
  if (static_strides.kind != LOOM_ATTR_I64_ARRAY ||
      static_strides.count != view_rank) {
    return loom_sanitizer_emit_static_extent_count_mismatch(
        emitter, op, IREE_SV("static_strides"), static_strides.count,
        view_rank);
  }

  bool has_non_zero_stride = false;
  for (uint16_t axis = 0; axis < view_rank; ++axis) {
    const int64_t extent = static_extents.i64_array[axis];
    if (extent <= 0) {
      return loom_sanitizer_emit_static_extent_constraint(emitter, op, extent);
    }
    const int64_t stride = static_strides.i64_array[axis];
    if (stride < 0) {
      return loom_sanitizer_emit_static_stride_constraint(emitter, op, stride);
    }
    has_non_zero_stride = has_non_zero_stride || stride != 0;

    if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
        axis >= static_indices.count) {
      continue;
    }
    const int64_t origin = static_indices.i64_array[axis];
    if (origin == INT64_MIN || loom_type_dim_is_dynamic_at(view_type, axis)) {
      continue;
    }
    int64_t final_offset = 0;
    if (!iree_checked_mul_i64(static_count - 1, stride, &final_offset)) {
      final_offset = INT64_MAX;
    }
    int64_t final_origin = 0;
    if (!iree_checked_add_i64(origin, final_offset, &final_origin)) {
      final_origin = INT64_MAX;
    }
    const int64_t bound = loom_type_dim_static_size_at(view_type, axis);
    if (origin < 0 || final_origin < 0 || final_origin > bound ||
        extent > bound - final_origin) {
      return loom_sanitizer_emit_static_access_out_of_bounds(
          emitter, op, axis, final_origin, extent, bound);
    }
  }
  if (static_count > 1 && !has_non_zero_stride) {
    return loom_sanitizer_emit_static_stride_progress_constraint(emitter, op);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Predicate assertions
//===----------------------------------------------------------------------===//

static bool loom_sanitizer_type_accepts_integer_predicates(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_type_accepts_float_predicates(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_sanitizer_type_accepts_predicate(loom_type_t type,
                                                  uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
      return loom_sanitizer_type_accepts_integer_predicates(type);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_sanitizer_type_accepts_float_predicates(type);
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static iree_string_view_t loom_sanitizer_predicate_expected_type(
    uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return IREE_SV("floating-point value");
    default:
      return IREE_SV("integer, index, or offset value");
  }
}

static bool loom_sanitizer_value_is_assert_operand(loom_value_slice_t values,
                                                   loom_value_id_t value_id) {
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] == value_id) return true;
  }
  return false;
}

static void loom_sanitizer_format_predicate_arg(char* buffer,
                                                iree_host_size_t capacity,
                                                uint16_t predicate_index,
                                                uint8_t argument_index) {
  iree_snprintf(buffer, capacity, "predicates[%u].arg[%u]", predicate_index,
                argument_index);
}

static iree_status_t loom_sanitizer_emit_unlisted_predicate_value(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, uint16_t predicate_index, uint8_t argument_index,
    iree_string_view_t expected_constraint) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_string(expected_constraint),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_predicate_value_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t predicate_index, uint8_t argument_index, loom_type_t actual_type,
    iree_string_view_t expected_type) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_verify_predicates_reference_values(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_slice_t values,
    loom_attribute_t predicates, iree_string_view_t expected_constraint) {
  for (uint16_t predicate_index = 0; predicate_index < predicates.count;
       ++predicate_index) {
    const loom_predicate_t* predicate =
        &predicates.predicate_list[predicate_index];
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE) continue;
      if (predicate->args[argument_index] < 0) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index,
            expected_constraint);
      }
      loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (!loom_sanitizer_value_is_assert_operand(values, value_id)) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index,
            expected_constraint);
      }
      loom_type_t value_type = loom_module_value_type(module, value_id);
      if (!loom_sanitizer_type_accepts_predicate(value_type, predicate->kind)) {
        return loom_sanitizer_emit_predicate_value_type(
            emitter, op, predicate_index, argument_index, value_type,
            loom_sanitizer_predicate_expected_type(predicate->kind));
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Layout assertions
//===----------------------------------------------------------------------===//

static iree_status_t loom_sanitizer_verify_type_has_encoding(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field_name, loom_type_t type) {
  if (!loom_type_is_view(type) || loom_type_has_encoding(type)) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(IREE_SV("view type")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_ENCODING_001, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_verify_static_dimensions(
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
    return loom_sanitizer_emit(emitter, op, LOOM_ERR_SHAPE_001, params,
                               IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

iree_status_t loom_sanitizer_assert_access_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_sanitizer_assert_access_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_sanitizer_assert_access_static_indices(op),
      loom_sanitizer_assert_access_indices(op).count));
  return loom_sanitizer_assert_access_verify_static_extents(
      op, emitter, view_type, loom_sanitizer_assert_access_static_indices(op),
      loom_sanitizer_assert_access_static_extents(op));
}

iree_status_t loom_sanitizer_assert_accesses_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_sanitizer_assert_accesses_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_sanitizer_assert_accesses_static_indices(op),
      loom_sanitizer_assert_accesses_indices(op).count));
  return loom_sanitizer_assert_accesses_verify_static_shape(
      op, emitter, view_type, loom_sanitizer_assert_accesses_static_indices(op),
      loom_sanitizer_assert_accesses_static_extents(op),
      loom_sanitizer_assert_accesses_static_strides(op),
      loom_sanitizer_assert_accesses_static_count(op));
}

iree_status_t loom_sanitizer_race_access_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_sanitizer_race_access_view(op));
  IREE_RETURN_IF_ERROR(loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_sanitizer_race_access_static_indices(op),
      loom_sanitizer_race_access_indices(op).count));

  const bool atomic = loom_sanitizer_race_access_atomic(op);
  const bool has_ordering = loom_sanitizer_optional_attr_is_present(
      op, loom_sanitizer_race_access_ordering_ATTR_INDEX);
  const bool has_scope = loom_sanitizer_optional_attr_is_present(
      op, loom_sanitizer_race_access_scope_ATTR_INDEX);
  if (atomic) {
    if (!has_ordering) {
      return loom_sanitizer_emit_attribute_value_constraint(
          emitter, op, IREE_SV("ordering"), /*actual_value=*/0,
          IREE_SV("present when atomic is true"));
    }
    if (!has_scope) {
      return loom_sanitizer_emit_attribute_value_constraint(
          emitter, op, IREE_SV("scope"), /*actual_value=*/0,
          IREE_SV("present when atomic is true"));
    }
    return iree_ok_status();
  }
  if (has_ordering) {
    return loom_sanitizer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"),
        loom_sanitizer_race_access_ordering(op),
        IREE_SV("absent when atomic is false"));
  }
  if (has_scope) {
    return loom_sanitizer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("scope"), loom_sanitizer_race_access_scope(op),
        IREE_SV("absent when atomic is false"));
  }
  return iree_ok_status();
}

iree_status_t loom_sanitizer_race_sync_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  (void)module;

  loom_atomic_ordering_t ordering = loom_sanitizer_race_sync_ordering(op);
  if (ordering == LOOM_ATOMIC_ORDERING_RELAXED) {
    return loom_sanitizer_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"), ordering,
        IREE_SV("acquire, release, acq_rel, or seq_cst ordering"));
  }
  return iree_ok_status();
}

iree_status_t loom_sanitizer_assert_value_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_sanitizer_verify_predicates_reference_values(
      module, op, emitter, loom_sanitizer_assert_value_values(op),
      loom_sanitizer_assert_value_predicates(op),
      IREE_SV("an asserted value operand"));
}

iree_status_t loom_sanitizer_assert_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_sanitizer_verify_predicates_reference_values(
      module, op, emitter, loom_sanitizer_assert_op_values(op),
      loom_sanitizer_assert_op_predicates(op), IREE_SV("an assertion operand"));
}

iree_status_t loom_sanitizer_assert_layout_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_result(op));

  IREE_RETURN_IF_ERROR(loom_sanitizer_verify_type_has_encoding(
      op, emitter, IREE_SV("view type layout"), source_type));
  IREE_RETURN_IF_ERROR(loom_sanitizer_verify_type_has_encoding(
      op, emitter, IREE_SV("result type layout"), result_type));

  if (!loom_type_is_view(source_type) || !loom_type_is_view(result_type) ||
      !loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_sanitizer_verify_static_dimensions(op, emitter, source_type,
                                                 result_type);
}
