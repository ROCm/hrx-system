// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/collective_arithmetic.h"

#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/features.h"

struct loom_spirv_extrema_row_t {
  // Scalar integer constant descriptor.
  uint32_t constant;
  // Integer vector constructor descriptor, unused for scalar carriers.
  uint32_t construct;
  // Floating-to-integer bitcast descriptor.
  uint32_t to_bits;
  // Integer-to-floating bitcast descriptor.
  uint32_t from_bits;
  // Integer magnitude/sign mask descriptor.
  uint32_t bitwise_and;
  // Integer sortable-key inversion descriptor.
  uint32_t bitwise_xor;
  // Arithmetic shift descriptor for sign extension.
  uint32_t shift_right;
  // Signed comparison descriptor for NaN detection.
  uint32_t greater_than;
  // Integer equality descriptor for empty-prefix detection.
  uint32_t equal;
  // Component-wise integer selection descriptor.
  uint32_t select;
  // Signed min/max descriptors indexed by mode, then minimum/maximum.
  uint32_t subgroup[4][2];
  // IEEE component width in bits.
  uint8_t bit_count;
  // Number of independent components in the native carrier.
  uint8_t lane_count;
};

#include "loom/target/arch/spirv/lower/collective_rows.inl"

const loom_spirv_extrema_row_t* loom_spirv_collective_extrema_row(
    loom_type_t source_type) {
  uint32_t type_index = 0;
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_F16:
      type_index = 0;
      break;
    case LOOM_SCALAR_TYPE_F32:
      type_index = 1;
      break;
    case LOOM_SCALAR_TYPE_F64:
      type_index = 2;
      break;
    default:
      return NULL;
  }
  uint64_t lane_count = 1;
  if (loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
      loom_type_is_all_static(source_type)) {
    lane_count = loom_type_dim_static_size_at(source_type, 0);
  } else if (!loom_type_is_scalar(source_type)) {
    return NULL;
  }
  return lane_count >= 1 && lane_count <= 4
             ? &kExtremaRows[type_index][lane_count - 1]
             : NULL;
}

static bool loom_spirv_extrema_is_minimum(loom_combining_kind_t kind) {
  return kind == LOOM_COMBINING_KIND_MINNUMF ||
         kind == LOOM_COMBINING_KIND_MINIMUMF;
}

static int64_t loom_spirv_extrema_magnitude_mask(
    const loom_spirv_extrema_row_t* row) {
  return (int64_t)((UINT64_C(1) << (row->bit_count - 1)) - 1);
}

static int64_t loom_spirv_extrema_identity(
    const loom_spirv_collective_extrema_t* arithmetic) {
  const int64_t mask = loom_spirv_extrema_magnitude_mask(arithmetic->row);
  return loom_spirv_extrema_is_minimum(arithmetic->kind) ? mask : -mask - 1;
}

static int64_t loom_spirv_extrema_infinity(
    const loom_spirv_extrema_row_t* row) {
  return row->bit_count == 16   ? INT64_C(0x7c00)
         : row->bit_count == 32 ? INT64_C(0x7f800000)
                                : INT64_C(0x7ff0000000000000);
}

static iree_status_t loom_spirv_collective_type(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    uint8_t lane_count, loom_type_t* out_type) {
  const loom_type_t type =
      lane_count == 1
          ? loom_type_scalar(element_type)
          : loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                                loom_dim_pack_static(lane_count), 0);
  return loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1, type, out_type);
}

iree_status_t loom_spirv_collective_extrema_initialize(
    loom_low_lower_context_t* context, const loom_spirv_extrema_row_t* row,
    loom_combining_kind_t kind,
    loom_spirv_collective_extrema_t* out_arithmetic) {
  *out_arithmetic = (loom_spirv_collective_extrema_t){.row = row, .kind = kind};
  const loom_scalar_type_t integer_type =
      row->bit_count == 16   ? LOOM_SCALAR_TYPE_I16
      : row->bit_count == 32 ? LOOM_SCALAR_TYPE_I32
                             : LOOM_SCALAR_TYPE_I64;
  const loom_scalar_type_t float_type =
      row->bit_count == 16   ? LOOM_SCALAR_TYPE_F16
      : row->bit_count == 32 ? LOOM_SCALAR_TYPE_F32
                             : LOOM_SCALAR_TYPE_F64;
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(
      context, float_type, row->lane_count, &out_arithmetic->float_type));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(
      context, integer_type, row->lane_count, &out_arithmetic->integer_type));
  IREE_RETURN_IF_ERROR(
      loom_spirv_collective_type(context, LOOM_SCALAR_TYPE_I1, row->lane_count,
                                 &out_arithmetic->predicate_type));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(
      context, integer_type, 1, &out_arithmetic->constant_type));
  const iree_string_view_t constant_key =
      row->bit_count == 16   ? IREE_SV("i16_value")
      : row->bit_count == 32 ? IREE_SV("i32_value")
                             : IREE_SV("i64_value");
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   constant_key, &out_arithmetic->constant_key);
}

static iree_status_t loom_spirv_collective_emit(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint32_t descriptor_ref, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attributes,
    loom_type_t result_type, loom_value_id_t* out_value) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ref)};
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count, attributes, &result_type,
      1, NULL, 0, location, &emitted));
  *out_value = loom_op_const_results(emitted)[0];
  return iree_ok_status();
}

static iree_status_t loom_spirv_collective_binary(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint32_t descriptor_ref, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_spirv_collective_emit(
      context, location, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), result_type, out_value);
}

static iree_status_t loom_spirv_extrema_constant(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic, int64_t literal,
    loom_value_id_t* out_value) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context),
          arithmetic->row->constant)};
  const loom_named_attr_t attribute = {.name_id = arithmetic->constant_key,
                                       .value = loom_attr_i64(literal)};
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&attribute, 1),
      arithmetic->constant_type, location, &emitted));
  const loom_value_id_t scalar = loom_op_const_results(emitted)[0];
  if (arithmetic->row->lane_count == 1) {
    *out_value = scalar;
    return iree_ok_status();
  }
  const loom_value_id_t components[] = {scalar, scalar, scalar, scalar};
  return loom_spirv_collective_emit(
      context, location, arithmetic->row->construct, components,
      arithmetic->row->lane_count, loom_named_attr_slice_empty(),
      arithmetic->integer_type, out_value);
}

// The sign transform is an involution: negative IEEE bit patterns reverse
// their magnitude order, while the sign bit retains signed integer ordering.
static iree_status_t loom_spirv_extrema_invert_sign(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic,
    loom_spirv_extrema_constants_t constants, loom_value_id_t value,
    loom_value_id_t* out_value) {
  loom_value_id_t sign = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
      context, location, arithmetic->row->shift_right, value,
      constants.sign_shift, arithmetic->integer_type, &sign));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
      context, location, arithmetic->row->bitwise_and, sign,
      constants.magnitude_mask, arithmetic->integer_type, &sign));
  return loom_spirv_collective_binary(context, location,
                                      arithmetic->row->bitwise_xor, value, sign,
                                      arithmetic->integer_type, out_value);
}

iree_status_t loom_spirv_collective_extrema_encode(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic, loom_value_id_t value,
    bool exclusive, loom_spirv_extrema_constants_t* out_constants,
    loom_value_id_t* out_key) {
  const loom_spirv_extrema_row_t* row = arithmetic->row;
  const int64_t mask = loom_spirv_extrema_magnitude_mask(row);
  IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
      context, location, arithmetic, mask, &out_constants->magnitude_mask));
  IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
      context, location, arithmetic, row->bit_count - 1,
      &out_constants->sign_shift));
  IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
      context, location, arithmetic, loom_spirv_extrema_infinity(row),
      &out_constants->infinity));
  loom_value_id_t bits, magnitude, is_nan, ordered;
  IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
      context, location, row->to_bits, &value, 1, loom_named_attr_slice_empty(),
      arithmetic->integer_type, &bits));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
      context, location, row->bitwise_and, bits, out_constants->magnitude_mask,
      arithmetic->integer_type, &magnitude));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
      context, location, row->greater_than, magnitude, out_constants->infinity,
      arithmetic->predicate_type, &is_nan));
  IREE_RETURN_IF_ERROR(loom_spirv_extrema_invert_sign(
      context, location, arithmetic, *out_constants, bits, &ordered));
  const bool minimum = loom_spirv_extrema_is_minimum(arithmetic->kind);
  const bool propagate_nan = arithmetic->kind == LOOM_COMBINING_KIND_MINIMUMF ||
                             arithmetic->kind == LOOM_COMBINING_KIND_MAXIMUMF;
  int64_t nan_literal = loom_spirv_extrema_identity(arithmetic);
  if (propagate_nan) {
    nan_literal = minimum ? -mask - 1 : mask;
  } else if (exclusive) {
    nan_literal += minimum ? -1 : 1;
  }
  loom_value_id_t nan_key = out_constants->magnitude_mask;
  if (nan_literal != mask) {
    IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
        context, location, arithmetic, nan_literal, &nan_key));
  }
  const loom_value_id_t operands[] = {is_nan, nan_key, ordered};
  return loom_spirv_collective_emit(
      context, location, row->select, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), arithmetic->integer_type, out_key);
}

iree_status_t loom_spirv_collective_extrema_decode(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic,
    loom_spirv_extrema_constants_t constants, loom_value_id_t key,
    bool exclusive, loom_value_id_t* out_value) {
  const loom_spirv_extrema_row_t* row = arithmetic->row;
  if (exclusive) {
    loom_value_id_t identity = constants.magnitude_mask;
    const bool minimum = loom_spirv_extrema_is_minimum(arithmetic->kind);
    if (!minimum) {
      IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
          context, location, arithmetic,
          loom_spirv_extrema_identity(arithmetic), &identity));
    }
    loom_value_id_t empty;
    IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
        context, location, row->equal, key, identity,
        arithmetic->predicate_type, &empty));
    loom_value_id_t infinity = constants.infinity;
    if (!minimum) {
      IREE_RETURN_IF_ERROR(loom_spirv_extrema_constant(
          context, location, arithmetic, -loom_spirv_extrema_infinity(row) - 1,
          &infinity));
    }
    const loom_value_id_t operands[] = {empty, infinity, key};
    IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
        context, location, row->select, operands, IREE_ARRAYSIZE(operands),
        loom_named_attr_slice_empty(), arithmetic->integer_type, &key));
  }
  loom_value_id_t bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_spirv_extrema_invert_sign(
      context, location, arithmetic, constants, key, &bits));
  return loom_spirv_collective_emit(context, location, row->from_bits, &bits, 1,
                                    loom_named_attr_slice_empty(),
                                    arithmetic->float_type, out_value);
}

typedef struct loom_spirv_subgroup_extrema_plan_t {
  // Typed arithmetic shared with workgroup partials and final results.
  loom_spirv_collective_extrema_t arithmetic;
  // Native group operation selected from the source collective shape.
  loom_spirv_collective_mode_t mode;
  // Explicit cluster extent, used only for clustered reductions.
  uint32_t cluster_size;
  // Immediate name for the retained cluster extent.
  loom_string_id_t cluster_key;
} loom_spirv_subgroup_extrema_plan_t;

iree_status_t loom_spirv_select_subgroup_extrema(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const bool scan = loom_kernel_subgroup_scan_isa(source_op);
  const loom_combining_kind_t kind =
      scan ? loom_kernel_subgroup_scan_kind(source_op)
           : loom_kernel_subgroup_reduce_kind(source_op);
  if (kind != LOOM_COMBINING_KIND_MINNUMF &&
      kind != LOOM_COMBINING_KIND_MAXNUMF &&
      kind != LOOM_COMBINING_KIND_MINIMUMF &&
      kind != LOOM_COMBINING_KIND_MAXIMUMF) {
    return iree_ok_status();
  }
  const loom_value_id_t value =
      scan ? loom_kernel_subgroup_scan_value(source_op)
           : loom_kernel_subgroup_reduce_value(source_op);
  const loom_spirv_extrema_row_t* row = loom_spirv_collective_extrema_row(
      loom_module_value_type(loom_low_lower_context_module(context), value));
  if (!row) {
    return iree_ok_status();
  }
  loom_spirv_subgroup_extrema_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_spirv_subgroup_extrema_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_spirv_collective_extrema_initialize(
      context, row, kind, &plan->arithmetic));
  if (scan) {
    plan->mode = loom_kernel_subgroup_scan_mode(source_op) ==
                         LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE
                     ? LOOM_SPIRV_COLLECTIVE_INCLUSIVE_SCAN
                     : LOOM_SPIRV_COLLECTIVE_EXCLUSIVE_SCAN;
  } else {
    plan->cluster_size =
        (uint32_t)loom_kernel_subgroup_reduce_cluster_size(source_op);
    const uint32_t subgroup_size =
        loom_low_lower_context_bundle(context)->snapshot->subgroup_size;
    if (plan->cluster_size != 0 && plan->cluster_size != subgroup_size) {
      plan->mode = LOOM_SPIRV_COLLECTIVE_CLUSTERED_REDUCE;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          loom_low_lower_context_module(context), IREE_SV("cluster_size"),
          &plan->cluster_key));
    }
  }
  const loom_value_id_t result =
      scan ? loom_kernel_subgroup_scan_result(source_op)
           : loom_kernel_subgroup_reduce_result(source_op);
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, result, plan->arithmetic.float_type));
  *out_plan = loom_low_lower_plan_make(
      scan ? LOOM_OP_KERNEL_SUBGROUP_SCAN : LOOM_OP_KERNEL_SUBGROUP_REDUCE,
      plan);
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_subgroup_extrema(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t selected_plan) {
  const loom_spirv_subgroup_extrema_plan_t* plan = selected_plan.target_data;
  const bool scan = loom_kernel_subgroup_scan_isa(source_op);
  const bool exclusive = plan->mode == LOOM_SPIRV_COLLECTIVE_EXCLUSIVE_SCAN;
  loom_value_id_t value = loom_low_lower_lookup_value(
      context, scan ? loom_kernel_subgroup_scan_value(source_op)
                    : loom_kernel_subgroup_reduce_value(source_op));
  loom_spirv_extrema_constants_t constants;
  IREE_RETURN_IF_ERROR(loom_spirv_collective_extrema_encode(
      context, source_op->location, &plan->arithmetic, value, exclusive,
      &constants, &value));
  const bool minimum = loom_spirv_extrema_is_minimum(plan->arithmetic.kind);
  const uint32_t descriptor =
      plan->arithmetic.row->subgroup[plan->mode][!minimum];
  const loom_named_attr_t cluster = {
      .name_id = plan->cluster_key, .value = loom_attr_i64(plan->cluster_size)};
  IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
      context, source_op->location, descriptor, &value, 1,
      loom_make_named_attr_slice(
          &cluster,
          plan->mode == LOOM_SPIRV_COLLECTIVE_CLUSTERED_REDUCE ? 1 : 0),
      plan->arithmetic.integer_type, &value));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_extrema_decode(
      context, source_op->location, &plan->arithmetic, constants, value,
      exclusive, &value));
  return loom_low_lower_bind_value(
      context,
      scan ? loom_kernel_subgroup_scan_result(source_op)
           : loom_kernel_subgroup_reduce_result(source_op),
      value);
}

struct loom_spirv_product_row_t {
  // Floating zero constant descriptor.
  uint32_t constant;
  // Ordered comparison against zero.
  uint32_t negative;
  // Floating subtraction used to negate a value without integer capabilities.
  uint32_t subtract;
  // Floating selection controlled by a Boolean sign.
  uint32_t select;
  // Scalar magnitude products indexed by native collective mode.
  uint32_t product[4];
  // Scalar extraction instructions for two-, three-, and four-lane packets.
  uint32_t extract[3];
  // Packet constructors for two-, three-, and four-lane results.
  uint32_t construct[3];
};

#define LOOM_SPIRV_PRODUCT_ROW(F)                                                            \
  {                                                                                          \
      .constant = SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_##F,                         \
      .negative = SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_F_ORD_LESS_THAN_##F,                  \
      .subtract = SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_FSUB_##F,                             \
      .select = SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_##F,                             \
      .product =                                                                             \
          {SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_FMUL_REDUCE_##F,           \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_FMUL_CLUSTERED_REDUCE_##F, \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_FMUL_INCLUSIVE_SCAN_##F,   \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_FMUL_EXCLUSIVE_SCAN_##F},  \
      .extract =                                                                             \
          {SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_EXTRACT_V2##F##_##F,               \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_EXTRACT_V3##F##_##F,               \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_EXTRACT_V4##F##_##F},              \
      .construct =                                                                           \
          {SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_CONSTRUCT_V2##F,                   \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_CONSTRUCT_V3##F,                   \
           SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COMPOSITE_CONSTRUCT_V4##F},                  \
  }
static const loom_spirv_product_row_t kProductRows[] = {
    LOOM_SPIRV_PRODUCT_ROW(F16),
    LOOM_SPIRV_PRODUCT_ROW(F32),
};
#undef LOOM_SPIRV_PRODUCT_ROW

static const uint32_t kProductSignParity[] = {
    SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_LOGICAL_XOR_REDUCE_BOOL,
    SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_LOGICAL_XOR_CLUSTERED_REDUCE_BOOL,
    SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_LOGICAL_XOR_INCLUSIVE_SCAN_BOOL,
    SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_LOGICAL_XOR_EXCLUSIVE_SCAN_BOOL,
};

bool loom_spirv_collective_product_needed(loom_low_lower_context_t* context,
                                          loom_combining_kind_t kind,
                                          loom_value_id_t source_value) {
  if (kind != LOOM_COMBINING_KIND_MULF) {
    return false;
  }
  const loom_scalar_type_t element =
      loom_type_element_type(loom_module_value_type(
          loom_low_lower_context_module(context), source_value));
  if (element != LOOM_SCALAR_TYPE_F16 && element != LOOM_SCALAR_TYPE_F32) {
    return false;
  }
  return loom_value_facts_is_subgroup_uniform(loom_value_fact_table_lookup(
      loom_low_lower_context_fact_table(context), source_value));
}

iree_status_t loom_spirv_collective_product_initialize(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_spirv_collective_product_t* out_arithmetic) {
  *out_arithmetic = (loom_spirv_collective_product_t){
      .row = &kProductRows[element_type == LOOM_SCALAR_TYPE_F32]};
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(context, element_type, 1,
                                                  &out_arithmetic->float_type));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(
      context, LOOM_SCALAR_TYPE_I1, 1, &out_arithmetic->predicate_type));
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   element_type == LOOM_SCALAR_TYPE_F16
                                       ? IREE_SV("f16_bits")
                                       : IREE_SV("f32_bits"),
                                   &out_arithmetic->constant_key);
}

iree_status_t loom_spirv_collective_product_emit(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_product_t* arithmetic,
    loom_spirv_collective_mode_t mode, loom_named_attr_slice_t attributes,
    loom_value_id_t value, loom_value_id_t* out_value) {
  const loom_spirv_product_row_t* row = arithmetic->row;
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), row->constant)};
  const loom_named_attr_t attribute = {.name_id = arithmetic->constant_key,
                                       .value = loom_attr_i64(0)};
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&attribute, 1),
      arithmetic->float_type, location, &constant));
  const loom_value_id_t zero = loom_op_const_results(constant)[0];
  loom_value_id_t negative, negated, magnitude, parity, product;
  IREE_RETURN_IF_ERROR(loom_spirv_collective_binary(
      context, location, row->negative, value, zero, arithmetic->predicate_type,
      &negative));
  IREE_RETURN_IF_ERROR(
      loom_spirv_collective_binary(context, location, row->subtract, zero,
                                   value, arithmetic->float_type, &negated));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
      context, location, row->select,
      (loom_value_id_t[]){negative, negated, value}, 3,
      loom_named_attr_slice_empty(), arithmetic->float_type, &magnitude));
  // Uniform products may be strength-reduced through log/exp by a driver.
  // Keep every floating collective input nonnegative and reduce its sign
  // separately, using the same active lanes and empty-prefix identity.
  IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
      context, location, row->product[mode], &magnitude, 1, attributes,
      arithmetic->float_type, &product));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_emit(
      context, location, kProductSignParity[mode], &negative, 1, attributes,
      arithmetic->predicate_type, &parity));
  IREE_RETURN_IF_ERROR(
      loom_spirv_collective_binary(context, location, row->subtract, zero,
                                   product, arithmetic->float_type, &negated));
  return loom_spirv_collective_emit(
      context, location, row->select,
      (loom_value_id_t[]){parity, negated, product}, 3,
      loom_named_attr_slice_empty(), arithmetic->float_type, out_value);
}

typedef struct loom_spirv_subgroup_product_plan_t {
  // Scalar arithmetic shared with workgroup partials.
  loom_spirv_collective_product_t arithmetic;
  // Native source/result packet type; vector1 uses a scalar register.
  loom_type_t packet_type;
  // Collective topology shared by magnitude and sign.
  loom_spirv_collective_mode_t mode;
  // Explicit cluster extent, zero for whole-subgroup operations.
  uint32_t cluster_size;
  // Interned immediate name for a contiguous cluster's extent.
  loom_string_id_t cluster_key;
  // Interned immediate name for packet component extraction.
  loom_string_id_t component_key;
  // Number of scalar components independently reduced.
  uint8_t component_count;
} loom_spirv_subgroup_product_plan_t;

iree_status_t loom_spirv_select_subgroup_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const bool scan = loom_kernel_subgroup_scan_isa(source_op);
  const loom_value_id_t value =
      scan ? loom_kernel_subgroup_scan_value(source_op)
           : loom_kernel_subgroup_reduce_value(source_op);
  if (!loom_spirv_collective_product_needed(context, LOOM_COMBINING_KIND_MULF,
                                            value) ||
      (source_op->instance_flags & LOOM_KERNEL_COLLECTIVEMATHFLAGS_PRESERVE)) {
    return iree_ok_status();
  }
  const uint32_t cluster_size =
      (uint32_t)(scan ? loom_kernel_subgroup_scan_cluster_size(source_op)
                      : loom_kernel_subgroup_reduce_cluster_size(source_op));
  const uint32_t stride =
      (uint32_t)(scan ? loom_kernel_subgroup_scan_cluster_stride(source_op)
                      : loom_kernel_subgroup_reduce_cluster_stride(source_op));
  const uint32_t subgroup_size =
      loom_low_lower_context_bundle(context)->snapshot->subgroup_size;
  const bool clustered = cluster_size != 0 && cluster_size != subgroup_size;
  if (stride > 1 ||
      (scan &&
       (clustered || loom_kernel_subgroup_scan_direction(source_op) !=
                         LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD)) ||
      (clustered && cluster_size != 1 &&
       (cluster_size > subgroup_size ||
        (cluster_size & (cluster_size - 1)) != 0))) {
    return iree_ok_status();
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t type = loom_module_value_type(module, value);
  const loom_scalar_type_t element = loom_type_element_type(type);
  const uint64_t lanes =
      loom_type_is_scalar(type) ? 1
      : loom_type_rank(type) == 1 && loom_type_is_all_static(type)
          ? loom_type_dim_static_size_at(type, 0)
          : 0;
  const uint64_t required_features =
      (clustered ? LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED
                 : LOOM_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC) |
      (element == LOOM_SCALAR_TYPE_F16
           ? LOOM_SPIRV_FEATURE_FLOAT16 |
                 LOOM_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES
           : 0);
  if (lanes == 0 || lanes > 4 ||
      (required_features & ~loom_low_lower_context_bundle(context)
                                ->config->contract_feature_bits)) {
    return iree_ok_status();
  }
  loom_spirv_subgroup_product_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_spirv_subgroup_product_plan_t){
      .component_count = (uint8_t)lanes,
      .cluster_size = cluster_size,
      .mode = clustered ? LOOM_SPIRV_COLLECTIVE_CLUSTERED_REDUCE
                        : LOOM_SPIRV_COLLECTIVE_REDUCE};
  if (scan) {
    plan->mode = loom_kernel_subgroup_scan_mode(source_op) ==
                         LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE
                     ? LOOM_SPIRV_COLLECTIVE_INCLUSIVE_SCAN
                     : LOOM_SPIRV_COLLECTIVE_EXCLUSIVE_SCAN;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_collective_product_initialize(
      context, element, &plan->arithmetic));
  IREE_RETURN_IF_ERROR(loom_spirv_collective_type(
      context, element, (uint8_t)lanes, &plan->packet_type));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("component_index"), &plan->component_key));
  if (clustered) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("cluster_size"), &plan->cluster_key));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context,
      scan ? loom_kernel_subgroup_scan_result(source_op)
           : loom_kernel_subgroup_reduce_result(source_op),
      plan->packet_type));
  *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_subgroup_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t selected_plan) {
  const loom_spirv_subgroup_product_plan_t* plan = selected_plan.target_data;
  const bool scan = loom_kernel_subgroup_scan_isa(source_op);
  const loom_value_id_t packet = loom_low_lower_lookup_value(
      context, scan ? loom_kernel_subgroup_scan_value(source_op)
                    : loom_kernel_subgroup_reduce_value(source_op));
  const loom_named_attr_t cluster = {
      .name_id = plan->cluster_key, .value = loom_attr_i64(plan->cluster_size)};
  const loom_named_attr_slice_t attributes = loom_make_named_attr_slice(
      &cluster, plan->mode == LOOM_SPIRV_COLLECTIVE_CLUSTERED_REDUCE ? 1 : 0);
  loom_value_id_t components[4];
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < plan->component_count && iree_status_is_ok(status);
       ++i) {
    loom_value_id_t value = packet;
    if (plan->component_count > 1) {
      const loom_named_attr_t component = {.name_id = plan->component_key,
                                           .value = loom_attr_i64(i)};
      status = loom_spirv_collective_emit(
          context, source_op->location,
          plan->arithmetic.row->extract[plan->component_count - 2], &packet, 1,
          loom_make_named_attr_slice(&component, 1),
          plan->arithmetic.float_type, &value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_collective_product_emit(
          context, source_op->location, &plan->arithmetic, plan->mode,
          attributes, value, &components[i]);
    }
  }
  if (iree_status_is_ok(status) && plan->component_count > 1) {
    status = loom_spirv_collective_emit(
        context, source_op->location,
        plan->arithmetic.row->construct[plan->component_count - 2], components,
        plan->component_count, loom_named_attr_slice_empty(), plan->packet_type,
        &components[0]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_bind_value(
        context,
        scan ? loom_kernel_subgroup_scan_result(source_op)
             : loom_kernel_subgroup_reduce_result(source_op),
        components[0]);
  }
  return status;
}
