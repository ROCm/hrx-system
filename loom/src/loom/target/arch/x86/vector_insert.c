// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/vector_insert.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/packet_insert_legalization.h"

// Selects the smallest native integer carrier that covers |lane_count| lanes
// with the same physical lane width as |result_type|. Predicate payloads use
// byte ordinals so their compare has a native packed representation.
static bool loom_x86_dynamic_insert_ordinal_type(
    const loom_target_vector_packet_policy_t* policy, loom_type_t result_type,
    uint32_t lane_count, loom_type_t* out_ordinal_type) {
  *out_ordinal_type = (loom_type_t){0};
  const loom_scalar_type_t result_element = loom_type_element_type(result_type);
  const int32_t result_element_bit_count =
      result_element == LOOM_SCALAR_TYPE_INDEX ? policy->index_bit_count
      : result_element == LOOM_SCALAR_TYPE_OFFSET
          ? policy->offset_bit_count
          : loom_scalar_type_bitwidth(result_element);
  const uint32_t result_bit_count =
      result_element_bit_count > 0
          ? lane_count * (uint32_t)result_element_bit_count
          : 0;

  uint32_t selected_bit_count = UINT32_MAX;
  int32_t element_bit_count = 0;
  if (result_element != LOOM_SCALAR_TYPE_I1) {
    for (uint8_t i = 0; i < policy->native_bit_count_count; ++i) {
      if (policy->native_bit_counts[i] == result_bit_count) {
        selected_bit_count = result_bit_count;
        element_bit_count = result_element_bit_count;
        break;
      }
    }
  }
  static const uint8_t kOrdinalElementBitCounts[] = {8u, 16u, 32u, 64u};
  if (element_bit_count == 0) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kOrdinalElementBitCounts);
         ++i) {
      const uint32_t candidate_element_bit_count = kOrdinalElementBitCounts[i];
      const uint32_t candidate_bit_count =
          lane_count * candidate_element_bit_count;
      for (uint8_t j = 0; j < policy->native_bit_count_count; ++j) {
        if (policy->native_bit_counts[j] == candidate_bit_count &&
            candidate_bit_count < selected_bit_count) {
          selected_bit_count = candidate_bit_count;
          element_bit_count = (int32_t)candidate_element_bit_count;
        }
      }
    }
  }
  if (element_bit_count == 0) {
    return false;
  }

  loom_scalar_type_t ordinal_element = LOOM_SCALAR_TYPE_NONE;
  switch (element_bit_count) {
    case 8:
      ordinal_element = LOOM_SCALAR_TYPE_I8;
      break;
    case 16:
      ordinal_element = LOOM_SCALAR_TYPE_I16;
      break;
    case 32:
      ordinal_element = LOOM_SCALAR_TYPE_I32;
      break;
    case 64:
      ordinal_element = LOOM_SCALAR_TYPE_I64;
      break;
    default:
      return false;
  }

  *out_ordinal_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, ordinal_element,
                          loom_dim_pack_static(lane_count), /*encoding_id=*/0);
  return true;
}

static bool loom_x86_dynamic_insert_position_requires_guard(
    const loom_target_legalization_context_t* context,
    loom_type_t ordinal_type) {
  return loom_scalar_type_bitwidth(loom_type_element_type(ordinal_type)) <
         context->vector_packet_policy->index_bit_count;
}

// Reports the exact operation count used by shared component planning. Native
// vectors use an ordinal mask; predicate payloads require two additional
// bitwise operations.
static uint8_t loom_x86_dynamic_insert_operation_count(
    void* user_data, const loom_target_legalization_context_t* context,
    loom_type_t result_type, uint32_t lane_offset) {
  (void)user_data;
  if (loom_type_is_scalar(result_type)) {
    return 3u;
  }
  const int64_t lane_count_i64 =
      loom_type_rank(result_type) == 1 && loom_type_is_all_static(result_type)
          ? loom_type_dim_static_size_at(result_type, 0)
          : 0;
  const uint32_t lane_count = lane_count_i64 > 0 && lane_count_i64 <= UINT32_MAX
                                  ? (uint32_t)lane_count_i64
                                  : 0;
  loom_type_t ordinal_type = {0};
  if (lane_count == 0 || !loom_x86_dynamic_insert_ordinal_type(
                             context->vector_packet_policy, result_type,
                             lane_count, &ordinal_type)) {
    return 0;
  }
  if (lane_count == 1) {
    return 5u;
  }
  const int32_t ordinal_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(ordinal_type));
  const uint8_t ordinal_mask_operation_count =
      8u + (ordinal_element_bit_count < 32 ? 1u : 0u) +
      (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1 ? 2u : 0u);
  const uint8_t local_position_operation_count = lane_offset == 0 ? 0u : 2u;
  const uint8_t range_guard_operation_count =
      loom_x86_dynamic_insert_position_requires_guard(context, ordinal_type)
          ? (lane_offset == 0 ? 4u : 7u)
          : 0u;
  return ordinal_mask_operation_count + local_position_operation_count +
         range_guard_operation_count;
}

static iree_status_t loom_x86_materialize_dynamic_scalar_insert(
    loom_target_legalization_context_t* context, loom_value_id_t value,
    loom_value_id_t dest, loom_value_id_t position, uint32_t lane_offset,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_type_t index_type =
      loom_module_value_type(context->module, position);
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(lane_offset), index_type, location, &lane_op));
  loom_op_t* condition_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder, LOOM_INDEX_CMP_PREDICATE_EQ, position,
      loom_index_constant_result(lane_op), location, &condition_op));

  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(builder, loom_index_cmp_result(condition_op), value,
                            dest, result_type, location, &select_op));
  *out_result = loom_scf_select_result(select_op);
  return iree_ok_status();
}

// Clears a narrowed ordinal comparison when the original signed index is
// outside this component. Unsigned bounds reject negative indices and prevent
// values separated by the narrow integer period from aliasing a valid lane.
static iree_status_t loom_x86_materialize_dynamic_insert_range_guard(
    loom_target_legalization_context_t* context, loom_value_id_t position,
    uint32_t lane_offset, uint32_t lane_count, loom_type_t condition_type,
    loom_value_id_t condition, loom_location_id_t location,
    loom_value_id_t* out_condition) {
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_type_t index_type =
      loom_module_value_type(context->module, position);
  loom_op_t* upper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64((int64_t)lane_offset + lane_count), index_type,
      location, &upper_op));
  loom_op_t* below_upper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder, LOOM_INDEX_CMP_PREDICATE_ULT, position,
      loom_index_constant_result(upper_op), location, &below_upper_op));
  loom_value_id_t in_range = loom_index_cmp_result(below_upper_op);
  if (lane_offset != 0) {
    loom_op_t* lower_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(lane_offset), index_type, location, &lower_op));
    loom_op_t* at_or_above_lower_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_UGE, position,
        loom_index_constant_result(lower_op), location, &at_or_above_lower_op));
    loom_op_t* in_range_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, loom_index_cmp_result(at_or_above_lower_op), in_range,
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), location, &in_range_op));
    in_range = loom_scalar_andi_result(in_range_op);
  }
  loom_op_t* range_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      builder, in_range, condition_type, location, &range_mask_op));
  loom_op_t* guarded_condition_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(
      builder, condition, loom_vector_splat_result(range_mask_op),
      condition_type, location, &guarded_condition_op));
  *out_condition = loom_vector_andi_result(guarded_condition_op);
  return iree_ok_status();
}

// Materializes a runtime-indexed scalar insertion as a total component-mask
// operation. x86 lane-insert instructions take immediate indices, while the
// comparison, splat, and select families accept runtime values across every
// native vector carrier. An out-of-range index preserves |dest|.
static iree_status_t loom_x86_materialize_dynamic_insert_component(
    void* user_data, loom_target_legalization_context_t* context,
    loom_value_id_t value, loom_value_id_t dest, loom_value_id_t position,
    uint32_t lane_offset, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result) {
  (void)user_data;
  if (loom_type_is_scalar(result_type)) {
    return loom_x86_materialize_dynamic_scalar_insert(
        context, value, dest, position, lane_offset, result_type, location,
        out_result);
  }

  const uint32_t lane_count =
      (uint32_t)loom_type_dim_static_size_at(result_type, 0);
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  if (lane_count == 1) {
    const loom_type_t index_type =
        loom_module_value_type(context->module, position);
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(lane_offset), index_type, location, &lane_op));
    loom_op_t* condition_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        builder, LOOM_INDEX_CMP_PREDICATE_EQ, position,
        loom_index_constant_result(lane_op), location, &condition_op));
    const int64_t zero_index = 0;
    const loom_type_t element_type =
        loom_type_scalar(loom_type_element_type(result_type));
    loom_op_t* previous_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        builder, dest, /*indices=*/NULL, /*indices_count=*/0, &zero_index,
        /*static_indices_count=*/1, element_type, location, &previous_op));
    loom_op_t* selected_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_select_build(builder, loom_index_cmp_result(condition_op),
                              value, loom_vector_extract_result(previous_op),
                              element_type, location, &selected_op));
    loom_op_t* result_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splat_build(builder, loom_scf_select_result(selected_op),
                                result_type, location, &result_op));
    *out_result = loom_vector_splat_result(result_op);
    return iree_ok_status();
  }

  loom_type_t ordinal_type = {0};
  const bool ordinal_type_selected = loom_x86_dynamic_insert_ordinal_type(
      context->vector_packet_policy, result_type, lane_count, &ordinal_type);
  IREE_ASSERT_TRUE(ordinal_type_selected);
  const loom_type_t ordinal_element_type =
      loom_type_scalar(loom_type_element_type(ordinal_type));
  const loom_type_t condition_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(lane_count),
      /*encoding_id=*/0);

  const int32_t ordinal_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(ordinal_type));
  const loom_type_t position_cast_type =
      loom_type_scalar(ordinal_element_bit_count == 64 ? LOOM_SCALAR_TYPE_I64
                                                       : LOOM_SCALAR_TYPE_I32);
  loom_op_t* position_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, position, loom_module_value_type(context->module, position),
      position_cast_type, location, &position_op));
  loom_value_id_t ordinal_position = loom_index_cast_result(position_op);
  if (!loom_type_equal(position_cast_type, ordinal_element_type)) {
    loom_op_t* truncate_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_trunci_build(builder, ordinal_position, position_cast_type,
                                 ordinal_element_type, location, &truncate_op));
    ordinal_position = loom_scalar_trunci_result(truncate_op);
  }
  if (lane_offset != 0) {
    loom_op_t* offset_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_constant_build(builder, loom_attr_i64(lane_offset),
                                   ordinal_element_type, location, &offset_op));
    loom_op_t* local_position_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        builder, /*instance_flags=*/0, ordinal_position,
        loom_scalar_constant_result(offset_op), ordinal_element_type, location,
        &local_position_op));
    ordinal_position = loom_scalar_subi_result(local_position_op);
  }

  loom_op_t* one_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(1), ordinal_element_type, location, &one_op));
  loom_op_t* relative_base_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_negi_build(builder, ordinal_position,
                                              ordinal_element_type, location,
                                              &relative_base_op));
  loom_op_t* ordinals_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_iota_build(builder, loom_scalar_negi_result(relative_base_op),
                             loom_scalar_constant_result(one_op), ordinal_type,
                             location, &ordinals_op));
  loom_op_t* zeros_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_splati_build(builder, 0, ordinal_type, location, &zeros_op));
  loom_op_t* condition_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
      builder, LOOM_VECTOR_CMPI_PREDICATE_EQ,
      loom_vector_iota_result(ordinals_op), loom_vector_splati_result(zeros_op),
      ordinal_type, condition_type, location, &condition_op));
  loom_value_id_t condition = loom_vector_cmpi_result(condition_op);
  if (loom_x86_dynamic_insert_position_requires_guard(context, ordinal_type)) {
    IREE_RETURN_IF_ERROR(loom_x86_materialize_dynamic_insert_range_guard(
        context, position, lane_offset, lane_count, condition_type, condition,
        location, &condition));
  }
  loom_op_t* values_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(builder, value, result_type,
                                               location, &values_op));
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    loom_op_t* delta_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_xori_build(
        builder, dest, loom_vector_splat_result(values_op), result_type,
        location, &delta_op));
    loom_op_t* masked_delta_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_andi_build(
        builder, loom_vector_xori_result(delta_op), condition, result_type,
        location, &masked_delta_op));
    loom_op_t* result_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_xori_build(
        builder, dest, loom_vector_andi_result(masked_delta_op), result_type,
        location, &result_op));
    *out_result = loom_vector_xori_result(result_op);
  } else {
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_select_build(
        builder, condition, loom_vector_splat_result(values_op), dest,
        result_type, location, &select_op));
    *out_result = loom_vector_select_result(select_op);
  }
  return iree_ok_status();
}

static const loom_vector_packet_insert_materializer_t
    kX86DynamicInsertMaterializer = {
        .operation_count = loom_x86_dynamic_insert_operation_count,
        .materialize_dynamic = loom_x86_materialize_dynamic_insert_component,
};

static iree_status_t loom_x86_legalize_dynamic_vector_insert(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  const loom_value_slice_t indices = loom_vector_insert_indices(op);
  const loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  const loom_value_id_t value = loom_vector_insert_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_insert_result(op));
  loom_type_t ordinal_type = {0};
  const uint32_t lane_count =
      loom_type_rank(result_type) == 1 && loom_type_is_all_static(result_type)
          ? (uint32_t)loom_type_dim_static_size_at(result_type, 0)
          : 0;
  if (indices.count != 1 || static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != 1 || static_indices.i64_array[0] != INT64_MIN ||
      !loom_type_is_scalar(value_type) || lane_count == 0 ||
      !loom_target_legalization_has_source_vector_carrier(
          context, result_type, loom_vector_insert_result(op)) ||
      !loom_x86_dynamic_insert_ordinal_type(context->vector_packet_policy,
                                            result_type, lane_count,
                                            &ordinal_type)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_materialize_dynamic_insert_component(
      /*user_data=*/NULL, context, value, loom_vector_insert_dest(op),
      indices.values[0], /*lane_offset=*/0, result_type, op->location,
      &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_x86_vector_insert_legalize(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_insert(
      context, op, context->vector_packet_policy,
      &kX86DynamicInsertMaterializer, out_rewritten));
  if (!*out_rewritten && loom_vector_insert_indices(op).count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_x86_legalize_dynamic_vector_insert(context, op, out_rewritten));
  }
  return iree_ok_status();
}
