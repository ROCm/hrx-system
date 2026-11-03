// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_selection.h"

#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_instructions.h"

static iree_status_t loom_spirv_selection_emit(
    loom_spirv_type_context_t* context, uint32_t opcode, uint32_t type_id,
    uint32_t result_id, const uint32_t* arguments, uint8_t argument_count) {
  uint32_t operands[6] = {type_id, result_id};
  memcpy(operands + 2, arguments, argument_count * sizeof(*arguments));
  return loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(context->builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      opcode, operands, 2 + argument_count);
}

static iree_status_t loom_spirv_selection_extract(
    loom_spirv_type_context_t* context, uint32_t type_id, uint32_t composite_id,
    uint32_t index, uint32_t* out_result_id) {
  *out_result_id = loom_spirv_module_builder_allocate_id(context->builder);
  const uint32_t arguments[] = {composite_id, index};
  return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_COMPOSITE_EXTRACT,
                                   type_id, *out_result_id, arguments, 2);
}

static iree_status_t loom_spirv_selection_splat_pair(
    loom_spirv_type_context_t* context, uint32_t type_id, uint32_t value_id,
    uint32_t* out_result_id) {
  *out_result_id = loom_spirv_module_builder_allocate_id(context->builder);
  const uint32_t arguments[] = {value_id, value_id};
  return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_COMPOSITE_CONSTRUCT,
                                   type_id, *out_result_id, arguments, 2);
}

// Bitcasts require equal total widths. A narrow float is paired with itself to
// fill a u32; a double uses two u32 words. Neither requires optional Int16 or
// Int64 support, and no floating-point conversion or arithmetic is introduced.
static iree_status_t loom_spirv_selection_float_component(
    loom_spirv_type_context_t* context, uint8_t bit_width,
    uint32_t float_type_id, uint32_t boolean_type_id, uint32_t word_type_id,
    uint32_t result_id, uint32_t condition_id, uint32_t true_id,
    uint32_t false_id) {
  uint32_t carrier_type_id = word_type_id;
  uint32_t packed_type_id = float_type_id;
  if (bit_width == 16) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(context, float_type_id, 2,
                                                     &packed_type_id));
    IREE_RETURN_IF_ERROR(loom_spirv_selection_splat_pair(
        context, packed_type_id, true_id, &true_id));
    IREE_RETURN_IF_ERROR(loom_spirv_selection_splat_pair(
        context, packed_type_id, false_id, &false_id));
  } else {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(context, word_type_id, 2,
                                                     &carrier_type_id));
    uint32_t mask_type_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(context, boolean_type_id,
                                                     2, &mask_type_id));
    IREE_RETURN_IF_ERROR(loom_spirv_selection_splat_pair(
        context, mask_type_id, condition_id, &condition_id));
  }
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      context->builder, LOOM_SPIRV_OP_BITCAST, carrier_type_id, true_id,
      &true_id));
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      context->builder, LOOM_SPIRV_OP_BITCAST, carrier_type_id, false_id,
      &false_id));
  const uint32_t selected_id =
      loom_spirv_module_builder_allocate_id(context->builder);
  const uint32_t arguments[] = {condition_id, true_id, false_id};
  IREE_RETURN_IF_ERROR(loom_spirv_selection_emit(context, LOOM_SPIRV_OP_SELECT,
                                                 carrier_type_id, selected_id,
                                                 arguments, 3));
  if (bit_width == 16) {
    uint32_t packed_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
        context->builder, LOOM_SPIRV_OP_BITCAST, packed_type_id, selected_id,
        &packed_id));
    const uint32_t extract_arguments[] = {packed_id, 0};
    return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_COMPOSITE_EXTRACT,
                                     float_type_id, result_id,
                                     extract_arguments, 2);
  }
  return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_BITCAST,
                                   float_type_id, result_id, &selected_id, 1);
}

iree_status_t loom_spirv_module_emit_selection(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t value_type,
    uint32_t result_type_id, uint32_t result_id, uint32_t condition_id,
    uint32_t true_id, uint32_t false_id) {
  const loom_spirv_scalar_type_descriptor_t* scalar = NULL;
  if (value_type.value_class == LOOM_SPIRV_VALUE_CLASS_SCALAR ||
      value_type.value_class == LOOM_SPIRV_VALUE_CLASS_VECTOR) {
    scalar = loom_spirv_scalar_type_descriptor(value_type.scalar_type);
  }
  if (!scalar || scalar->kind != LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT) {
    const uint32_t arguments[] = {condition_id, true_id, false_id};
    return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_SELECT,
                                     result_type_id, result_id, arguments, 3);
  }

  // A floating OpSelect inherits NSZ/NotInf/NotNaN in Vulkan's default
  // environment. Integer selection preserves the chosen payload even when a
  // driver folds adjacent arithmetic into the selection.
  const uint16_t lane_count =
      value_type.value_class == LOOM_SPIRV_VALUE_CLASS_VECTOR
          ? value_type.vector.lane_count
          : 1;
  uint32_t word_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u32(context, &word_type_id));
  if (scalar->bit_width == 32) {
    uint32_t carrier_type_id = word_type_id;
    if (lane_count > 1) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
          context, word_type_id, lane_count, &carrier_type_id));
    }
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
        context->builder, LOOM_SPIRV_OP_BITCAST, carrier_type_id, true_id,
        &true_id));
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
        context->builder, LOOM_SPIRV_OP_BITCAST, carrier_type_id, false_id,
        &false_id));
    const uint32_t selected_id =
        loom_spirv_module_builder_allocate_id(context->builder);
    const uint32_t arguments[] = {condition_id, true_id, false_id};
    IREE_RETURN_IF_ERROR(
        loom_spirv_selection_emit(context, LOOM_SPIRV_OP_SELECT,
                                  carrier_type_id, selected_id, arguments, 3));
    return loom_spirv_selection_emit(context, LOOM_SPIRV_OP_BITCAST,
                                     result_type_id, result_id, &selected_id,
                                     1);
  }

  uint32_t component_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_scalar(
      context, value_type.scalar_type, &component_type_id));
  uint32_t boolean_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_bool(context, &boolean_type_id));
  if (lane_count == 1) {
    return loom_spirv_selection_float_component(
        context, scalar->bit_width, component_type_id, boolean_type_id,
        word_type_id, result_id, condition_id, true_id, false_id);
  }
  uint32_t components[4] = {0};
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < lane_count && iree_status_is_ok(status); ++i) {
    uint32_t condition = 0;
    uint32_t true_component = 0;
    uint32_t false_component = 0;
    status = loom_spirv_selection_extract(context, boolean_type_id,
                                          condition_id, i, &condition);
    if (iree_status_is_ok(status)) {
      status = loom_spirv_selection_extract(context, component_type_id, true_id,
                                            i, &true_component);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_selection_extract(context, component_type_id,
                                            false_id, i, &false_component);
    }
    if (iree_status_is_ok(status)) {
      components[i] = loom_spirv_module_builder_allocate_id(context->builder);
      status = loom_spirv_selection_float_component(
          context, scalar->bit_width, component_type_id, boolean_type_id,
          word_type_id, components[i], condition, true_component,
          false_component);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_selection_emit(
        context, LOOM_SPIRV_OP_COMPOSITE_CONSTRUCT, result_type_id, result_id,
        components, (uint8_t)lane_count);
  }
  return status;
}
