// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/conversion.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/scalarization.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"

// Callback plan IDs occupy a target-local domain outside descriptor ordinals.
static const uint64_t kLoomVmConversionPlanId = UINT64_C(0x100000400);

typedef struct loom_vm_conversion_lowering_step_t {
  // Dense Core VM descriptor ordinal emitted for this step.
  uint16_t descriptor_ordinal;
  // Fixed selector value selecting the concrete conversion.
  uint8_t selector_value;
  // Logical scalar type carried by the step result.
  loom_scalar_type_t result_scalar_type;
} loom_vm_conversion_lowering_step_t;
static_assert(sizeof(loom_vm_conversion_lowering_step_t) == 4,
              "conversion lowering steps must remain compact");

typedef struct loom_vm_conversion_lowering_t {
  // First step in kVmConversionLoweringSteps.
  uint16_t step_start;
  // Exact source scalar type accepted by this lowering.
  loom_scalar_type_t source_scalar_type;
  // Exact result scalar type produced by this lowering.
  loom_scalar_type_t result_scalar_type;
  // Number of consecutive steps in the lowering.
  uint8_t step_count;
} loom_vm_conversion_lowering_t;
static_assert(sizeof(loom_vm_conversion_lowering_t) == 6,
              "conversion lowerings must remain compact");

typedef struct loom_vm_conversion_lowering_range_t {
  // First lowering in kVmConversionLowerings for the source operation.
  uint16_t lowering_start;
  // Number of concrete source/result type pairs for the source operation.
  uint8_t lowering_count;
} loom_vm_conversion_lowering_range_t;
static_assert(sizeof(loom_vm_conversion_lowering_range_t) == 4,
              "conversion lowering ranges must remain compact");

static const loom_vm_conversion_lowering_step_t kVmConversionLoweringSteps[] = {
#define LOOM_VM_CONVERSION_LOWERING_STEP_ROW(               \
    descriptor_ordinal, selector_value, result_scalar_type) \
  {descriptor_ordinal, selector_value, result_scalar_type},
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_CONVERSION_LOWERING_STEP_ROW
};

static const loom_vm_conversion_lowering_t kVmConversionLowerings[] = {
#define LOOM_VM_CONVERSION_LOWERING_ROW(                            \
    source_scalar_type, result_scalar_type, step_start, step_count) \
  {step_start, source_scalar_type, result_scalar_type, step_count},
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_CONVERSION_LOWERING_ROW
};

#define LOOM_VM_CONVERSION_LOWERING_DEFINE_RANGES
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_CONVERSION_LOWERING_DEFINE_RANGES

#define LOOM_VM_CONVERSION_LOWERING_LIMITS(maximum_step_count)       \
  enum {                                                             \
    LOOM_VM_CONVERSION_LOWERING_MAX_STEP_COUNT = maximum_step_count, \
  };
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_CONVERSION_LOWERING_LIMITS

// Returns the scalar operation defining the per-lane semantics of |op_kind|.
static loom_op_kind_t loom_vm_conversion_scalar_op_kind(
    loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) == LOOM_DIALECT_SCALAR) return op_kind;
  const loom_vector_scalarization_t* scalarization =
      loom_vector_scalarization_lookup(op_kind);
  return scalarization != NULL ? scalarization->lane_op_kind
                               : LOOM_OP_KIND_UNKNOWN;
}

// Classifies a scalar conversion or a same-shape static-vector conversion.
static bool loom_vm_conversion_try_get_lane_types(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_scalar_type_t* out_source_scalar_type,
    loom_scalar_type_t* out_result_scalar_type, uint16_t* out_lane_count) {
  *out_source_scalar_type = LOOM_SCALAR_TYPE_NONE;
  *out_result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  *out_lane_count = 0;
  if (source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }

  const loom_value_id_t source_value = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result_value = loom_op_const_results(source_op)[0];
  const loom_type_t source_type = loom_module_value_type(module, source_value);
  const loom_type_t result_type = loom_module_value_type(module, result_value);
  if (loom_type_is_scalar(source_type) && loom_type_is_scalar(result_type)) {
    *out_lane_count = 1;
  } else {
    uint64_t source_lane_count = 0;
    uint64_t result_lane_count = 0;
    if (!loom_type_is_vector(source_type) ||
        !loom_type_is_vector(result_type) ||
        !loom_type_shape_equals(source_type, result_type) ||
        !loom_type_static_element_count(source_type, &source_lane_count) ||
        !loom_type_static_element_count(result_type, &result_lane_count) ||
        source_lane_count == 0 || source_lane_count != result_lane_count ||
        source_lane_count > LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT) {
      return false;
    }
    *out_lane_count = (uint16_t)source_lane_count;
  }
  *out_source_scalar_type = loom_type_element_type(source_type);
  *out_result_scalar_type = loom_type_element_type(result_type);
  return true;
}

static const loom_vm_conversion_lowering_t* loom_vm_conversion_find_lowering(
    const loom_module_t* module, const loom_op_t* source_op) {
  const loom_op_kind_t scalar_op_kind =
      loom_vm_conversion_scalar_op_kind(source_op->kind);
  if (loom_op_dialect_id(scalar_op_kind) != LOOM_DIALECT_SCALAR) return NULL;
  const uint8_t op_index = loom_op_dialect_index(scalar_op_kind);
  if (op_index >= IREE_ARRAYSIZE(kVmConversionLoweringRanges)) return NULL;
  const loom_vm_conversion_lowering_range_t range =
      kVmConversionLoweringRanges[op_index];
  if (range.lowering_count == 0) return NULL;

  loom_scalar_type_t source_scalar_type = LOOM_SCALAR_TYPE_NONE;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  uint16_t lane_count = 0;
  if (!loom_vm_conversion_try_get_lane_types(
          module, source_op, &source_scalar_type, &result_scalar_type,
          &lane_count)) {
    return NULL;
  }
  for (uint8_t i = 0; i < range.lowering_count; ++i) {
    const loom_vm_conversion_lowering_t* lowering =
        &kVmConversionLowerings[range.lowering_start + i];
    if (lowering->source_scalar_type == source_scalar_type &&
        lowering->result_scalar_type == result_scalar_type) {
      return lowering;
    }
  }
  return NULL;
}

bool loom_vm_conversion_try_select_op(const loom_module_t* module,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t* out_plan) {
  const loom_vm_conversion_lowering_t* lowering =
      loom_vm_conversion_find_lowering(module, source_op);
  if (lowering == NULL) return false;
  *out_plan = loom_low_lower_plan_make(kLoomVmConversionPlanId, lowering);
  return true;
}

static iree_status_t loom_vm_conversion_make_selector_attr(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    const loom_vm_conversion_lowering_step_t* step,
    loom_named_attr_t* out_selector_attr) {
  *out_selector_attr = (loom_named_attr_t){0};
  IREE_ASSERT_EQ(descriptor->immediate_count, 1);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(loom_low_lower_context_builder(context),
                                 immediate_name, &out_selector_attr->name_id));
  out_selector_attr->value = loom_attr_i64(step->selector_value);
  return iree_ok_status();
}

typedef struct loom_vm_conversion_emission_step_t {
  // Descriptor and target-owned resolution state for the instruction.
  loom_low_lower_resolved_descriptor_t resolved_descriptor;
  // Interned selector attribute when the instruction has one.
  loom_named_attr_t selector_attr;
  // Low scalar register type produced by the instruction.
  loom_type_t result_type;
} loom_vm_conversion_emission_step_t;

// Resolves the descriptors, selectors, and result types used by one lowering.
static iree_status_t loom_vm_conversion_prepare_emission(
    loom_low_lower_context_t* context,
    const loom_vm_conversion_lowering_t* lowering,
    loom_type_t final_result_type,
    loom_vm_conversion_emission_step_t* out_emission_steps) {
  IREE_ASSERT_LE(lowering->step_count,
                 LOOM_VM_CONVERSION_LOWERING_MAX_STEP_COUNT);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  for (uint8_t step_ordinal = 0; step_ordinal < lowering->step_count;
       ++step_ordinal) {
    const loom_vm_conversion_lowering_step_t* step =
        &kVmConversionLoweringSteps[lowering->step_start + step_ordinal];
    loom_vm_conversion_emission_step_t* emission_step =
        &out_emission_steps[step_ordinal];
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              step->descriptor_ordinal);
    IREE_ASSERT(descriptor != NULL);
    emission_step->resolved_descriptor = (loom_low_lower_resolved_descriptor_t){
        .descriptor = descriptor,
    };

    emission_step->selector_attr = (loom_named_attr_t){0};
    IREE_RETURN_IF_ERROR(loom_vm_conversion_make_selector_attr(
        context, descriptor, step, &emission_step->selector_attr));

    emission_step->result_type = final_result_type;
    if (step_ordinal + 1 < lowering->step_count) {
      IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
          context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
          loom_type_scalar(step->result_scalar_type),
          &emission_step->result_type));
    }
  }
  return iree_ok_status();
}

// Applies one prepared scalar conversion lowering to one lane.
static iree_status_t loom_vm_conversion_emit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_vm_conversion_lowering_t* lowering,
    const loom_vm_conversion_emission_step_t* emission_steps,
    loom_value_id_t source_value, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t value = source_value;
  for (uint8_t step_ordinal = 0; step_ordinal < lowering->step_count;
       ++step_ordinal) {
    const loom_vm_conversion_emission_step_t* emission_step =
        &emission_steps[step_ordinal];
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &emission_step->resolved_descriptor, &value,
        /*operand_count=*/1,
        loom_make_named_attr_slice(&emission_step->selector_attr, 1),
        &emission_step->result_type,
        /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    value = loom_op_const_results(low_op)[0];
  }
  *out_result = value;
  return iree_ok_status();
}

static iree_status_t loom_vm_conversion_emit_scalar(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_vm_conversion_lowering_t* lowering) {
  const loom_value_id_t source_value = loom_op_const_operands(source_op)[0];
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_value_id_t source_result = loom_op_const_results(source_op)[0];
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, source_result, &low_result_type));
  loom_vm_conversion_emission_step_t
      emission_steps[LOOM_VM_CONVERSION_LOWERING_MAX_STEP_COUNT];
  IREE_RETURN_IF_ERROR(loom_vm_conversion_prepare_emission(
      context, lowering, low_result_type, emission_steps));
  IREE_RETURN_IF_ERROR(loom_vm_conversion_emit_lane(
      context, source_op, lowering, emission_steps, low_value, &low_value));
  return loom_low_lower_bind_value(context, source_result, low_value);
}

static iree_status_t loom_vm_conversion_emit_vector(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_vm_conversion_lowering_t* lowering, uint16_t lane_count) {
  const loom_value_id_t source_value = loom_op_const_operands(source_op)[0];
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_source));

  loom_type_t low_source_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(lowering->source_scalar_type), &low_source_lane_type));
  loom_type_t low_result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(lowering->result_scalar_type), &low_result_lane_type));
  loom_vm_conversion_emission_step_t
      emission_steps[LOOM_VM_CONVERSION_LOWERING_MAX_STEP_COUNT];
  IREE_RETURN_IF_ERROR(loom_vm_conversion_prepare_emission(
      context, lowering, low_result_lane_type, emission_steps));

  loom_value_id_t low_lanes[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  for (uint16_t lane_ordinal = 0; lane_ordinal < lane_count; ++lane_ordinal) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(
        loom_low_lower_context_builder(context), low_source, lane_ordinal,
        low_source_lane_type, source_op->location, &slice_op));
    IREE_RETURN_IF_ERROR(loom_vm_conversion_emit_lane(
        context, source_op, lowering, emission_steps,
        loom_low_slice_result(slice_op), &low_lanes[lane_ordinal]));
  }

  const loom_value_id_t source_result = loom_op_const_results(source_op)[0];
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, source_result, &low_result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_lanes, lane_count,
      low_result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_vm_conversion_emit_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan,
                                         bool* out_handled) {
  *out_handled = plan.id == kLoomVmConversionPlanId;
  if (!*out_handled) return iree_ok_status();
  const loom_vm_conversion_lowering_t* lowering = plan.target_data;
  IREE_ASSERT(lowering != NULL);

  loom_scalar_type_t source_scalar_type = LOOM_SCALAR_TYPE_NONE;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  uint16_t lane_count = 0;
  const bool has_lane_types = loom_vm_conversion_try_get_lane_types(
      loom_low_lower_context_module(context), source_op, &source_scalar_type,
      &result_scalar_type, &lane_count);
  IREE_ASSERT(has_lane_types);
  IREE_ASSERT_EQ(source_scalar_type, lowering->source_scalar_type);
  IREE_ASSERT_EQ(result_scalar_type, lowering->result_scalar_type);
  const loom_type_t source_type =
      loom_module_value_type(loom_low_lower_context_module(context),
                             loom_op_const_operands(source_op)[0]);
  if (loom_type_is_scalar(source_type)) {
    return loom_vm_conversion_emit_scalar(context, source_op, lowering);
  }
  return loom_vm_conversion_emit_vector(context, source_op, lowering,
                                        lane_count);
}
