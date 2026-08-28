// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/vector.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/records/target_records.h"
#include "loom/target/registers.h"

typedef uint8_t loom_vm_vector_plan_kind_t;
enum loom_vm_vector_plan_kind_e {
  LOOM_VM_VECTOR_PLAN_KIND_NONE = 0,
  LOOM_VM_VECTOR_PLAN_KIND_FROM_ELEMENTS = 1,
  LOOM_VM_VECTOR_PLAN_KIND_EXTRACT = 2,
  LOOM_VM_VECTOR_PLAN_KIND_INSERT = 3,
  LOOM_VM_VECTOR_PLAN_KIND_SLICE = 4,
  LOOM_VM_VECTOR_PLAN_KIND_CONCAT = 5,
  LOOM_VM_VECTOR_PLAN_KIND_CONSTANT = 6,
  LOOM_VM_VECTOR_PLAN_KIND_SPLAT = 7,
};

#define LOOM_VM_VECTOR_OP_INDEX(op_kind) ((op_kind) & 0xFFu)

static const loom_vm_vector_plan_kind_t
    kLoomVmVectorPlanKinds[LOOM_OP_VECTOR_COUNT_] = {
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_CONSTANT)] =
            LOOM_VM_VECTOR_PLAN_KIND_CONSTANT,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_SPLAT)] =
            LOOM_VM_VECTOR_PLAN_KIND_SPLAT,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_FROM_ELEMENTS)] =
            LOOM_VM_VECTOR_PLAN_KIND_FROM_ELEMENTS,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_EXTRACT)] =
            LOOM_VM_VECTOR_PLAN_KIND_EXTRACT,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_INSERT)] =
            LOOM_VM_VECTOR_PLAN_KIND_INSERT,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_SLICE)] =
            LOOM_VM_VECTOR_PLAN_KIND_SLICE,
        [LOOM_VM_VECTOR_OP_INDEX(LOOM_OP_VECTOR_CONCAT)] =
            LOOM_VM_VECTOR_PLAN_KIND_CONCAT,
};

#undef LOOM_VM_VECTOR_OP_INDEX

// Callback plan IDs occupy a target-local domain outside descriptor ordinals.
static const uint64_t kLoomVmVectorPlanIdBase = UINT64_C(0x100000100);

typedef struct loom_vm_vector_run_t {
  // Source operand ordinal supplying this consecutive range.
  uint16_t source_ordinal;
  // First VM value unit selected from the source operand.
  uint16_t unit_offset;
  // Number of consecutive VM value units selected from the source operand.
  uint16_t unit_count;
} loom_vm_vector_run_t;

static loom_vm_vector_plan_kind_t loom_vm_vector_plan_kind_for_op(
    const loom_op_t* source_op) {
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_VECTOR) {
    return LOOM_VM_VECTOR_PLAN_KIND_NONE;
  }
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  return op_index < IREE_ARRAYSIZE(kLoomVmVectorPlanKinds)
             ? kLoomVmVectorPlanKinds[op_index]
             : LOOM_VM_VECTOR_PLAN_KIND_NONE;
}

static bool loom_vm_vector_plan_is_selected(loom_low_lower_plan_t plan) {
  return plan.id >= kLoomVmVectorPlanIdBase &&
         plan.id <
             kLoomVmVectorPlanIdBase + IREE_ARRAYSIZE(kLoomVmVectorPlanKinds) &&
         kLoomVmVectorPlanKinds[plan.id - kLoomVmVectorPlanIdBase] !=
             LOOM_VM_VECTOR_PLAN_KIND_NONE;
}

static loom_vm_vector_plan_kind_t loom_vm_vector_plan_kind_for_plan(
    loom_low_lower_plan_t plan) {
  IREE_ASSERT(loom_vm_vector_plan_is_selected(plan));
  return kLoomVmVectorPlanKinds[plan.id - kLoomVmVectorPlanIdBase];
}

static bool loom_vm_vector_try_get_unit_count(const loom_module_t* module,
                                              loom_type_t type,
                                              uint16_t* out_unit_count) {
  *out_unit_count = 0;
  loom_vm_call_abi_register_layout_t layout = {0};
  if (!loom_type_is_vector(type) ||
      !loom_vm_call_abi_try_classify_logical_type(module, type, &layout) ||
      layout.bank != LOOM_VM_CALL_ABI_BANK_VALUE) {
    return false;
  }
  *out_unit_count = layout.unit_count;
  return true;
}

static bool loom_vm_vector_try_get_value_unit_count(const loom_module_t* module,
                                                    loom_type_t type,
                                                    uint16_t* out_unit_count) {
  *out_unit_count = 0;
  loom_vm_call_abi_register_layout_t layout = {0};
  if (!loom_vm_call_abi_try_classify_logical_type(module, type, &layout) ||
      layout.bank != LOOM_VM_CALL_ABI_BANK_VALUE) {
    return false;
  }
  *out_unit_count = layout.unit_count;
  return true;
}

static bool loom_vm_vector_attr_is_static(loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_I64_ARRAY) return false;
  for (uint16_t i = 0; i < attr.count; ++i) {
    if (attr.i64_array[i] == INT64_MIN) return false;
  }
  return true;
}

// Computes the row-major unit offset selected by leading vector indices.
static bool loom_vm_vector_try_get_leading_unit_offset(
    loom_type_t type, loom_attribute_t static_indices,
    uint16_t* out_unit_offset) {
  *out_unit_offset = 0;
  if (!loom_vm_vector_attr_is_static(static_indices) ||
      static_indices.count > loom_type_rank(type)) {
    return false;
  }
  uint64_t unit_offset = 0;
  uint64_t unit_stride = 1;
  for (uint8_t axis = loom_type_rank(type); axis > 0; --axis) {
    const uint8_t source_axis = (uint8_t)(axis - 1);
    const int64_t extent = loom_type_dim_static_size_at(type, source_axis);
    if (extent <= 0) return false;
    if (source_axis < static_indices.count) {
      const int64_t index = static_indices.i64_array[source_axis];
      if (index < 0 || index >= extent) return false;
      unit_offset += (uint64_t)index * unit_stride;
    }
    unit_stride *= (uint64_t)extent;
  }
  if (unit_offset > UINT16_MAX) return false;
  *out_unit_offset = (uint16_t)unit_offset;
  return true;
}

static bool loom_vm_vector_can_lower(const loom_module_t* module,
                                     const loom_op_t* source_op,
                                     loom_vm_vector_plan_kind_t plan_kind) {
  switch (plan_kind) {
    case LOOM_VM_VECTOR_PLAN_KIND_CONSTANT: {
      uint16_t result_unit_count = 0;
      return loom_vm_vector_try_get_unit_count(
          module,
          loom_module_value_type(module,
                                 loom_vector_constant_result(source_op)),
          &result_unit_count);
    }
    case LOOM_VM_VECTOR_PLAN_KIND_SPLAT: {
      uint16_t result_unit_count = 0;
      uint16_t scalar_unit_count = 0;
      return loom_vm_vector_try_get_unit_count(
                 module,
                 loom_module_value_type(module,
                                        loom_vector_splat_result(source_op)),
                 &result_unit_count) &&
             loom_vm_vector_try_get_value_unit_count(
                 module,
                 loom_module_value_type(module,
                                        loom_vector_splat_scalar(source_op)),
                 &scalar_unit_count) &&
             scalar_unit_count == 1;
    }
    case LOOM_VM_VECTOR_PLAN_KIND_FROM_ELEMENTS: {
      const loom_value_id_t result =
          loom_vector_from_elements_result(source_op);
      uint16_t result_unit_count = 0;
      if (!loom_vm_vector_try_get_unit_count(
              module, loom_module_value_type(module, result),
              &result_unit_count)) {
        return false;
      }
      const loom_value_slice_t elements =
          loom_vector_from_elements_elements(source_op);
      if (elements.count != result_unit_count) return false;
      for (uint16_t i = 0; i < elements.count; ++i) {
        uint16_t element_unit_count = 0;
        if (!loom_vm_vector_try_get_value_unit_count(
                module, loom_module_value_type(module, elements.values[i]),
                &element_unit_count) ||
            element_unit_count != 1) {
          return false;
        }
      }
      return true;
    }
    case LOOM_VM_VECTOR_PLAN_KIND_EXTRACT: {
      if (loom_vector_extract_indices(source_op).count != 0) return false;
      const loom_type_t source_type =
          loom_module_value_type(module, loom_vector_extract_source(source_op));
      const loom_type_t result_type =
          loom_module_value_type(module, loom_vector_extract_result(source_op));
      uint16_t source_unit_count = 0;
      uint16_t result_unit_count = 0;
      uint16_t unit_offset = 0;
      return loom_vm_vector_try_get_unit_count(module, source_type,
                                               &source_unit_count) &&
             loom_vm_vector_try_get_value_unit_count(module, result_type,
                                                     &result_unit_count) &&
             loom_vm_vector_try_get_leading_unit_offset(
                 source_type, loom_vector_extract_static_indices(source_op),
                 &unit_offset) &&
             unit_offset <= source_unit_count &&
             result_unit_count <= source_unit_count - unit_offset;
    }
    case LOOM_VM_VECTOR_PLAN_KIND_INSERT: {
      if (loom_vector_insert_indices(source_op).count != 0) return false;
      const loom_type_t value_type =
          loom_module_value_type(module, loom_vector_insert_value(source_op));
      const loom_type_t dest_type =
          loom_module_value_type(module, loom_vector_insert_dest(source_op));
      uint16_t value_unit_count = 0;
      uint16_t dest_unit_count = 0;
      uint16_t unit_offset = 0;
      return loom_vm_vector_try_get_value_unit_count(module, value_type,
                                                     &value_unit_count) &&
             loom_vm_vector_try_get_unit_count(module, dest_type,
                                               &dest_unit_count) &&
             loom_vm_vector_try_get_leading_unit_offset(
                 dest_type, loom_vector_insert_static_indices(source_op),
                 &unit_offset) &&
             unit_offset <= dest_unit_count &&
             value_unit_count <= dest_unit_count - unit_offset;
    }
    case LOOM_VM_VECTOR_PLAN_KIND_SLICE: {
      if (loom_vector_slice_offsets(source_op).count != 0) return false;
      const loom_type_t source_type =
          loom_module_value_type(module, loom_vector_slice_source(source_op));
      const loom_type_t result_type =
          loom_module_value_type(module, loom_vector_slice_result(source_op));
      uint16_t source_unit_count = 0;
      uint16_t result_unit_count = 0;
      const loom_attribute_t static_offsets =
          loom_vector_slice_static_offsets(source_op);
      return loom_vm_vector_try_get_unit_count(module, source_type,
                                               &source_unit_count) &&
             loom_vm_vector_try_get_unit_count(module, result_type,
                                               &result_unit_count) &&
             loom_vm_vector_attr_is_static(static_offsets) &&
             static_offsets.count == loom_type_rank(source_type);
    }
    case LOOM_VM_VECTOR_PLAN_KIND_CONCAT: {
      const loom_value_id_t result = loom_vector_concat_result(source_op);
      const loom_type_t result_type = loom_module_value_type(module, result);
      uint16_t result_unit_count = 0;
      const int64_t axis = loom_vector_concat_axis(source_op);
      if (!loom_vm_vector_try_get_unit_count(module, result_type,
                                             &result_unit_count) ||
          axis < 0 || axis >= loom_type_rank(result_type)) {
        return false;
      }
      const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
      if (inputs.count == 0) return false;
      for (uint16_t i = 0; i < inputs.count; ++i) {
        uint16_t input_unit_count = 0;
        if (!loom_vm_vector_try_get_unit_count(
                module, loom_module_value_type(module, inputs.values[i]),
                &input_unit_count)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

bool loom_vm_vector_try_select_op(const loom_module_t* module,
                                  const loom_op_t* source_op,
                                  loom_low_lower_plan_t* out_plan) {
  const loom_vm_vector_plan_kind_t plan_kind =
      loom_vm_vector_plan_kind_for_op(source_op);
  if (!loom_vm_vector_can_lower(module, source_op, plan_kind)) return false;
  *out_plan = loom_low_lower_plan_make(
      kLoomVmVectorPlanIdBase + loom_op_dialect_index(source_op->kind), NULL);
  return true;
}

static iree_status_t loom_vm_vector_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_vm_target_bundle_is_core(
          loom_target_low_legality_bundle(context))) {
    return iree_ok_status();
  }
  loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
  *out_handled = loom_vm_vector_try_select_op(
      loom_target_low_legality_module(context), source_op, &plan);
  return iree_ok_status();
}

const loom_target_low_legality_provider_t loom_vm_vector_low_legality_provider =
    {
        .name = IREE_SVL("vm-vector-carriers"),
        .builtin_dialect_bits = UINT64_C(1) << LOOM_DIALECT_VECTOR,
        .try_verify_op = loom_vm_vector_try_verify_op,
};

static iree_status_t loom_vm_vector_make_fragment_type(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    uint16_t unit_count, loom_type_t* out_type) {
  const loom_type_t fragment_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(unit_count),
      /*encoding_id=*/0);
  return loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, unit_count, fragment_type, out_type);
}

static iree_status_t loom_vm_vector_emit_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t unit_offset, loom_type_t result_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_source, unit_offset,
      result_type, source_op->location, &slice_op));
  *out_result = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_vm_vector_emit_fragment_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t unit_offset, uint16_t unit_count,
    loom_scalar_type_t element_type, loom_value_id_t* out_result) {
  loom_type_t fragment_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vm_vector_make_fragment_type(
      context, element_type, unit_count, &fragment_type));
  return loom_vm_vector_emit_slice(context, source_op, low_source, unit_offset,
                                   fragment_type, out_result);
}

static iree_status_t loom_vm_vector_emit_concat(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* low_sources, uint16_t source_count,
    loom_value_id_t source_result) {
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, source_result, &low_result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_sources, source_count,
      low_result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static bool loom_vm_vector_append_run(uint16_t source_ordinal,
                                      uint16_t unit_offset, uint16_t unit_count,
                                      loom_vm_vector_run_t* runs,
                                      uint16_t* inout_run_count) {
  if (unit_count == 0 ||
      *inout_run_count >= LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT) {
    return false;
  }
  if (*inout_run_count != 0) {
    loom_vm_vector_run_t* tail = &runs[*inout_run_count - 1];
    if (tail->source_ordinal == source_ordinal &&
        tail->unit_offset + tail->unit_count == unit_offset) {
      tail->unit_count = (uint16_t)(tail->unit_count + unit_count);
      return true;
    }
  }
  runs[*inout_run_count] = (loom_vm_vector_run_t){
      .source_ordinal = source_ordinal,
      .unit_offset = unit_offset,
      .unit_count = unit_count,
  };
  ++*inout_run_count;
  return true;
}

static bool loom_vm_vector_collect_slice_runs(loom_type_t source_type,
                                              loom_type_t result_type,
                                              loom_attribute_t static_offsets,
                                              loom_vm_vector_run_t* runs,
                                              uint16_t* out_run_count) {
  *out_run_count = 0;
  uint64_t result_element_count = 0;
  if (!loom_type_static_element_count(result_type, &result_element_count) ||
      result_element_count == 0 ||
      result_element_count > LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT) {
    return false;
  }
  const uint8_t rank = loom_type_rank(result_type);
  for (uint16_t result_ordinal = 0; result_ordinal < result_element_count;
       ++result_ordinal) {
    uint32_t remaining = result_ordinal;
    uint32_t source_stride = 1;
    uint32_t source_ordinal = 0;
    for (uint8_t axis = rank; axis > 0; --axis) {
      const uint8_t source_axis = (uint8_t)(axis - 1);
      const uint32_t result_extent =
          (uint32_t)loom_type_dim_static_size_at(result_type, source_axis);
      const uint32_t source_extent =
          (uint32_t)loom_type_dim_static_size_at(source_type, source_axis);
      const uint32_t result_coordinate = remaining % result_extent;
      remaining /= result_extent;
      const int64_t static_offset = static_offsets.i64_array[source_axis];
      if (static_offset < 0 ||
          (uint64_t)static_offset + result_coordinate >= source_extent) {
        return false;
      }
      source_ordinal +=
          ((uint32_t)static_offset + result_coordinate) * source_stride;
      source_stride *= source_extent;
    }
    if (source_ordinal > UINT16_MAX ||
        !loom_vm_vector_append_run(/*source_ordinal=*/0,
                                   (uint16_t)source_ordinal,
                                   /*unit_count=*/1, runs, out_run_count)) {
      return false;
    }
  }
  return true;
}

static bool loom_vm_vector_collect_concat_runs(const loom_module_t* module,
                                               const loom_op_t* source_op,
                                               loom_vm_vector_run_t* runs,
                                               uint16_t* out_run_count) {
  *out_run_count = 0;
  const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(source_op));
  const uint8_t axis = (uint8_t)loom_vector_concat_axis(source_op);
  uint16_t outer_count = 1;
  uint16_t inner_unit_count = 1;
  for (uint8_t i = 0; i < axis; ++i) {
    outer_count =
        (uint16_t)(outer_count * loom_type_dim_static_size_at(result_type, i));
  }
  for (uint8_t i = (uint8_t)(axis + 1); i < loom_type_rank(result_type); ++i) {
    inner_unit_count = (uint16_t)(inner_unit_count *
                                  loom_type_dim_static_size_at(result_type, i));
  }
  for (uint16_t outer_ordinal = 0; outer_ordinal < outer_count;
       ++outer_ordinal) {
    for (uint16_t input_ordinal = 0; input_ordinal < inputs.count;
         ++input_ordinal) {
      const loom_type_t input_type =
          loom_module_value_type(module, inputs.values[input_ordinal]);
      const uint16_t unit_count =
          (uint16_t)(loom_type_dim_static_size_at(input_type, axis) *
                     inner_unit_count);
      const uint16_t unit_offset = (uint16_t)(outer_ordinal * unit_count);
      if (!loom_vm_vector_append_run(input_ordinal, unit_offset, unit_count,
                                     runs, out_run_count)) {
        return false;
      }
    }
  }
  return true;
}

static iree_status_t loom_vm_vector_emit_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  loom_value_id_t low_elements[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  for (uint16_t i = 0; i < elements.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, elements.values[i], &low_elements[i]));
  }
  return loom_vm_vector_emit_concat(
      context, source_op, low_elements, elements.count,
      loom_vector_from_elements_result(source_op));
}

static iree_status_t loom_vm_vector_emit_splat_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_scalar, loom_value_id_t source_result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint16_t result_unit_count = 0;
  const bool has_unit_count = loom_vm_vector_try_get_unit_count(
      module, loom_module_value_type(module, source_result),
      &result_unit_count);
  IREE_ASSERT(has_unit_count);
  (void)has_unit_count;
  loom_value_id_t low_elements[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  for (uint16_t i = 0; i < result_unit_count; ++i) {
    low_elements[i] = low_scalar;
  }
  return loom_vm_vector_emit_concat(context, source_op, low_elements,
                                    result_unit_count, source_result);
}

static iree_status_t loom_vm_vector_emit_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  loom_type_t low_scalar_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, VM_CORE_REG_CLASS_ID_VALUE, /*unit_count=*/1,
      loom_type_scalar(element_type), &low_scalar_type));
  const uint64_t bits = loom_vm_constant_bits_from_scalar_attr(
      element_type, loom_vector_constant_value(source_op));
  loom_value_id_t low_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_inline_constant_build(
      loom_low_lower_context_builder(context), bits, low_scalar_type,
      source_op->location, &low_scalar));
  return loom_vm_vector_emit_splat_value(context, source_op, low_scalar,
                                         result);
}

static iree_status_t loom_vm_vector_emit_splat(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_splat_scalar(source_op), &low_scalar));
  return loom_vm_vector_emit_splat_value(context, source_op, low_scalar,
                                         loom_vector_splat_result(source_op));
}

static iree_status_t loom_vm_vector_emit_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_vector_extract_source(source_op);
  uint16_t unit_offset = 0;
  const bool has_unit_offset = loom_vm_vector_try_get_leading_unit_offset(
      loom_module_value_type(module, source),
      loom_vector_extract_static_indices(source_op), &unit_offset);
  IREE_ASSERT(has_unit_offset);
  (void)has_unit_offset;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source));
  const loom_value_id_t result = loom_vector_extract_result(source_op);
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, result, &low_result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_vector_emit_slice(context, source_op, low_source,
                                                 unit_offset, low_result_type,
                                                 &low_result));
  return loom_low_lower_bind_value(context, result, low_result);
}

static iree_status_t loom_vm_vector_emit_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_vector_insert_value(source_op);
  const loom_value_id_t dest = loom_vector_insert_dest(source_op);
  const loom_type_t dest_type = loom_module_value_type(module, dest);
  uint16_t dest_unit_count = 0;
  uint16_t value_unit_count = 0;
  uint16_t unit_offset = 0;
  const bool has_layout =
      loom_vm_vector_try_get_unit_count(module, dest_type, &dest_unit_count) &&
      loom_vm_vector_try_get_value_unit_count(
          module, loom_module_value_type(module, value), &value_unit_count) &&
      loom_vm_vector_try_get_leading_unit_offset(
          dest_type, loom_vector_insert_static_indices(source_op),
          &unit_offset);
  IREE_ASSERT(has_layout);
  (void)has_layout;

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, value, &low_value));
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, dest, &low_dest));

  loom_value_id_t fragments[3];
  uint16_t fragment_count = 0;
  const loom_scalar_type_t element_type = loom_type_element_type(dest_type);
  if (unit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_vm_vector_emit_fragment_slice(
        context, source_op, low_dest, /*unit_offset=*/0, unit_offset,
        element_type, &fragments[fragment_count++]));
  }
  fragments[fragment_count++] = low_value;
  const uint16_t suffix_offset = (uint16_t)(unit_offset + value_unit_count);
  if (suffix_offset < dest_unit_count) {
    IREE_RETURN_IF_ERROR(loom_vm_vector_emit_fragment_slice(
        context, source_op, low_dest, suffix_offset,
        (uint16_t)(dest_unit_count - suffix_offset), element_type,
        &fragments[fragment_count++]));
  }
  return loom_vm_vector_emit_concat(context, source_op, fragments,
                                    fragment_count,
                                    loom_vector_insert_result(source_op));
}

static iree_status_t loom_vm_vector_emit_slice_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_vector_slice_source(source_op);
  const loom_value_id_t result = loom_vector_slice_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  loom_vm_vector_run_t runs[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  uint16_t run_count = 0;
  const bool has_runs = loom_vm_vector_collect_slice_runs(
      source_type, result_type, loom_vector_slice_static_offsets(source_op),
      runs, &run_count);
  IREE_ASSERT(has_runs);
  (void)has_runs;

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source));
  if (run_count == 1) {
    loom_type_t low_result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_value(context, source_op, result, &low_result_type));
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vm_vector_emit_slice(
        context, source_op, low_source, runs[0].unit_offset, low_result_type,
        &low_result));
    return loom_low_lower_bind_value(context, result, low_result);
  }

  loom_value_id_t fragments[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  for (uint16_t i = 0; i < run_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_vector_emit_fragment_slice(
        context, source_op, low_source, runs[i].unit_offset, runs[i].unit_count,
        element_type, &fragments[i]));
  }
  return loom_vm_vector_emit_concat(context, source_op, fragments, run_count,
                                    result);
}

static iree_status_t loom_vm_vector_emit_concat_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  loom_value_id_t low_inputs[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  uint16_t input_unit_counts[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  for (uint16_t i = 0; i < inputs.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, inputs.values[i], &low_inputs[i]));
    const bool has_unit_count = loom_vm_vector_try_get_unit_count(
        module, loom_module_value_type(module, inputs.values[i]),
        &input_unit_counts[i]);
    IREE_ASSERT(has_unit_count);
    (void)has_unit_count;
  }

  loom_vm_vector_run_t runs[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  uint16_t run_count = 0;
  const bool has_runs =
      loom_vm_vector_collect_concat_runs(module, source_op, runs, &run_count);
  IREE_ASSERT(has_runs);
  (void)has_runs;

  loom_value_id_t fragments[LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT];
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(source_op));
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  for (uint16_t i = 0; i < run_count; ++i) {
    const loom_vm_vector_run_t run = runs[i];
    if (run.unit_offset == 0 &&
        run.unit_count == input_unit_counts[run.source_ordinal]) {
      fragments[i] = low_inputs[run.source_ordinal];
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vm_vector_emit_fragment_slice(
        context, source_op, low_inputs[run.source_ordinal], run.unit_offset,
        run.unit_count, element_type, &fragments[i]));
  }
  return loom_vm_vector_emit_concat(context, source_op, fragments, run_count,
                                    loom_vector_concat_result(source_op));
}

iree_status_t loom_vm_vector_emit_op(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     bool* out_handled) {
  *out_handled = loom_vm_vector_plan_is_selected(plan);
  if (!*out_handled) return iree_ok_status();
  switch (loom_vm_vector_plan_kind_for_plan(plan)) {
    case LOOM_VM_VECTOR_PLAN_KIND_CONSTANT:
      return loom_vm_vector_emit_constant(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_SPLAT:
      return loom_vm_vector_emit_splat(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_FROM_ELEMENTS:
      return loom_vm_vector_emit_from_elements(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_EXTRACT:
      return loom_vm_vector_emit_extract(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_INSERT:
      return loom_vm_vector_emit_insert(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_SLICE:
      return loom_vm_vector_emit_slice_op(context, source_op);
    case LOOM_VM_VECTOR_PLAN_KIND_CONCAT:
      return loom_vm_vector_emit_concat_op(context, source_op);
    default:
      IREE_ASSERT_UNREACHABLE("selected VM vector lowering plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}
