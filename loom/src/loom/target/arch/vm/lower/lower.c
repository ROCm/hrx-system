// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/lower.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func/reference.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/lower/control.h"
#include "loom/target/arch/vm/lower/types.h"

#define LOOM_VM_SOURCE_LOWERING_LIMITS(max_operand_count, max_result_count) \
  enum {                                                                    \
    LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT = max_operand_count,          \
    LOOM_VM_SOURCE_LOWERING_MAX_RESULT_COUNT = max_result_count,            \
    LOOM_VM_SOURCE_LOWERING_MAX_TYPE_COUNT =                                \
        max_operand_count + max_result_count,                               \
  };
#define LOOM_VM_SOURCE_LOWERING_ROW(...)
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_SOURCE_LOWERING_ROW
#undef LOOM_VM_SOURCE_LOWERING_LIMITS

typedef struct loom_vm_source_lowering_row_t {
  // Source operation kind matched by this row.
  loom_op_kind_t source_op_kind;
  // Dense Core VM descriptor ordinal selected by this row.
  uint16_t descriptor_ordinal;
  // Descriptor-local selector immediate ordinal, or UINT8_MAX when absent.
  uint8_t selector_immediate_ordinal;
  // Encoded selector value when |selector_immediate_ordinal| is present.
  uint8_t selector_value;
  // Operand types followed by result types at the generated maximum offsets.
  loom_scalar_type_t scalar_types[LOOM_VM_SOURCE_LOWERING_MAX_TYPE_COUNT];
} loom_vm_source_lowering_row_t;

typedef struct loom_vm_source_constant_t {
  // Source SSA result receiving the constant value.
  loom_value_id_t result;
  // Canonical VM value-cell bits; narrow payloads are zero-extended.
  uint64_t bits;
} loom_vm_source_constant_t;

static const loom_vm_source_lowering_row_t kVmSourceLoweringRows[] = {
#define LOOM_VM_SOURCE_LOWERING_LIMITS(...)
#define LOOM_VM_SOURCE_LOWERING_ROW(source_op_kind, descriptor_ordinal, \
                                    selector_immediate_ordinal,         \
                                    selector_value, ...)                \
  {source_op_kind,                                                      \
   descriptor_ordinal,                                                  \
   selector_immediate_ordinal,                                          \
   selector_value,                                                      \
   {__VA_ARGS__}},
#include "loom/target/arch/vm/lowering_rows.inl"
#undef LOOM_VM_SOURCE_LOWERING_ROW
#undef LOOM_VM_SOURCE_LOWERING_LIMITS
};

static bool loom_vm_source_lowering_row_matches(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_vm_source_lowering_row_t* row) {
  if (source_op->kind != row->source_op_kind) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, operands[i]);
    if (!loom_type_is_scalar(type) ||
        loom_type_element_type(type) != row->scalar_types[i]) {
      return false;
    }
  }
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, results[i]);
    if (!loom_type_is_scalar(type) ||
        loom_type_element_type(type) !=
            row->scalar_types[LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT + i]) {
      return false;
    }
  }
  return true;
}

static uint64_t loom_vm_float_constant_bits(loom_scalar_type_t scalar_type,
                                            double value) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return iree_math_f32_to_f8e4m3fn((float)value);
    case LOOM_SCALAR_TYPE_F8E5M2:
      return iree_math_f32_to_f8e5m2((float)value);
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16((float)value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16((float)value);
    case LOOM_SCALAR_TYPE_F32: {
      const float f32_value = (float)value;
      uint32_t bits = 0;
      memcpy(&bits, &f32_value, sizeof(bits));
      return bits;
    }
    case LOOM_SCALAR_TYPE_F64: {
      uint64_t bits = 0;
      memcpy(&bits, &value, sizeof(bits));
      return bits;
    }
    default:
      IREE_ASSERT_UNREACHABLE("verified VM float constant type");
      return 0;
  }
}

static uint64_t loom_vm_constant_bits(loom_scalar_type_t scalar_type,
                                      loom_attribute_t value) {
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_vm_float_constant_bits(scalar_type, loom_attr_as_f64(value));
  }
  const int64_t integer_value =
      scalar_type == LOOM_SCALAR_TYPE_I1 && value.kind == LOOM_ATTR_BOOL
          ? (loom_attr_as_bool(value) ? 1 : 0)
          : loom_attr_as_i64(value);
  const int32_t bit_width = loom_scalar_type_bitwidth(scalar_type);
  IREE_ASSERT_GT(bit_width, 0);
  return iree_math_mask_low_bits_u64((uint64_t)integer_value, bit_width);
}

static bool loom_vm_try_get_source_constant(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_vm_source_constant_t* out_constant) {
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_attribute_t value = loom_attr_absent();
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      result = loom_index_constant_result(source_op);
      value = loom_index_constant_value(source_op);
      break;
    case LOOM_OP_SCALAR_CONSTANT:
      result = loom_scalar_constant_result(source_op);
      value = loom_scalar_constant_value(source_op);
      break;
    default:
      return false;
  }
  const loom_type_t result_type = loom_module_value_type(module, result);
  *out_constant = (loom_vm_source_constant_t){
      .result = result,
      .bits = loom_vm_constant_bits(loom_type_element_type(result_type), value),
  };
  return true;
}

static uint16_t loom_vm_source_type_register_class(const loom_module_t* module,
                                                   loom_type_t source_type) {
  if (loom_type_is_scalar(source_type)) return VM_CORE_REG_CLASS_ID_VALUE;
  if (loom_func_ref_type_isa(source_type)) {
    const loom_type_t signature =
        loom_func_ref_resolve_signature(module, source_type);
    return loom_type_is_function(signature)
               ? VM_CORE_REG_CLASS_ID_FUNCTION
               : LOOM_LOW_REGISTER_CLASS_ID_INVALID;
  }
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_resolve(module, source_type);
  return descriptor != NULL && descriptor->semantics.semantic ==
                                   LOOM_TYPE_SEMANTIC_MANAGED_REFERENCE
             ? VM_CORE_REG_CLASS_ID_REF
             : LOOM_LOW_REGISTER_CLASS_ID_INVALID;
}

static iree_status_t loom_vm_map_type(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  (void)user_data;
  const uint16_t register_class = loom_vm_source_type_register_class(
      loom_low_lower_context_module(context), source_type);
  if (register_class != LOOM_LOW_REGISTER_CLASS_ID_INVALID) {
    return loom_low_lower_make_typed_register_type(
        context, register_class, /*unit_count=*/1, source_type, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static bool loom_vm_source_type_supported(void* user_data,
                                          const loom_module_t* module,
                                          loom_type_t source_type) {
  (void)user_data;
  return loom_vm_source_type_register_class(module, source_type) !=
         LOOM_LOW_REGISTER_CLASS_ID_INVALID;
}

static iree_status_t loom_vm_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  *out_abi_layout = loom_named_attr_slice_empty();
  if (layout_kind != LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC) {
    return iree_ok_status();
  }
  loom_attribute_t layout_attr = loom_attr_absent();
  IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_make_attr(
      loom_low_lower_context_module(context), argument_types, argument_count,
      result_types, result_count, &layout_attr));
  *out_abi_layout = loom_attr_as_dict(layout_attr);
  return iree_ok_status();
}

static iree_status_t loom_vm_select_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_vm_source_constant_t constant = {0};
  if (loom_vm_try_get_source_constant(module, source_op, &constant)) {
    *out_plan = loom_low_lower_plan_make(
        loom_vm_constant_descriptor_ordinal(constant.bits), NULL);
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVmSourceLoweringRows); ++i) {
    const loom_vm_source_lowering_row_t* row = &kVmSourceLoweringRows[i];
    if (loom_vm_source_lowering_row_matches(module, source_op, row)) {
      *out_plan = loom_low_lower_plan_make(row->descriptor_ordinal, row);
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_constant(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan) {
  loom_vm_source_constant_t constant = {0};
  const bool is_constant = loom_vm_try_get_source_constant(
      loom_low_lower_context_module(context), source_op, &constant);
  IREE_ASSERT(is_constant);
  (void)is_constant;
  IREE_ASSERT_EQ(plan.id, loom_vm_constant_descriptor_ordinal(constant.bits));

  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_op, constant.result, &low_result_type));
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_constant_build(
      loom_low_lower_context_builder(context), constant.bits, low_result_type,
      source_op->location, &low_value));
  return loom_low_lower_bind_value(context, constant.result, low_value);
}

static iree_status_t loom_vm_emit_op(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan) {
  (void)user_data;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, (uint32_t)plan.id);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  if (descriptor->op_kind == LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
    return loom_vm_emit_constant(context, source_op, plan);
  }

  loom_value_id_t low_operands[LOOM_VM_SOURCE_LOWERING_MAX_OPERAND_COUNT];
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, source_operands[i], &low_operands[i]));
  }

  loom_type_t low_result_types[LOOM_VM_SOURCE_LOWERING_MAX_RESULT_COUNT];
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
        context, source_op, source_results[i], &low_result_types[i]));
  }

  const loom_vm_source_lowering_row_t* row = plan.target_data;
  IREE_ASSERT(row != NULL);
  loom_named_attr_t selector_attr = {0};
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (row->selector_immediate_ordinal != UINT8_MAX) {
    IREE_ASSERT_LT(row->selector_immediate_ordinal,
                   descriptor->immediate_count);
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    row->selector_immediate_ordinal];
    const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
        descriptor_set, immediate->field_name_string_offset);
    IREE_RETURN_IF_ERROR(
        loom_builder_intern_string(loom_low_lower_context_builder(context),
                                   immediate_name, &selector_attr.name_id));
    selector_attr.value = loom_attr_i64(row->selector_value);
    attrs = loom_make_named_attr_slice(&selector_attr, 1);
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, low_operands, source_op->operand_count,
      attrs, low_result_types, source_op->result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  const loom_value_id_t* low_results = loom_op_const_results(low_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static const loom_low_lower_policy_t kVmCoreLowLowerPolicy = {
    .name = IREE_SVL("vm-core-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_vm_map_type, .user_data = NULL},
    .source_type_supported = {.fn = loom_vm_source_type_supported,
                              .user_data = NULL},
    .map_abi_layout = {.fn = loom_vm_map_abi_layout, .user_data = NULL},
    .switch_lowering =
        {
            .can_emit = loom_vm_switch_lowering_can_emit,
            .emit = loom_vm_switch_lowering_emit,
            .user_data = NULL,
        },
    .select_op = {.fn = loom_vm_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_vm_emit_op, .user_data = NULL},
};

void loom_vm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("vm.core"),
          .policy = &kVmCoreLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
