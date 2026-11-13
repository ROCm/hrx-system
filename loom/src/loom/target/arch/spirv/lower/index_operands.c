// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/index_operands.h"

#include "loom/codegen/low/lower/representation_observer.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/lower/index.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/registers.h"
#include "loom/util/fact_table.h"

static bool loom_spirv_index_type_is_wide(loom_type_t type) {
  const loom_type_t* payload = loom_type_register_value_type(type);
  return payload && loom_type_element_type(*payload) == LOOM_SCALAR_TYPE_I64;
}

iree_status_t loom_spirv_prepare_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_type_t actual_type, loom_type_t required_type, const void** out_plan) {
  (void)user_data;
  (void)source_op;
  static const uint32_t kConversions[] = {
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I32_I64,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_S_CONVERT_I64_I32,
  };
  static const uint32_t kNativeIndexCast =
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_COPY_OBJECT_I32;
  static const uint32_t kOffsetToIndex =
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_OFFSET64_I64;
  static const uint32_t kZeroExtend =
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_CONVERT_U32_U64;
  const bool actual_wide = loom_spirv_index_type_is_wide(actual_type);
  const bool required_wide = loom_spirv_index_type_is_wide(required_type);
  if (loom_low_register_type_class_id(actual_type) ==
          SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64 &&
      required_wide) {
    *out_plan = &kOffsetToIndex;
    return iree_ok_status();
  }
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      loom_low_lower_context_fact_table(context), source_value);
  if (!actual_wide && required_wide && facts.range_lo >= 0 &&
      facts.range_hi > INT32_MAX) {
    *out_plan = &kZeroExtend;
    return iree_ok_status();
  }
  *out_plan = actual_wide != required_wide ? &kConversions[actual_wide]
              : loom_type_equal(actual_type, required_type) ? NULL
                                                            : &kNativeIndexCast;
  return iree_ok_status();
}

static iree_status_t loom_spirv_index_emit_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t descriptor_id, const loom_value_id_t* operands,
    uint16_t operand_count, loom_type_t result_type,
    loom_value_id_t* out_value) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_id)};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0,
      source_op->location, &op));
  *out_value = loom_op_const_results(op)[0];
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_value_id_t low_value, loom_type_t required_type, const void* plan,
    loom_value_id_t* out_value) {
  (void)user_data;
  (void)source_value;
  if (plan == NULL) {
    *out_value = low_value;
    return iree_ok_status();
  }
  if (*(const uint32_t*)plan ==
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_U_CONVERT_U32_U64) {
    const loom_type_t native_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_value);
    IREE_RETURN_IF_ERROR(loom_spirv_index_emit_descriptor(
        context, source_op,
        SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_I32_U32, &low_value, 1,
        native_type, &low_value));
    IREE_RETURN_IF_ERROR(loom_spirv_index_emit_descriptor(
        context, source_op, *(const uint32_t*)plan, &low_value, 1,
        required_type, &low_value));
    return loom_spirv_index_emit_descriptor(
        context, source_op,
        SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_BITCAST_U64_I64, &low_value, 1,
        required_type, out_value);
  }
  return loom_spirv_index_emit_descriptor(context, source_op,
                                          *(const uint32_t*)plan, &low_value, 1,
                                          required_type, out_value);
}

typedef struct loom_spirv_index_operand_plan_t {
  // Exact typed carrier produced by this use's materialization.
  loom_type_t type;
  // Retained conversion selected after the producer's carrier is final.
  const void* conversion;
} loom_spirv_index_operand_plan_t;

iree_status_t loom_spirv_can_materialize_native_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize) {
  (void)context;
  (void)source_op;
  (void)source_value;
  *out_can_materialize = true;
  return iree_ok_status();
}

static bool loom_spirv_index_arithmetic_is_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < source_op->operand_count + source_op->result_count;
       ++i) {
    const loom_value_id_t value =
        i < source_op->operand_count
            ? loom_op_const_operands(source_op)[i]
            : loom_op_const_results(source_op)[i - source_op->operand_count];
    if (!loom_type_equal(loom_module_value_type(module, value),
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
      continue;
    }
    const loom_value_facts_t value_facts =
        loom_value_fact_table_lookup(facts, value);
    if (value_facts.range_lo >= INT32_MIN && value_facts.range_hi > INT32_MAX) {
      return false;
    }
    if (!loom_value_facts_fit_signed_bit_count(value_facts, 32)) {
      loom_low_representation_id_t representation =
          LOOM_LOW_REPRESENTATION_ID_NONE;
      loom_low_lower_representation_lookup_if_ready(context, value,
                                                    &representation);
      if (representation == LOOM_SPIRV_INDEX_REPRESENTATION_WIDE) {
        return false;
      }
    }
  }
  return true;
}

iree_status_t loom_spirv_can_materialize_native_index_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize) {
  (void)source_value;
  *out_can_materialize =
      loom_spirv_index_arithmetic_is_native(context, source_op);
  return iree_ok_status();
}

iree_status_t loom_spirv_can_materialize_wide_index_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize) {
  (void)source_value;
  *out_can_materialize =
      !loom_spirv_index_arithmetic_is_native(context, source_op);
  return iree_ok_status();
}

static iree_status_t loom_spirv_prepare_index_operand(
    loom_low_lower_context_t* context, loom_value_id_t source_value, bool wide,
    const void** out_plan) {
  *out_plan = NULL;
  const loom_type_t actual_type =
      loom_low_lower_value_binding_type(context, source_value);
  if (loom_spirv_index_type_is_wide(actual_type) == wide) {
    return iree_ok_status();
  }
  loom_spirv_index_operand_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1,
      loom_type_scalar(wide ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_INDEX),
      &plan->type));
  IREE_RETURN_IF_ERROR(loom_spirv_prepare_control_operand(
      NULL, context, NULL, source_value, actual_type, plan->type,
      &plan->conversion));
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_spirv_prepare_native_index(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value,
                                              const void** out_plan) {
  return loom_spirv_prepare_index_operand(context, source_value, false,
                                          out_plan);
}

iree_status_t loom_spirv_prepare_wide_index(loom_low_lower_context_t* context,
                                            loom_value_id_t source_value,
                                            const void** out_plan) {
  return loom_spirv_prepare_index_operand(context, source_value, true,
                                          out_plan);
}

loom_type_t loom_spirv_materialized_index_operand_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* opaque_plan) {
  const loom_spirv_index_operand_plan_t* plan = opaque_plan;
  return plan ? plan->type
              : loom_low_lower_value_binding_type(context, source_value);
}

iree_status_t loom_spirv_emit_index_operand(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_value_id_t source_value,
                                            const void* opaque_plan,
                                            loom_value_id_t* out_low_value) {
  const loom_value_id_t value =
      loom_low_lower_lookup_value(context, source_value);
  const loom_spirv_index_operand_plan_t* plan = opaque_plan;
  if (plan == NULL) {
    *out_low_value = value;
    return iree_ok_status();
  }
  return loom_spirv_emit_control_operand(NULL, context, source_op, source_value,
                                         value, plan->type, plan->conversion,
                                         out_low_value);
}
