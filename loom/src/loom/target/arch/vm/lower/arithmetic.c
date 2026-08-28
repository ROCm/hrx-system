// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower/arithmetic.h"

#include <stdint.h>

#include "loom/ops/index/ops.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/records/target_records.h"

// Callback plan IDs occupy a target-local domain outside descriptor ordinals.
static const uint64_t kLoomVmArithmeticPlanId = UINT64_C(0x100000300);

static iree_status_t loom_vm_arithmetic_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  (void)provider;
  *out_handled =
      loom_vm_target_bundle_is_core(loom_target_low_legality_bundle(context)) &&
      source_op->kind == LOOM_OP_INDEX_MADD;
  return iree_ok_status();
}

const loom_target_low_legality_provider_t
    loom_vm_arithmetic_low_legality_provider = {
        .name = IREE_SVL("vm-arithmetic-expansions"),
        .builtin_dialect_bits = UINT64_C(1) << LOOM_DIALECT_INDEX,
        .try_verify_op = loom_vm_arithmetic_try_verify_op,
};

bool loom_vm_arithmetic_try_select_op(const loom_op_t* source_op,
                                      loom_low_lower_plan_t* out_plan) {
  if (source_op->kind != LOOM_OP_INDEX_MADD) return false;
  *out_plan = loom_low_lower_plan_make(kLoomVmArithmeticPlanId,
                                       /*target_data=*/NULL);
  return true;
}

static iree_status_t loom_vm_arithmetic_emit_binary(
    loom_low_lower_context_t* context, uint16_t descriptor_ordinal,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  const loom_low_lower_resolved_descriptor_t resolved_descriptor = {
      .descriptor = descriptor,
  };
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &low_op));
  *out_result = loom_op_const_results(low_op)[0];
  return iree_ok_status();
}

iree_status_t loom_vm_arithmetic_emit_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan,
                                         bool* out_handled) {
  *out_handled = plan.id == kLoomVmArithmeticPlanId;
  if (!*out_handled) return iree_ok_status();
  IREE_ASSERT_EQ(source_op->kind, LOOM_OP_INDEX_MADD);

  loom_value_id_t a = LOOM_VALUE_ID_INVALID;
  loom_value_id_t b = LOOM_VALUE_ID_INVALID;
  loom_value_id_t c = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, loom_index_madd_a(source_op), &a));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, loom_index_madd_b(source_op), &b));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, loom_index_madd_c(source_op), &c));

  const loom_value_id_t source_result = loom_index_madd_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, &result_type));
  loom_value_id_t product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_arithmetic_emit_binary(
      context, VM_CORE_DESCRIPTOR_REF_INTEGER_MUL_I64, a, b, result_type,
      source_op->location, &product));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vm_arithmetic_emit_binary(
      context, VM_CORE_DESCRIPTOR_REF_INTEGER_ADD_I64, product, c, result_type,
      source_op->location, &result));
  return loom_low_lower_bind_value(context, source_result, result);
}
