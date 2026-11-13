// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/integer_division_dispatch.h"

iree_status_t loom_amdgpu_select_integer_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_integer_division_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_integer_division_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_wide_division_plan(
      context, source_op, &out_plan->payload.constant_wide, out_selected));
  if (*out_selected) {
    out_plan->kind = LOOM_AMDGPU_INTEGER_DIVISION_PLAN_CONSTANT_WIDE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_dynamic_u32_division_plan(
      context, source_op, &out_plan->payload.dynamic_u32, out_selected));
  if (*out_selected) {
    out_plan->kind = LOOM_AMDGPU_INTEGER_DIVISION_PLAN_DYNAMIC_U32;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_integer_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_integer_division_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_CONSTANT_WIDE:
      return loom_amdgpu_finalize_wide_division_plan(
          context, &plan->payload.constant_wide);
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_DYNAMIC_U32:
      return loom_amdgpu_finalize_dynamic_u32_division_plan(
          context, source_op, &plan->payload.dynamic_u32);
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU integer division plan kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

void loom_amdgpu_mark_integer_division_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_amdgpu_integer_division_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_CONSTANT_WIDE:
      loom_low_lower_require_source_value_storage(
          context, plan->payload.constant_wide.source);
      return;
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_DYNAMIC_U32:
      loom_amdgpu_mark_dynamic_u32_division_plan_storage_demands(
          context, &plan->payload.dynamic_u32);
      return;
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU integer division plan kind");
  }
}

iree_status_t loom_amdgpu_lower_integer_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_division_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_CONSTANT_WIDE:
      return loom_amdgpu_lower_wide_division(context, source_op,
                                             &plan->payload.constant_wide);
    case LOOM_AMDGPU_INTEGER_DIVISION_PLAN_DYNAMIC_U32:
      return loom_amdgpu_lower_dynamic_u32_division(context, source_op,
                                                    &plan->payload.dynamic_u32);
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU integer division plan kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
