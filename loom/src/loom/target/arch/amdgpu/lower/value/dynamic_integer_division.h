// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_DYNAMIC_INTEGER_DIVISION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_DYNAMIC_INTEGER_DIVISION_H_

#include "loom/codegen/low/lower/realization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains one exact runtime-u32 division recipe in the per-operation plan
// arena.
typedef struct loom_amdgpu_dynamic_u32_division_plan_t {
  // Dividend whose complete payload participates in division.
  loom_value_id_t lhs;
  // Runtime divisor whose complete payload participates in division.
  loom_value_id_t rhs;
  // Quotient result, or invalid when the source operation omits it.
  loom_value_id_t quotient;
  // Remainder result, or invalid when the source operation omits it.
  loom_value_id_t remainder;
  // Selected SGPR or VGPR carrier for the computation.
  uint16_t register_class_id;
  // Shared reciprocal placed outside a loop, or NULL when the divisor or
  // source control domain requires direct preparation at the division.
  const loom_low_lower_realization_t* reciprocal;
} loom_amdgpu_dynamic_u32_division_plan_t;

// Selects exact signed or unsigned runtime-u32 division.
iree_status_t loom_amdgpu_select_dynamic_u32_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_dynamic_u32_division_plan_t* out_plan, bool* out_selected);

// Publishes result carriers and requests the divisor's pure preparation.
iree_status_t loom_amdgpu_finalize_dynamic_u32_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_dynamic_u32_division_plan_t* plan);

// Marks the dividend and runtime divisor consumed during emission.
void loom_amdgpu_mark_dynamic_u32_division_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_amdgpu_dynamic_u32_division_plan_t* plan);

// Emits exact runtime-u32 quotient and remainder arithmetic.
iree_status_t loom_amdgpu_lower_dynamic_u32_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_dynamic_u32_division_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_DYNAMIC_INTEGER_DIVISION_H_
