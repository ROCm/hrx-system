// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER_DIVISION_DISPATCH_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER_DIVISION_DISPATCH_H_

#include "loom/target/arch/amdgpu/lower/value/dynamic_integer_division.h"
#include "loom/target/arch/amdgpu/lower/value/integer_division.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_amdgpu_integer_division_plan_kind_t;
enum loom_amdgpu_integer_division_plan_kind_e {
  // Exact wide quotient and remainder by a compile-time constant.
  LOOM_AMDGPU_INTEGER_DIVISION_PLAN_CONSTANT_WIDE = 0,
  // Exact i32 or u32-index quotient and remainder by a runtime value.
  LOOM_AMDGPU_INTEGER_DIVISION_PLAN_DYNAMIC_U32 = 1,
};

// Retains one exact constant-wide or runtime-u32 division plan.
typedef struct loom_amdgpu_integer_division_plan_t {
  // Selected plan family identifying the active payload member.
  loom_amdgpu_integer_division_plan_kind_t kind;
  // Family-specific plan payload.
  union {
    // Compile-time constant wide division plan.
    loom_amdgpu_wide_division_plan_t constant_wide;
    // Runtime u32 division plan.
    loom_amdgpu_dynamic_u32_division_plan_t dynamic_u32;
  } payload;
} loom_amdgpu_integer_division_plan_t;

// Selects exact constant-wide or runtime-u32 division.
iree_status_t loom_amdgpu_select_integer_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_integer_division_plan_t* out_plan, bool* out_selected);

// Publishes result carriers and retains reusable divisor preparation.
iree_status_t loom_amdgpu_finalize_integer_division_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_integer_division_plan_t* plan);

// Marks source values consumed by the retained division plan.
void loom_amdgpu_mark_integer_division_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_amdgpu_integer_division_plan_t* plan);

// Emits the retained constant-wide or runtime-u32 recipe.
iree_status_t loom_amdgpu_lower_integer_division(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_division_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER_DIVISION_DISPATCH_H_
