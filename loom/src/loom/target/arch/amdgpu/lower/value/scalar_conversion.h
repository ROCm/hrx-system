// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for ordinary scalar numeric conversions.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_SCALAR_CONVERSION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_SCALAR_CONVERSION_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_scalar_conversion_op_group_e {
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI = 0,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTF,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_SITOFP,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI,
  LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_,
} loom_amdgpu_scalar_conversion_op_group_t;

// Returns the scalar numeric conversion family for |op_kind|.
loom_amdgpu_scalar_conversion_op_group_t loom_amdgpu_scalar_conversion_op_group(
    loom_op_kind_t op_kind);

// Selects an AMDGPU scalar conversion plan.
iree_status_t loom_amdgpu_select_scalar_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU scalar conversion plan.
iree_status_t loom_amdgpu_lower_scalar_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan);

// Verifies AMDGPU low legality for scalar conversions.
iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_SCALAR_CONVERSION_H_
