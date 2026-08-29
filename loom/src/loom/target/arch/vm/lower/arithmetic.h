// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM arithmetic expansions that require multiple instructions.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_ARITHMETIC_H_
#define LOOM_TARGET_ARCH_VM_LOWER_ARITHMETIC_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source legality for Core VM multi-instruction arithmetic expansions.
extern const loom_target_low_legality_provider_t
    loom_vm_arithmetic_low_legality_provider;

// Emits an i64 multiply using values already mapped into the Low function.
iree_status_t loom_vm_arithmetic_emit_mul_i64(loom_low_lower_context_t* context,
                                              loom_value_id_t lhs,
                                              loom_value_id_t rhs,
                                              loom_type_t result_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_result);

// Emits an i64 multiply-add using values already mapped into the Low function.
iree_status_t loom_vm_arithmetic_emit_madd_i64(
    loom_low_lower_context_t* context, loom_value_id_t a, loom_value_id_t b,
    loom_value_id_t c, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_result);

// Selects a Core VM arithmetic expansion when available.
bool loom_vm_arithmetic_try_select_op(const loom_op_t* source_op,
                                      loom_low_lower_plan_t* out_plan);

// Emits one selected Core VM arithmetic expansion.
//
// |out_handled| is true when |plan| belongs to this lowering.
iree_status_t loom_vm_arithmetic_emit_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan,
                                         bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOWER_ARITHMETIC_H_
