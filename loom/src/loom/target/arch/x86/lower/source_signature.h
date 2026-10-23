// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86-64 SysV projection of retained C source object boundaries.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_SOURCE_SIGNATURE_H_
#define LOOM_TARGET_ARCH_X86_LOWER_SOURCE_SIGNATURE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Plans exact physical SysV carriers and placement for one verified C source
// signature.
iree_status_t loom_x86_plan_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    const loom_c_abi_signature_layout_t* layout,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    loom_low_lower_source_signature_target_plan_t* out_plan);

// Emits byte-exact pack and unpack operations for a retained x86 source
// signature plan.
iree_status_t loom_x86_emit_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_SOURCE_SIGNATURE_H_
