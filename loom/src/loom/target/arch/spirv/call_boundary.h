// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source callable admission before SPIR-V physical signature selection.

#ifndef LOOM_TARGET_ARCH_SPIRV_CALL_BOUNDARY_H_
#define LOOM_TARGET_ARCH_SPIRV_CALL_BOUNDARY_H_

#include "loom/target/legalization.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Storage-bearing signatures and resource-producing bodies require expansion
// before the function-local SPIR-V storage planner selects pointer types.
bool loom_spirv_source_callable_requires_inline(const loom_module_t* module,
                                                loom_func_like_t function);

// Rejects opaque fragments crossing a retained source call or return before
// boundary projection can split their logical vector spelling into packets.
// The matcher selects only unsupported boundaries, before contract queries or
// report construction. Ordinary calls and other targets need no legalizer work.
bool loom_spirv_match_call_boundary(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op);

// Reports an unsupported boundary selected by loom_spirv_match_call_boundary.
iree_status_t loom_spirv_legalize_call_boundary(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Enforces the same boundary contract for direct source-to-Low consumers.
extern const loom_target_low_legality_provider_t
    loom_spirv_call_boundary_legality_provider;

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_CALL_BOUNDARY_H_
