// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 one-pointer view ABI planning and materialization.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_VIEW_ABI_H_
#define LOOM_TARGET_ARCH_X86_LOWER_VIEW_ABI_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains view-address plans encountered by the common source traversal.
extern const loom_low_lower_source_plan_observer_t
    loom_x86_view_abi_source_plan_observer;

// Materializes one-pointer view operands at ordinary function-call boundaries.
iree_status_t loom_x86_materialize_view_abi_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_VIEW_ABI_H_
