// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for control-flow source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Plans divergent branch expansion before the source body is emitted.
iree_status_t loom_amdgpu_prepare_branch(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_terminator);

// Materializes branch payload values to the register class selected for the
// destination block argument.
iree_status_t loom_amdgpu_materialize_branch_arg(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator, uint8_t successor_index,
    uint16_t arg_index, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id, loom_type_t required_low_type,
    loom_value_id_t* out_low_value_id);

// Emits a conditional branch, using EXEC narrowing for divergent SGPR masks.
iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_H_
