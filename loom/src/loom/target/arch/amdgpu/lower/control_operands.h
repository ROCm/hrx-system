// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU carrier conversions and defined-storage contracts at control edges.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_OPERANDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_OPERANDS_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Admits lossless native argument conversions from retained distribution and
// range facts. This does not choose or change either endpoint's carrier.
bool loom_amdgpu_control_operand_accepts(loom_type_t source_type,
                                         loom_value_facts_t source_facts,
                                         loom_type_t actual_type,
                                         loom_type_t required_type);

// Retains branch/call/return literal, extension, or active-mask choices using
// the receiving contract's established proof. Selecting one lane from a
// vector-register producer requires subgroup uniformity.
iree_status_t loom_amdgpu_prepare_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator, loom_value_id_t source_value,
    loom_type_t required_type, const void** out_plan);

// Emits the selected carrier conversion without source analysis. The canonical
// source binding is preserved, so each edge can use its own receiving carrier.
iree_status_t loom_amdgpu_emit_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id, loom_type_t required_low_type,
    const void* plan, loom_value_id_t* out_low_value_id);

// Defines every stored register part after prepared carrier conversion. This
// consumes only Low operands and also applies to structured control and calls.
iree_status_t loom_amdgpu_materialize_structural_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_OPERANDS_H_
