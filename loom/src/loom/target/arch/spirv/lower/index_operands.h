// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Index operand conversions from selected SPIR-V carriers.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_OPERANDS_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_OPERANDS_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Every index carrier can supply its native coordinate bits.
iree_status_t loom_spirv_can_materialize_native_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize);

// Matches arithmetic whose facts and retained representation use native width.
iree_status_t loom_spirv_can_materialize_native_index_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize);

// Matches arithmetic requiring the wider coordinate domain.
iree_status_t loom_spirv_can_materialize_wide_index_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, bool* out_can_materialize);

// Retains the native coordinate type when a wide carrier needs projection.
iree_status_t loom_spirv_prepare_native_index(loom_low_lower_context_t* context,
                                              loom_value_id_t source_value,
                                              const void** out_plan);

// Retains extension to the wider coordinate type when needed by the use.
iree_status_t loom_spirv_prepare_wide_index(loom_low_lower_context_t* context,
                                            loom_value_id_t source_value,
                                            const void** out_plan);

// Returns the type retained by preparation, or the producer's existing carrier.
loom_type_t loom_spirv_materialized_index_operand_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void* plan);

// Applies the retained extension or projection, or reuses an unchanged carrier.
iree_status_t loom_spirv_emit_index_operand(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_value_id_t source_value,
                                            const void* plan,
                                            loom_value_id_t* out_low_value);

// Prepares the selected receiving carrier after producer bindings are final.
iree_status_t loom_spirv_prepare_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_type_t actual_type, loom_type_t required_type, const void** out_plan);

// Emits the retained receiving-carrier conversion without source fact queries.
iree_status_t loom_spirv_emit_control_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value,
    loom_value_id_t low_value, loom_type_t required_type, const void* plan,
    loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_INDEX_OPERANDS_H_
