// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V function-section instruction helpers.
//
// These helpers emit small, common instructions whose only state requirement is
// the sectioned module builder. They intentionally do not know about low IR,
// ABI plans, packet rows, or value-table ownership.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_INSTRUCTIONS_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_INSTRUCTIONS_H_

#include "iree/base/api.h"
#include "loom/target/emit/spirv/module_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits OpAccessChain into the function section.
iree_status_t loom_spirv_module_emit_access_chain(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t base_pointer_id, const uint32_t* index_ids, uint8_t index_count,
    uint32_t* out_result_id);

// Emits OpLoad with an Aligned memory operand into the function section.
iree_status_t loom_spirv_module_emit_load_aligned(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t pointer_id, uint32_t alignment, uint32_t* out_result_id);

// Emits a unary result instruction into the function section.
iree_status_t loom_spirv_module_emit_unary_result(
    loom_spirv_module_builder_t* builder, uint32_t opcode,
    uint32_t result_type_id, uint32_t operand_id, uint32_t* out_result_id);

// Emits a binary result instruction into the function section.
iree_status_t loom_spirv_module_emit_binary_result(
    loom_spirv_module_builder_t* builder, uint32_t opcode,
    uint32_t result_type_id, uint32_t lhs_id, uint32_t rhs_id,
    uint32_t* out_result_id);

// Returns |operand_id| when it already has |result_type_id|, otherwise emits an
// OpBitcast into the function section.
iree_status_t loom_spirv_module_emit_bitcast_if_needed(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t operand_type_id, uint32_t operand_id, uint32_t* out_result_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_INSTRUCTIONS_H_
