// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private helpers shared by the descriptor-backed low text-asm environment.

#ifndef LOOM_CODEGEN_LOW_TEXT_ASM_INTERNAL_H_
#define LOOM_CODEGEN_LOW_TEXT_ASM_INTERNAL_H_

#include "loom/codegen/low/text_asm.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_low_descriptor_text_asm_lookup_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t field_name, const loom_named_attr_t** out_attr);

iree_status_t loom_low_descriptor_text_asm_structural_attr_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor);

iree_status_t loom_low_descriptor_text_asm_build_structural(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t key,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, int64_t offset, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

iree_status_t loom_low_descriptor_text_asm_describe_structural_operation(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TEXT_ASM_INTERNAL_H_
