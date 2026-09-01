// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source and target-Low function boundary construction.

#ifndef LOOM_CODEGEN_LOW_LOWER_FUNCTION_H_
#define LOOM_CODEGEN_LOW_LOWER_FUNCTION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Asserts source-function lowering preconditions owned by the caller.
void loom_low_lower_assert_options(const loom_module_t* module,
                                   loom_func_like_t source_function,
                                   const loom_low_lower_options_t* options);

// Maps and validates one source SSA value type.
iree_status_t loom_low_lower_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type);

// Validates and caches the active source function boundary mapping.
iree_status_t loom_low_lower_check_function_signature(
    loom_low_lower_context_t* context, loom_region_t* source_body);

// Returns the number of source arguments passed directly through the Low ABI.
uint16_t loom_low_lower_direct_argument_count(
    const loom_low_lower_context_t* context);

// Creates the target Low function shell and maps its public boundary.
iree_status_t loom_low_lower_create_function_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref);

// Translates source function predicates through the source-to-Low value map.
iree_status_t loom_low_lower_remap_function_predicates(
    loom_low_lower_context_t* context);

// Copies source presentation retained on the replacement Low function.
iree_status_t loom_low_lower_copy_function_source_presentation(
    loom_low_lower_context_t* context);

// Emits resource-backed source arguments into the Low function entry block.
iree_status_t loom_low_lower_emit_argument_resource_imports(
    loom_low_lower_context_t* context);

// Emits explicit ABI-to-semantic argument transfers when required by policy.
iree_status_t loom_low_lower_emit_direct_argument_transfers(
    loom_low_lower_context_t* context);

// Emits target-owned function preamble operations.
iree_status_t loom_low_lower_emit_preamble(loom_low_lower_context_t* context);

// Emits target-owned function entry setup after argument materialization.
iree_status_t loom_low_lower_emit_entry_setup(
    loom_low_lower_context_t* context);

// Finalizes target-owned state after the Low function body has been emitted.
iree_status_t loom_low_lower_finalize_function(
    loom_low_lower_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_FUNCTION_H_
