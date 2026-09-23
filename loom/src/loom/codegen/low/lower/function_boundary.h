// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low callable boundary lowering.
//
// A source function boundary is mapped once and then consumed throughout
// lowering. Validation establishes argument mappings before source planning;
// the source-plan traversal joins return carriers into retained result types.
// Definition creation materializes the target-Low callable,
// entry binding connects direct arguments, resource emission materializes
// arguments omitted from the direct ABI, and predicate remapping translates
// source value references after those bindings exist.
//
// Function declarations use the same type and metadata mapping without a body.
// They are lowered independently by the module source-to-Low pass before
// function definitions.

#ifndef LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_
#define LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries the native ABI representation of a source function argument without
// emitting diagnostics or recording a required boundary mapping. An unsupported
// argument has abi_type none; other output fields are then unused. The default
// direct ABI uses the native value query when the policy has no argument
// mapper. A target may support a native value without supporting it as an ABI
// argument. Allocation failures while constructing a native type propagate
// normally.
iree_status_t loom_low_lower_query_argument(
    loom_low_lower_context_t* context, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

// Validates the source callable boundary, maps arguments, and allocates empty
// result mappings. This must run before source-plan construction.
iree_status_t loom_low_lower_function_boundary_validate(
    loom_low_lower_context_t* context);

// Joins one return's native value carriers into the retained callable result
// types. Called once per return by the existing source-plan traversal.
iree_status_t loom_low_lower_function_boundary_observe_return(
    loom_low_lower_context_t* context, const loom_op_t* return_op);

// Completes result mappings after all returns have been observed. A callable
// without returning paths retains the target mapping of its declared types.
iree_status_t loom_low_lower_function_boundary_finalize(
    loom_low_lower_context_t* context);

// Creates the target-Low function or kernel definition for the mapped source
// callable. The definition is inserted immediately before the source op and
// recorded in the lowering context and result.
iree_status_t loom_low_lower_function_boundary_create(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref);

// Binds source entry arguments represented directly in the target ABI to the
// corresponding target-Low entry block arguments.
iree_status_t loom_low_lower_function_boundary_bind_entry_arguments(
    loom_low_lower_context_t* context, const loom_block_t* source_entry_block,
    loom_block_t* low_entry_block);

// Remaps source callable predicates through the completed source-to-Low value
// map and attaches them to the target-Low definition.
iree_status_t loom_low_lower_function_boundary_remap_predicates(
    loom_low_lower_context_t* context);

// Emits target-Low resource imports for source arguments excluded from the
// direct callable signature and binds their source values.
iree_status_t loom_low_lower_function_boundary_emit_resource_imports(
    loom_low_lower_context_t* context);

// Lowers one target-bound function declaration into a low.func.decl.
//
// The emitted low declaration preserves source symbol identity and callable
// metadata and maps its signature through |options->policy|. Runtime imports
// record the policy import kind and resolved code symbol. Ordinary declarations
// remain unresolved Loom symbols for a subsequent IR link.
iree_status_t loom_low_lower_declaration(
    loom_module_t* module, loom_func_like_t source_declaration,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_
