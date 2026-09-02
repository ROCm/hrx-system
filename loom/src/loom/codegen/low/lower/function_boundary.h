// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low callable boundary lowering.
//
// A source function boundary is mapped once and then consumed throughout
// lowering. Validation establishes the argument/result map before source-plan
// construction. Definition creation materializes the target-Low callable,
// entry binding connects direct arguments, resource emission materializes
// arguments omitted from the direct ABI, and predicate remapping translates
// source value references after those bindings exist.
//
// Import declarations use the same type and metadata mapping without a body.
// They are lowered independently by the module source-to-Low pass before
// function definitions.

#ifndef LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_
#define LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates and maps the source callable signature into function-local state.
// This must run before source-plan construction and low callable creation.
iree_status_t loom_low_lower_function_boundary_validate(
    loom_low_lower_context_t* context, loom_region_t* source_body);

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

// Lowers one target-bound external function declaration into a low.func.decl.
//
// The emitted low declaration preserves source symbol identity and callable
// metadata, maps its signature through |options->policy|, and records the
// policy import kind plus the resolved import symbol as the low code symbol.
iree_status_t loom_low_lower_import_declaration(
    loom_module_t* module, loom_func_like_t source_declaration,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_FUNCTION_BOUNDARY_H_
