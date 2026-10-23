// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-callable projection from retained source object ABIs to physical Low
// signatures.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_SIGNATURE_PROJECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_SIGNATURE_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_signature_projection_plan_t
    loom_low_source_signature_projection_plan_t;

// Sparse module-symbol index shared by declaration, definition, and call
// planning in one source-to-Low transaction. Empty indexes allocate nothing.
typedef struct loom_low_source_signature_projection_index_t {
  // Borrowed plan vector in module planning order.
  const loom_low_source_signature_projection_plan_t* const* plans;
  // One-based plan ordinal by module symbol ID; zero means no projection.
  const uint16_t* ordinals;
  // Number of symbol ordinals available.
  iree_host_size_t symbol_count;
} loom_low_source_signature_projection_index_t;

// Returns true when the callable carries a verified C source signature
// contract.
bool loom_low_source_signature_projection_required(loom_func_like_t function);

// Builds an O(1) module-local symbol index over retained projection plans.
iree_status_t loom_low_source_signature_projection_index_build(
    const loom_module_t* module,
    const loom_low_source_signature_projection_plan_t* const* plans,
    iree_host_size_t plan_count, iree_arena_allocator_t* arena,
    loom_low_source_signature_projection_index_t* out_index);

// Returns the plan for |symbol_id|, or NULL when the symbol has no projected
// source signature. A NULL index represents standalone lowering.
const loom_low_source_signature_projection_plan_t*
loom_low_source_signature_projection_index_find(
    const loom_low_source_signature_projection_index_t* index,
    loom_symbol_id_t symbol_id);

// Plans one verified C source signature without publishing IR. Logical carriers
// are mapped once, the target classifies the whole object signature, and the
// final physical ABI layout is canonicalized into module storage. Diagnostics
// leave |out_plan| NULL; status is reserved for infrastructure failures.
iree_status_t loom_low_plan_source_signature_projection(
    loom_module_t* module, loom_func_like_t function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_result_t* out_result,
    const loom_low_source_signature_projection_plan_t** out_plan);

const loom_type_t* loom_low_source_signature_projection_logical_argument_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count);

const loom_type_t* loom_low_source_signature_projection_logical_result_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count);

const loom_type_t* loom_low_source_signature_projection_physical_argument_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count);

const loom_type_t* loom_low_source_signature_projection_physical_result_types(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count);

const loom_tied_result_t* loom_low_source_signature_projection_tied_results(
    const loom_low_source_signature_projection_plan_t* plan,
    uint16_t* out_count);

loom_named_attr_slice_t loom_low_source_signature_projection_abi_layout(
    const loom_low_source_signature_projection_plan_t* plan);

// Executes one retained target pack/unpack phase. |state_arena| owns any state
// produced at definition entry or call argument emission; the paired exit or
// result phase receives it through |inout_state|.
iree_status_t loom_low_source_signature_projection_emit(
    const loom_low_source_signature_projection_plan_t* plan,
    loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const loom_value_id_t* input_values,
    uint16_t input_count, loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_SIGNATURE_PROJECTION_H_
