// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low target contract queries.
//
// Source lowering and target legalization both need to ask whether a source
// operation has a target contract without emitting target-low IR. This
// component adapts the lowering policy and function analyses to the common
// target contract query interface. Full lowering borrows its active context;
// target legalization creates a read-only scope with the same adapter.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_source_query_scope_t
    loom_low_lower_source_query_scope_t;
typedef struct loom_low_source_representation_plan_t
    loom_low_source_representation_plan_t;
typedef struct loom_source_program_t loom_source_program_t;

// Populates a target contract query environment from |context|.
//
// The returned environment borrows the active source function, value domain,
// source program, dataflow result, view-region analysis, and target state
// allocator from |context|. These retained inputs form one coherent analysis
// scope and must not be replaced independently.
iree_status_t loom_low_lower_source_query_environment_initialize(
    loom_low_lower_context_t* context,
    loom_target_contract_query_environment_t* out_environment);

// Returns a target contract query callback backed by |context|.
loom_target_contract_query_callback_t loom_low_lower_source_query_callback(
    loom_low_lower_context_t* context);

// Creates a read-only target contract query scope for |source_function|.
//
// The scope uses the same generated rules, policy callbacks, descriptor
// resolution, and function analyses as source-to-Low lowering. It borrows
// |module| and |options| and allocates its object storage from |arena|. Callers
// must deinitialize the scope before releasing those inputs or resetting
// |arena|.
iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope);

// Releases analyses owned by |scope|. The scope object remains arena-owned.
void loom_low_lower_source_query_scope_deinitialize(
    loom_low_lower_source_query_scope_t* scope);

// Returns a target contract query callback backed by |scope|.
loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope);

// Returns the function-local value domain owned by |scope|, or NULL when the
// source function has no body.
const loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the immutable source-program index owned by |scope|, or NULL when
// the source function has no body.
const loom_source_program_t* loom_low_lower_source_query_scope_program(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the descriptor set shared by every analysis in |scope|.
const loom_low_descriptor_set_t*
loom_low_lower_source_query_scope_descriptor_set(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the retained physical source-value dataflow result, or NULL when the
// active policy has no provider.
const loom_source_dataflow_result_t* loom_low_lower_source_query_scope_dataflow(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the retained physical source-representation plan, or NULL when the
// active policy has no provider.
const loom_low_source_representation_plan_t*
loom_low_lower_source_query_scope_representation_plan(
    const loom_low_lower_source_query_scope_t* scope);

// Returns the lazily analyzed source view regions owned by |scope|.
iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_
