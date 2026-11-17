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

// Populates a target contract query environment from |context|.
//
// The returned environment borrows the active source function, value domain,
// view-region analysis, and target state allocator from |context|. Queries made
// through the source callback always retain state in the context's arena and
// state allocator, including when callers supply a different query environment.
iree_status_t loom_low_lower_source_query_environment_initialize(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_target_contract_query_environment_t* out_environment);

// Returns a target contract query callback backed by |context|.
loom_target_contract_query_callback_t loom_low_lower_source_query_callback(
    loom_low_lower_context_t* context);

// Retains a deferred capability rejection for immutable source. The rejection
// borrows the query environment's arena; a sparse index in the context's
// analysis arena preserves its identity until materialization demand is known.
// A NULL rejection retains the unavailable operation without diagnostic detail.
iree_status_t loom_low_lower_source_query_retain_rejection(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_rejection_t* rejection);

// Returns whether the source operation has a deferred capability rejection and
// its original diagnostic detail, which may be NULL. An unavailable operation
// cannot select an executable plan even when a target callback could produce
// one; only elimination after demand analysis can discharge the rejection.
bool loom_low_lower_source_query_deferred_rejection(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_rejection_t** out_rejection);

// Creates a read-only target contract query scope for |source_function|.
//
// The scope uses the same generated rules, policy callbacks, descriptor
// binding, and function analyses as source-to-Low lowering. It borrows
// |module|, |options|, and |binding| and allocates its object storage from
// |arena|. Callers must deinitialize the scope before releasing those inputs or
// resetting |arena|. Reference/access discovery may reach source callees. All
// queries in the scope finish before mutating represented source; a rewrite
// retires the scope's analyses before another query can observe the changed
// program.
iree_status_t loom_low_lower_source_query_scope_allocate(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options,
    loom_target_low_source_binding_t binding, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope);

// Releases analyses owned by |scope|. The scope object remains arena-owned.
void loom_low_lower_source_query_scope_deinitialize(
    loom_low_lower_source_query_scope_t* scope);

// Retires all analyses and target state after a source mutation, reusing the
// object storage for an empty query epoch. The source function and target
// configuration stay fixed; the new epoch borrows the current options facts.
// Previously returned analysis pointers and environments become invalid.
void loom_low_lower_source_query_scope_reset(
    loom_low_lower_source_query_scope_t* scope);

// Returns a target contract query callback backed by |scope|.
loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope);

// Populates an environment borrowing the scope's source analyses and retained
// target state. The environment remains valid until |scope| is reset or
// deinitialized.
iree_status_t loom_low_lower_source_query_scope_environment_initialize(
    loom_low_lower_source_query_scope_t* scope,
    loom_target_contract_query_environment_t* out_environment);

// Returns the function-local value domain after function analyses have been
// acquired, or NULL while the scope has only answered point queries.
loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    loom_low_lower_source_query_scope_t* scope);

// Returns the lazily analyzed source view regions owned by |scope|.
iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_
