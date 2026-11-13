// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transactional type propagation through generated op constraints.
//
// This utility interprets table-driven semantic constraints as a monotonic
// refinement problem. Candidate type narrowings are collected in scratch state,
// expanded across the connected constraint/use closure, and committed through
// the rewriter only if the whole transaction is consistent. The propagator
// consumes verified IR and retained analysis facts; contradictory candidate
// closures leave the module unchanged. Callable inputs, call results, and
// returned values retain their verified boundary types unless a whole-module
// owner explicitly permits refinement and reconciles the signature. Local
// refinement can also proceed after inlining removes that boundary.

#ifndef LOOM_REWRITE_TYPE_PROPAGATION_H_
#define LOOM_REWRITE_TYPE_PROPAGATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_type_propagator_t loom_type_propagator_t;

// Cumulative activity across all transactions run by one propagator.
typedef struct loom_type_propagator_statistics_t {
  // Number of candidate closures rejected as inconsistent.
  uint64_t conflict_count;

  // Number of candidate seeds suppressed by an identical prior rejection.
  uint64_t rejection_cache_hit_count;
} loom_type_propagator_statistics_t;

// Optional ownership query for callable boundary refinement. The callback
// borrows its state for the propagator's lifetime. Without a callback, callable
// boundary types remain fixed.
typedef struct loom_type_propagator_boundary_callback_t {
  // Returns true only when the owner will reconcile the signature and all
  // callers after local refinement. |op| is a callable definition or call.
  bool (*fn)(void* user_data, loom_op_t* op);

  // Borrowed whole-module boundary owner state.
  void* user_data;
} loom_type_propagator_boundary_callback_t;

// Type properties that one propagation instance may refine.
typedef enum loom_type_propagator_scope_e {
  // Refine every type property carried by facts and op constraints.
  LOOM_TYPE_PROPAGATOR_SCOPE_ALL = 0,
  // Refine encoding attachments while preserving dynamic SSA dimensions.
  LOOM_TYPE_PROPAGATOR_SCOPE_ENCODING = 1,
} loom_type_propagator_scope_t;

// Context passed to op-specific semantic type-transfer hooks. Hooks can inspect
// the current transactional candidate type of a value and seed new candidates,
// but they cannot commit mutations themselves.
typedef struct loom_type_transfer_context_t loom_type_transfer_context_t;

// Allocates a type propagator from |arena|. The propagator owns compact scratch
// indexed by region-local value ordinals and may be reused across many op
// transactions in the same pass/region run.
iree_status_t loom_type_propagator_allocate(
    loom_module_t* module,
    loom_type_propagator_boundary_callback_t refine_boundary,
    loom_type_propagator_scope_t scope, iree_arena_allocator_t* arena,
    loom_type_propagator_t** out_propagator);

// Releases the propagator's active function-local value domain. The propagator
// object itself is arena-allocated and has no heap ownership.
void loom_type_propagator_deinitialize(loom_type_propagator_t* propagator);

// Acquires the local value domain for an explicit region tree. |parent_op|
// supplies the logical owner only when the root region is detached; attached
// nested regions retain their owners directly. New values created by rewrites
// are admitted to the same region-local domain on demand.
iree_status_t loom_type_propagator_prepare_region(
    loom_type_propagator_t* propagator, loom_region_t* region,
    loom_op_t* parent_op);

// Acquires a function's body-local value domain. Convenience wrapper around
// loom_type_propagator_prepare_region.
iree_status_t loom_type_propagator_prepare_function(
    loom_type_propagator_t* propagator, loom_func_like_t function);

// Returns the active region-local value domain. The propagator retains
// ownership; sibling analyses may register rewrite-created values but must not
// release the domain.
loom_local_value_domain_t* loom_type_propagator_value_domain(
    loom_type_propagator_t* propagator);

// Starts a new enclosing fixed-point iteration. Exact candidates rejected in
// an earlier iteration are reconsidered because intervening rewrites may have
// changed their constraint closure.
void loom_type_propagator_begin_iteration(loom_type_propagator_t* propagator);

// Returns cumulative propagation activity for reporting and diagnostics.
loom_type_propagator_statistics_t loom_type_propagator_statistics(
    const loom_type_propagator_t* propagator);

// Returns true when applying the type propagator to |op| may commit a type
// change. This is a cheap prefilter for pass hot paths: false means the op has
// no generated type-refining metadata and, when analysis facts are available,
// none of its directly declared values carry dynamic type properties that facts
// can narrow. |vtable| may be NULL when the op kind is unknown.
bool loom_type_propagator_may_apply_op(const loom_type_propagator_t* propagator,
                                       const loom_rewriter_t* rewriter,
                                       const loom_op_t* op,
                                       const loom_op_vtable_t* vtable);

// Applies one transactional propagation seeded at |op|. The transaction uses
// generated constraints on |op| and every affected user/defining op reached by
// candidate type changes. If the closure is consistent, all narrowed value
// types are committed through |rewriter| and |*out_changed| is set. If any
// constraint conflicts, no IR is mutated and |*out_changed| is false.
iree_status_t loom_type_propagator_apply_op(loom_type_propagator_t* propagator,
                                            loom_rewriter_t* rewriter,
                                            loom_op_t* op, bool* out_changed);

// Returns the current transactional type for |value_id|, including any
// candidates already accepted earlier in the active transaction. Invalid value
// ids return the none type.
loom_type_t loom_type_transfer_value_type(
    const loom_type_transfer_context_t* context, loom_value_id_t value_id);

// Rejects the current transaction when candidate types violate an op's semantic
// constraint. No candidates are committed; retained facts remain available to
// lowering without redeclaring the value types.
void loom_type_transfer_reject(loom_type_transfer_context_t* context);

// Seeds a candidate refinement for |value_id|. The transaction owns conflict
// handling and commit; callers must not mutate the IR directly.
iree_status_t loom_type_transfer_seed_candidate(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type, loom_constraint_property_t property);

// Seeds explicit candidate dimensions for |value_id|. The rank and dimensions
// are interpreted as shape-only information and may include dynamic SSA dims.
iree_status_t loom_type_transfer_seed_shape_dims(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    const uint64_t* candidate_dimensions, uint8_t candidate_rank);

// Seeds an encoding/layout attachment for |value_id|.
iree_status_t loom_type_transfer_seed_encoding_attachment(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    uint16_t candidate_encoding_id,
    loom_encoding_flags_t candidate_encoding_flags);

// Seeds only static structural information from |candidate_type| into
// |value_id|: kind, element/role, static dimensions, and static encoding
// attachments. Dynamic dimension and SSA encoding value ids are intentionally
// not transferred, making this helper safe across region/call boundaries that
// remap dynamic type variables.
iree_status_t loom_type_transfer_seed_static_structure_from_type(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_TYPE_PROPAGATION_H_
