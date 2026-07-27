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
// the rewriter only if the whole transaction is consistent. Malformed or
// contradictory IR leaves the module unchanged; the verifier remains
// responsible for structured user diagnostics.

#ifndef LOOM_REWRITE_TYPE_PROPAGATION_H_
#define LOOM_REWRITE_TYPE_PROPAGATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_type_propagator_t loom_type_propagator_t;

// Context passed to op-specific semantic type-transfer hooks. Hooks can inspect
// the current transactional candidate type of a value and seed new candidates,
// but they cannot commit mutations themselves.
typedef struct loom_type_transfer_context_t loom_type_transfer_context_t;

// Allocates a type propagator from |arena|. The propagator owns compact scratch
// indexed by region-local value ordinals and may be reused across many op
// transactions in the same pass/region run.
iree_status_t loom_type_propagator_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_type_propagator_t** out_propagator);

// Releases the propagator's active function-local value domain. The propagator
// object itself is arena-allocated and has no heap ownership.
void loom_type_propagator_deinitialize(loom_type_propagator_t* propagator);

// Refreshes region-local block-argument ownership metadata. Call once before
// processing an explicit region tree. |parent_op| owns the root entry block
// arguments when provided. New values created by rewrites are admitted to the
// same region-local domain on demand; new region-owner ops are recorded when
// their constraints are processed.
iree_status_t loom_type_propagator_prepare_region(
    loom_type_propagator_t* propagator, loom_region_t* region,
    loom_op_t* parent_op);

// Refreshes function-body block-argument ownership metadata. Convenience
// wrapper around loom_type_propagator_prepare_region.
iree_status_t loom_type_propagator_prepare_function(
    loom_type_propagator_t* propagator, loom_func_like_t function);

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
