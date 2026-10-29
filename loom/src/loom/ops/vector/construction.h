// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_VECTOR_CONSTRUCTION_H_
#define LOOM_OPS_VECTOR_CONSTRUCTION_H_

#include "loom/ops/vector/ops.h"
#include "loom/rewrite/pattern_registry.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true for a literal integer or floating-point vector splat. Operand
// vector.splat is deliberately excluded: it has an SSA source, not an
// immediate payload.
static inline bool loom_vector_literal_splat_isa(const loom_op_t* op) {
  return loom_vector_splati_isa(op) || loom_vector_splatf_isa(op);
}

// Plans in-bounds constant coordinate conversion for extract, insert, or slice.
// Only exact coordinates are retained; dynamic operand IDs are read when
// applied.
iree_status_t loom_vector_prepare_coordinates(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, iree_arena_allocator_t* arena,
    loom_rewrite_action_t** out_action);

// Resolves newly exposed coordinates during ordinary observing rewrites. Shares
// the same proof and application code as batch preparation; no-match allocates
// nothing. The caller provides coherent facts.
iree_status_t loom_vector_normalize_coordinates(loom_op_t* op,
                                                loom_rewriter_t* rewriter,
                                                bool* out_changed);

// Folds extract/insert/slice through their direct structural producers without
// querying facts. Suitable both for observing rewrites and unpublished batches.
iree_status_t loom_vector_project_structure(loom_op_t* op,
                                            loom_rewriter_t* rewriter,
                                            bool* out_changed);

// Projects a static rank-one slice or scalar extraction into one retained
// regular vector component. The indexed partition preserves the component's
// SSA value instead of rematerializing its producer. Spanning or dynamic
// projections and sources without a regular partition remain unchanged.
iree_status_t loom_vector_project_regular_partition(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed);

// Folds a pure single-vector-result operation to vector.from_elements when
// retained small-static-lane facts provide an exact constant for every lane.
// The caller selects operations whose constant construction is canonical;
// unknown lanes and NaN facts without retained payloads remain unchanged.
iree_status_t loom_vector_fold_constant_lanes(loom_op_t* op,
                                              loom_rewriter_t* rewriter,
                                              bool* out_changed);

// Folds a vector.from_elements of every unmodified source lane, in order, to
// that source. This identity introduces no operation and is valid at any phase.
// Nonmatching constructions remain unchanged and allocate no scratch storage.
iree_status_t loom_vector_from_elements_fold_source(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed);

// Combines identical scalar rounding/conversion chains on ordered source lanes
// into elementwise vector operations. Only run before target legalization:
// scalarization deliberately decomposes these operations into the same chains.
// The bounded matcher borrows verified IR and allocates only on a match.
iree_status_t loom_vector_from_elements_combine_lanes(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed);

// Combines a complete rank-one chain of static scalar vector.insert operations
// into vector.from_elements. The outermost write to each lane wins; lanes not
// written by the chain can be retained from vector.splat or
// vector.from_elements bases. Only run before target legalization, which may
// deliberately linearize vector.from_elements back into scalar insertions.
iree_status_t loom_vector_insert_combine_elements(loom_op_t* op,
                                                  loom_rewriter_t* rewriter,
                                                  bool* out_changed);

// Forwards the sole result to an existing value and erases the construction.
iree_status_t loom_vector_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement);

// Builds a splat, transfers the result name, and replaces the construction.
iree_status_t loom_vector_replace_single_result_with_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t scalar,
    loom_type_t result_type);

// Transfers the result name only onto newly built values, then replaces the
// old construction with the sole result of the replacement operation.
iree_status_t loom_vector_replace_single_result_with_new_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* new_op,
    loom_value_id_t value_checkpoint);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_CONSTRUCTION_H_
