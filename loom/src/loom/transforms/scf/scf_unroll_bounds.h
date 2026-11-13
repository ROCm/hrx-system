// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-domain preservation and representable main/tail splits for SCF unroll.

#ifndef LOOM_TRANSFORMS_SCF_SCF_UNROLL_BOUNDS_H_
#define LOOM_TRANSFORMS_SCF_SCF_UNROLL_BOUNDS_H_

#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds an upper-bound-guard envelope around the supplied lower endpoint and
// divisor, which the caller has already established through arithmetic or
// explicit relations. The guard alone retains only upper - 1, not the source
// loop's last reachable induction value.
loom_value_facts_t loom_scf_unroll_guarded_index_facts(
    loom_value_facts_t source_facts, loom_value_facts_t upper_facts);

// Preserves source integer refinements beyond |inferred_facts| on |indices|,
// replacing their identities in place. Every index must already be proven to
// denote an executed source iteration. Optional |lower_bound| and |upper_bound|
// add inclusive-lower and exclusive-upper relations guaranteed by the caller's
// construction; INVALID omits a relation. No analysis of fresh values is run.
// Equal-typed indices share an assumption, split at the IR's segment limits.
// Singleton emission uses stack scratch; groups borrow |scratch_arena|.
iree_status_t loom_scf_unroll_refine_indices(
    loom_rewriter_t* rewriter, loom_value_facts_t source_facts,
    loom_value_facts_t inferred_facts, loom_value_id_t lower_bound,
    loom_value_id_t upper_bound, uint32_t index_count, loom_value_id_t* indices,
    loom_location_id_t location, iree_arena_allocator_t* scratch_arena);

// Materialized split coordinate and the facts retained by its arithmetic.
typedef struct loom_scf_unroll_split_t {
  // Shared main-loop upper bound and tail-loop lower bound.
  loom_value_id_t value;
  // Conservative facts retained by the split expression's own arithmetic.
  loom_value_facts_t facts;
} loom_scf_unroll_split_t;

// Builds the shared main-loop upper bound and tail-loop lower bound for a
// verified positive-step loop. The caller has proved step * factor fits the
// selected address carrier and materialized it as |scaled_step|. The main loop
// visits complete tiles and the tail visits the remaining source iterations.
// When rounding up past the source upper bound could overflow, the split stays
// between min(lower, upper) and upper. Original facts remain valid while this
// appends arithmetic at |builder|'s insertion point.
iree_status_t loom_scf_unroll_build_dynamic_split(
    loom_builder_t* builder, const loom_value_fact_table_t* facts,
    const loom_op_t* op, int64_t step, uint32_t factor,
    loom_value_id_t scaled_step, loom_scf_unroll_split_t* out_split);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_SCF_UNROLL_BOUNDS_H_
