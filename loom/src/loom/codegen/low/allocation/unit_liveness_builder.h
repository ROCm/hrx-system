// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Construction and refinement of per-allocation-unit storage lifetimes.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_BUILDER_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_BUILDER_H_

#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes |out_unit_liveness| from value-granular liveness and IR use
// structure over the canonical |cfg_graph|. The resulting points refine
// register intervals down to target allocation units across CFG boundaries,
// low.slice uses, descriptor early-clobber hazards, and structured backedges.
// Native result units occupy their definition point even without readers.
// Structural transports write only units with nonempty destination lifetimes.
// The complete register-unit extent is bounded before point allocation;
// retained-read construction consumes subsets of this same bounded domain.
// Published point arrays are owned by |result_arena|; query metadata and
// physical access indexes are owned by |decision_arena| through final physical
// numbering. The arenas must be distinct. Construction scratch borrows the
// result arena's tail and is released before returning.
iree_status_t loom_low_allocation_unit_liveness_initialize(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    loom_low_call_contract_provider_t call_contracts,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena,
    loom_low_allocation_unit_liveness_t* out_unit_liveness);

// Completes physical lifetime facts for mandatory tied-storage components.
// Each ancestor acquires storage by its earliest mandatory descendant start.
// Component origins retain every member's physical unit lifetime and sparse
// segments so destructive-reuse refinement can query exact old-content
// observations before deciding which optional relations remain aliasable.
// Published segments use |result_arena|; query ranges use |decision_arena|.
iree_status_t loom_low_allocation_unit_liveness_retain_tied_storage(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena);

// Propagates storage starts across the final structural placement relations.
// Sources flow into tied results, and contiguous aggregate parts carry source
// starts into accepted result reservations. Call after optional alias
// permissions have been refined.
void loom_low_allocation_unit_liveness_propagate_storage_relations(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_UNIT_LIVENESS_BUILDER_H_
