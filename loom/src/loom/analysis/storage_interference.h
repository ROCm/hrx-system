// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent interference analysis for function-local storage roots.

#ifndef LOOM_ANALYSIS_STORAGE_INTERFERENCE_H_
#define LOOM_ANALYSIS_STORAGE_INTERFERENCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/storage_access.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_storage_interference_t loom_storage_interference_t;

// Analyzes the storage roots and accesses in |function|.
//
// |fact_table| must contain the populated facts and retained CFG snapshots for
// |function|. |value_domain| must be acquired for the function body and remain
// active for the analysis lifetime. The returned analysis and all retained
// provenance, access, and control facts are allocated from |arena|.
// |access_scope| owns the immutable source snapshot shared with callable effect
// queries. Construction may temporarily release and restore |value_domain|'s
// scratch borrow without changing its numbering or the caller's value facts.
//
// Reference transport and classified uses come from the canonical access graph.
// This analysis adds allocation memberships and lifetime proofs. Functions
// without local allocation roots require no access graph or per-value tables.
// Consumers query its indexed result instead of reconstructing aliases or
// access footprints. Calls expose only roots reachable through their reference
// operands. Channel bindings expose their backing storage until realization
// makes borrowed accesses and their lifetimes explicit. Unmodeled non-call
// memory effects remain conservative for every root.
iree_status_t loom_storage_interference_analyze_function(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_local_value_domain_t* value_domain, loom_func_like_t function,
    loom_storage_access_scope_t* access_scope, iree_arena_allocator_t* arena,
    loom_storage_interference_t** out_analysis);

// Returns true when |root_value_id| may have a memory footprint.
//
// Complete roots with no recorded accesses return false. Missing roots,
// incomplete provenance, and unknown memory effects conservatively return
// true. Storage layout consumers use this to omit provably unused allocations
// without repeating source traversal or demand analysis.
bool loom_storage_interference_root_may_be_accessed(
    const loom_storage_interference_t* analysis, loom_value_id_t root_value_id);

// Returns true when one static storage instance can represent every execution
// of the allocation. Acyclic allocations are single instances. Repeated roots
// require complete provenance, no older live alias at allocation entry, and no
// asynchronous access crossing that boundary. Accessed workgroup roots require
// either a single-invocation launch or a uniform workgroup rendezvous at a
// point separating accesses to different instances across invocations. In CFG
// functions every path from an access through reallocation to another access
// must cross such a rendezvous; different paths may use different barriers.
// Declarations and alias construction do not access storage. Structured
// allocations use a cut in the allocation block.
// The proof is retained during construction; this query neither traverses IR
// nor computes liveness.
bool loom_storage_interference_root_has_single_live_instance(
    const loom_storage_interference_t* analysis, loom_value_id_t root_value_id);

// Proves that two workgroup buffer.alloca roots never require their bytes at
// the same time. The proof accepts workgroup-uniform mutually exclusive
// control or a verified workgroup acq_rel barrier separating all synchronous
// and asynchronous access completions. Missing provenance, unknown memory
// effects, weaker barriers, and cyclic phase reversal fail the proof.
iree_status_t loom_storage_interference_prove_workgroup_nonoverlap(
    loom_storage_interference_t* analysis, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_proven);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_STORAGE_INTERFERENCE_H_
