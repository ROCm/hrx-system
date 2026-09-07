// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type refinement driven by value facts.
//
// This layer is intentionally outside loom/ir because it consumes the reusable
// value fact table. The IR helper owns the monotonic type-meet rules; this
// analysis helper only projects proven facts into candidate dimensions or
// encoding attachments.

#ifndef LOOM_ANALYSIS_TYPE_REFINEMENT_H_
#define LOOM_ANALYSIS_TYPE_REFINEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/type_refinement.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;

// Refines |current_type| using exact facts referenced by the type itself.
//
// Dynamic dimensions narrow when their SSA value has exact non-negative integer
// facts. SSA encoding attachments narrow when their encoding-summary fact names
// an exact static encoding spec. Contradictory exact facts, such as a negative
// dimension extent, report LOOM_TYPE_REFINEMENT_CONFLICT and leave |out_type|
// equal to |current_type|.
iree_status_t loom_type_refine_with_value_facts(
    loom_type_t current_type, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

// Specializes |current_type| using exact facts and canonical module encodings.
//
// This includes ordinary dimension/static-spec refinement and additionally
// interns a lossless static representation for exact composed encodings such
// as dynamic strided physical storage. Callers must own a module rewrite or
// transformation boundary because the module encoding table may grow.
iree_status_t loom_type_specialize_with_value_facts(
    loom_module_t* module, loom_type_t current_type,
    const loom_value_fact_table_t* fact_table, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_type_refinement_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_TYPE_REFINEMENT_H_
