// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared component planning for shape-preserving vector packet legalization.

#ifndef LOOM_TRANSFORMS_VECTOR_COMPONENT_PACKET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_COMPONENT_PACKET_LEGALIZATION_H_

#include "iree/base/api.h"
#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vector_component_packet_rewrite_callback_t {
  // Callback state borrowed for each completed source rewrite.
  void* user_data;
  // Optional observer for cold reporting and accounting.
  iree_status_t (*fn)(void* user_data, const loom_op_t* source_op,
                      uint64_t created_op_count, uint64_t erased_op_count);
} loom_vector_component_packet_rewrite_callback_t;

typedef struct loom_vector_component_plan_t loom_vector_component_plan_t;

// Borrowed authored contract results retained by one completed component plan.
// The cache remains valid until the legalization query scope or source IR is
// changed. An empty cache has no retained plan.
typedef struct loom_vector_component_packet_query_cache_t {
  // Arena-owned component plan retaining the indexed authored query results.
  const loom_vector_component_plan_t* plan;
} loom_vector_component_packet_query_cache_t;

// Returns an empty component packet query cache.
static inline loom_vector_component_packet_query_cache_t
loom_vector_component_packet_query_cache_empty(void) {
  return (loom_vector_component_packet_query_cache_t){0};
}

// Looks up the authored target contract result retained for |op|. Returns NULL
// when |op| was outside the decomposable component family covered by the plan.
const loom_target_contract_query_result_t*
loom_vector_component_packet_query_cache_lookup(
    const loom_vector_component_packet_query_cache_t* cache,
    const loom_op_t* op);

// Plans and rewrites every profitable decomposable vector component in
// |region|. The target contributes candidate lane counts through
// |context->vector_packet_policy|; the target contract remains the sole source
// of native legality. Authored-legal producers remain target snapshots and
// narrower consumers slice their results without replaying native work. A
// contract that consumes whole-vector lane provenance retains its producer
// closure so each projected packet establishes a complete local origin.
// Components requiring legalization are emitted packet-major so direct SSA
// edges remain packet-local across mixed operations and fanout.
//
// |rewrite_callback| observes each authored operation only after its component
// rewrite has completed. Missing callback functions disable observation.
//
// |out_query_cache| retains every authored target contract result established
// by the plan so later legalization consumes the same classification. The
// caller clears the cache before changing the IR or query scope.
//
// Returns the number of authored operations replaced through
// |out_rewritten_op_count|. A zero count means that no component admitted a
// native candidate under the active legalization policy.
iree_status_t loom_vector_component_packet_legalize(
    loom_target_legalization_context_t* context, loom_region_t* region,
    loom_vector_component_packet_rewrite_callback_t rewrite_callback,
    loom_vector_component_packet_query_cache_t* out_query_cache,
    uint32_t* out_rewritten_op_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_COMPONENT_PACKET_LEGALIZATION_H_
