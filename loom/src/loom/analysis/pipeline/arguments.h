// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_ARGUMENTS_H_
#define LOOM_ANALYSIS_PIPELINE_ARGUMENTS_H_

#include "loom/analysis/pipeline/resources.h"

#ifdef __cplusplus
extern "C" {
#endif

// Actual channel correspondence at a construction or worker boundary. Storage
// and protocol identities remain owned by the construction that created them.
typedef struct loom_pipeline_channel_binding_t {
  // Value in the receiving source scope.
  loom_value_id_t value_id;
  // Construction occurrence owning the channel and its allocation placements.
  const loom_pipeline_resources_t* resources;
  // Binding in resources, independent of its storage address or source symbol.
  const loom_pipeline_resource_channel_t* channel;
} loom_pipeline_channel_binding_t;

typedef struct loom_pipeline_arguments_t {
  // Closed literal producers indexed by formal argument, or NULL when all
  // captures use value facts or runtime bindings. Producers remain immutable.
  const loom_op_t* const* literal_captures;
  // Canonical protocols selected by the actual operands, in formal ID order.
  const loom_pipeline_channel_binding_t* channels;
  // Number of channel formals.
  iree_host_size_t channel_count;
  // Canonical backing stores selected by pool operands, in formal ID order.
  const loom_pipeline_resource_pool_binding_t* pools;
  // Number of resolved pool formals.
  iree_host_size_t pool_count;
} loom_pipeline_arguments_t;

// Binds a verified ordinary worker call or child composition. Captured facts
// are immutable seeds for the callee's own inference. Channel and pool formals
// retain the actual construction identities; equal backing addresses never
// merge protocols. Incoming channels are sorted by value_id and can originate
// several construction scopes above this call.
//
// The caller initializes callee_facts with the callee's execution environment
// and computes its body after this binding. Source facts and resources stay
// immutable through consumption. All new storage belongs to arena. An absent
// channel binding is an unsupported authored boundary, reported through the
// diagnostic emitter with out_valid false. Statuses carry allocation or sink
// failure. An unbound pool is rejected by resource construction only if the
// child uses it to allocate. No source mutation or callee-body traversal occurs
// here.
iree_status_t loom_pipeline_arguments_bind(
    loom_module_t* module, loom_op_t* call, loom_func_like_t callee,
    const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* caller_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count, loom_value_fact_table_t* callee_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_arguments_t* out_arguments, bool* out_valid);

// Returns the explicit correspondence for a formal, or NULL when absent.
const loom_pipeline_channel_binding_t* loom_pipeline_arguments_lookup_channel(
    const loom_pipeline_channel_binding_t* bindings,
    iree_host_size_t binding_count, loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_ARGUMENTS_H_
