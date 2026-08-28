// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Index-backed publication of independent kernel compilation requests.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_REQUEST_PRODUCER_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_REQUEST_PRODUCER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/link/materialization_environment.h"
#include "loom/link/module_index.h"
#include "loom/transforms/kernel/kernel_class_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct loom_kernel_request_producer_t loom_kernel_request_producer_t;

// One independently owned source request for a live kernel class.
typedef struct loom_kernel_request_t {
  // Exact index-wide kernel definition ordinal rooted by this request.
  iree_host_size_t source_symbol_ordinal;

  // Dense class ordinal within this kernel publication.
  loom_decision_class_ordinal_t class_ordinal;

  // Number of launch sites assigned to this class.
  iree_host_size_t member_count;

  // Independently owned ordinary Loom source module and kernel root.
  loom_kernel_class_product_t product;
} loom_kernel_request_t;

// Accepts ownership of one request at callback entry.
//
// The callback must release or transfer |request.product| even when returning
// an error. A non-OK status stops publication before another class is
// materialized.
typedef iree_status_t (*loom_kernel_request_publish_fn_t)(
    void* user_data, loom_kernel_request_t request);

// Required sink for one publication operation.
typedef struct loom_kernel_request_sink_t {
  // Callback accepting each request.
  loom_kernel_request_publish_fn_t publish;

  // Opaque value passed to |publish|.
  void* user_data;
} loom_kernel_request_sink_t;

// Allocates a reusable producer over an immutable provider index.
//
// The producer copies |*environment| and borrows |index| plus every resource
// referenced by the copied environment for its lifetime. The caller may
// discard the environment struct after this call, but its context, block pool,
// low-representation codec, callbacks and their user data, and allocator
// backing state must remain valid. Template provider headers are loaded lazily
// and cached across kernel publications. Implementation bodies are
// materialized only for the source kernel passed to publish.
iree_status_t loom_kernel_request_producer_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_kernel_request_producer_t** out_producer);

// Frees |producer| and its persistent provider-header cache.
void loom_kernel_request_producer_free(
    loom_kernel_request_producer_t* producer);

// Publishes every live semantic class of one indexed kernel definition.
//
// |sites| is the complete launch-site set for this kernel across the enclosing
// product. Publication begins only after the bounded class collection closes,
// because accepting a later site may conservatively collapse prior classes.
// Each request is transferred before the next class is materialized.
//
// The returned collection is allocated from |scratch_arena| and remains valid
// until that arena is reset. It maps the input sites to the published class
// ordinals without retaining any product or source-module storage.
iree_status_t loom_kernel_request_producer_publish(
    loom_kernel_request_producer_t* producer,
    iree_host_size_t source_symbol_ordinal,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    loom_kernel_class_collection_t* out_collection);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_REQUEST_PRODUCER_H_
