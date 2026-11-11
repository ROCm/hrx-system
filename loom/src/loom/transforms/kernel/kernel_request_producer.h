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
typedef struct loom_kernel_request_source_t loom_kernel_request_source_t;
typedef struct loom_kernel_request_preparation_t
    loom_kernel_request_preparation_t;

// One borrowed live-class request prepared for optional materialization.
//
// |source| is the callback-live source/configuration scope used for early
// product resolution. |class_ordinal| identifies the exact accepted decision
// trace within that scope. The record and everything it references remain valid
// only for the duration of the publication callback.
typedef struct loom_kernel_request_t {
  // Exact index-wide kernel definition ordinal rooted by this request.
  iree_host_size_t source_symbol_ordinal;

  // Dense class ordinal within this kernel publication.
  loom_decision_class_ordinal_t class_ordinal;

  // Number of launch sites assigned to this class.
  iree_host_size_t member_count;

  // Callback-live source/configuration scope owned by the publication.
  const loom_kernel_request_source_t* source;
} loom_kernel_request_t;

// Visits one borrowed live-class request.
//
// A non-OK status stops publication before another class is visited. The
// callback may resolve the process-local class without materializing it, or
// call loom_kernel_request_materialize when an independently owned source
// product is required.
typedef iree_status_t (*loom_kernel_request_publish_fn_t)(
    void* user_data, const loom_kernel_request_t* request);

// Required sink for one publication operation.
typedef struct loom_kernel_request_sink_t {
  // Callback accepting each request.
  loom_kernel_request_publish_fn_t publish;

  // Opaque value passed to |publish|.
  void* user_data;
} loom_kernel_request_sink_t;

// Invocation-local provider of complete source-class publications. The callback
// owns preparation lifetime through every sink callback. Requests borrow that
// preparation; explicitly materialized products own independent storage. The
// returned site map uses caller scratch and retains no preparation state. This
// lets an embedding retain immutable prepared sources while custom
// materialization environments keep their ordinary one-shot lifetime.
typedef struct loom_kernel_request_provider_t {
  // Publishes all classes after collecting the complete source-kernel site set.
  iree_status_t (*publish)(
      void* user_data,
      const loom_link_plan_materialization_environment_t* environment,
      iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
      const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
      const loom_kernel_class_collection_options_t* collection_options,
      loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
      const loom_decision_class_ordinal_t** out_site_classes);

  // Borrowed state valid for the enclosing command-product operation.
  void* user_data;
} loom_kernel_request_provider_t;

// Prepares one immutable source independently of invocation scratch. Owns the
// linked module, classifier analyses, and candidate-header backing. Borrows the
// fixed index, context, and block pool until freed; the environment's callbacks
// and user data are used only during this call and are never retained.
iree_status_t loom_kernel_request_preparation_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal,
    loom_kernel_request_preparation_t** out_preparation);

// Releases every object owned by |preparation| before its backing pool retires.
void loom_kernel_request_preparation_free(
    loom_kernel_request_preparation_t* preparation);

// Returns names from the finalized configuration dependency closure, including
// defaults and unresolved declarations. Names belong to |preparation|. The
// producer derives them from the final link plan and its symbol correspondence.
iree_string_view_list_t loom_kernel_request_preparation_config_keys(
    const loom_kernel_request_preparation_t* preparation);

// Collects invocation-local classes and publishes them from an immutable
// source. Concurrent calls use distinct scratch arenas and independently owned
// sinks. The returned site map uses |scratch_arena| and retains no preparation
// state or site literals; failure leaves it NULL.
iree_status_t loom_kernel_request_preparation_publish(
    const loom_kernel_request_preparation_t* preparation,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes);

// Allocates an invocation-local producer over an immutable provider index.
//
// The producer borrows |index|, the environment context, and its block pool for
// its lifetime and copies the allocator. It lazily caches only template
// provider headers shared by source kernels in the enclosing product build. It
// does not retain diagnostics, specialization callbacks, or user data.
//
// Each source kernel is prepared once by the command planner, which groups all
// of its launch sites before publication. Source modules and classifiers remain
// invocation scratch and are released before the next kernel is prepared.
iree_status_t loom_kernel_request_producer_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_kernel_request_producer_t** out_producer);

// Frees |producer| and its provider-header cache.
void loom_kernel_request_producer_free(
    loom_kernel_request_producer_t* producer);

// Materializes one independently owned source product on a cache miss.
//
// |request| must be live in its publication callback. The returned product
// retains no producer, collection, or request storage and must be released with
// loom_kernel_class_product_deinitialize.
iree_status_t loom_kernel_request_materialize(
    const loom_kernel_request_t* request, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_kernel_class_product_t* out_product);

// Publishes every live semantic class of one indexed kernel definition.
//
// |sites| is the complete launch-site set for this kernel across the enclosing
// product. Publication begins only after the bounded class collection closes,
// because accepting a later site may conservatively collapse prior classes.
// |environment| supplies invocation-local diagnostics and specialization.
//
// The returned site map is allocated from |scratch_arena| and remains valid
// until that arena is reset. It maps every input site to a published class
// ordinal without retaining products, source state, or site literals. Failure
// leaves the map NULL. Staged literal tuples borrow |site_module| only through
// publication; ordinary kernels without staged arguments do not use it.
iree_status_t loom_kernel_request_producer_publish(
    loom_kernel_request_producer_t* producer,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_REQUEST_PRODUCER_H_
