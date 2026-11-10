// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cleanup rewrite pattern composition.
//
// Cleanup separates fact-independent rewrites from four observing phases.
// Fact-independent patterns may prepare root-local actions from coherent facts
// once, apply nested roots inside-out, then run structural callbacks in the
// driver's non-observing batch.
// Region-initialization patterns run once in region preorder before the first
// worklist is processed.
// Universal pre-fold patterns run before constant folding, universal post-type
// patterns run after type propagation, and source-combine patterns run after
// structural canonicalization only while source representations remain legal.
//
// Region-initialization patterns establish outer-to-inner invariants without
// erasing or replacing walked roots. Providers that must maintain those
// invariants for roots created or changed during rewriting also register the
// same rooted patterns in an ordinary fixed-point phase.

#ifndef LOOM_TRANSFORMS_CLEANUP_PATTERNS_H_
#define LOOM_TRANSFORMS_CLEANUP_PATTERNS_H_

#include "iree/base/api.h"
#include "loom/rewrite/pattern_registry.h"
#include "loom/transforms/cleanup/fact_refinement_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_symbolic_expr_context_t loom_symbolic_expr_context_t;
typedef struct loom_cleanup_special_value_policy_t
    loom_cleanup_special_value_policy_t;

// Invocation-local state available to cleanup rewrite patterns.
typedef struct loom_cleanup_pattern_context_t {
  // Symbolic expression context maintained by the canonicalizer driver.
  loom_symbolic_expr_context_t* symbolic_expression_context;
} loom_cleanup_pattern_context_t;

// Ordered provider lists selected by one compiler composition.
typedef struct loom_cleanup_pattern_provider_set_t {
  // Structural patterns with optional initial read-only preparation.
  loom_rewrite_pattern_provider_list_t fact_independent;
  // Patterns applied once in outer-to-inner order before worklist processing.
  loom_rewrite_pattern_provider_list_t region_initialization;
  // Universal patterns applied before fact-backed constant folding.
  loom_rewrite_pattern_provider_list_t universal_pre_fold;
  // Universal patterns applied after type propagation.
  loom_rewrite_pattern_provider_list_t universal_post_type;
  // Source representation combines disabled after target legalization.
  loom_rewrite_pattern_provider_list_t source_combine;
  // Dialect-backed special-value recognition and builders, or NULL.
  const loom_cleanup_special_value_policy_t* special_value_policy;
  // Dialect-backed SSA fact and carrier refinement builders, or NULL.
  const loom_fact_refinement_policy_t* fact_refinement_policy;
} loom_cleanup_pattern_provider_set_t;

// Indexed pattern registries prepared for one compiler invocation.
typedef struct loom_cleanup_pattern_registry_t {
  // Structural patterns with optional initial read-only preparation.
  const loom_rewrite_pattern_registry_t* fact_independent;
  // Patterns applied once in outer-to-inner order before worklist processing.
  const loom_rewrite_pattern_registry_t* region_initialization;
  // Universal patterns applied before fact-backed constant folding.
  const loom_rewrite_pattern_registry_t* universal_pre_fold;
  // Universal patterns applied after type propagation.
  const loom_rewrite_pattern_registry_t* universal_post_type;
  // Source representation combines disabled after target legalization.
  const loom_rewrite_pattern_registry_t* source_combine;
  // Dialect-backed special-value recognition and builders, or NULL.
  const loom_cleanup_special_value_policy_t* special_value_policy;
  // Dialect-backed SSA fact and carrier refinement builders, or NULL.
  const loom_fact_refinement_policy_t* fact_refinement_policy;
} loom_cleanup_pattern_registry_t;

// Owned storage for the indexed cleanup pattern registries. Provider
// descriptors and callback data remain borrowed.
typedef struct loom_cleanup_pattern_registry_storage_t {
  // Storage for the fact-independent registry.
  loom_rewrite_pattern_registry_storage_t fact_independent_storage;
  // Storage for the region-initialization registry.
  loom_rewrite_pattern_registry_storage_t region_initialization_storage;
  // Storage for the universal pre-fold registry.
  loom_rewrite_pattern_registry_storage_t universal_pre_fold_storage;
  // Storage for the universal post-type registry.
  loom_rewrite_pattern_registry_storage_t universal_post_type_storage;
  // Storage for the source-combine registry.
  loom_rewrite_pattern_registry_storage_t source_combine_storage;
  // Borrowed registry view over the storage fields.
  loom_cleanup_pattern_registry_t registry;
} loom_cleanup_pattern_registry_storage_t;

// Composes each cleanup phase into an indexed registry while preserving the
// provider and pattern order selected by |provider_set|.
iree_status_t loom_cleanup_pattern_registry_storage_initialize(
    const loom_cleanup_pattern_provider_set_t* provider_set,
    iree_allocator_t allocator,
    loom_cleanup_pattern_registry_storage_t* out_storage);

// Releases registry storage. Borrowed provider data is never released.
void loom_cleanup_pattern_registry_storage_deinitialize(
    loom_cleanup_pattern_registry_storage_t* storage);

// Returns the immutable registry view owned by |storage|.
const loom_cleanup_pattern_registry_t*
loom_cleanup_pattern_registry_storage_registry(
    const loom_cleanup_pattern_registry_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_PATTERNS_H_
