// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end materialization from a provider-backed module index.

#ifndef LOOM_LINK_INDEX_MATERIALIZER_H_
#define LOOM_LINK_INDEX_MATERIALIZER_H_

#include "iree/base/api.h"
#include "loom/link/plan_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned output of one index materialization.
typedef struct loom_link_index_materialization_t {
  // Stable plan including template providers needed by later specialization.
  loom_link_plan_t* plan;
  // Standalone linked product and its exact index projections.
  loom_link_plan_materialization_t product;
  // Storage retaining every projection in |product|.
  iree_arena_allocator_t arena;
} loom_link_index_materialization_t;

// Releases every owned object in |materialization|.
void loom_link_index_materialization_deinitialize(
    loom_link_index_materialization_t* materialization);

// Plans and materializes one provider-backed index for later specialization.
// Merge plans materialize directly. Link plans repeatedly materialize
// ordinary reachability, evaluate headers for providers in reachable template
// families, and retain providers whose choice still depends on caller facts.
// Proven choices retain only the selected provider. Nested applications enter
// the next ordinary closure; providers are retained once by index ordinal.
// Template applicability remains open to later specialization regardless of
// unresolved-symbol policy. That policy governs missing symbol definitions,
// not unknown provider predicates. Root retention and outward linkage come
// exclusively from |plan_options|. Rejection emits diagnostics and returns OK
// with an empty materialization. Status reports only infrastructure failure.
iree_status_t loom_link_index_materialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_index_materialization_t* out_materialization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_INDEX_MATERIALIZER_H_
