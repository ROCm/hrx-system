// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end selective materialization from a provider-backed module index.

#ifndef LOOM_LINK_INDEX_MATERIALIZER_H_
#define LOOM_LINK_INDEX_MATERIALIZER_H_

#include "iree/base/api.h"
#include "loom/link/plan_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owned output of one index materialization.
typedef struct loom_link_index_materialization_t {
  // Stable final plan including every selected template provider.
  loom_link_plan_t* plan;
  // Standalone linked output module.
  loom_module_t* module;
} loom_link_index_materialization_t;

// Releases every owned object in |materialization|.
void loom_link_index_materialization_deinitialize(
    loom_link_index_materialization_t* materialization);

// Plans, specializes, and materializes one provider-backed index.
//
// Archive plans materialize directly. Selective plans repeatedly materialize
// ordinary reachability, evaluate headers for providers in reachable template
// families, and add each exact selected provider as an ordinary root. Nested
// applications therefore enter the next ordinary closure, while duplicate
// transitive providers are selected once by index ordinal. When unresolved
// symbols are allowed, the stable partial module preserves unresolved imports
// and template applications for a later link.
iree_status_t loom_link_index_materialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_options_t* plan_options,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name, iree_string_view_list_t output_roots,
    loom_link_index_materialization_t* out_materialization);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_INDEX_MATERIALIZER_H_
