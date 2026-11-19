// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_SOURCE_STABILITY_H_
#define LOOM_ANALYSIS_PIPELINE_SOURCE_STABILITY_H_

#include "loom/analysis/pipeline/invocations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Determines whether each repeated transport can read its source ahead of the
// worker. The caller supplies a closed inventory of worker occurrences. NULL
// invocations denotes a single construction with only concurrent strands.
// Alias partitions, hidden effects, and the lifetimes of enclosing workers
// all participate. The scope owns the temporary memory and access summaries.
iree_status_t loom_pipeline_prove_source_stability(
    loom_storage_access_scope_t* storage_access,
    const loom_pipeline_invocations_t* invocations,
    loom_pipeline_worker_t* workers, iree_host_size_t worker_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_SOURCE_STABILITY_H_
