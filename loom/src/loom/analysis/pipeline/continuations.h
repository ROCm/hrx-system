// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_CONTINUATIONS_H_
#define LOOM_ANALYSIS_PIPELINE_CONTINUATIONS_H_

#include "loom/analysis/pipeline/progress.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical observation of a canonical channel's record arrival. The admitting
// execution owner supplies one unambiguous read site for each eligible
// endpoint.
typedef struct loom_pipeline_progress_site_t {
  // Existing instruction worker, in canonical invocation order.
  uint32_t worker;
  // Read action in that worker's retained channel plan.
  uint32_t action;
} loom_pipeline_progress_site_t;

// One existing worker can observe every phase of a borrowed input lifetime.
// Admission precedes its first dependent read. The final source phase's last
// arrival proves source reclamation, independently of copied local lifetimes.
typedef struct loom_pipeline_continuation_t {
  // Selected existing worker, or UINT32_MAX when this proof is unavailable.
  uint32_t worker;
  // Read action per source phase, in the selected worker's channel plan.
  const uint32_t* phase_actions;
  // Source phase supplying this worker's first demand within each lifetime.
  uint32_t first_phase;
  // Consumer loop matching one source lifetime, or LOOP_NEST_NONE for a
  // single lifetime spanning the complete worker invocation.
  uint16_t scope;
} loom_pipeline_continuation_t;

// Chooses a common continuation owner for each requested transport input.
// inputs contains source worker ordinals; every source has a retained input.
// observations is the common progress owner's complete population for each
// requested phase destination. sites indexes those canonical observations.
// The callback ranks source-worker/observer-worker pairs, not channel pairs.
//
// Selection consumes canonical channel identity, exact counts, retained loop
// structure, and channel action order. Equal cardinalities never identify a
// phase. It neither walks source IR nor expands runtime record visits. Retained
// block order is indexed once per participating worker and shared by requests.
// Physical transport admission and visibility remain the target owner's job.
// Returned rows are request-indexed and owned by arena. Status carries only
// allocation failure; absence of a common proved owner is an empty selection.
iree_status_t loom_pipeline_continuations_select(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* workers,
    const loom_pipeline_progress_observations_t* observations,
    const loom_pipeline_progress_site_t* sites, const uint32_t* inputs,
    iree_host_size_t input_count, loom_pipeline_progress_callback_t callback,
    iree_arena_allocator_t* arena,
    loom_pipeline_continuation_t** out_selections);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_CONTINUATIONS_H_
