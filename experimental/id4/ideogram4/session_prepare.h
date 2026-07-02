// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PREPARE_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PREPARE_H_

#include "experimental/id4/ideogram4/session_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Retains a prepared generation bundle for another owner.
void id4_ideogram4_generation_bundle_retain(
    id4_ideogram4_generation_bundle_t* bundle);

// Initializes stack-backed phase-bundle state before acquiring stage bundles.
void id4_ideogram4_generation_phase_bundle_initialize(
    id4_ideogram4_generation_phase_mask_t phase_mask,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle);

// Releases any stage-bundle refs owned by a phase bundle.
iree_status_t id4_ideogram4_generation_phase_bundle_deinitialize(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle);

// Acquires a stage bundle for issue, reusing resident bundles when available.
iree_status_t id4_ideogram4_generation_acquire_stage_bundle_ref(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_stage_bundle_ref_t* out_stage_bundle_ref);

// Releases a stage-bundle ref, waiting for unissued preparation work if needed.
iree_status_t id4_ideogram4_generation_release_stage_bundle_ref(
    id4_ideogram4_generation_stage_bundle_ref_t* stage_bundle_ref);

// Queries resident stage-bundle readiness and parameter-load failures.
iree_status_t id4_ideogram4_generation_bundle_check_resident_failures(
    const id4_ideogram4_generation_bundle_t* bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Acquires all stage bundles needed by one generation issue phase.
iree_status_t id4_ideogram4_generation_prepare_phase_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_phase_bundle_t* out_phase_bundle);

// Returns the stage bundle acquired in a prepared phase bundle.
id4_pipeline_bundle_t* id4_ideogram4_generation_phase_stage_bundle(
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_PREPARE_H_
