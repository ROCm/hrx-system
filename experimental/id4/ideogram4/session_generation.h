// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_GENERATION_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_GENERATION_H_

#include "experimental/id4/ideogram4/session_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  // Number of boundary aliases connecting coarse generation stages.
  ID4_IDEOGRAM4_GENERATION_BOUNDARY_ALIAS_COUNT = 8,
};

// Static descriptors for every coarse generation stage ordinal.
extern const id4_ideogram4_generation_stage_descriptor_t
    id4_ideogram4_generation_stage_descriptors
        [ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];

// Static descriptors for every public generation issue phase.
extern const id4_ideogram4_generation_phase_descriptor_t
    id4_ideogram4_generation_phase_descriptors
        [ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];

// Static producer/consumer aliases between stage boundary tensors.
extern const id4_ideogram4_generation_boundary_alias_t
    id4_ideogram4_generation_boundary_aliases
        [ID4_IDEOGRAM4_GENERATION_BOUNDARY_ALIAS_COUNT];

// Returns the descriptor for |ordinal|, or NULL when invalid.
const id4_ideogram4_generation_stage_descriptor_t*
id4_ideogram4_generation_stage_descriptor(
    id4_ideogram4_generation_stage_ordinal_t ordinal);

// Returns the descriptor for |stage_key|, or NULL when unknown.
const id4_ideogram4_generation_stage_descriptor_t*
id4_ideogram4_generation_stage_descriptor_for_key(iree_string_view_t stage_key);

// Returns the planned stage selected by |ordinal|, or NULL when absent.
const id4_pipeline_plan_t* id4_ideogram4_generation_stage_plan(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t ordinal);

// Returns the diffusion latent shape implied by a generation request.
id4_pipeline_program_shape_t
id4_ideogram4_generation_request_diffusion_latent_shape(
    const id4_ideogram4_request_t* request);

// Validates the request shape and model compatibility for generation.
iree_status_t id4_ideogram4_validate_generation_request(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_request_t* request);

// Validates a public generation phase mask.
iree_status_t id4_ideogram4_validate_generation_phase_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask);

// Validates that a phase mask identifies exactly one public generation phase.
iree_status_t id4_ideogram4_validate_single_generation_phase_mask(
    id4_ideogram4_generation_phase_mask_t phase_mask);

// Validates a public generation resident-stage mask.
iree_status_t id4_ideogram4_validate_generation_resident_stage_mask(
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask);

// Validates residency options used while preparing a generation.
iree_status_t id4_ideogram4_validate_generation_prepare_residency_options(
    id4_ideogram4_generation_residency_mode_t mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask);

// Validates residency state stored by a prepared generation bundle.
iree_status_t id4_ideogram4_validate_generation_bundle_residency(
    id4_ideogram4_generation_residency_mode_t mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_GENERATION_H_
