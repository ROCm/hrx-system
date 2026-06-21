// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_SAMPLER_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_SAMPLER_PROGRAM_H_

#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current sampler denoise kernel flattened element-count limit.
#define ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT 1048576u

// Dynamic request dimensions used when authoring one denoise step.
typedef struct id4_sampler_denoise_request_config_t {
  // Latent tensor shape shared by cond_out, uncond_out, x_t, and denoised.
  id4_pipeline_program_shape_t latent_shape;
} id4_sampler_denoise_request_config_t;

// Options for authoring one sampler denoise-step semantic program.
typedef struct id4_sampler_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic request dimensions.
  id4_sampler_denoise_request_config_t request;
} id4_sampler_program_options_t;

// Authors one device-side sampler denoise step into |builder|.
iree_status_t id4_sampler_program_author_denoise_step(
    const id4_sampler_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Returns the stable boundary name for the conditional model prediction.
iree_string_view_t id4_sampler_program_cond_out_boundary_name(void);

// Returns the stable boundary name for the unconditional model prediction.
iree_string_view_t id4_sampler_program_uncond_out_boundary_name(void);

// Returns the stable boundary name for the input latent.
iree_string_view_t id4_sampler_program_x_t_boundary_name(void);

// Returns the stable boundary name for sampler scaling constants.
iree_string_view_t id4_sampler_program_scalings_boundary_name(void);

// Returns the stable boundary name for sampler guidance constants.
iree_string_view_t id4_sampler_program_guidance_boundary_name(void);

// Returns the stable boundary name for the denoised output latent.
iree_string_view_t id4_sampler_program_denoised_boundary_name(void);

// Returns the diagnostic tap name for the guided prediction tensor.
iree_string_view_t id4_sampler_program_guided_pred_tap_name(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_SAMPLER_PROGRAM_H_
