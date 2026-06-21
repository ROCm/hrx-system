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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_SAMPLER_PROGRAM_H_
