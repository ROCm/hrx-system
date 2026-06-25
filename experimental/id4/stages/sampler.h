// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_SAMPLER_H_
#define EXPERIMENTAL_ID4_STAGES_SAMPLER_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/sampler_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable reference-comparable stage name for one sampler denoise step.
#define ID4_SAMPLER_DENOISE_STAGE_NAME "sampler.denoise_step"

// Stable reference-comparable stage name for initial latent noise generation.
#define ID4_SAMPLER_NOISE_STAGE_NAME "sampler.noise"

// Options for creating the concrete sampler denoise-step stage.
typedef struct id4_sampler_denoise_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing sampler kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
} id4_sampler_denoise_stage_create_options_t;

// Stage-specific plan extension carrying sampler request dimensions.
typedef struct id4_sampler_denoise_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic request dimensions for this plan.
  id4_sampler_denoise_request_config_t request;
} id4_sampler_denoise_stage_plan_options_t;

// Options for creating the concrete sampler noise stage.
typedef struct id4_sampler_noise_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing sampler kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
} id4_sampler_noise_stage_create_options_t;

// Stage-specific plan extension carrying sampler noise request dimensions.
typedef struct id4_sampler_noise_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic request dimensions for this plan.
  id4_sampler_noise_request_config_t request;
} id4_sampler_noise_stage_plan_options_t;

// Creates a sampler denoise-step stage.
iree_status_t id4_sampler_denoise_stage_create(
    const id4_sampler_denoise_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

// Creates a sampler initial-latent noise stage.
iree_status_t id4_sampler_noise_stage_create(
    const id4_sampler_noise_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_SAMPLER_H_
