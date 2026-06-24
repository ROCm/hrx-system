// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_VAE_H_
#define EXPERIMENTAL_ID4_STAGES_VAE_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/vae_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable reference-comparable stage name for VAE decode.
#define ID4_VAE_DECODE_STAGE_NAME "vae.decode"

// Options for creating a concrete reusable VAE decode stage.
typedef struct id4_vae_stage_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Services retained by the base pipeline stage.
  id4_pipeline_stage_services_t services;
  // Loom kernel cache used when preparing VAE kernels.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter provider scope containing VAE weights; empty selects the
  // anonymous scope. The stage copies this string during creation.
  iree_string_view_t parameter_scope;
  // Static VAE implementation capabilities.
  id4_vae_model_config_t model;
  // Activation storage format selected during VAE program authoring.
  id4_vae_activation_format_t activation_format;
} id4_vae_stage_create_options_t;

// Stage-specific plan extension carrying dynamic VAE decode dimensions.
typedef struct id4_vae_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Dynamic VAE decode request dimensions and tiling policy.
  id4_vae_decode_request_config_t request;
} id4_vae_stage_plan_options_t;

// Creates a reusable VAE decode stage.
iree_status_t id4_vae_stage_create(
    const id4_vae_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_VAE_H_
