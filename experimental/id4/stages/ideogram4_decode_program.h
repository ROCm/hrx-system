// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DECODE_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DECODE_PROGRAM_H_

#include <stdint.h>

#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/stages/vae_program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current flattened element limit for Ideogram4 diffusion latent tensors.
#define ID4_IDEOGRAM4_DECODE_MAX_DIFFUSION_ELEMENT_COUNT 1048576u

// Static Ideogram4 decode model contract.
typedef struct id4_ideogram4_decode_model_config_t {
  // Reusable VAE decode model configuration.
  id4_vae_model_config_t vae;
} id4_ideogram4_decode_model_config_t;

// Dynamic Ideogram4 decode request dimensions.
typedef struct id4_ideogram4_decode_request_config_t {
  // Diffusion latent tensor shape in row-major WHCB order.
  id4_pipeline_program_shape_t diffusion_latent_shape;
  // VAE tiling policy for the diffusion latent.
  id4_vae_tiling_config_t vae_tiling;
} id4_ideogram4_decode_request_config_t;

// Options for authoring an Ideogram4 latent-to-image decode program.
typedef struct id4_ideogram4_decode_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Static decode model contract.
  id4_ideogram4_decode_model_config_t model;
  // Dynamic decode request dimensions.
  id4_ideogram4_decode_request_config_t request;
  // Provider source scope used when loading decode parameters.
  iree_string_view_t parameter_scope;
  // Activation storage format for internal VAE intermediates.
  id4_vae_activation_format_t vae_activation_format;
} id4_ideogram4_decode_program_options_t;

// Validates an Ideogram4 diffusion latent against the VAE public contract.
iree_status_t id4_ideogram4_decode_program_validate_diffusion_latent_shape(
    id4_ideogram4_decode_model_config_t model,
    id4_pipeline_program_shape_t diffusion_latent_shape);

// Authors the Ideogram4 latent-to-image decode program into |builder|.
iree_status_t id4_ideogram4_decode_program_author_decode(
    const id4_ideogram4_decode_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Returns the Ideogram4 decode contract using the Flux2-format VAE.
const id4_ideogram4_decode_model_config_t*
id4_ideogram4_decode_program_ideogram4_model_config(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DECODE_PROGRAM_H_
