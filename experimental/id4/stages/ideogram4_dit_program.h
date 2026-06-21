// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_

#include <stdint.h>

#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current image-token prelude flattened token-count limit.
#define ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT 65536u

// Current combined text+image prelude flattened token-count limit.
#define ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT 131072u

// Conditioning path selected for an Ideogram4 DiT forward request.
typedef enum id4_ideogram4_dit_conditioning_mode_e {
  // Invalid conditioning mode.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_INVALID = 0,
  // Request uses only image tokens and the unconditioned DiT weights.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED = 1,
  // Request imports Qwen condition tokens and runs the conditioned prelude.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED = 2,
} id4_ideogram4_dit_conditioning_mode_t;

// Ideogram4 DiT model dimensions used when authoring the forward program.
typedef struct id4_ideogram4_dit_model_config_t {
  // Number of transformer blocks in the DiT.
  uint32_t layer_count;
  // Channel count of each VAE latent image token.
  uint32_t input_channel_count;
  // Transformer hidden-state channel count.
  uint32_t hidden_size;
  // Feed-forward intermediate channel count.
  uint32_t intermediate_size;
  // Transformer attention head count.
  uint32_t attention_head_count;
  // AdaLN conditioning vector channel count.
  uint32_t adaln_size;
  // Qwen condition feature channel count consumed by llm_cond_norm.
  uint32_t llm_feature_count;
  // Number of image-indicator embedding rows.
  uint32_t image_indicator_count;
} id4_ideogram4_dit_model_config_t;

// Dynamic request dimensions used when authoring the DiT forward program.
typedef struct id4_ideogram4_dit_request_config_t {
  // Latent tensor shape supplied by the sampler.
  id4_pipeline_program_shape_t latent_shape;
  // Conditioning path for this DiT request.
  id4_ideogram4_dit_conditioning_mode_t conditioning_mode;
  // Number of imported Qwen condition token positions.
  uint32_t text_token_count;
} id4_ideogram4_dit_request_config_t;

// Options for authoring an Ideogram4 DiT forward semantic program.
typedef struct id4_ideogram4_dit_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Static model dimensions.
  id4_ideogram4_dit_model_config_t model;
  // Dynamic request dimensions.
  id4_ideogram4_dit_request_config_t request;
} id4_ideogram4_dit_program_options_t;

// Authors the Ideogram4 DiT forward program into |builder|.
iree_status_t id4_ideogram4_dit_program_author_forward(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Returns the Ideogram 4 DiT model configuration.
const id4_ideogram4_dit_model_config_t*
id4_ideogram4_dit_program_ideogram4_model_config(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_
