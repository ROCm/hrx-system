// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_VAE_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_VAE_PROGRAM_H_

#include <stdint.h>

#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current decode output element-count limit for the VAE scheduling smoke.
#define ID4_VAE_DECODE_MAX_ELEMENT_COUNT 268435456u

// VAE implementation capability flags.
typedef uint32_t id4_vae_capability_flags_t;

// VAE implementation capability flag bits.
typedef enum id4_vae_capability_flag_bits_e {
  // Implementation can decode latent tensors to image tensors.
  ID4_VAE_CAPABILITY_DECODE = 1u << 0,
  // Implementation can encode image tensors to latent tensors.
  ID4_VAE_CAPABILITY_ENCODE = 1u << 1,
  // Implementation supports spatial tiling for image-sized tensors.
  ID4_VAE_CAPABILITY_SPATIAL_TILING = 1u << 2,
  // Implementation supports temporal tiling for video-shaped tensors.
  ID4_VAE_CAPABILITY_TEMPORAL_TILING = 1u << 3,
} id4_vae_capability_flag_bits_t;

// Policy used to resolve VAE spatial tiling during planning.
typedef enum id4_vae_tiling_mode_e {
  // Invalid tiling mode.
  ID4_VAE_TILING_MODE_INVALID = 0,
  // Decode the entire latent image as one logical tile.
  ID4_VAE_TILING_MODE_DISABLED = 1,
  // Use caller-provided latent tile dimensions.
  ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE = 2,
  // Derive latent tile dimensions from relative factors.
  ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE = 3,
  // Select the largest legal latent tile shape under a byte budget.
  ID4_VAE_TILING_MODE_MEMORY_BUDGET = 4,
} id4_vae_tiling_mode_t;

// VAE implementation selected by the model configuration.
typedef enum id4_vae_implementation_e {
  // Direct test implementation with no model-specific latent prelude.
  ID4_VAE_IMPLEMENTATION_DIRECT = 0,
  // Flux2 AutoEncoderKL implementation used by Ideogram 4.
  ID4_VAE_IMPLEMENTATION_FLUX2 = 1,
} id4_vae_implementation_t;

// Static VAE model and implementation capabilities.
typedef struct id4_vae_model_config_t {
  // Latent-to-image scale factor along the width axis.
  uint32_t scale_x;
  // Latent-to-image scale factor along the height axis.
  uint32_t scale_y;
  // Latent-to-media scale factor along the temporal axis.
  uint32_t scale_t;
  // Channel count in latent tensors accepted by the VAE.
  uint32_t latent_channel_count;
  // Channel count in decoded image tensors produced by the VAE.
  uint32_t decoded_channel_count;
  // Minimum legal latent tile width.
  uint32_t min_tile_size_x;
  // Minimum legal latent tile height.
  uint32_t min_tile_size_y;
  // Default latent tile width used by explicit-size policy.
  uint32_t default_tile_size_x;
  // Default latent tile height used by explicit-size policy.
  uint32_t default_tile_size_y;
  // Maximum legal fractional tile overlap.
  float max_overlap;
  // Capability bits supported by this VAE implementation.
  id4_vae_capability_flags_t capabilities;
  // Concrete VAE implementation selected by this model.
  id4_vae_implementation_t implementation;
} id4_vae_model_config_t;

// User or model policy for resolving VAE tiling.
typedef struct id4_vae_tiling_config_t {
  // Tiling policy selected for this request.
  id4_vae_tiling_mode_t mode;
  // Requested latent tile width for explicit-size policy.
  uint32_t tile_size_x;
  // Requested latent tile height for explicit-size policy.
  uint32_t tile_size_y;
  // Relative latent width factor or tile-count hint.
  float relative_size_x;
  // Relative latent height factor or tile-count hint.
  float relative_size_y;
  // Requested fractional tile overlap.
  float overlap;
  // Maximum transient bytes for memory-budget policy.
  iree_device_size_t memory_budget;
} id4_vae_tiling_config_t;

// Dynamic VAE decode request dimensions and tiling policy.
typedef struct id4_vae_decode_request_config_t {
  // Latent tensor shape in row-major WHCB order.
  id4_pipeline_program_shape_t latent_shape;
  // Tiling policy for this decode request.
  id4_vae_tiling_config_t tiling;
} id4_vae_decode_request_config_t;

// Concrete VAE tiling facts derived during planning.
typedef struct id4_vae_decode_tiling_plan_t {
  // Batch count in the latent tensor.
  uint32_t batch_count;
  // Latent tensor height.
  uint32_t latent_height;
  // Latent tensor width.
  uint32_t latent_width;
  // Latent tensor channel count.
  uint32_t latent_channel_count;
  // Decoded image height.
  uint32_t decoded_height;
  // Decoded image width.
  uint32_t decoded_width;
  // Decoded image channel count.
  uint32_t decoded_channel_count;
  // Resolved latent tile width.
  uint32_t tile_size_x;
  // Resolved latent tile height.
  uint32_t tile_size_y;
  // Number of tiles covering the latent width.
  uint32_t tile_count_x;
  // Number of tiles covering the latent height.
  uint32_t tile_count_y;
  // Resolved fractional width-axis tile overlap.
  float overlap_x;
  // Resolved fractional height-axis tile overlap.
  float overlap_y;
  // Resolved overlap represented in thousandths for kernel config.
  uint32_t overlap_milli;
  // Dense decoded output element count.
  uint64_t decoded_element_count;
  // Dense latent input element count.
  uint64_t latent_element_count;
  // Dense decoded tile scratch element count.
  uint64_t tile_element_count;
  // Estimated scratch bytes represented by the current placeholder program.
  iree_device_size_t estimated_transient_peak;
} id4_vae_decode_tiling_plan_t;

// Options for authoring a VAE decode semantic program.
typedef struct id4_vae_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Static VAE implementation capabilities.
  id4_vae_model_config_t model;
  // Dynamic decode request dimensions and tiling policy.
  id4_vae_decode_request_config_t request;
} id4_vae_program_options_t;

// Resolves |request| into concrete VAE decode tiling facts.
iree_status_t id4_vae_program_resolve_decode_tiling(
    id4_vae_model_config_t model, id4_vae_decode_request_config_t request,
    id4_vae_decode_tiling_plan_t* out_tiling_plan);

// Authors the VAE decode program into |builder|.
iree_status_t id4_vae_program_author_decode(
    const id4_vae_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Authors VAE decode from an initialized in-program latent tensor.
iree_status_t id4_vae_program_author_decode_from_tensor(
    const id4_vae_program_options_t* options,
    id4_pipeline_program_tensor_t latent, iree_string_view_t decoded_image_name,
    id4_pipeline_program_builder_t* builder,
    id4_pipeline_program_tensor_t* out_decoded_image);

// Returns the Flux2-format VAE configuration used by Ideogram 4.
const id4_vae_model_config_t* id4_vae_program_flux2_model_config(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_VAE_PROGRAM_H_
