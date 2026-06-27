// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_REQUEST_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_REQUEST_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parsed request behavior flags.
typedef uint32_t id4_ideogram4_request_flags_t;

// Parsed request behavior flag bits.
typedef enum id4_ideogram4_request_flag_bits_e {
  // Request includes full-generation dimensions and sampling parameters.
  ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION = 1u << 0,
} id4_ideogram4_request_flag_bits_t;

// Full-generation dimensions and sampling parameters parsed from JSON.
typedef struct id4_ideogram4_request_generation_t {
  // Diffusion latent tensor width in latent-token positions.
  uint32_t latent_width;
  // Diffusion latent tensor height in latent-token positions.
  uint32_t latent_height;
  // Number of denoise steps in the host-controlled sampler loop.
  uint32_t denoise_step_count;
  // Request seed used by the deterministic noise producer.
  uint64_t seed;
  // Text guidance scale consumed by classifier-free guidance.
  float guidance_scale;
} id4_ideogram4_request_generation_t;

// Parsed high-level Ideogram 4 generation request.
typedef struct id4_ideogram4_request_t {
  // Parsed request behavior flags.
  id4_ideogram4_request_flags_t flags;
  // Canonical prompt payload text owned by the request.
  iree_string_view_t prompt_payload;
  // Qwen chat-wrapped prompt text owned by the request.
  iree_string_view_t qwen_prompt;
  // Full-generation dimensions and sampling parameters when present.
  id4_ideogram4_request_generation_t generation;
} id4_ideogram4_request_t;

// Host-side Qwen3-VL input tensors lowered from a generation request.
typedef struct id4_ideogram4_qwen_inputs_t {
  // Number of token positions in every tensor.
  uint32_t token_count;
  // Rank-1 I32 token ID tensor with token_count elements.
  iree_tokenizer_token_id_t* token_ids;
  // Rank-2 F32 additive attention mask with token_count x token_count elements.
  float* attention_mask;
  // Rank-1 F32 token weighting tensor with token_count elements.
  float* token_weights;
} id4_ideogram4_qwen_inputs_t;

// Number of F32 denoise coefficients consumed by the sampler stage.
#define ID4_IDEOGRAM4_DENOISE_SCALING_COUNT 3

// Number of F32 sigma values consumed by one Euler sampler step.
#define ID4_IDEOGRAM4_DENOISE_SIGMA_COUNT 2

// Number of F32 guidance values consumed by the sampler stage.
#define ID4_IDEOGRAM4_GUIDANCE_VALUE_COUNT 3

// Host-side scalar tensors lowered for one denoise step.
typedef struct id4_ideogram4_denoise_step_t {
  // DiT timestep scalar consumed by timestep embedding.
  float timestep;
  // Sampler denoise coefficient vector in `{c_skip, c_out, c_in}` order.
  float scalings[ID4_IDEOGRAM4_DENOISE_SCALING_COUNT];
  // Euler sigma vector in `{sigma, next_sigma}` order.
  float sigmas[ID4_IDEOGRAM4_DENOISE_SIGMA_COUNT];
  // Sampler guidance vector; element 0 is the text CFG scale.
  float guidance[ID4_IDEOGRAM4_GUIDANCE_VALUE_COUNT];
} id4_ideogram4_denoise_step_t;

// Host-side denoise schedule lowered from generation metadata.
typedef struct id4_ideogram4_denoise_schedule_t {
  // Number of denoise steps in |steps|.
  uint32_t step_count;
  // Heap-allocated step table with |step_count| entries.
  id4_ideogram4_denoise_step_t* steps;
} id4_ideogram4_denoise_schedule_t;

// Host-side request metadata tensors for one DiT branch.
typedef struct id4_ideogram4_dit_branch_inputs_t {
  // Number of token positions represented by every branch tensor.
  uint32_t token_count;
  // Rank-2 I32 image-indicator tensor with token_count x 1 elements.
  int32_t* image_indicator;
  // Byte length of |image_indicator|.
  iree_host_size_t image_indicator_byte_length;
  // Packed rank-4 F32 rotary tensor with 2 x 2 x head_size/2 x token_count
  // elements.
  float* position_embedding;
  // Byte length of |position_embedding|.
  iree_host_size_t position_embedding_byte_length;
} id4_ideogram4_dit_branch_inputs_t;

// Host-side DiT metadata tensors lowered from generation dimensions.
typedef struct id4_ideogram4_dit_inputs_t {
  // Number of text conditioning token positions in the conditioned branch.
  uint32_t text_token_count;
  // Number of image latent token positions in both branches.
  uint32_t image_token_count;
  // Metadata tensors for the conditioned branch.
  id4_ideogram4_dit_branch_inputs_t conditioned;
  // Metadata tensors for the unconditioned branch.
  id4_ideogram4_dit_branch_inputs_t unconditioned;
} id4_ideogram4_dit_inputs_t;

// Options controlling request-to-Qwen input lowering.
typedef struct id4_ideogram4_qwen_lowering_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Tokenizer used to encode the Qwen chat-wrapped prompt text.
  const iree_tokenizer_t* tokenizer;
  // Parsed request to lower.
  const id4_ideogram4_request_t* request;
  // Tokenizer flags used for the prompt encode operation.
  iree_tokenizer_encode_flags_t tokenizer_flags;
  // Maximum token count accepted by the caller's planned request shape.
  uint32_t max_token_count;
} id4_ideogram4_qwen_lowering_options_t;

// Options controlling request-to-DiT metadata lowering.
typedef struct id4_ideogram4_dit_lowering_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Full-generation dimensions and sampling metadata.
  const id4_ideogram4_request_generation_t* generation;
  // Number of Qwen text conditioning token positions.
  uint32_t text_token_count;
  // DiT attention head channel count used by MRoPE.
  uint32_t attention_head_size;
} id4_ideogram4_dit_lowering_options_t;

// Parses a strict Ideogram 4 request JSON object.
iree_status_t id4_ideogram4_request_parse_json(
    iree_string_view_t json, iree_allocator_t host_allocator,
    id4_ideogram4_request_t* out_request);

// Releases storage owned by |request|.
void id4_ideogram4_request_deinitialize(id4_ideogram4_request_t* request,
                                        iree_allocator_t host_allocator);

// Lowers a parsed request into host-side Qwen3-VL input tensors.
iree_status_t id4_ideogram4_request_lower_qwen_inputs(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_qwen_inputs_t* out_inputs);

// Counts the Qwen3-VL prompt tokens produced by |options| without materializing
// request tensor payloads.
iree_status_t id4_ideogram4_request_count_qwen_tokens(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, uint32_t* out_token_count);

// Lowers generation metadata into the scalar tensors used by each denoise step.
iree_status_t id4_ideogram4_request_generation_lower_denoise_schedule(
    const id4_ideogram4_request_generation_t* generation,
    iree_allocator_t host_allocator,
    id4_ideogram4_denoise_schedule_t* out_schedule);

// Lowers generation metadata into the host-side DiT branch metadata tensors.
iree_status_t id4_ideogram4_request_lower_dit_inputs(
    const id4_ideogram4_dit_lowering_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_dit_inputs_t* out_inputs);

// Releases storage owned by |inputs|.
void id4_ideogram4_qwen_inputs_deinitialize(id4_ideogram4_qwen_inputs_t* inputs,
                                            iree_allocator_t host_allocator);

// Releases storage owned by |inputs|.
void id4_ideogram4_dit_inputs_deinitialize(id4_ideogram4_dit_inputs_t* inputs,
                                           iree_allocator_t host_allocator);

// Releases storage owned by |schedule|.
void id4_ideogram4_denoise_schedule_deinitialize(
    id4_ideogram4_denoise_schedule_t* schedule,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_REQUEST_H_
