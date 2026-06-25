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

// Parsed high-level Ideogram 4 generation request.
typedef struct id4_ideogram4_request_t {
  // Prompt text owned by the request.
  iree_string_view_t prompt;
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

// Options controlling request-to-Qwen input lowering.
typedef struct id4_ideogram4_qwen_lowering_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Tokenizer used to encode the prompt text.
  const iree_tokenizer_t* tokenizer;
  // Parsed request to lower.
  const id4_ideogram4_request_t* request;
  // Tokenizer flags used for the prompt encode operation.
  iree_tokenizer_encode_flags_t tokenizer_flags;
  // Maximum token count accepted by the caller's planned request shape.
  uint32_t max_token_count;
} id4_ideogram4_qwen_lowering_options_t;

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

// Releases storage owned by |inputs|.
void id4_ideogram4_qwen_inputs_deinitialize(id4_ideogram4_qwen_inputs_t* inputs,
                                            iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_REQUEST_H_
