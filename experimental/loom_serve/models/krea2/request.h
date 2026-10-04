// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_REQUEST_H_
#define EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_REQUEST_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold request preparation for the distilled eight-step still-image model.
// These are image/model options, not shared serving configuration.
typedef struct loom_serve_krea2_request_options_t {
  // Output pixel height, divisible by 16.
  uint32_t height;
  // Output pixel width, divisible by 16.
  uint32_t width;
  // Retained text extent, divisible by 16; includes padding and suffix.
  uint32_t text_tokens;
  // Philox4x32-10 key; not the PyTorch CPU generator's seed convention.
  uint64_t seed;
  // Finite adapter strength; ignored by the non-adapted command.
  float strength;
} loom_serve_krea2_request_options_t;

// Immutable input spans in sample_image_adapted binding order. The base command
// omits STRENGTH. Output and reflected workspace are separate device
// allocations.
enum loom_serve_krea2_input_e {
  LOOM_SERVE_KREA2_INPUT_NOISE,
  LOOM_SERVE_KREA2_INPUT_TOKEN_IDS,
  LOOM_SERVE_KREA2_INPUT_ENCODER_COSINE,
  LOOM_SERVE_KREA2_INPUT_ENCODER_SINE,
  LOOM_SERVE_KREA2_INPUT_ENCODER_MASK,
  LOOM_SERVE_KREA2_INPUT_TIMES,
  LOOM_SERVE_KREA2_INPUT_COSINE,
  LOOM_SERVE_KREA2_INPUT_SINE,
  LOOM_SERVE_KREA2_INPUT_DELTAS,
  LOOM_SERVE_KREA2_INPUT_STRENGTH,
  LOOM_SERVE_KREA2_INPUT_AFFINE,
  LOOM_SERVE_KREA2_INPUT_COUNT,
};
typedef enum loom_serve_krea2_input_e loom_serve_krea2_input_t;

typedef struct loom_serve_krea2_request_t loom_serve_krea2_request_t;

// Validates external geometry and measures the exact input byte lengths. Model
// residency and request allocation share this layout; no tokenizer or device
// is needed to size the retained input buffers. Failure leaves sizes unchanged.
iree_status_t loom_serve_krea2_request_measure(
    uint32_t height, uint32_t width, uint32_t text_tokens,
    iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]);

// Validates geometry/strength and encodes prefix+prompt with right truncation,
// middle padding and a live suffix using the model's immutable tokenizer.
// Builds the encoder/DiT rotary, shifted Euler schedule, packed BF16 normal
// noise and VAE affine in one owned slab. IDs are checked against the embedding
// vocabulary. The tokenizer is borrowed only for this call. No checkpoint or
// device access. On failure *out_request is NULL and partial ownership is
// released.
iree_status_t loom_serve_krea2_request_create(
    const iree_tokenizer_t* tokenizer,
    loom_serve_krea2_request_options_t options, iree_string_view_t prompt,
    iree_allocator_t host_allocator, loom_serve_krea2_request_t** out_request);

// Releases the slab. Any queued upload borrowing its spans must retire first,
// including accepted work preceding a later submission failure. NULL is
// allowed.
void loom_serve_krea2_request_destroy(loom_serve_krea2_request_t* request);

// Borrowed little-endian input bytes, immutable for the request's lifetime.
iree_const_byte_span_t loom_serve_krea2_request_input(
    const loom_serve_krea2_request_t* request, loom_serve_krea2_input_t input);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_REQUEST_H_
