// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_MODELS_KREA_REQUEST_H_
#define EXPERIMENTAL_LOOM_SERVE_MODELS_KREA_REQUEST_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-image options for the distilled eight-step still-image model.
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

// Immutable inputs shared by generate_image and generate_image_adapted. The
// source generates noise/tables/padding in its reflected workspace. Output and
// workspace are separate retained device allocations.
enum loom_serve_krea2_input_e {
  // Little-endian seed:u64, retained-prefix-count:u32, strength:f32.
  LOOM_SERVE_KREA2_INPUT_HEADER,
  // Prefix+prompt immediately followed by five suffix IDs, then unused zeros.
  LOOM_SERVE_KREA2_INPUT_TOKEN_IDS,
  LOOM_SERVE_KREA2_INPUT_COUNT,
};
typedef enum loom_serve_krea2_input_e loom_serve_krea2_input_t;

// Immutable bounded prefix+prompt IDs with the model's live suffix.
// Preparation has no image geometry, checkpoint or device ownership.
typedef struct loom_serve_krea2_prompt_t loom_serve_krea2_prompt_t;

typedef struct loom_serve_krea2_request_t loom_serve_krea2_request_t;

// Encodes the whole framing prefix plus text once, retaining at most
// maximum_text_tokens+29 IDs. The maximum is a supported 16-aligned text
// extent. Validates the template, padding token and retained IDs against the
// embedding vocabulary. The tokenizer and text are borrowed only for this call.
// On failure *out_prompt is NULL and partial ownership is released.
iree_status_t loom_serve_krea2_prompt_create(
    const iree_tokenizer_t* tokenizer, uint32_t maximum_text_tokens,
    iree_string_view_t text, loom_serve_krea2_prompt_t** out_prompt,
    iree_allocator_t host_allocator);

// Actual retained combined prefix+prompt count, excluding padding and suffix.
// Separately encoding the prefix does not establish a lower bound on this
// count.
uint32_t loom_serve_krea2_prompt_token_count(
    const loom_serve_krea2_prompt_t* prompt);

// Releases the prepared IDs. Materialized requests retain no references to this
// object, so it may be destroyed before their uploads are submitted. NULL is
// allowed.
void loom_serve_krea2_prompt_destroy(loom_serve_krea2_prompt_t* prompt);

// Validates external geometry and measures the exact input byte lengths. Model
// residency and request allocation share this layout; no tokenizer or device
// is needed to size the retained input buffers. Failure leaves sizes unchanged.
iree_status_t loom_serve_krea2_request_measure(
    uint32_t height, uint32_t width, uint32_t text_tokens,
    iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]);

// Validates geometry/strength and packs the header and unpadded IDs. IDs must
// fit within text_tokens+29; materialization never truncates them further. The
// prepared prompt is borrowed only for this call. The command generates all
// numerical inputs and places the suffix after the middle padding on device.
// One owned slab, with no tokenizer, checkpoint or device access. On failure
// *out_request is NULL and partial ownership is released.
iree_status_t loom_serve_krea2_request_create(
    const loom_serve_krea2_prompt_t* prompt,
    loom_serve_krea2_request_options_t options,
    loom_serve_krea2_request_t** out_request, iree_allocator_t host_allocator);

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

#endif  // EXPERIMENTAL_LOOM_SERVE_MODELS_KREA_REQUEST_H_
