// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_REQUEST_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_REQUEST_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decoded single-image input, independent of tokenizers and model arithmetic.
typedef struct loom_serve_image_request_t {
  // Owned decoded UTF-8 prompt, including any embedded NUL bytes.
  iree_string_builder_t prompt;
  // Unsigned native noise key; zero when omitted.
  uint64_t seed;
  // Finite adapter multiplier; one when omitted.
  float strength;
} loom_serve_image_request_t;

// Decodes one bounded JSON/JSONC body for a fixed model and pixel geometry.
// Requires prompt; optional model/size must exactly match the supplied literal
// names. Only n=1 and response_format="b64_json" are supported. Seed accepts an
// unsigned integer or literal decimal string. Unknown, duplicate, wrongly
// typed and trailing fields fail. No body storage is borrowed after return.
// Failure releases partial ownership and leaves a deinitializable empty value.
iree_status_t loom_serve_image_request_initialize(
    iree_string_view_t body, iree_string_view_t model, uint32_t width,
    uint32_t height, loom_serve_image_request_t* out_request,
    iree_allocator_t host_allocator);

// Releases decoded prompt storage. An empty value is accepted.
void loom_serve_image_request_deinitialize(loom_serve_image_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_REQUEST_H_
