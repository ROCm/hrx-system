// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_MODEL_H_
#define EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_MODEL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable model/adapter residency with serialized reusable image storage.
// The application owns serialization; simultaneous generate calls are invalid.
typedef struct loom_serve_krea2_model_t loom_serve_krea2_model_t;

typedef struct loom_serve_krea2_model_options_t {
  // Source catalog directory, borrowed only during creation.
  iree_string_view_t source_directory;
  // Official checkpoint directory, borrowed only during creation.
  iree_string_view_t checkpoint_directory;
  // Optional adapter safetensors path, borrowed only during creation.
  iree_string_view_t adapter_path;
  // Output image height in pixels, fixed for this residency.
  uint32_t height;
  // Output image width in pixels, fixed for this residency.
  uint32_t width;
  // Retained text extent, fixed for this residency.
  uint32_t text_tokens;
} loom_serve_krea2_model_options_t;

// Loads one tokenizer and source-JIT command, streams each immutable parameter
// domain once, and allocates all device input/output/workspace backing. Failure
// releases partial ownership and leaves *out_model NULL.
iree_status_t loom_serve_krea2_model_create(
    const loom_serve_krea2_model_options_t* options,
    loom_serve_krea2_model_t** out_model, iree_allocator_t host_allocator);

// Drains accepted work before releasing buffers, commands and device services.
// No generate call or borrowed RGB view may remain active. NULL is accepted.
iree_status_t loom_serve_krea2_model_destroy(loom_serve_krea2_model_t* model);

// Prepares a fresh request, uploads once, executes the retained source command
// and waits for final RGB. There is no warm JIT, weight load or device backing
// allocation. Prompt is borrowed only during the call. Strength must be finite;
// a model without an adapter requires strength one. Request errors before
// submission leave the model reusable; execution failure is terminal for its
// owner. Every return retires accepted operations borrowing request storage.
// Success returns little-endian F32 NCHW RGB, borrowed until the next generate
// call or model destruction. Failure leaves *out_rgb empty.
iree_status_t loom_serve_krea2_model_generate(loom_serve_krea2_model_t* model,
                                              iree_string_view_t prompt,
                                              uint64_t seed, float strength,
                                              iree_const_byte_span_t* out_rgb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_MODELS_KREA2_MODEL_H_
