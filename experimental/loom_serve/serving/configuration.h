// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_CONFIGURATION_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_CONFIGURATION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_model_kind_e {
  LOOM_SERVE_MODEL_KIND_TEXT = 1,
  LOOM_SERVE_MODEL_KIND_IMAGE = 2,
  LOOM_SERVE_MODEL_KIND_EMBEDDING = 3,
  LOOM_SERVE_MODEL_KIND_DECISION = 4,
} loom_serve_model_kind_t;

typedef struct loom_serve_text_configuration_t {
  // Owned checkpoint file path.
  iree_string_view_t weights;
  // Owned tokenizer JSON file path.
  iree_string_view_t tokenizer;
  // Maximum simultaneous execution rows for this model.
  iree_host_size_t rows;
  // Successful overtakes per blocked oldest text request; zero is FIFO.
  iree_host_size_t admission_overtakes;
  // Logical context ceiling per row, not eagerly committed storage.
  iree_host_size_t context_capacity;
  // Shared token positions in this model's logical cache pool.
  iree_host_size_t pool_capacity;
  // Largest automatically specialized packed token shape.
  iree_host_size_t prefill_capacity;
  // Explicit pinned endpoint slots, independent of active rows.
  iree_host_size_t checkpoint_capacity;
  // Evictable idle endpoints retained beyond execution rows; zero disables.
  // Native storage includes one additional endpoint for an admission exchange.
  iree_host_size_t session_cache_capacity;
  // Exact intermediate token-prefix endpoints shared across requests. Zero
  // disables capture; these evictable slots are independent of other caches.
  iree_host_size_t prefix_cache_capacity;
  // Default output limit when a request omits max_tokens.
  iree_host_size_t max_tokens;
  // Speculative proposal depth, zero or three.
  iree_host_size_t mtp_depth;
  // Device-fed epochs per scheduling turn, one or two.
  iree_host_size_t continuation_epochs;
} loom_serve_text_configuration_t;

typedef struct loom_serve_image_configuration_t {
  // Owned checkpoint asset directory, interpreted by the source bootstrap.
  iree_string_view_t checkpoint;
  // Optional owned adapter asset path; empty means no adapter.
  iree_string_view_t adapter;
  // Fixed output pixel width, validated for the architecture by its source.
  uint32_t width;
  // Fixed output pixel height, validated for the architecture by its source.
  uint32_t height;
  // Text capacity requested from the source; its meaning is model-defined.
  uint32_t text_tokens;
  // Soft command-count target for barrier-aligned image submissions; zero is
  // one whole-program submission. This is not a wall-clock latency bound.
  uint32_t command_quantum;
} loom_serve_image_configuration_t;

typedef struct loom_serve_finite_configuration_t {
  // Owned asset directory passed to source prepare.
  iree_string_view_t assets;
  // Owned JSON object interpreted by source prepare, not the native scheduler.
  iree_string_view_t options;
} loom_serve_finite_configuration_t;

// One source-defined deployment. Entries have independent logical state and
// admission capacity; the executable supplies their shared device. Paths are
// interpreted relative to the process working directory.
typedef struct loom_serve_model_configuration_t {
  // Allocator owning decoded strings, retained when this declaration is moved.
  iree_allocator_t allocator;
  // Selects the native modality contract and the configuration union member.
  loom_serve_model_kind_t kind;
  // Owned visible ASCII identifier used in JSON and HTTP header selectors.
  iree_string_view_t name;
  // Owned source package directory path.
  iree_string_view_t source;
  // Bounded host request credit. Text/image exclude the active invocation;
  // finite vector services include all requests in the active cohort.
  iree_host_size_t pending_requests;
  // Register metadata without preparing the model until admission opens.
  bool start_drained;
  // Only the member selected by kind is initialized and owned.
  union {
    // Autoregressive text deployment capacity and checkpoint inputs.
    loom_serve_text_configuration_t text;
    // Finite image deployment geometry and checkpoint inputs.
    loom_serve_image_configuration_t image;
    // Finite vector deployment assets and source-defined policy.
    loom_serve_finite_configuration_t finite;
  };
} loom_serve_model_configuration_t;

typedef struct loom_serve_configuration_t {
  // Allocator owning the model array and decoded strings.
  iree_allocator_t allocator;
  // Number of initialized model records.
  iree_host_size_t model_count;
  // Owned deployment declarations; discovery order is not a priority contract.
  loom_serve_model_configuration_t* models;
} loom_serve_configuration_t;

// Parses one complete model object with the catalog's schema. Optional
// admission="open" (default) or "drained" selects initial admission only.
// Strings are copied; failure releases partial state and zeros the output.
iree_status_t loom_serve_model_configuration_initialize(
    iree_string_view_t json, loom_serve_model_configuration_t* out_model,
    iree_allocator_t host_allocator);
void loom_serve_model_configuration_deinitialize(
    loom_serve_model_configuration_t* model);

// Parses one complete JSON/JSONC {"models":[...]} catalog; the array may be
// empty. Every model declares kind="text", "image", "embedding" or "decision",
// name and source. Text
// entries require weights and tokenizer; image entries require a checkpoint
// directory. Embedding/decision entries require assets and accept a
// source-owned options object. Unknown/duplicate fields, fields from another
// modality, duplicate names and invalid native options reject before model
// creation. Model-specific geometry is validated by its source during creation.
// Strings are decoded and owned independently of input. Failure releases
// partial state.
iree_status_t loom_serve_configuration_initialize(
    iree_string_view_t json, loom_serve_configuration_t* out_configuration,
    iree_allocator_t host_allocator);
void loom_serve_configuration_deinitialize(
    loom_serve_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_CONFIGURATION_H_
