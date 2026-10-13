// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_CONFIGURATION_H_
#define LOOM_SERVE_SERVER_CONFIGURATION_H_

#include "loom/serve/runtime/program/sources.h"

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
  // Proposed tokens after each anchor (0-7), validated by the model source.
  iree_host_size_t draft_depth;
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
} loom_serve_finite_configuration_t;

// One source-defined deployment. Entries have independent logical state and
// admission capacity; the executable supplies their device group. Authored
// paths are retained unchanged, with a separate local directory base.
typedef struct loom_serve_model_configuration_t {
  // Allocator owning decoded strings, retained when this declaration is moved.
  iree_allocator_t allocator;
  // Selects the native modality contract and the configuration union member.
  loom_serve_model_kind_t kind;
  // Owned visible ASCII identifier used in JSON and HTTP header selectors.
  iree_string_view_t name;
  // Device-group ordinal, or IREE_HOST_SIZE_MAX when the document omits it.
  // The registry replaces omission with zero only for a one-device group.
  iree_host_size_t device_index;
  // Owned program role paths, provider array/strings, and local directory base.
  loom_serve_sources_t sources;
  // Owned JSON object interpreted by the source prepare program.
  iree_string_view_t options;
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

// Parses a complete {name, definition, device?, admission?, base_directory?}
// model document. The definition owns kind, prepare/control paths, sources
// array, modality-specific inputs/capacities, and source-owned options.
// Unknown, duplicate, wrong-kind and invalid fields reject before any model
// I/O. Admission defaults to "open"; "drained" defers preparation. Device
// selection is a nonnegative group ordinal; membership is resolved separately.
// document_directory supplies the local base for file input. HTTP passes an
// empty base: relative references then require an explicit absolute
// base_directory in the document. That explicit base overrides the file base.
// All strings are copied. Failure releases partial state and zeros the output.
iree_status_t loom_serve_model_configuration_initialize(
    iree_string_view_t json, iree_string_view_t document_directory,
    loom_serve_model_configuration_t* out_model,
    iree_allocator_t host_allocator);
void loom_serve_model_configuration_deinitialize(
    loom_serve_model_configuration_t* model);

// Reads one model document and supplies its directory to the shared parser.
// This cold file-I/O entry belongs on a startup thread or preparation worker,
// never on the HTTP owner. No program or parameter payload is read here.
iree_status_t loom_serve_model_configuration_load(
    iree_string_view_t path, loom_serve_model_configuration_t* out_model,
    iree_allocator_t host_allocator);

// Resolves this deployment's device against the caller's immutable group size.
// Omission is accepted only with exactly one device. Both explicit and implied
// bindings must name an available ordinal. Does not mutate the declaration or
// access source files/devices. Call once at admission, before publishing a
// model.
iree_status_t loom_serve_model_configuration_resolve_device(
    const loom_serve_model_configuration_t* model,
    iree_host_size_t device_count, iree_host_size_t* out_device_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_CONFIGURATION_H_
