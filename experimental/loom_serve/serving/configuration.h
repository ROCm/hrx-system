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

// One source-defined text deployment. Entries have independent logical state
// and admission capacity; the executable supplies their shared device. Paths
// are interpreted relative to the process working directory.
typedef struct loom_serve_model_configuration_t {
  // Owned visible ASCII identifier used in JSON and HTTP header selectors.
  iree_string_view_t name;
  // Owned source package directory path.
  iree_string_view_t source;
  // Owned checkpoint file path.
  iree_string_view_t weights;
  // Owned tokenizer JSON file path.
  iree_string_view_t tokenizer;
  // Maximum simultaneous retained rows for this model.
  iree_host_size_t rows;
  // Logical context ceiling per row, not eagerly committed storage.
  iree_host_size_t context_capacity;
  // Shared token positions in this model's logical cache pool.
  iree_host_size_t pool_capacity;
  // Largest automatically specialized packed token shape.
  iree_host_size_t prefill_capacity;
  // Explicit pinned endpoint slots, independent of active rows.
  iree_host_size_t checkpoint_capacity;
  // Maximum host requests waiting for admission without a row.
  iree_host_size_t pending_requests;
  // Default output limit when a request omits max_tokens.
  iree_host_size_t max_tokens;
  // Speculative proposal depth, zero or three.
  iree_host_size_t mtp_depth;
  // Device-fed epochs per scheduling turn, one or two.
  iree_host_size_t continuation_epochs;
} loom_serve_model_configuration_t;

typedef struct loom_serve_configuration_t {
  // Allocator owning the model array and decoded strings.
  iree_allocator_t allocator;
  // Number of initialized model records.
  iree_host_size_t model_count;
  // Owned deployment records in scheduling order.
  loom_serve_model_configuration_t* models;
} loom_serve_configuration_t;

// Parses one complete JSON/JSONC {"models":[...]} catalog. Every model declares
// kind="text", name, source, weights and tokenizer. Unknown/duplicate fields,
// duplicate names and invalid native options reject before model creation.
// Model-specific geometry is validated by its source during creation. Strings
// are decoded and owned independently of input. Failure releases partial state.
iree_status_t loom_serve_configuration_initialize(
    iree_string_view_t json, loom_serve_configuration_t* out_configuration,
    iree_allocator_t host_allocator);
void loom_serve_configuration_deinitialize(
    loom_serve_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_CONFIGURATION_H_
