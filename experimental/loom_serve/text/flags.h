// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_MODELS_TEXT_FLAGS_H_
#define EXPERIMENTAL_LOOM_SERVE_MODELS_TEXT_FLAGS_H_

#include "experimental/loom_serve/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Launch defaults supplied by the entry point, independent of explicit flags.
typedef struct loom_serve_text_flag_defaults_t {
  // Resident state slots selected by the entry point's row-count option.
  iree_host_size_t row_count;
  // Shared KV positions when --pool_capacity is omitted; zero keeps dense KV.
  iree_host_size_t pool_capacity;
  // Generate packed shapes when --epoch is omitted. Explicit pooling or MTP
  // also generates shapes because both use the packed addressing path.
  bool automatic_shapes;
} loom_serve_text_flag_defaults_t;

// Shared source, weight, and specialization flags for every text entry point.
// The caller parses ordinary IREE flags before invoking these functions.
// Explicit count excludes the generated catalog; experiment tools use it to
// require operator-selected shapes rather than changing their control arm.
iree_host_size_t loom_serve_text_explicit_shape_count_from_flags(void);
bool loom_serve_text_mtp_from_flags(void);
iree_status_t loom_serve_text_model_create_from_flags(
    const loom_serve_text_flag_defaults_t* defaults,
    loom_serve_text_model_t** out_model, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_MODELS_TEXT_FLAGS_H_
