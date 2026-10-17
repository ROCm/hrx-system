// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/kind.h"

typedef struct loom_serve_model_kind_descriptor_t {
  // Exact declaration and discovery spelling with process lifetime.
  const char* name;
  // Primary managed HTTP endpoint, or NULL for native-only inference.
  const char* endpoint;
} loom_serve_model_kind_descriptor_t;

static const loom_serve_model_kind_descriptor_t loom_serve_model_kinds[] = {
    [LOOM_SERVE_MODEL_KIND_TEXT] = {"text", "/v1/chat/completions"},
    [LOOM_SERVE_MODEL_KIND_IMAGE] = {"image", "/v1/images/generations"},
    [LOOM_SERVE_MODEL_KIND_EMBEDDING] = {"embedding", "/v1/embeddings"},
    [LOOM_SERVE_MODEL_KIND_DECISION] = {"decision", "/v1/systemone"},
    [LOOM_SERVE_MODEL_KIND_SPEECH] = {"speech", "/v1/audio/speech"},
    [LOOM_SERVE_MODEL_KIND_RECOGNITION] = {"recognition",
                                           "/v1/audio/transcriptions"},
    [LOOM_SERVE_MODEL_KIND_ACTIVITY] = {"activity", NULL},
};

bool loom_serve_model_kind_parse(iree_string_view_t value,
                                 loom_serve_model_kind_t* out_kind) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(loom_serve_model_kinds);
       ++i) {
    if (iree_string_view_equal(
            value, iree_make_cstring_view(loom_serve_model_kinds[i].name))) {
      *out_kind = (loom_serve_model_kind_t)i;
      return true;
    }
  }
  return false;
}

const char* loom_serve_model_kind_name(loom_serve_model_kind_t kind) {
  return loom_serve_model_kinds[kind].name;
}

const char* loom_serve_model_kind_endpoint(loom_serve_model_kind_t kind) {
  return loom_serve_model_kinds[kind].endpoint;
}
