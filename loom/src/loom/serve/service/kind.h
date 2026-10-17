// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_KIND_H_
#define LOOM_SERVE_SERVICE_KIND_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Deployment role, retained from declaration through prepared service and
// retirement. Source capabilities may provide additional native outputs.
typedef enum loom_serve_model_kind_e {
  LOOM_SERVE_MODEL_KIND_TEXT = 1,
  LOOM_SERVE_MODEL_KIND_IMAGE = 2,
  LOOM_SERVE_MODEL_KIND_EMBEDDING = 3,
  LOOM_SERVE_MODEL_KIND_DECISION = 4,
  LOOM_SERVE_MODEL_KIND_SPEECH = 5,
  LOOM_SERVE_MODEL_KIND_RECOGNITION = 6,
  LOOM_SERVE_MODEL_KIND_ACTIVITY = 7,
} loom_serve_model_kind_t;

// Parses an exact deployment spelling. Failure leaves out_kind unchanged.
bool loom_serve_model_kind_parse(iree_string_view_t value,
                                 loom_serve_model_kind_t* out_kind);

// Process-lifetime spelling of a valid kind.
const char* loom_serve_model_kind_name(loom_serve_model_kind_t kind);

// Canonical primary HTTP inference endpoint, or NULL for a native-only role.
// Management, tokenizer utilities, and modality controls are separate routes.
const char* loom_serve_model_kind_endpoint(loom_serve_model_kind_t kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_KIND_H_
