// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_KIND_H_
#define LOOM_SERVICE_KIND_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantic role, retained from declaration through prepared service and
// retirement. Source capabilities may provide additional native outputs.
typedef enum loom_service_kind_e {
  LOOM_SERVICE_KIND_TEXT = 1,
  LOOM_SERVICE_KIND_IMAGE = 2,
  LOOM_SERVICE_KIND_EMBEDDING = 3,
  LOOM_SERVICE_KIND_DECISION = 4,
  LOOM_SERVICE_KIND_SPEECH = 5,
  LOOM_SERVICE_KIND_RECOGNITION = 6,
  LOOM_SERVICE_KIND_ACTIVITY = 7,
  LOOM_SERVICE_KIND_SEQUENCE = 8,
} loom_service_kind_t;

// Parses an exact deployment spelling. Failure leaves out_kind unchanged.
bool loom_service_kind_parse(iree_string_view_t value,
                             loom_service_kind_t* out_kind);

// Process-lifetime spelling of a valid kind.
const char* loom_service_kind_name(loom_service_kind_t kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_KIND_H_
