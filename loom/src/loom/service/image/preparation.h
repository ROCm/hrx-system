// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_IMAGE_PREPARATION_H_
#define LOOM_SERVICE_IMAGE_PREPARATION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copied source-selected cohort facts, valid after successful batch
// preparation. No pointer into the source VM or its request payload escapes in
// this value.
typedef struct loom_service_image_prepared_t {
  // Retained command index selected by the source program.
  uint64_t stage;
  // Source-declared stage tag; its units and interpretation are model-defined.
  int64_t tag;
  // Bytes of opaque prepared input consumed by this invocation.
  uint64_t input_bytes;
} loom_service_image_prepared_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_IMAGE_PREPARATION_H_
