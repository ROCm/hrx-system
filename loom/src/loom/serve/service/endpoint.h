// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_ENDPOINT_H_
#define LOOM_SERVE_SERVICE_ENDPOINT_H_

#include "loom/service/kind.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical primary HTTP inference endpoint for a valid semantic role, or NULL
// for native-only inference. Management and modality controls are separate
// routes.
const char* loom_serve_service_endpoint(loom_service_kind_t kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_ENDPOINT_H_
